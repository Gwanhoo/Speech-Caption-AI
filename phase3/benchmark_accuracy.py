"""Reproducible 3s/2s GPU audit; never treats model output as ground truth.

References are optional JSON {sample_id: {text: str, kind: 'human_verified'}}.
Only single_A/single_B and silence have a single unambiguous reference stream.
Saved FLOAT windows and raw hypotheses can be replayed without GPU inference.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import importlib.util
import importlib.metadata
import json
from math import gcd
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase4"))
from stt_context import LIVE_WHISPER_OPTIONS, transcribe_base
from subtitle_assembler import (
    SubtitleAssembler,
    SpeakerSubtitleState,
    normalize_for_matching,
)
from speaker_tracking import PersistentSpeakerTracker
from secondary_leakage_diagnostics import (
    build_window_diagnostic,
    secondary_transcript_suppression_reason,
)
from benchmark_stt_context import edit_distance, adjacent_repeated_token_count

SR = 16000


def text_scores(reference: str, hypothesis: str) -> dict:
    ref, hyp = normalize_for_matching(reference), normalize_for_matching(hypothesis)
    # Backtrace for explicit deletions/insertions; these are edit counts, not a
    # semantic detector of hallucinations or omissions.
    table = [[(0, 0, 0, 0)] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(1, len(ref) + 1):
        table[i][0] = (i, 0, i, 0)
    for j in range(1, len(hyp) + 1):
        table[0][j] = (j, 0, 0, j)
    for i, a in enumerate(ref, 1):
        for j, b in enumerate(hyp, 1):
            d, s, deletion, ins = table[i - 1][j - 1]
            choices = [(d + (a != b), s + (a != b), deletion, ins)]
            d, s, deletion, ins = table[i - 1][j]
            choices.append((d + 1, s, deletion + 1, ins))
            d, s, deletion, ins = table[i][j - 1]
            choices.append((d + 1, s, deletion, ins + 1))
            table[i][j] = min(choices, key=lambda x: x[0])
    distance, substitutions, deletions, insertions = table[-1][-1]
    return {
        "cer": distance / len(ref) if ref else None,
        "reference_characters": len(ref),
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "wer": (
            edit_distance(reference.split(), hypothesis.split())
            / len(reference.split())
            if reference.split()
            else None
        ),
        "silence_output_characters": len(hyp) if not ref else None,
    }


def levels(audio):
    return {
        "rms": float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))),
        "peak": float(np.max(np.abs(audio))),
        "samples_at_or_above_one": int(np.sum(np.abs(audio) >= 1)),
        "finite": bool(np.isfinite(audio).all()),
    }


def replay(records, assembler_class=SubtitleAssembler):
    assemblers = [assembler_class(i) for i in (0, 1)]
    states = [SpeakerSubtitleState(i) for i in (0, 1)]
    events, assembly_events = [], []
    for row in records:
        for i, slot in enumerate(row["slots"]):
            assembly = assemblers[i].process(row["window"], slot["text"])
            assembly_events.append(assembly.to_dict())
            for event in states[i].process(
                row["window"],
                assembly.utterance_hypothesis,
                slot["active"],
                row["end"],
                confirmed_prefix_length=getattr(
                    assembly, "confirmed_prefix_length", None
                ),
            ):
                events.append(event.to_dict())
                if event.status == "final":
                    assemblers[i].reset_utterance()
    for state in states:
        event = state.flush(records[-1]["window"], records[-1]["end"])
        if event:
            events.append(event.to_dict())
    return {
        "transcripts": [" ".join(s.final_segments) for s in states],
        "events": events,
        "assembly": assembly_events,
        "adjacent_repeated_tokens": sum(
            adjacent_repeated_token_count(" ".join(s.final_segments)) for s in states
        ),
    }


def distribution(values):
    return (
        {
            "mean": float(np.mean(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(max(values)),
            "count": len(values),
        }
        if values
        else None
    )


def load_audio(path, seconds):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    audio = resample_poly(audio, SR // gcd(sr, SR), sr // gcd(sr, SR)).astype(
        np.float32
    )
    return np.ascontiguousarray(audio[: round(seconds * SR)])


def prepare(args):
    from silero_vad import load_silero_vad
    from validate_end_to_end_gpu import (
        load_separator,
        separate_amp,
        detect_speech_activity,
    )
    from separation_recovery import (
        finite_audio_stats,
        is_pre_separation_silence,
        silent_separation_output,
    )

    separator, device = load_separator()
    vad = load_silero_vad()
    tracks = {
        name: load_audio(ROOT / f"phase3/test_audio/{name}.wav", args.seconds)
        for name in ["A", "B"]
    }
    length = min(map(len, tracks.values()))
    samples = {
        "single_A": tracks["A"],
        "single_B": tracks["B"],
        "mixture": (tracks["A"][:length] + tracks["B"][:length]) * 0.5,
        "silence": np.zeros(5 * SR, dtype=np.float32),
        "noise": np.random.default_rng(42).normal(0, 0.002, 5 * SR).astype(np.float32),
    }
    prepared = {}
    for name, audio in samples.items():
        tracker = PersistentSpeakerTracker(overlap_samples=SR)
        rows = []
        # Match production complete-window schedule exactly; do not add a
        # specially shifted final window (which would increase the overlap).
        for index, start in enumerate(range(0, len(audio) - 3 * SR + 1, 2 * SR)):
            clip = audio[start : start + 3 * SR]
            silence = is_pre_separation_silence(finite_audio_stats(clip))
            result = None if silence else separate_amp(separator, device, clip)
            output = silent_separation_output(len(clip)) if silence else result.output
            raw = tuple(np.ascontiguousarray(output[i, 0, :]) for i in (0, 1))
            assignment = tracker.assign(
                window=index,
                raw_speakers=raw,
                mixture=clip,
                pre_separation_silence=silence,
            )
            arrays = [clip, *assignment.speakers]
            paths = []
            for slot, array in enumerate(arrays):
                path = args.output.parent / "audio" / f"{name}_{index:03d}_{slot}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, array, SR, subtype="FLOAT")
                paths.append(str(path.relative_to(ROOT)))
            activities = [detect_speech_activity(vad, array, 200) for array in arrays]
            corrs = [
                (
                    float(abs(np.corrcoef(clip, array)[0, 1]))
                    if np.std(clip) > 1e-8 and np.std(array) > 1e-8
                    else None
                )
                for array in arrays[1:]
            ]
            row = {
                "window": index,
                "start": start / SR,
                "end": start / SR + 3,
                "paths": paths,
                "vad": activities,
                "levels": [levels(a) for a in arrays],
                "source_correlation": corrs,
                "assignment": assignment.diagnostic,
                "separation_seconds": 0 if result is None else result.total_seconds,
                "fp32_fallback": False if result is None else result.fallback_attempted,
            }
            rows.append(row)
            print("PREPARED", name, index, flush=True)
        prepared[name] = {
            "source_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
            "windows": rows,
            "evaluated_seconds": rows[-1]["end"],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = args.output.parent / "prepared.json"
    cache.write_text(json.dumps(prepared, ensure_ascii=False, indent=2))
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["base", "small", "medium", "large-v3-turbo", "large-v3"],
    )
    parser.add_argument("--seconds", type=int, default=15)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "phase3/output/accuracy_audit/benchmark.json",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--replay-only",
        action="store_true",
        help="Reconcile saved raw hypotheses without running GPU models",
    )
    parser.add_argument("--references", type=Path)
    parser.add_argument("--before-assembler", type=Path)
    args = parser.parse_args()
    if args.seconds < 3:
        parser.error("--seconds must be >=3")
    if args.replay_only:
        report = json.loads(args.output.read_text())
        for model in report["models"]:
            for sample in model["samples"]:
                sample["after"] = replay(sample["windows"])
                if sample["sample"] == "noise":
                    sample["reference"] = {"text": "", "kind": "known_synthetic_noise"}
                    sample["scores"]["before"] = text_scores(
                        "", " ".join(sample["before"]["transcripts"])
                    )
                ref = sample["reference"]
                sample["scores"]["after"] = (
                    text_scores(ref["text"], " ".join(sample["after"]["transcripts"]))
                    if ref
                    else None
                )
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        return
    cache = args.output.parent / "prepared.json"
    prepared = json.loads(cache.read_text()) if cache.exists() else prepare(args)
    for name in ("A", "B"):
        requested = load_audio(ROOT / f"phase3/test_audio/{name}.wav", args.seconds)
        if (
            hashlib.sha256(requested.tobytes()).hexdigest()
            != prepared[f"single_{name}"]["source_sha256"]
        ):
            raise ValueError(
                "Cached audio differs from requested input/duration; use a new --output directory"
            )
    if args.prepare_only:
        return
    refs = json.loads(args.references.read_text()) if args.references else {}
    before = SubtitleAssembler
    if args.before_assembler:
        spec = importlib.util.spec_from_file_location(
            "assembler_before", args.before_assembler
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        before = module.SubtitleAssembler
    from faster_whisper import WhisperModel
    from faster_whisper.utils import available_models
    from validate_end_to_end_gpu import MemoryMonitor, query_gpu_memory

    report = {
        "options": LIVE_WHISPER_OPTIONS,
        "window": 3,
        "stride": 2,
        "reference_note": "No automatic pseudo-reference; missing human reference yields null quality score.",
        "versions": {
            n: importlib.metadata.version(n)
            for n in [
                "faster-whisper",
                "ctranslate2",
                "torch",
                "silero-vad",
                "clearvoice",
            ]
        },
        "prepared": prepared,
        "models": [],
    }
    for name in args.models:
        if name not in available_models():
            raise ValueError(f"Unsupported model: {name}")
        print("MODEL", name, flush=True)
        model = WhisperModel(
            name,
            device="cuda",
            compute_type="float16",
            download_root=str(ROOT / "checkpoints/faster-whisper"),
        )
        transcribe_base(model, np.zeros(SR, dtype=np.float32))
        monitor = MemoryMonitor()
        monitor.start()
        model_record = {
            "model": name,
            "memory_after_load": query_gpu_memory(),
            "samples": [],
        }
        try:
            for sample, prepared_sample in prepared.items():
                routes = {"direct": [], "separated": []}
                all_calls = []
                for window in prepared_sample["windows"]:
                    arrays = [
                        sf.read(ROOT / p, dtype="float32")[0] for p in window["paths"]
                    ]
                    calls = []
                    for array, activity in zip(arrays, window["vad"]):
                        stt = (
                            transcribe_base(model, array)
                            if activity["speech_detected"]
                            else None
                        )
                        calls.append(
                            {
                                "text": stt.text if stt else "",
                                "active": activity["speech_detected"],
                                "stt": asdict(stt) if stt else None,
                            }
                        )
                    diagnostic = build_window_diagnostic(
                        speaker_0=arrays[1],
                        speaker_1=arrays[2],
                        vad_results=window["vad"][1:],
                        transcripts=[c["text"] for c in calls[1:]],
                    )
                    reason = secondary_transcript_suppression_reason(
                        diagnostic, calls[2]["text"]
                    )
                    separated_calls = [dict(c) for c in calls[1:]]
                    if reason:
                        separated_calls[1]["text"] = ""
                        separated_calls[1]["active"] = False
                    routes["direct"].append(
                        {**window, "slots": [calls[0], {"text": "", "active": False}]}
                    )
                    routes["separated"].append(
                        {**window, "slots": separated_calls, "leakage": diagnostic}
                    )
                    all_calls.append({"window": window["window"], "calls": calls})
                for route, rows in routes.items():
                    old, new = replay(rows, before), replay(rows)
                    reference = refs.get(sample)
                    if sample in {"silence", "noise"}:
                        reference = {
                            "text": "",
                            "kind": (
                                "known_digital_silence"
                                if sample == "silence"
                                else "known_synthetic_noise"
                            ),
                        }
                    if reference and reference.get("kind") not in (
                        "human_verified",
                        "known_digital_silence",
                        "known_synthetic_noise",
                    ):
                        raise ValueError("Reference must be explicitly human_verified")
                    if sample == "mixture":
                        reference = None
                    # In single-source samples include both output transcripts so
                    # residual/leakage hallucinations are not hidden by selecting a best slot.
                    metrics = {}
                    for label, value in [("before", old), ("after", new)]:
                        metrics[label] = (
                            text_scores(
                                reference["text"], " ".join(value["transcripts"])
                            )
                            if reference
                            else None
                        )
                    seconds = [
                        sum(
                            c["stt"]["elapsed_seconds"] if c.get("stt") else 0
                            for c in row["slots"]
                        )
                        for row in rows
                    ]
                    processing = [
                        seconds[i]
                        + (row["separation_seconds"] if route == "separated" else 0)
                        + sum(
                            v["processing_seconds"]
                            for v in (
                                row["vad"][1:]
                                if route == "separated"
                                else row["vad"][:1]
                            )
                        )
                        for i, row in enumerate(rows)
                    ]
                    record = {
                        "sample": sample,
                        "route": route,
                        "reference": reference,
                        "scores": metrics,
                        "before": old,
                        "after": new,
                        "raw_calls": all_calls,
                        "windows": rows,
                        "stt_seconds_per_window": distribution(seconds),
                        "offline_processing_seconds": distribution(processing),
                        "over_stride_count": sum(t > 2 for t in processing),
                        "queue_backlog": None,
                        "network_latency": None,
                    }
                    model_record["samples"].append(record)
                    print(
                        name,
                        sample,
                        route,
                        new["transcripts"],
                        distribution(seconds),
                        flush=True,
                    )
        finally:
            monitor.stop()
        model_record["peak_memory_device_process_mib"] = monitor.peaks_since(0)
        report["models"].append(model_record)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        del model
        gc.collect()
    print("SAVED", args.output, flush=True)


if __name__ == "__main__":
    main()

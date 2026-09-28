from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel

from stt_context import LIVE_WHISPER_OPTIONS, SpeakerPromptContext, transcribe_base
from subtitle_assembler import (
    SpeakerSubtitleState,
    SubtitleAssembler,
    normalize_for_matching,
)


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "phase3" / "output"
DIAGNOSTIC_DIR = OUTPUT_DIR / "diagnostic"
DEFAULT_OUTPUT = OUTPUT_DIR / "stt_context_benchmark.json"
LIVE_RESULT = OUTPUT_DIR / "live_secondary_validity_75s.json"
SCHEDULE_PATH = OUTPUT_DIR / "live_secondary_validity_75s_schedule.json"
MODEL_DIR = (
    ROOT
    / "checkpoints"
    / "faster-whisper"
    / "models--Systran--faster-whisper-base"
    / "snapshots"
    / "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66"
)
SAMPLE_RATE = 16_000
STRIDE_SECONDS = 2.0
PROMPT_TAIL_CHARACTERS = 80
TRACKS = {
    "original": DIAGNOSTIC_DIR / "original.wav",
    "logical_speaker_0": DIAGNOSTIC_DIR / "logical_speaker_0.wav",
    "logical_speaker_1": DIAGNOSTIC_DIR / "logical_speaker_1.wav",
}
SPEECH_REGIONS = (
    {"name": "A_only", "start": 5.0, "end": 25.0},
    {"name": "B_only", "start": 30.0, "end": 50.0},
    {"name": "A+B", "start": 55.0, "end": 70.0},
)
SILENCE_REGIONS = (
    {"name": "silence_0", "start": 0.0, "end": 5.0},
    {"name": "silence_1", "start": 25.0, "end": 30.0},
    {"name": "silence_2", "start": 50.0, "end": 55.0},
    {"name": "silence_3", "start": 70.0, "end": 75.0},
)
CONFIGS = (
    {"name": "A_baseline_3s", "window_seconds": 3.0, "prompt": False},
    {"name": "B_longer_5s", "window_seconds": 5.0, "prompt": False},
    {"name": "C_prompt_3s", "window_seconds": 3.0, "prompt": True},
    {"name": "D_longer_prompt_5s", "window_seconds": 5.0, "prompt": True},
)


def read_audio(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if sample_rate != SAMPLE_RATE or audio.ndim != 1:
        raise ValueError(f"Expected 16kHz mono WAV: {path}")
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio: {path}")
    return np.ascontiguousarray(audio)


def window_starts(duration: float, window: float, stride: float = STRIDE_SECONDS) -> list[float]:
    if duration < window:
        return []
    starts = [float(value) for value in np.arange(0.0, duration - window + 1e-9, stride)]
    final_start = duration - window
    if not starts or abs(starts[-1] - final_start) > 1e-6:
        starts.append(final_start)
    return starts


def slice_audio(audio: np.ndarray, start: float, end: float) -> np.ndarray:
    first = round(start * SAMPLE_RATE)
    last = round(end * SAMPLE_RATE)
    return np.ascontiguousarray(audio[first:last])


def expected_active(track: str, region_name: str) -> bool:
    if region_name.startswith("silence_"):
        return False
    if track in {"original", "logical_speaker_0"}:
        return True
    return track == "logical_speaker_1" and region_name == "A+B"


def evaluation_category(track: str, region_name: str) -> str:
    if region_name.startswith("silence_"):
        return "silence"
    if track == "original":
        return "original_mixture_overlap" if region_name == "A+B" else "original_mixture_single"
    if not expected_active(track, region_name):
        return "separated_residual"
    return "separated_overlap" if region_name == "A+B" else "separated_single"


def edit_distance(left: list[str] | str, right: list[str] | str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, 1):
        current = [left_index]
        for right_index, right_item in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def score_text(reference: str, hypothesis: str) -> dict[str, Any]:
    reference_characters = normalize_for_matching(reference)
    hypothesis_characters = normalize_for_matching(hypothesis)
    reference_words = re.findall(r"\S+", reference.strip())
    hypothesis_words = re.findall(r"\S+", hypothesis.strip())
    character_edits = edit_distance(reference_characters, hypothesis_characters)
    word_edits = edit_distance(reference_words, hypothesis_words)
    return {
        "character_edits": character_edits,
        "reference_characters": len(reference_characters),
        "pseudo_reference_cer": (
            character_edits / len(reference_characters) if reference_characters else None
        ),
        "word_edits": word_edits,
        "reference_words": len(reference_words),
        "pseudo_reference_wer": word_edits / len(reference_words) if reference_words else None,
    }


def distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": float(np.percentile(values, 95)),
        "max": max(values),
    }


def adjacent_repeated_token_count(text: str) -> int:
    tokens = re.findall(r"[0-9A-Za-z가-힣]+", text.lower())
    return sum(left == right for left, right in zip(tokens, tokens[1:]))


def aggregate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [sample for sample in samples if sample["score"] is not None]
    character_edits = sum(sample["score"]["character_edits"] for sample in scored)
    reference_characters = sum(sample["score"]["reference_characters"] for sample in scored)
    word_edits = sum(sample["score"]["word_edits"] for sample in scored)
    reference_words = sum(sample["score"]["reference_words"] for sample in scored)
    calls = [call for sample in samples for call in sample["calls"]]
    elapsed = [call["elapsed_seconds"] for call in calls]
    rtfs = [call["rtf"] for call in calls]
    return {
        "sample_count": len(samples),
        "scored_sample_count": len(scored),
        "pseudo_reference_cer": (
            character_edits / reference_characters if reference_characters else None
        ),
        "pseudo_reference_wer": word_edits / reference_words if reference_words else None,
        "stt_seconds": distribution(elapsed),
        "rtf": distribution(rtfs),
        "call_count": len(calls),
        "nonempty_inactive_transcript_count": sum(
            not sample["expected_active"] and bool(sample["assembled_text"])
            for sample in samples
        ),
        "prompt_applied_to_inactive_count": sum(
            not sample["expected_active"] and call["prompt_applied"]
            for sample in samples
            for call in sample["calls"]
        ),
        "adjacent_repeated_token_count": sum(
            sample["adjacent_repeated_token_count"] for sample in samples
        ),
        "boundary_match_type_counts": dict(
            Counter(
                event["match_type"]
                for sample in samples
                for event in sample["assembly_events"]
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline base-model STT context benchmark")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if not (MODEL_DIR / "model.bin").is_file():
        raise FileNotFoundError(MODEL_DIR / "model.bin")

    audio_by_track = {name: read_audio(path) for name, path in TRACKS.items()}
    for name, audio in audio_by_track.items():
        if len(audio) < 75 * SAMPLE_RATE:
            raise ValueError(f"{name} is shorter than 75 seconds")

    print("Loading faster-whisper base on CUDA FP16...", flush=True)
    model = WhisperModel(str(MODEL_DIR), device="cuda", compute_type="float16")
    transcribe_base(model, np.zeros(SAMPLE_RATE, dtype=np.float32))

    pseudo_references: dict[str, dict[str, Any]] = {}
    print("Building long-region pseudo-references (not ground truth)...", flush=True)
    for track, audio in audio_by_track.items():
        pseudo_references[track] = {}
        for region in SPEECH_REGIONS:
            if not expected_active(track, region["name"]):
                continue
            clip = slice_audio(audio, region["start"], region["end"])
            result = transcribe_base(model, clip)
            pseudo_references[track][region["name"]] = {
                "text": result.text,
                "duration_seconds": len(clip) / SAMPLE_RATE,
                "processing_seconds": result.elapsed_seconds,
                "rtf": result.elapsed_seconds / (len(clip) / SAMPLE_RATE),
            }
            print(f"  pseudo {track}/{region['name']}: {result.text}", flush=True)

    config_results: list[dict[str, Any]] = []
    all_regions = (*SPEECH_REGIONS, *SILENCE_REGIONS)
    for config in CONFIGS:
        print(f"Running {config['name']}...", flush=True)
        samples: list[dict[str, Any]] = []
        for track_index, (track, audio) in enumerate(audio_by_track.items()):
            prompt_context = SpeakerPromptContext(
                speaker_count=len(TRACKS), maximum_characters=PROMPT_TAIL_CHARACTERS
            )
            for region in all_regions:
                active = expected_active(track, region["name"])
                prompt_context.reset(track_index)
                assembler = SubtitleAssembler(speaker=track_index)
                subtitle_state = SpeakerSubtitleState(speaker=track_index)
                duration = region["end"] - region["start"]
                calls: list[dict[str, Any]] = []
                assembly_events: list[dict[str, Any]] = []
                for window_index, relative_start in enumerate(
                    window_starts(duration, config["window_seconds"])
                ):
                    start = region["start"] + relative_start
                    end = start + config["window_seconds"]
                    clip = slice_audio(audio, start, end)
                    prompt = (
                        prompt_context.prompt(track_index, active)
                        if config["prompt"]
                        else None
                    )
                    result = transcribe_base(model, clip, initial_prompt=prompt)
                    assembly = assembler.process(window_index, result.text)
                    state_events = subtitle_state.process(
                        window_index,
                        assembly.utterance_hypothesis,
                        active,
                        end,
                    )
                    if config["prompt"]:
                        stable_text = (
                            state_events[-1].stable_text if state_events else ""
                        )
                        prompt_context.observe(
                            track_index, stable_text, active
                        )
                    calls.append(
                        {
                            "window": window_index,
                            "start_seconds": start,
                            "end_seconds": end,
                            "audio_rms": float(np.sqrt(np.mean(clip * clip))),
                            "transcript": result.text,
                            "prompt": prompt,
                            "prompt_applied": bool(prompt),
                            "elapsed_seconds": result.elapsed_seconds,
                            "rtf": result.elapsed_seconds / config["window_seconds"],
                            "segment_count": result.segment_count,
                            "average_log_probability": result.average_log_probability,
                            "maximum_no_speech_probability": result.maximum_no_speech_probability,
                            "maximum_compression_ratio": result.maximum_compression_ratio,
                        }
                    )
                    assembly_events.append(assembly.to_dict())

                assembled = assembler.utterance_hypothesis
                reference = pseudo_references.get(track, {}).get(region["name"], {}).get("text")
                samples.append(
                    {
                        "track": track,
                        "region": region["name"],
                        "category": evaluation_category(track, region["name"]),
                        "expected_active": active,
                        "reference_kind": "long_region_pseudo_reference" if reference else None,
                        "pseudo_reference": reference,
                        "assembled_text": assembled,
                        "score": score_text(reference, assembled) if reference else None,
                        "adjacent_repeated_token_count": adjacent_repeated_token_count(assembled),
                        "assembly_events": assembly_events,
                        "calls": calls,
                    }
                )

        categories = sorted({sample["category"] for sample in samples})
        config_results.append(
            {
                **config,
                "summary": aggregate_samples(samples),
                "by_category": {
                    category: aggregate_samples(
                        [sample for sample in samples if sample["category"] == category]
                    )
                    for category in categories
                },
                "samples": samples,
            }
        )

    live_json = json.loads(LIVE_RESULT.read_text(encoding="utf-8"))
    report = {
        "phase": "P2-A",
        "benchmark_kind": "offline_saved_live_audio",
        "production_pipeline_modified": False,
        "model": {
            "name": "faster-whisper-base",
            "path": str(MODEL_DIR),
            "device": "cuda",
            "compute_type": "float16",
            "installed_faster_whisper_version": "1.2.1",
        },
        "production_transcribe_options": {
            **LIVE_WHISPER_OPTIONS,
            "best_of": 5,
            "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "initial_prompt": None,
            "vad_filter": False,
            "word_timestamps": False,
            "note": "Only language, beam_size, and condition_on_previous_text are explicitly passed; remaining values are faster-whisper 1.2.1 defaults.",
        },
        "audio": {
            name: {
                "path": str(TRACKS[name]),
                "duration_seconds": len(audio) / SAMPLE_RATE,
                "sample_rate": SAMPLE_RATE,
            }
            for name, audio in audio_by_track.items()
        },
        "ground_truth": {
            "real_reference_available": False,
            "reason": "No A.wav/B.wav source transcript metadata was found in the repository.",
            "metric_label": "pseudo-reference CER/WER; not ground-truth accuracy",
            "pseudo_reference_method": "One full region transcribe using the same base model and production options.",
            "existing_full_diagnostic_transcripts": live_json.get("diagnostic_audio", {}),
        },
        "schedule_path": str(SCHEDULE_PATH),
        "stride_seconds": STRIDE_SECONDS,
        "prompt_tail_characters": PROMPT_TAIL_CHARACTERS,
        "prompt_guard": "No prompt is supplied to expected-inactive silence/residual samples; context resets per utterance region and remains track-isolated.",
        "pseudo_references": pseudo_references,
        "configs": config_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"JSON: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

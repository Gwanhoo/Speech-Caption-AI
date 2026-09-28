from __future__ import annotations

"""P2-B1: sequential base/small/medium comparison on the saved P2-A WAVs."""

import argparse
import gc
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from faster_whisper import WhisperModel

from analyze_stt_context_benchmark import analyze_samples
from benchmark_stt_context import (
    DEFAULT_OUTPUT as P2A_OUTPUT,
    SAMPLE_RATE,
    SILENCE_REGIONS,
    SPEECH_REGIONS,
    STRIDE_SECONDS,
    TRACKS,
    aggregate_samples,
    adjacent_repeated_token_count,
    evaluation_category,
    expected_active,
    read_audio,
    score_text,
    slice_audio,
    window_starts,
)
from stt_context import LIVE_WHISPER_OPTIONS, transcribe_base
from subtitle_assembler import SpeakerSubtitleState, SubtitleAssembler


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "phase3" / "output"
DEFAULT_OUTPUT = OUTPUT_DIR / "stt_model_size_benchmark.json"
MODELS = ("base", "small", "medium")
WINDOW_SECONDS = 3.0
CHECKPOINT_ROOT = ROOT / "checkpoints" / "faster-whisper"


def read_p2a_pseudo_references(path: Path = P2A_OUTPUT) -> dict[str, dict[str, str]]:
    """Use the fixed P2-A base-model regional references for every candidate."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    references = raw.get("pseudo_references")
    if not isinstance(references, dict):
        raise ValueError(f"P2-A pseudo_references missing from {path}")
    result: dict[str, dict[str, str]] = {}
    for track, regions in references.items():
        if not isinstance(regions, dict):
            raise ValueError(f"Invalid pseudo-reference regions for {track}")
        result[track] = {}
        for name, record in regions.items():
            text = record.get("text") if isinstance(record, dict) else None
            if not isinstance(text, str):
                raise ValueError(f"Invalid pseudo-reference text for {track}/{name}")
            result[track][name] = text
    return result


def gpu_memory_mib() -> dict[str, float | None]:
    """nvidia-smi includes CTranslate2 allocations, unlike torch alone."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        values = completed.stdout.splitlines()[0].split(",")
        return {"used_mib": float(values[0].strip()), "total_mib": float(values[1].strip())}
    except (IndexError, OSError, subprocess.SubprocessError, ValueError):
        return {"used_mib": None, "total_mib": None}


def torch_memory_mib() -> dict[str, float | None]:
    if not torch.cuda.is_available():
        return {"allocated_mib": None, "reserved_mib": None, "peak_allocated_mib": None}
    scale = 1024 * 1024
    return {
        "allocated_mib": torch.cuda.memory_allocated() / scale,
        "reserved_mib": torch.cuda.memory_reserved() / scale,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / scale,
    }


def model_source(model_name: str) -> str:
    """Prefer P2-A's pinned base snapshot; let faster-whisper cache missing sizes."""
    if model_name == "base":
        base = (
            CHECKPOINT_ROOT
            / "models--Systran--faster-whisper-base"
            / "snapshots"
            / "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66"
        )
        if (base / "model.bin").is_file():
            return str(base)
    return model_name


def run_model(
    model_name: str,
    audio_by_track: dict[str, np.ndarray],
    pseudo_references: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Load exactly one model, benchmark it, then release its CUDA allocations."""
    model: WhisperModel | None = None
    result: dict[str, Any] = {
        "name": model_name,
        "source": model_source(model_name),
        "status": "error",
        "memory_before_load": gpu_memory_mib(),
    }
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"Loading {model_name} on CUDA FP16...", flush=True)
        model = WhisperModel(
            result["source"],
            device="cuda",
            compute_type="float16",
            download_root=str(CHECKPOINT_ROOT),
        )
        result["memory_after_load"] = gpu_memory_mib()
        result["torch_after_load"] = torch_memory_mib()
        # Materialize lazy generator/model execution before measuring benchmark calls.
        transcribe_base(model, np.zeros(SAMPLE_RATE, dtype=np.float32))
        torch.cuda.reset_peak_memory_stats()

        samples: list[dict[str, Any]] = []
        # Include silence and logical residual samples to measure hallucinations too.
        all_regions = (*SPEECH_REGIONS, *SILENCE_REGIONS)
        for track_index, (track, audio) in enumerate(audio_by_track.items()):
            for region in all_regions:
                active = expected_active(track, region["name"])
                assembler = SubtitleAssembler(speaker=track_index)
                state = SpeakerSubtitleState(speaker=track_index)
                calls: list[dict[str, Any]] = []
                assembly_events: list[dict[str, Any]] = []
                for window_index, relative_start in enumerate(
                    window_starts(region["end"] - region["start"], WINDOW_SECONDS)
                ):
                    start = region["start"] + relative_start
                    end = start + WINDOW_SECONDS
                    clip = slice_audio(audio, start, end)
                    stt = transcribe_base(model, clip)
                    assembly = assembler.process(window_index, stt.text)
                    state.process(window_index, assembly.utterance_hypothesis, active, end)
                    calls.append(
                        {
                            "window": window_index,
                            "start_seconds": start,
                            "end_seconds": end,
                            "audio_rms": float(np.sqrt(np.mean(clip * clip))),
                            "transcript": stt.text,
                            "prompt": None,
                            "prompt_applied": False,
                            "elapsed_seconds": stt.elapsed_seconds,
                            "rtf": stt.elapsed_seconds / WINDOW_SECONDS,
                            "segment_count": stt.segment_count,
                            "average_log_probability": stt.average_log_probability,
                            "maximum_no_speech_probability": stt.maximum_no_speech_probability,
                            "maximum_compression_ratio": stt.maximum_compression_ratio,
                        }
                    )
                    assembly_events.append(assembly.to_dict())
                reference = pseudo_references.get(track, {}).get(region["name"])
                samples.append(
                    {
                        "track": track,
                        "region": region["name"],
                        "category": evaluation_category(track, region["name"]),
                        "expected_active": active,
                        "reference_kind": "fixed_P2-A_base_long_region_pseudo_reference" if reference else None,
                        "pseudo_reference": reference,
                        "assembled_text": assembler.utterance_hypothesis,
                        "score": score_text(reference, assembler.utterance_hypothesis) if reference else None,
                        "adjacent_repeated_token_count": adjacent_repeated_token_count(assembler.utterance_hypothesis),
                        "assembly_events": assembly_events,
                        "calls": calls,
                    }
                )
        categories = sorted({sample["category"] for sample in samples})
        result.update(
            {
                "status": "ok",
                "summary": aggregate_samples(samples),
                "analysis": analyze_samples(samples),
                "by_category": {
                    category: aggregate_samples([sample for sample in samples if sample["category"] == category])
                    for category in categories
                },
                "samples": samples,
                "memory_after_benchmark": gpu_memory_mib(),
                "torch_after_benchmark": torch_memory_mib(),
            }
        )
    except Exception as error:  # Keep OOM/CUDA failures visible and continue with the next size.
        result.update(
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "memory_at_error": gpu_memory_mib(),
                "torch_at_error": torch_memory_mib(),
            }
        )
        print(f"{model_name} failed: {type(error).__name__}: {error}", flush=True)
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        result["memory_after_unload"] = gpu_memory_mib()
        result["torch_after_unload"] = torch_memory_mib()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="P2-B1 sequential faster-whisper model-size benchmark")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--p2a-input", type=Path, default=P2A_OUTPUT)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    references = read_p2a_pseudo_references(args.p2a_input)
    audio_by_track = {name: read_audio(path) for name, path in TRACKS.items()}
    for name, audio in audio_by_track.items():
        if len(audio) < 75 * SAMPLE_RATE:
            raise ValueError(f"{name} is shorter than 75 seconds")

    results = [run_model(name, audio_by_track, references) for name in args.models]
    report = {
        "phase": "P2-B1",
        "benchmark_kind": "offline_saved_live_audio_model_size_comparison",
        "production_pipeline_modified": False,
        "comparison_policy": "One CUDA FP16 model is loaded, benchmarked, released, and CUDA-cache-cleared before the next model.",
        "models_requested": args.models,
        "window_seconds": WINDOW_SECONDS,
        "stride_seconds": STRIDE_SECONDS,
        "production_transcribe_options": {**LIVE_WHISPER_OPTIONS, "initial_prompt": None},
        "ground_truth": {
            "real_reference_available": False,
            "metric_label": "pseudo-reference CER/WER; not ground-truth accuracy",
            "pseudo_reference_method": "Fixed long-region base-model references saved by P2-A, shared across every candidate.",
        },
        "audio": {name: {"path": str(TRACKS[name]), "duration_seconds": len(audio) / SAMPLE_RATE} for name, audio in audio_by_track.items()},
        "models": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"JSON: {args.output}", flush=True)
    return 0 if all(item["status"] == "ok" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

"""Compare faster-whisper models on identical, persisted separation outputs."""

import argparse
import gc
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase3"))

from stt_context import LIVE_WHISPER_OPTIONS  # noqa: E402


SAMPLE_RATE = 16_000
DEFAULT_MODELS = ("small", "large-v3-turbo")
DEFAULT_OUTPUT = ROOT / "phase3" / "output" / "stt_model_benchmark" / "benchmark.json"
DEFAULT_SEPARATED_DIR = DEFAULT_OUTPUT.parent / "separated"
CHECKPOINT_ROOT = ROOT / "checkpoints" / "faster-whisper"
MIB = 1024 * 1024
TRANSCRIBE_OPTIONS: dict[str, Any] = {**LIVE_WHISPER_OPTIONS, "initial_prompt": None}


def finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_audio(path: Path) -> tuple[np.ndarray, float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if sample_rate != SAMPLE_RATE or audio.ndim != 1:
        raise ValueError(f"Expected mono 16 kHz WAV: {path}")
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"WAV must contain finite audio samples: {path}")
    return np.ascontiguousarray(audio), len(audio) / SAMPLE_RATE


def audio_record(path: Path, audio: np.ndarray) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "sample_count": len(audio),
        "duration_seconds": len(audio) / SAMPLE_RATE,
        "rms": float(np.sqrt(np.mean(audio * audio))),
        "peak": float(np.max(np.abs(audio))),
    }


def prepare_separated_audio(mixed_path: Path, output_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    """Run the production MossFormer2 helper once and persist lossless float WAVs."""
    mixed, duration = read_audio(mixed_path)
    from validate_end_to_end_gpu import load_separator, separate_amp

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    separator, device = load_separator()
    load_seconds = time.perf_counter() - started
    inference_started = time.perf_counter()
    separation = separate_amp(separator, device, mixed)
    inference_seconds = time.perf_counter() - inference_started

    paths: list[Path] = []
    for slot in (0, 1):
        value = np.ascontiguousarray(separation.output[slot, 0, :], dtype=np.float32)
        if len(value) != len(mixed) or not np.isfinite(value).all():
            raise ValueError(f"Invalid separation output for raw slot {slot}: shape={value.shape}")
        path = output_dir / f"raw_slot_{slot}.wav"
        sf.write(path, value, SAMPLE_RATE, subtype="FLOAT")
        paths.append(path)

    # Every candidate is fed the bytes decoded from these fixed files, not the
    # in-memory separator output.
    return paths, {
        "mode": "generated_once_from_mixed_wav",
        "mixed_audio": audio_record(mixed_path, mixed),
        "separator": "ClearVoice MossFormer2_SS_16K",
        "device": str(device),
        "model_load_seconds": load_seconds,
        "separation_wall_seconds": inference_seconds,
        "separation_reported_seconds": finite_or_none(separation.total_seconds),
        "amp_seconds": finite_or_none(separation.amp_seconds),
        "fp32_seconds": finite_or_none(separation.fp32_seconds),
        "fp32_fallback_attempted": bool(separation.fallback_attempted),
        "input_duration_seconds": duration,
    }


def query_gpu_memory() -> dict[str, int | None]:
    """Measure whole-device and current-process VRAM through nvidia-smi.

    CTranslate2 allocations are not represented by torch.cuda memory counters.
    """
    result = {"device_used_mib": None, "device_total_mib": None, "process_used_mib": None}
    try:
        device = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if device.returncode == 0:
            used, total = device.stdout.splitlines()[0].split(",")[:2]
            result["device_used_mib"] = int(used.strip())
            result["device_total_mib"] = int(total.strip())
        processes = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if processes.returncode == 0:
            for line in processes.stdout.splitlines():
                fields = [field.strip() for field in line.split(",")]
                if len(fields) == 2 and int(fields[0]) == os.getpid():
                    result["process_used_mib"] = int(fields[1])
                    break
    except (IndexError, OSError, subprocess.SubprocessError, ValueError):
        pass
    return result


class MemoryMonitor:
    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, int | None]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="stt-vram-monitor", daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.interval_seconds)

    def sample(self) -> dict[str, int | None]:
        value = query_gpu_memory()
        self.samples.append(value)
        return value

    def start(self) -> None:
        self.sample()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        self.sample()

    def peak_since(self, index: int) -> dict[str, int | None]:
        selected = self.samples[index:]
        result: dict[str, int | None] = {}
        for key in ("device_used_mib", "device_total_mib", "process_used_mib"):
            values = [sample[key] for sample in selected if sample.get(key) is not None]
            result[key] = max(values) if values else None
        return result


def repetition_metrics(text: str, maximum_ngram: int = 4) -> dict[str, Any]:
    """Describe repetition without filtering or rewriting the transcript."""
    tokens = re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)
    adjacent_duplicates = sum(left == right for left, right in zip(tokens, tokens[1:]))
    maximum_token_run = 0
    current_run = 0
    previous: str | None = None
    for token in tokens:
        current_run = current_run + 1 if token == previous else 1
        maximum_token_run = max(maximum_token_run, current_run)
        previous = token

    best_repeats = 0
    best_phrase: list[str] = []
    for width in range(1, min(maximum_ngram, len(tokens)) + 1):
        for start in range(0, len(tokens) - width + 1):
            phrase = tokens[start : start + width]
            repeats = 1
            cursor = start + width
            while tokens[cursor : cursor + width] == phrase:
                repeats += 1
                cursor += width
            if repeats > best_repeats:
                best_repeats = repeats
                best_phrase = phrase
    return {
        "token_count": len(tokens),
        "unique_token_count": len(set(tokens)),
        "unique_token_ratio": len(set(tokens)) / len(tokens) if tokens else None,
        "adjacent_duplicate_token_count": adjacent_duplicates,
        "adjacent_duplicate_token_ratio": adjacent_duplicates / (len(tokens) - 1) if len(tokens) > 1 else 0.0,
        "maximum_identical_token_run": maximum_token_run,
        "maximum_consecutive_ngram_repetitions": best_repeats,
        "most_repeated_consecutive_ngram": best_phrase,
    }


def segment_record(segment: Any) -> dict[str, Any]:
    return {
        "id": getattr(segment, "id", None),
        "start": finite_or_none(getattr(segment, "start", None)),
        "end": finite_or_none(getattr(segment, "end", None)),
        "text": str(getattr(segment, "text", "")),
        "avg_logprob": finite_or_none(getattr(segment, "avg_logprob", None)),
        "no_speech_prob": finite_or_none(getattr(segment, "no_speech_prob", None)),
        "compression_ratio": finite_or_none(getattr(segment, "compression_ratio", None)),
    }


def transcribe_audio(model: Any, audio: np.ndarray, duration: float) -> dict[str, Any]:
    total_started = time.perf_counter()
    inference_started = time.perf_counter()
    segments, info = model.transcribe(audio, **TRANSCRIBE_OPTIONS)
    materialized = list(segments)
    inference_seconds = time.perf_counter() - inference_started
    records = [segment_record(segment) for segment in materialized]
    transcript = " ".join(record["text"].strip() for record in records).strip()
    return {
        "raw_transcript": transcript,
        "segments": records,
        "segment_count": len(records),
        "detected_language": getattr(info, "language", None),
        "language_probability": finite_or_none(getattr(info, "language_probability", None)),
        "inference_latency_seconds": inference_seconds,
        "real_time_factor": inference_seconds / duration,
        "total_processing_seconds": time.perf_counter() - total_started,
        "repetition": repetition_metrics(transcript),
    }


def model_source(model_name: str) -> str:
    if model_name != "base":
        return model_name
    model_cache = CHECKPOINT_ROOT / "models--Systran--faster-whisper-base"
    candidates = [CHECKPOINT_ROOT, model_cache]
    snapshots = model_cache / "snapshots"
    if snapshots.is_dir():
        candidates.extend(path for path in snapshots.iterdir() if path.is_dir())
    for candidate in candidates:
        if (candidate / "model.bin").is_file():
            return str(candidate)
    return model_name


def torch_memory() -> dict[str, float | None]:
    if not torch.cuda.is_available():
        return {"allocated_mib": None, "reserved_mib": None, "peak_allocated_mib": None}
    return {
        "allocated_mib": torch.cuda.memory_allocated() / MIB,
        "reserved_mib": torch.cuda.memory_reserved() / MIB,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
    }


def run_model(
    model_name: str,
    inputs: list[tuple[Path, np.ndarray, float]],
    *,
    warm_up: bool,
) -> dict[str, Any]:
    source = model_source(model_name)
    result: dict[str, Any] = {
        "model_name": model_name,
        "model_source": source,
        "status": "error",
        "device": "cuda",
        "compute_type": "float16",
        "transcribe_options": dict(TRANSCRIBE_OPTIONS),
        "vad_filter": {
            "passed_to_faster_whisper": False,
            "effective_default": False,
        },
        "memory_before_load": query_gpu_memory(),
    }
    model: WhisperModel | None = None
    monitor = MemoryMonitor()
    monitor.start()
    model_started = time.perf_counter()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        load_sample = len(monitor.samples)
        load_started = time.perf_counter()
        print(f"[{model_name}] loading CUDA FP16 model...", flush=True)
        model = WhisperModel(
            source,
            device="cuda",
            compute_type="float16",
            download_root=str(CHECKPOINT_ROOT),
        )
        result["actual_device"] = str(model.model.device)
        result["actual_compute_type"] = str(model.model.compute_type)
        result["model_load_seconds"] = time.perf_counter() - load_started
        result["memory_after_load"] = monitor.sample()
        result["memory_peak_during_load"] = monitor.peak_since(load_sample)

        if warm_up:
            warm_started = time.perf_counter()
            list(
                model.transcribe(
                    np.zeros(SAMPLE_RATE * 3, dtype=np.float32),
                    **TRANSCRIBE_OPTIONS,
                )[0]
            )
            result["warm_up_seconds"] = time.perf_counter() - warm_started
        else:
            result["warm_up_seconds"] = None

        input_results: list[dict[str, Any]] = []
        for path, audio, duration in inputs:
            sample_index = len(monitor.samples)
            monitor.sample()
            print(f"[{model_name}] transcribing {path.name} ({duration:.3f}s)...", flush=True)
            transcribed = transcribe_audio(model, audio, duration)
            memory_after = monitor.sample()
            transcribed.update(
                {
                    "audio_path": str(path.resolve()),
                    "audio_sha256": sha256_file(path),
                    "audio_duration_seconds": duration,
                    "gpu_memory_after_inference": memory_after,
                    "gpu_memory_peak_during_inference": monitor.peak_since(sample_index),
                }
            )
            input_results.append(transcribed)
        result.update(
            {
                "status": "ok",
                "inputs": input_results,
                "concatenated_raw_transcript": "\n".join(
                    item["raw_transcript"] for item in input_results
                ).strip(),
                "torch_memory_after_inference": torch_memory(),
                "total_processing_seconds": time.perf_counter() - model_started,
                "total_inference_latency_seconds": sum(
                    item["inference_latency_seconds"] for item in input_results
                ),
                "mean_real_time_factor": (
                    sum(item["real_time_factor"] for item in input_results)
                    / len(input_results)
                ),
            }
        )
        result["overall_repetition"] = repetition_metrics(result["concatenated_raw_transcript"])
    except Exception as exc:
        result.update(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "memory_at_error": query_gpu_memory(),
                "torch_memory_at_error": torch_memory(),
                "total_processing_seconds": time.perf_counter() - model_started,
            }
        )
        print(f"[{model_name}] failed: {type(exc).__name__}: {exc}", flush=True)
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        monitor.stop()
        result["memory_peak_total"] = monitor.peak_since(0)
        result["memory_after_unload"] = query_gpu_memory()
    return result


def load_fixed_inputs(paths: Iterable[Path]) -> list[tuple[Path, np.ndarray, float]]:
    return [(path, *read_audio(path)) for path in paths]


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def side_by_side_comparison(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Arrange unmodified model outputs by input for direct human comparison."""
    by_audio: dict[str, dict[str, Any]] = {}
    for model in models:
        if model.get("status") != "ok":
            continue
        for item in model["inputs"]:
            row = by_audio.setdefault(
                item["audio_sha256"],
                {
                    "audio_path": item["audio_path"],
                    "audio_sha256": item["audio_sha256"],
                    "audio_duration_seconds": item["audio_duration_seconds"],
                    "models": {},
                },
            )
            row["models"][model["model_name"]] = {
                "raw_transcript": item["raw_transcript"],
                "repetition": item["repetition"],
                "inference_latency_seconds": item["inference_latency_seconds"],
                "real_time_factor": item["real_time_factor"],
                "segments": item["segments"],
            }
    return list(by_audio.values())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare faster-whisper models on fixed MossFormer2 output WAVs"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", type=Path, help="Mixed mono 16 kHz WAV to separate once")
    source.add_argument(
        "--separated-wav",
        type=Path,
        nargs="+",
        help="One or more already-separated mono 16 kHz WAVs",
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--separated-dir", type=Path, default=DEFAULT_SEPARATED_DIR)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="With --audio, save fixed separation WAVs and stop before STT",
    )
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args(argv)
    if args.prepare_only and args.audio is None:
        parser.error("--prepare-only requires --audio")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    benchmark_started = time.perf_counter()
    if args.audio is not None:
        paths, preparation = prepare_separated_audio(args.audio, args.separated_dir)
    else:
        paths = list(args.separated_wav)
        preparation = {"mode": "pre_separated_wavs"}

    fixed_inputs = load_fixed_inputs(paths)
    preparation["fixed_separated_audio"] = [
        audio_record(path, audio) for path, audio, _ in fixed_inputs
    ]
    report: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_kind": "isolated_raw_faster_whisper_model_comparison",
        "production_pipeline_modified": False,
        "comparison_policy": (
            "MossFormer2 runs at most once. Every model receives arrays decoded from the "
            "same persisted FLOAT WAV files, identified by SHA-256. Models run sequentially."
        ),
        "models_requested": list(args.models),
        "production_transcribe_options": dict(TRANSCRIBE_OPTIONS),
        "vad_policy": (
            "No Silero gate is applied inside this isolated benchmark. faster-whisper "
            "vad_filter is not passed and therefore remains false, matching production."
        ),
        "preparation": preparation,
        "models": [],
    }
    if args.prepare_only:
        report["total_benchmark_seconds"] = time.perf_counter() - benchmark_started
        write_report(args.output, report)
        print(f"Prepared fixed WAVs: {', '.join(str(path) for path in paths)}", flush=True)
        print(f"JSON: {args.output}", flush=True)
        return 0

    for model_name in args.models:
        report["models"].append(
            run_model(model_name, fixed_inputs, warm_up=not args.no_warmup)
        )
        report["side_by_side"] = side_by_side_comparison(report["models"])
        report["total_benchmark_seconds"] = time.perf_counter() - benchmark_started
        write_report(args.output, report)

    print(f"JSON: {args.output}", flush=True)
    return 0 if all(model["status"] == "ok" for model in report["models"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

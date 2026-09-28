from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from funasr import AutoModel


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "phase3" / "output"
DIAGNOSTIC_DIR = OUTPUT_DIR / "diagnostic"
RESULT_PATH = OUTPUT_DIR / "stt_benchmark.json"
SAMPLE_RATE = 16_000
WHISPER_MODEL_DIR = (
    ROOT
    / "checkpoints"
    / "faster-whisper"
    / "models--Systran--faster-whisper-base"
    / "snapshots"
    / "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66"
)
WAVS = (
    ("ORIGINAL", DIAGNOSTIC_DIR / "original.wav"),
    ("SPEAKER 0", DIAGNOSTIC_DIR / "speaker_0.wav"),
    ("SPEAKER 1", DIAGNOSTIC_DIR / "speaker_1.wav"),
)


def gpu_memory_mib() -> dict[str, float | None]:
    if not torch.cuda.is_available():
        return {"allocated_mib": None, "reserved_mib": None, "peak_allocated_mib": None}
    mib = 1024 * 1024
    return {
        "allocated_mib": torch.cuda.memory_allocated() / mib,
        "reserved_mib": torch.cuda.memory_reserved() / mib,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / mib,
    }


def read_wav(path: Path) -> tuple[np.ndarray, dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    audio, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != SAMPLE_RATE or audio.ndim != 1:
        raise ValueError(f"Expected 16kHz mono WAV: {path}")
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"Invalid WAV data: {path}")
    return np.ascontiguousarray(audio), {
        "duration": len(audio) / sample_rate,
        "rms": float(np.sqrt(np.mean(audio * audio))),
        "peak": float(np.max(np.abs(audio))),
    }


def transcribe_whisper(model: WhisperModel, audio: np.ndarray) -> tuple[str, float]:
    started = time.perf_counter()
    segments, _ = model.transcribe(
        audio,
        language="ko",
        beam_size=5,
        condition_on_previous_text=False,
    )
    return " ".join(segment.text.strip() for segment in segments).strip(), time.perf_counter() - started


def transcribe_sensevoice(model: Any, path: Path) -> tuple[str, float]:
    started = time.perf_counter()
    result = model.generate(input=str(path), language="ko", use_itn=True)
    text = " ".join(str(item.get("text", "")).strip() for item in result).strip()
    return text, time.perf_counter() - started


def record_result(
    model_name: str,
    path: Path,
    stats: dict[str, float],
    transcript: str,
    elapsed: float,
    memory_before: dict[str, float | None],
    memory_after: dict[str, float | None],
) -> dict[str, Any]:
    return {
        "model": model_name,
        "audio_file": str(path),
        "duration": stats["duration"],
        "rms": stats["rms"],
        "peak": stats["peak"],
        "processing_time": elapsed,
        "rtf": elapsed / stats["duration"],
        "transcript": transcript,
        "gpu_memory": {"before": memory_before, "after": memory_after},
    }


def print_result(model_name: str, result: dict[str, Any]) -> None:
    print(f"\n--- {model_name} ---")
    print(f"duration: {result['duration']:.3f}s")
    print(f"RMS: {result['rms']:.6f}")
    print(f"peak: {result['peak']:.6f}")
    print(f"processing time: {result['processing_time']:.3f}s")
    print(f"RTF: {result['rtf']:.3f}")
    print(f"GPU allocated before/after/peak: "
          f"{result['gpu_memory']['before']['allocated_mib']:.1f}/"
          f"{result['gpu_memory']['after']['allocated_mib']:.1f}/"
          f"{result['gpu_memory']['after']['peak_allocated_mib']:.1f} MiB")
    print(f"transcript: {result['transcript']}")


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 4-C requires CUDA")
    print("Loading faster-whisper base on CUDA float16...", flush=True)
    whisper_model = WhisperModel(str(WHISPER_MODEL_DIR), device="cuda", compute_type="float16")
    print("Loading FunASR SenseVoiceSmall on CUDA...", flush=True)
    sensevoice_model = AutoModel(
        model="FunAudioLLM/SenseVoiceSmall",
        hub="hf",
        device="cuda",
        disable_update=True,
    )

    results: list[dict[str, Any]] = []
    print("\n========== Phase 4-C STT Benchmark ==========", flush=True)
    for label, path in WAVS:
        audio, stats = read_wav(path)
        print(f"\n[{label}]", flush=True)

        torch.cuda.reset_peak_memory_stats()
        whisper_before = gpu_memory_mib()
        whisper_text, whisper_elapsed = transcribe_whisper(whisper_model, audio)
        whisper_after = gpu_memory_mib()
        whisper_result = record_result(
            "faster-whisper base",
            path,
            stats,
            whisper_text,
            whisper_elapsed,
            whisper_before,
            whisper_after,
        )
        results.append(whisper_result)
        print_result("faster-whisper base", whisper_result)

        torch.cuda.reset_peak_memory_stats()
        sensevoice_before = gpu_memory_mib()
        sensevoice_text, sensevoice_elapsed = transcribe_sensevoice(sensevoice_model, path)
        sensevoice_after = gpu_memory_mib()
        sensevoice_result = record_result(
            "SenseVoiceSmall",
            path,
            stats,
            sensevoice_text,
            sensevoice_elapsed,
            sensevoice_before,
            sensevoice_after,
        )
        results.append(sensevoice_result)
        print_result("SenseVoiceSmall", sensevoice_result)

    summary = {
        "phase": "4-C",
        "faster_whisper": {"model": "base", "device": "cuda", "compute_type": "float16", "language": "ko"},
        "sensevoice": {"model": "FunAudioLLM/SenseVoiceSmall", "device": "cuda", "language": "ko"},
        "results": results,
    }
    RESULT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n==============================================")
    print(f"JSON: {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

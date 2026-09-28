from __future__ import annotations

import json
import statistics
import subprocess
import sys
import threading
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel


ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "phase2" / "output" / "short_1s_retry"
OUTPUT_PATH = ROOT / "phase3" / "output" / "transcripts_1s.json"
MODEL_DIR = (
    ROOT
    / "checkpoints"
    / "faster-whisper"
    / "models--Systran--faster-whisper-base"
    / "snapshots"
    / "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66"
)
SAMPLE_RATE = 16000
CHUNK_COUNT = 20


def gpu_used_mib() -> int | None:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode:
        return None
    try:
        return int(result.stdout.strip().splitlines()[0])
    except (IndexError, ValueError):
        return None


def main() -> int:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)
    if not (MODEL_DIR / "model.bin").is_file():
        raise FileNotFoundError(MODEL_DIR / "model.bin")

    paths = [INPUT_DIR / f"chunk_{chunk:03d}_speaker_{speaker}.wav" for chunk in range(CHUNK_COUNT) for speaker in (0, 1)]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    baseline_memory = gpu_used_mib()
    memory_samples: list[int] = []
    stop_monitor = threading.Event()

    def monitor_memory() -> None:
        while not stop_monitor.is_set():
            used = gpu_used_mib()
            if used is not None:
                memory_samples.append(used)
            stop_monitor.wait(0.25)

    monitor = threading.Thread(target=monitor_memory, name="gpu_memory_monitor")
    monitor.start()
    try:
        load_start = perf_counter()
        model = WhisperModel(str(MODEL_DIR), device="cuda", compute_type="float16")
        model_load_seconds = perf_counter() - load_start
        device = model.model.device
        compute_type = model.model.compute_type
        print(f"STT: faster-whisper base; device={device}; compute_type={compute_type}", flush=True)
        print(f"Model load count: 1; load time={model_load_seconds:.3f} sec", flush=True)
        print(f"GPU memory before load={baseline_memory} MiB; after load={gpu_used_mib()} MiB", flush=True)

        results = []
        times = []
        for chunk in range(CHUNK_COUNT):
            pair = []
            for speaker in (0, 1):
                path = INPUT_DIR / f"chunk_{chunk:03d}_speaker_{speaker}.wav"
                audio, sample_rate = sf.read(path, dtype="float32")
                if sample_rate != SAMPLE_RATE or audio.ndim != 1 or len(audio) != SAMPLE_RATE:
                    raise ValueError(f"Expected 1-second 16 kHz mono WAV: {path}")
                if not np.isfinite(audio).all() or not np.any(audio):
                    raise ValueError(f"Invalid audio: {path}")

                started = perf_counter()
                segments, _ = model.transcribe(
                    audio,
                    language="ko",
                    beam_size=5,
                    condition_on_previous_text=False,
                )
                segment_list = list(segments)
                elapsed = perf_counter() - started
                transcript = " ".join(segment.text.strip() for segment in segment_list).strip()
                pair.append({"speaker": speaker, "text": transcript, "seconds": elapsed})
                times.append(elapsed)
                print(f"chunk {chunk:03d} speaker_{speaker}: {transcript!r} ({elapsed:.3f} sec)", flush=True)
            results.append({"chunk": chunk, "speakers": pair})

        summary = {
            "implementation": "faster-whisper",
            "model": "base",
            "device": device,
            "compute_type": compute_type,
            "model_load_seconds": model_load_seconds,
            "gpu_memory_baseline_mib": baseline_memory,
            "gpu_memory_observed_peak_mib": max(memory_samples) if memory_samples else None,
            "wav_count": len(times),
            "mean_stt_seconds": statistics.mean(times),
            "median_stt_seconds": statistics.median(times),
            "min_stt_seconds": min(times),
            "max_stt_seconds": max(times),
            "empty_count": sum(not item["text"] for row in results for item in row["speakers"]),
            "chunks": results,
        }
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved: {OUTPUT_PATH}", flush=True)
        print(
            f"STT mean={summary['mean_stt_seconds']:.3f} sec, median={summary['median_stt_seconds']:.3f} sec, "
            f"min={summary['min_stt_seconds']:.3f} sec, max={summary['max_stt_seconds']:.3f} sec; "
            f"empty={summary['empty_count']}; GPU observed peak={summary['gpu_memory_observed_peak_mib']} MiB",
            flush=True,
        )
    finally:
        stop_monitor.set()
        monitor.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

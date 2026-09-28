from __future__ import annotations

import gc
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path

import ctranslate2
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

from stt_context import LIVE_WHISPER_OPTIONS, transcribe_base


ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "base"
MODEL_ROOT = ROOT / "checkpoints" / "faster-whisper"
INPUT_PATHS = (
    ROOT / "phase2" / "output" / "amp_single_speaker_0.wav",
    ROOT / "phase2" / "output" / "amp_single_speaker_1.wav",
)
SAMPLE_RATE = 16_000
MIB = 1024 * 1024


def query_gpu_memory() -> tuple[int | None, int | None]:
    """Return device used MiB and this process' compute allocation, if visible."""
    device = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    device_used = None
    if device.returncode == 0:
        try:
            device_used = int(device.stdout.strip().splitlines()[0])
        except (IndexError, ValueError):
            pass

    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    process_used = None
    if processes.returncode == 0:
        for line in processes.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 2:
                continue
            try:
                if int(fields[0]) == os.getpid():
                    process_used = int(fields[1])
                    break
            except ValueError:
                continue
    return device_used, process_used


class MemoryMonitor:
    def __init__(self) -> None:
        self.samples: list[tuple[int | None, int | None]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="nvidia-smi-monitor")

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(query_gpu_memory())
            self._stop.wait(0.1)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def peaks_since(self, first_sample: int) -> tuple[int | None, int | None]:
        samples = self.samples[first_sample:]
        device_values = [device for device, _ in samples if device is not None]
        process_values = [process for _, process in samples if process is not None]
        return (
            max(device_values) if device_values else None,
            max(process_values) if process_values else None,
        )


def read_audio(path: Path) -> tuple[np.ndarray, float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if sample_rate != SAMPLE_RATE or audio.ndim != 1:
        raise ValueError(f"Expected 16 kHz mono WAV: {path}")
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio: {path}")
    return np.ascontiguousarray(audio), len(audio) / sample_rate


def format_memory(value: int | None) -> str:
    return "unavailable" if value is None else f"{value} MiB"


def main() -> int:
    if ctranslate2.get_cuda_device_count() < 1:
        raise RuntimeError("CTranslate2 does not detect a CUDA device")

    inputs = [(path, *read_audio(path)) for path in INPUT_PATHS]
    before_load = query_gpu_memory()
    monitor = MemoryMonitor()
    monitor.start()
    try:
        load_started = time.perf_counter()
        model = WhisperModel(
            MODEL_NAME,
            device="cuda",
            compute_type="float16",
            download_root=str(MODEL_ROOT),
        )
        load_seconds = time.perf_counter() - load_started
        after_load = query_gpu_memory()

        print("=== faster-whisper CUDA benchmark ===", flush=True)
        print(
            f"model={MODEL_NAME}; device={model.model.device}; "
            f"compute_type={model.model.compute_type}",
            flush=True,
        )
        print(f"options={LIVE_WHISPER_OPTIONS}", flush=True)
        print(f"model_load_seconds={load_seconds:.6f}", flush=True)
        print(
            "memory_before_load: "
            f"device_used={format_memory(before_load[0])}; "
            f"process_used={format_memory(before_load[1])}",
            flush=True,
        )
        print(
            "memory_after_load: "
            f"device_used={format_memory(after_load[0])}; "
            f"process_used={format_memory(after_load[1])}",
            flush=True,
        )

        latencies: list[float] = []
        rtfs: list[float] = []
        for index, (path, audio, duration) in enumerate(inputs, 1):
            first_sample = len(monitor.samples)
            result = transcribe_base(model, audio)
            peak_device, peak_process = monitor.peaks_since(first_sample)
            rtf = result.elapsed_seconds / duration
            latencies.append(result.elapsed_seconds)
            rtfs.append(rtf)
            print(f"--- input {index} ---", flush=True)
            print(f"file={path}", flush=True)
            print(f"duration_seconds={duration:.6f}", flush=True)
            print(f"inference_seconds={result.elapsed_seconds:.6f}", flush=True)
            print(f"rtf={rtf:.6f}", flush=True)
            print(f"raw_transcript={result.text!r}", flush=True)
            print(
                "memory_inference_peak: "
                f"device_used={format_memory(peak_device)}; "
                f"process_used={format_memory(peak_process)}",
                flush=True,
            )

        print(f"mean_inference_seconds={statistics.mean(latencies):.6f}", flush=True)
        print(f"mean_rtf={statistics.mean(rtfs):.6f}", flush=True)
        del model
        gc.collect()
        after_release = query_gpu_memory()
        print(
            "memory_after_model_release: "
            f"device_used={format_memory(after_release[0])}; "
            f"process_used={format_memory(after_release[1])}",
            flush=True,
        )
    finally:
        monitor.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

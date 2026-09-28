from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "phase2" / "input" / "chunk_000_mixed_16k.wav"
OUTPUT_DIR = ROOT / "phase2" / "output"
SAMPLE_RATE = 16000
CHUNK_SECONDS = 5
MIB = 2**20
sys.path.insert(0, str(ROOT / "phase0"))

from separate import CHECKPOINT_MARKER, MODEL_NAME, _load_clearvoice  # noqa: E402


def gpu_status() -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else f"unavailable: {result.stderr.strip()}"


def run_once(separator, batched_audio: np.ndarray, device: torch.device) -> tuple[np.ndarray, float]:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            separated = np.asarray(separator(batched_audio, False))
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    if separated.ndim != 3 or separated.shape[0] < 2 or separated.shape[1] != 1:
        raise RuntimeError(f"Unexpected separation output: {separated.shape}")
    if not np.isfinite(separated[:2]).all():
        raise ValueError("AMP separation output contains NaN or Inf")
    return separated, elapsed


def print_memory(device: torch.device) -> None:
    free, total = torch.cuda.mem_get_info(device)
    print(
        f"GPU memory MiB: allocated={torch.cuda.memory_allocated(device) / MIB:.1f}, "
        f"reserved={torch.cuda.memory_reserved(device) / MIB:.1f}, "
        f"peak_allocated={torch.cuda.max_memory_allocated(device) / MIB:.1f}, "
        f"peak_reserved={torch.cuda.max_memory_reserved(device) / MIB:.1f}, "
        f"CUDA_free/total={free / MIB:.1f}/{total / MIB:.1f}",
        flush=True,
    )
    print(f"nvidia-smi total,used,free MiB; utilization %; temperature C: {gpu_status()}", flush=True)


def save_and_compare(separated: np.ndarray) -> None:
    for source in (0, 1):
        path = OUTPUT_DIR / f"amp_single_speaker_{source}.wav"
        if path.exists():
            raise FileExistsError(path)
        audio = separated[source, 0, :]
        if len(audio) != CHUNK_SECONDS * SAMPLE_RATE:
            raise ValueError(f"Unexpected speaker {source} length: {len(audio)}")
        sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
        saved, sample_rate = sf.read(path, dtype="float32")
        peak = float(np.max(np.abs(saved)))
        rms = float(np.sqrt(np.mean(saved * saved)))
        if sample_rate != SAMPLE_RATE or saved.ndim != 1 or not np.isfinite(saved).all() or rms == 0:
            raise ValueError(f"Invalid output WAV: {path}")
        baseline_path = OUTPUT_DIR / f"gpu_chunk_000_speaker_{source}.wav"
        baseline, baseline_rate = sf.read(baseline_path, dtype="float32")
        if baseline_rate != SAMPLE_RATE or baseline.shape != saved.shape:
            raise ValueError(f"Incompatible FP32 output: {baseline_path}")
        correlation = float(np.corrcoef(saved, baseline)[0, 1])
        baseline_rms = float(np.sqrt(np.mean(baseline * baseline)))
        print(
            f"speaker_{source}: {len(saved) / sample_rate:.3f} sec, peak={peak:.6f}, "
            f"RMS={rms:.6f}, FP32 correlation={correlation:.6f}, "
            f"RMS ratio={rms / baseline_rms:.4f}; saved={path}",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("single", "repeat"))
    mode = parser.parse_args().mode
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    audio, sample_rate = sf.read(INPUT_PATH, dtype="float32")
    if sample_rate != SAMPLE_RATE or audio.ndim != 1 or len(audio) != CHUNK_SECONDS * SAMPLE_RATE:
        raise ValueError(f"Expected 5-second 16 kHz mono input: {INPUT_PATH}")
    if not np.isfinite(audio).all():
        raise ValueError(f"Non-finite input: {INPUT_PATH}")
    batched_audio = audio.reshape(1, -1)

    ClearVoice = _load_clearvoice()
    separator = ClearVoice(task="speech_separation", model_names=[MODEL_NAME])
    speech_model = separator.models[0]
    marker = Path(speech_model.args.checkpoint_dir) / CHECKPOINT_MARKER
    if not marker.is_file():
        raise RuntimeError(f"Pretrained checkpoint not found: {marker}")
    parameter = next(speech_model.model.parameters())
    device = parameter.device
    if speech_model.device.type != "cuda" or device.type != "cuda" or parameter.dtype != torch.float32:
        raise RuntimeError(f"Unexpected model device or dtype: {device}, {parameter.dtype}")
    torch.cuda.synchronize(device)
    print(f"Mode={mode}; model load count=1; device={device}; model dtype={parameter.dtype}", flush=True)
    print(f"Before nvidia-smi total,used,free MiB; utilization %; temperature C: {gpu_status()}", flush=True)

    if mode == "single":
        separated, elapsed = run_once(separator, batched_audio, device)
        print(f"Single AMP inference: {elapsed:.3f} sec, RTF={elapsed / CHUNK_SECONDS:.3f}", flush=True)
        print_memory(device)
        save_and_compare(separated)
        return 0

    warmup_times = []
    measured_times = []
    for index in range(7):
        separated, elapsed = run_once(separator, batched_audio, device)
        if index < 2:
            warmup_times.append(elapsed)
            print(f"Warm-up {index + 1}: {elapsed:.3f} sec", flush=True)
        else:
            measured_times.append(elapsed)
            print(
                f"Measured {index - 1}: {elapsed:.3f} sec, RTF={elapsed / CHUNK_SECONDS:.3f}, "
                f"peak_allocated={torch.cuda.max_memory_allocated(device) / MIB:.1f} MiB",
                flush=True,
            )
        print_memory(device)
        if elapsed > 30:
            print("Stopped: inference exceeded 30 seconds", flush=True)
            return 2
    print(f"Warm-up times: {warmup_times}", flush=True)
    print(f"Mean RTF: {statistics.mean(measured_times) / CHUNK_SECONDS:.3f}", flush=True)
    print(f"Median RTF: {statistics.median(measured_times) / CHUNK_SECONDS:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

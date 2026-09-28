from __future__ import annotations

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
WARMUP_COUNT = 3
MEASURE_COUNT = 10
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
    if result.returncode:
        return f"unavailable ({result.stderr.strip()})"
    return result.stdout.strip()


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    output_paths = [OUTPUT_DIR / f"latency_chunk_000_speaker_{source}.wav" for source in (0, 1)]
    for path in output_paths:
        if path.exists():
            raise FileExistsError(path)

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
    device = next(speech_model.model.parameters()).device
    if speech_model.device.type != "cuda" or device.type != "cuda":
        raise RuntimeError(f"MossFormer2 did not load on CUDA: wrapper={speech_model.device}, model={device}")
    torch.cuda.synchronize(device)

    print(f"Model load count: 1; device: {device}; allocator: {torch.cuda.memory.get_allocator_backend()}", flush=True)
    print("nvidia-smi fields: total MiB, used MiB, free MiB, GPU %, temperature C", flush=True)
    print(f"Before: {gpu_status()}", flush=True)

    warmup_times = []
    measured_times = []
    last_result = None
    for index in range(WARMUP_COUNT + MEASURE_COUNT):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        result = np.asarray(separator(batched_audio, False))
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        if result.ndim != 3 or result.shape[0] < 2 or result.shape[1] != 1:
            raise RuntimeError(f"Unexpected separation output: {result.shape}")
        if not np.isfinite(result[:2]).all():
            raise ValueError(f"Non-finite separation output on run {index}")

        allocated = torch.cuda.memory_allocated(device) / MIB
        reserved = torch.cuda.memory_reserved(device) / MIB
        peak_allocated = torch.cuda.max_memory_allocated(device) / MIB
        peak_reserved = torch.cuda.max_memory_reserved(device) / MIB
        free, total = torch.cuda.mem_get_info(device)
        status = gpu_status()
        memory = (
            f"allocated={allocated:.1f}, reserved={reserved:.1f}, "
            f"peak_allocated={peak_allocated:.1f}, peak_reserved={peak_reserved:.1f} MiB; "
            f"cuda_free/total={free / MIB:.1f}/{total / MIB:.1f} MiB; "
            f"nvidia-smi={status}"
        )
        if index < WARMUP_COUNT:
            warmup_times.append(elapsed)
            print(f"Warm-up {index + 1}: {elapsed:.3f} sec; {memory}", flush=True)
        else:
            measured_times.append(elapsed)
            print(f"Measured {index - WARMUP_COUNT + 1:02d}: {elapsed:.3f} sec, RTF={elapsed / CHUNK_SECONDS:.3f}; {memory}", flush=True)
        last_result = result

    print(f"After: {gpu_status()}", flush=True)
    for source, path in enumerate(output_paths):
        sf.write(path, last_result[source, 0, :], SAMPLE_RATE, subtype="PCM_16")
        saved, saved_rate = sf.read(path, dtype="float32")
        rms = float(np.sqrt(np.mean(saved * saved)))
        if saved_rate != SAMPLE_RATE or saved.ndim != 1 or not np.isfinite(saved).all() or rms == 0:
            raise ValueError(f"Invalid saved WAV: {path}")
        print(f"Saved {path.name}: {len(saved) / saved_rate:.3f} sec, RMS={rms:.6f}", flush=True)

    mean = statistics.mean(measured_times)
    median = statistics.median(measured_times)
    print(f"Warm-up times: {warmup_times}", flush=True)
    print(f"Mean: {mean:.3f} sec, RTF={mean / CHUNK_SECONDS:.3f}", flush=True)
    print(f"Median: {median:.3f} sec, RTF={median / CHUNK_SECONDS:.3f}", flush=True)
    print(f"Min/Max: {min(measured_times):.3f}/{max(measured_times):.3f} sec", flush=True)
    print(f"RTF < 1: {sum(seconds < CHUNK_SECONDS for seconds in measured_times)}; RTF >= 1: {sum(seconds >= CHUNK_SECONDS for seconds in measured_times)}", flush=True)
    print(f"Spikes >= 8 sec: {sum(seconds >= 8 for seconds in measured_times)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

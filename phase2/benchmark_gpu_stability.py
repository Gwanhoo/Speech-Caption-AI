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
SAMPLE_RATE = 16_000
CHUNK_SECONDS = 5
WARMUP_COUNT = 2
MEASURE_COUNT = 10
MIB = 2**20
sys.path.insert(0, str(ROOT / "phase0"))

from separate import CHECKPOINT_MARKER, MODEL_NAME, _load_clearvoice  # noqa: E402
from benchmark_cuda_amp import run_once  # noqa: E402


def nvidia_smi(query: str) -> str:
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode:
        return f"unavailable ({result.stderr.strip()})"
    return result.stdout.strip()


def gpu_snapshot() -> str:
    fields = (
        "memory.used,memory.free,utilization.gpu,temperature.gpu,"
        "power.draw,clocks.current.sm,pstate"
    )
    return nvidia_smi(fields)


def print_environment() -> None:
    properties = torch.cuda.get_device_properties(0)
    print(f"GPU: {properties.name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(
        f"cuDNN: available={torch.backends.cudnn.is_available()}, "
        f"enabled={torch.backends.cudnn.enabled}, version={torch.backends.cudnn.version()}"
    )
    print(f"CUDA device: {torch.cuda.current_device()}")
    print(f"Total VRAM: {properties.total_memory / MIB:.1f} MiB")
    print(
        f"Start PyTorch VRAM: allocated={torch.cuda.memory_allocated() / MIB:.1f} MiB, "
        f"reserved={torch.cuda.memory_reserved() / MIB:.1f} MiB"
    )
    print(
        "nvidia-smi fields: used MiB, free MiB, GPU %, temperature C, "
        "power W, SM clock MHz, pstate"
    )
    print(f"Start nvidia-smi: {gpu_snapshot()}")
    print(f"Driver: {nvidia_smi('driver_version')}")


def main() -> int:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    print_environment()
    audio, sample_rate = sf.read(INPUT_PATH, dtype="float32")
    expected_samples = CHUNK_SECONDS * SAMPLE_RATE
    if sample_rate != SAMPLE_RATE or audio.ndim != 1 or len(audio) != expected_samples:
        raise ValueError(f"Expected 5-second 16 kHz mono input: {INPUT_PATH}")
    if not np.isfinite(audio).all():
        raise ValueError(f"Non-finite input: {INPUT_PATH}")
    batched_audio = np.ascontiguousarray(audio.reshape(1, -1), dtype=np.float32)
    print(f"Input: {INPUT_PATH}; shape={batched_audio.shape}; dtype={batched_audio.dtype}")

    ClearVoice = _load_clearvoice()
    load_started = time.perf_counter()
    separator = ClearVoice(task="speech_separation", model_names=[MODEL_NAME])
    load_seconds = time.perf_counter() - load_started
    speech_model = separator.models[0]
    marker = Path(speech_model.args.checkpoint_dir) / CHECKPOINT_MARKER
    if not marker.is_file():
        raise RuntimeError(f"Pretrained checkpoint not found: {marker}")
    parameter = next(speech_model.model.parameters())
    device = parameter.device
    if speech_model.device.type != "cuda" or device.type != "cuda":
        raise RuntimeError(
            f"MossFormer2 did not load on CUDA: wrapper={speech_model.device}, model={device}"
        )
    torch.cuda.synchronize(device)
    print(
        f"Model: {MODEL_NAME}; load count=1; load time={load_seconds:.3f} sec; "
        f"device={device}; weights={parameter.dtype}; AMP=float16; inference_mode=True"
    )
    print(f"After model load nvidia-smi: {gpu_snapshot()}")

    for index in range(WARMUP_COUNT):
        separated, elapsed = run_once(separator, batched_audio, device)
        if separated.shape[0] < 2 or not np.isfinite(separated[:2]).all():
            raise ValueError(f"Invalid warm-up output: {separated.shape}")
        print(
            f"warm-up {index + 1:02d}: time={elapsed:.3f} sec, "
            f"RTF={elapsed / CHUNK_SECONDS:.3f}; nvidia-smi={gpu_snapshot()}",
            flush=True,
        )

    times: list[float] = []
    peaks: list[float] = []
    for index in range(MEASURE_COUNT):
        separated, elapsed = run_once(separator, batched_audio, device)
        if separated.shape[0] < 2 or not np.isfinite(separated[:2]).all():
            raise ValueError(f"Invalid measured output: {separated.shape}")
        allocated = torch.cuda.memory_allocated(device) / MIB
        reserved = torch.cuda.memory_reserved(device) / MIB
        peak_allocated = torch.cuda.max_memory_allocated(device) / MIB
        times.append(elapsed)
        peaks.append(peak_allocated)
        print(
            f"run {index + 1:02d}: separation={elapsed:.3f} sec, "
            f"RTF={elapsed / CHUNK_SECONDS:.3f}, allocated={allocated:.1f} MiB, "
            f"reserved={reserved:.1f} MiB, peak_allocated={peak_allocated:.1f} MiB; "
            f"nvidia-smi={gpu_snapshot()}",
            flush=True,
        )

    mean = statistics.mean(times)
    median = statistics.median(times)
    minimum = min(times)
    maximum = max(times)
    stdev = statistics.pstdev(times)
    rtf_values = [elapsed / CHUNK_SECONDS for elapsed in times]
    print("\n========== Phase 2-F summary ==========")
    print(f"mean={mean:.3f} sec")
    print(f"median={median:.3f} sec")
    print(f"min={minimum:.3f} sec")
    print(f"max={maximum:.3f} sec")
    print(f"stdev={stdev:.3f} sec")
    print(f"mean_RTF={statistics.mean(rtf_values):.3f}")
    print(f"median_RTF={statistics.median(rtf_values):.3f}")
    print(f"peak_VRAM={max(peaks):.1f} MiB")
    print(f"RTF_lt_1={sum(rtf < 1 for rtf in rtf_values)}/{MEASURE_COUNT}")
    print(f"VRAM_end_allocated={torch.cuda.memory_allocated(device) / MIB:.1f} MiB")
    print(f"VRAM_end_reserved={torch.cuda.memory_reserved(device) / MIB:.1f} MiB")
    print(f"End nvidia-smi: {gpu_snapshot()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

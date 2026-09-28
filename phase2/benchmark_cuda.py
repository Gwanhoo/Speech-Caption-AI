from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "phase2" / "input"
OUTPUT_DIR = ROOT / "phase2" / "output"
SAMPLE_RATE = 16000
CHUNK_SECONDS = 5
CHUNK_COUNT = 3
sys.path.insert(0, str(ROOT / "phase0"))

from separate import CHECKPOINT_MARKER, MODEL_NAME, _load_clearvoice  # noqa: E402


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this PyTorch installation")

    inputs = [INPUT_DIR / f"chunk_{index:03d}_mixed_16k.wav" for index in range(CHUNK_COUNT)]
    outputs = [
        OUTPUT_DIR / f"gpu_chunk_{index:03d}_speaker_{source}.wav"
        for index in range(CHUNK_COUNT)
        for source in (0, 1)
    ]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in outputs:
        if path.exists():
            raise FileExistsError(path)

    ClearVoice = _load_clearvoice()
    load_start = time.perf_counter()
    separator = ClearVoice(task="speech_separation", model_names=[MODEL_NAME])
    model_load_seconds = time.perf_counter() - load_start
    speech_model = separator.models[0]
    marker = Path(speech_model.args.checkpoint_dir) / CHECKPOINT_MARKER
    if not marker.is_file():
        raise RuntimeError(f"Pretrained checkpoint not found: {marker}")
    model_device = next(speech_model.model.parameters()).device
    if speech_model.device.type != "cuda" or model_device.type != "cuda":
        raise RuntimeError(f"MossFormer2 did not load on CUDA: wrapper={speech_model.device}, model={model_device}")

    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(model_device)}", flush=True)
    print(f"MossFormer2 device: {model_device}", flush=True)
    print(f"Model load count: 1, time: {model_load_seconds:.3f} sec", flush=True)

    times = []
    for index, path in enumerate(inputs):
        audio, sample_rate = sf.read(path, dtype="float32")
        if sample_rate != SAMPLE_RATE or audio.ndim != 1 or len(audio) != CHUNK_SECONDS * SAMPLE_RATE:
            raise ValueError(f"Expected 5-second 16 kHz mono input: {path}")
        if not np.isfinite(audio).all():
            raise ValueError(f"Non-finite input: {path}")

        torch.cuda.synchronize(model_device)
        start = time.perf_counter()
        separated = np.asarray(separator(audio.reshape(1, -1).astype(np.float32), False))
        torch.cuda.synchronize(model_device)
        elapsed = time.perf_counter() - start
        if separated.ndim != 3 or separated.shape[0] < 2 or separated.shape[1] != 1:
            raise RuntimeError(f"Unexpected separation output for chunk {index}: {separated.shape}")

        for source in (0, 1):
            result = separated[source, 0, :]
            if result.size == 0 or not np.isfinite(result).all():
                raise ValueError(f"Invalid speaker {source} in chunk {index}")
            output_path = OUTPUT_DIR / f"gpu_chunk_{index:03d}_speaker_{source}.wav"
            sf.write(output_path, result, SAMPLE_RATE, subtype="PCM_16")
            saved, saved_rate = sf.read(output_path, dtype="float32")
            rms = float(np.sqrt(np.mean(saved * saved)))
            if saved_rate != SAMPLE_RATE or saved.ndim != 1 or rms == 0:
                raise ValueError(f"Invalid saved speaker WAV: {output_path}")
            print(f"  speaker_{source}: {len(saved) / saved_rate:.3f} sec, RMS={rms:.6f}", flush=True)

        times.append(elapsed)
        print(f"Chunk {index:03d}: inference={elapsed:.3f} sec, RTF={elapsed / CHUNK_SECONDS:.3f}", flush=True)

    average_rtf = float(np.mean(times)) / CHUNK_SECONDS
    print(f"Mean GPU RTF: {average_rtf:.3f}", flush=True)
    print(f"CPU baseline / GPU speedup: {4.15 / average_rtf:.2f}x", flush=True)
    print(f"RTF < 1: {average_rtf < 1}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

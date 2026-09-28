from __future__ import annotations

import sys
from math import gcd
from pathlib import Path

import numpy as np
import soundcard as sc
import soundfile as sf
from scipy.signal import resample_poly


RECORD_SECONDS = 5
CAPTURE_SAMPLE_RATE = 48_000
TARGET_SAMPLE_RATE = 16_000
OUTPUT_PATH = Path(__file__).resolve().parent / "output" / "captured.wav"


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)


def _resample(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return audio.astype(np.float32, copy=False)

    divisor = gcd(source_sr, target_sr)
    up = target_sr // divisor
    down = source_sr // divisor
    return resample_poly(audio, up, down).astype(np.float32)


def _fit_length(audio: np.ndarray, sample_rate: int, seconds: int) -> np.ndarray:
    target_frames = sample_rate * seconds
    if len(audio) > target_frames:
        return audio[:target_frames]
    if len(audio) < target_frames:
        return np.pad(audio, (0, target_frames - len(audio)))
    return audio


def capture_system_audio() -> tuple[Path, float, float, float]:
    speaker = sc.default_speaker()
    loopback = sc.get_microphone(speaker.name, include_loopback=True)

    print(f"System audio device: {speaker.name}", flush=True)
    print(f"Recording system audio for {RECORD_SECONDS} seconds...", flush=True)

    with loopback.recorder(samplerate=CAPTURE_SAMPLE_RATE) as recorder:
        captured = recorder.record(numframes=CAPTURE_SAMPLE_RATE * RECORD_SECONDS)

    audio = np.asarray(captured, dtype=np.float32)
    audio = _to_mono(audio)
    audio = _resample(audio, CAPTURE_SAMPLE_RATE, TARGET_SAMPLE_RATE)
    audio = _fit_length(audio, TARGET_SAMPLE_RATE, RECORD_SECONDS)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    duration = len(audio) / TARGET_SAMPLE_RATE if TARGET_SAMPLE_RATE else 0.0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sf.write(OUTPUT_PATH, audio, TARGET_SAMPLE_RATE, subtype="PCM_16")

    return OUTPUT_PATH, duration, peak, rms


def main() -> int:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    if sys.platform != "win32":
        print("[ERROR] Phase 1-A requires Windows WASAPI loopback capture.")
        return 1

    try:
        output_path, duration, peak, rms = capture_system_audio()
    except Exception as exc:
        print(f"[ERROR] System audio capture failed: {type(exc).__name__}: {exc}")
        return 1

    relative_output = output_path.relative_to(Path.cwd())
    print(f"Saved: {relative_output}")
    print(f"Duration: {duration:.2f} sec")
    print(f"Sample rate: {TARGET_SAMPLE_RATE} Hz")
    print("Channels: 1")
    print(f"Peak amplitude: {peak:.6f}")
    print(f"RMS amplitude: {rms:.6f}")
    print(f"Silent: {'yes' if peak < 1e-5 else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

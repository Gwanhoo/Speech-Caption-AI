from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


TARGET_SAMPLE_RATE = 16_000
OUTPUT_PEAK = 0.98


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)


def _resample(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return audio

    divisor = gcd(source_sr, target_sr)
    up = target_sr // divisor
    down = source_sr // divisor
    return resample_poly(audio, up, down).astype(np.float32)


def load_wav_mono_16k(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32")
    audio = _to_mono(audio)
    audio = _resample(audio, sample_rate, TARGET_SAMPLE_RATE)
    return np.asarray(audio, dtype=np.float32)


def mix_wavs(
    speaker_a_path: Path,
    speaker_b_path: Path,
    output_path: Path,
) -> Path:
    speaker_a = load_wav_mono_16k(speaker_a_path)
    speaker_b = load_wav_mono_16k(speaker_b_path)

    output_length = max(len(speaker_a), len(speaker_b))
    padded_a = np.zeros(output_length, dtype=np.float32)
    padded_b = np.zeros(output_length, dtype=np.float32)
    padded_a[: len(speaker_a)] = speaker_a
    padded_b[: len(speaker_b)] = speaker_b

    mixed = 0.5 * padded_a + 0.5 * padded_b
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > OUTPUT_PEAK:
        mixed *= OUTPUT_PEAK / peak

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, mixed, TARGET_SAMPLE_RATE, subtype="PCM_16")
    return output_path


def main() -> int:
    phase0_dir = Path(__file__).resolve().parent
    input_dir = phase0_dir / "input"
    output_dir = phase0_dir / "output"

    speaker_a_path = input_dir / "speaker_a.wav"
    speaker_b_path = input_dir / "speaker_b.wav"
    mixed_path = output_dir / "mixed.wav"

    if not speaker_a_path.exists() or not speaker_b_path.exists():
        print("phase0/input/speaker_a.wav와 speaker_b.wav를 넣어주세요.")
        return 1

    mix_wavs(speaker_a_path, speaker_b_path, mixed_path)
    print(f"mixed.wav 생성 완료: {mixed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

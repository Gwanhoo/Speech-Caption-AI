from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DIR = ROOT / "phase1" / "reference"
OUTPUT_DIR = ROOT / "phase3" / "test_audio"
SAMPLE_RATE = 16000
SPEECH_SECONDS = 5
SILENCE_SECONDS = 3


def create_fixture(source: Path, destination: Path) -> None:
    audio, sample_rate = sf.read(source, dtype="float32")
    if sample_rate != SAMPLE_RATE or audio.ndim != 1:
        raise ValueError(f"Expected {SAMPLE_RATE}Hz mono WAV: {source}")
    speech_samples = SPEECH_SECONDS * SAMPLE_RATE
    if len(audio) < speech_samples:
        raise ValueError(f"Reference is shorter than {SPEECH_SECONDS}s: {source}")
    speech = np.ascontiguousarray(audio[:speech_samples], dtype=np.float32)
    silence = np.zeros(SILENCE_SECONDS * SAMPLE_RATE, dtype=np.float32)
    fixture = np.concatenate((speech, silence, speech, silence, speech))
    sf.write(destination, fixture, SAMPLE_RATE, subtype="PCM_16")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for speaker in ("a", "b"):
        destination = OUTPUT_DIR / f"speaker_{speaker}_silence_test.wav"
        create_fixture(REFERENCE_DIR / f"speaker_{speaker}.wav", destination)
        info = sf.info(destination)
        print(
            f"{destination}: {info.duration:.1f}s, {info.samplerate}Hz, "
            f"channels={info.channels}, subtype={info.subtype}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

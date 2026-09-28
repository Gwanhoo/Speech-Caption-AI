from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import soundfile as sf
from faster_whisper import WhisperModel


MODEL_NAME = "base"
SAMPLE_RATE = 16_000


def main() -> int:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Transcribe separated Phase 0 audio")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    phase0_dir = Path(__file__).resolve().parent
    output_dir = phase0_dir / "output"
    paths = [output_dir / f"speaker_{number}.wav" for number in (1, 2)]
    for path in paths:
        if not path.is_file():
            parser.error(f"Input file not found: {path}")

    started = perf_counter()
    model = WhisperModel(
        MODEL_NAME,
        device=args.device,
        compute_type="float16" if args.device == "cuda" else "int8",
        download_root=phase0_dir.parent / "checkpoints" / "faster-whisper",
    )
    print(f"STT model: faster-whisper {MODEL_NAME} ({args.device})", flush=True)

    for number, path in enumerate(paths, start=1):
        audio, sample_rate = sf.read(path, dtype="float32")
        if sample_rate != SAMPLE_RATE or audio.ndim != 1:
            raise ValueError(f"Expected mono 16 kHz WAV: {path}")

        speaker_started = perf_counter()
        segments, _ = model.transcribe(
            audio,
            language="ko",
            beam_size=5,
            condition_on_previous_text=False,
        )
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
        text_path = path.with_suffix(".txt")
        text_path.write_text(transcript + "\n", encoding="utf-8")
        print(f"\n[Speaker {number}]\n{transcript}", flush=True)
        print(f"Saved: {text_path}", flush=True)
        print(f"STT time: {perf_counter() - speaker_started:.2f}s", flush=True)

    print(f"Total time (including model load): {perf_counter() - started:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

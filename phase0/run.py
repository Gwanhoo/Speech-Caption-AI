from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter


def main() -> int:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    total_started = perf_counter()
    phase0_dir = Path(__file__).resolve().parent
    input_dir = phase0_dir / "input"
    output_dir = phase0_dir / "output"

    speaker_a_path = input_dir / "speaker_a.wav"
    speaker_b_path = input_dir / "speaker_b.wav"
    mixed_path = output_dir / "mixed.wav"
    stage = "Input validation"

    try:
        for path in (speaker_a_path, speaker_b_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        from mix import mix_wavs
        from separate import separate_wav

        stage = "Mixing"
        print("[1/3] Mixing audio...", flush=True)
        started = perf_counter()
        mix_wavs(speaker_a_path, speaker_b_path, mixed_path)
        if not mixed_path.is_file():
            raise FileNotFoundError(mixed_path)
        mixing_time = perf_counter() - started

        stage = "Separation"
        print("[2/3] Separating speakers...", flush=True)
        started = perf_counter()
        speaker_1_path, speaker_2_path = separate_wav(mixed_path, output_dir)
        for path in (speaker_1_path, speaker_2_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        separation_time = perf_counter() - started

        stage = "STT"
        print("[3/3] Transcribing speakers...", flush=True)
        started = perf_counter()
        from transcribe import main as transcribe_main

        if transcribe_main() != 0:
            raise RuntimeError("Transcription failed")
        transcripts = []
        for path in (speaker_1_path, speaker_2_path):
            text_path = path.with_suffix(".txt")
            if not text_path.is_file():
                raise FileNotFoundError(text_path)
            transcripts.append(text_path.read_text(encoding="utf-8").strip())
        stt_time = perf_counter() - started
    except Exception as exc:
        print(f"[ERROR] {stage}: {type(exc).__name__}: {exc}")
        return 1

    total_time = perf_counter() - total_started
    print("\n========== RESULT ==========")
    for number, transcript in enumerate(transcripts, start=1):
        print(f"\n[Speaker {number}]\n{transcript}")
    print("\n============================")
    print(f"Mixing: {mixing_time:.2f} sec")
    print(f"Separation: {separation_time:.2f} sec")
    print(f"STT: {stt_time:.2f} sec")
    print(f"Total: {total_time:.2f} sec")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

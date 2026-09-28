from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "phase1" / "output"
CAPTURED_PATH = OUTPUT_DIR / "captured.wav"
sys.path.insert(0, str(ROOT / "phase0"))


def inspect_wav(path: Path) -> tuple[np.ndarray, int, float, float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    audio, sample_rate = sf.read(path, dtype="float32")
    if audio.ndim != 1 or sample_rate != 16_000:
        raise ValueError(f"Expected mono 16 kHz WAV: {path}")
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError(f"Empty or non-finite audio: {path}")
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio))))
    print(
        f"{path.name}: {sample_rate} Hz, 1 channel, {len(audio) / sample_rate:.2f} sec, "
        f"peak={peak:.6f}, RMS={rms:.6f}, silent={'yes' if peak == 0 else 'no'}",
        flush=True,
    )
    return audio, sample_rate, peak, rms


def main() -> int:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    total_started = perf_counter()
    stage = "Input validation"
    try:
        print("[1/3] Checking captured audio...", flush=True)
        input_started = perf_counter()
        _, _, peak, _ = inspect_wav(CAPTURED_PATH)
        if peak == 0:
            raise ValueError("Captured audio is silent")
        input_time = perf_counter() - input_started
        print(f"Input check: {input_time:.2f} sec", flush=True)

        stage = "Separation"
        print("[2/3] Separating speakers...", flush=True)
        from separate import MODEL_NAME as SEPARATION_MODEL_NAME, separate_wav

        print(f"Separation model: {SEPARATION_MODEL_NAME} (cpu)", flush=True)
        separation_started = perf_counter()
        speaker_paths = separate_wav(CAPTURED_PATH, OUTPUT_DIR)
        for path in speaker_paths:
            inspect_wav(path)
        separation_time = perf_counter() - separation_started
        print(f"Separation: {separation_time:.2f} sec", flush=True)

        stage = "STT"
        print("[3/3] Transcribing speakers...", flush=True)
        from faster_whisper import WhisperModel
        from transcribe import MODEL_NAME as STT_MODEL_NAME

        stt_started = perf_counter()
        model = WhisperModel(
            STT_MODEL_NAME,
            device="cpu",
            compute_type="int8",
            download_root=ROOT / "checkpoints" / "faster-whisper",
        )
        print(f"STT model: faster-whisper {STT_MODEL_NAME} (cpu)", flush=True)
        transcripts = []
        for number, path in enumerate(speaker_paths, start=1):
            audio, _, _, _ = inspect_wav(path)
            segments, _ = model.transcribe(
                audio,
                language="ko",
                beam_size=5,
                condition_on_previous_text=False,
            )
            transcript = " ".join(segment.text.strip() for segment in segments).strip()
            text_path = path.with_suffix(".txt")
            text_path.write_text(transcript + "\n", encoding="utf-8")
            transcripts.append(transcript)
            print(f"Speaker {number}: {transcript}", flush=True)
            print(f"Saved: {text_path}", flush=True)
        stt_time = perf_counter() - stt_started
    except Exception as exc:
        print(f"[ERROR] {stage}: {type(exc).__name__}: {exc}", flush=True)
        return 1

    print(f"Separation: {separation_time:.2f} sec")
    print(f"STT: {stt_time:.2f} sec")
    print(f"Total: {perf_counter() - total_started:.2f} sec")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundcard as sc
import soundfile as sf

from playback_capture import load_reference


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "phase1"))

from capture import CAPTURE_SAMPLE_RATE, TARGET_SAMPLE_RATE, _resample, _to_mono  # noqa: E402


def capture_one(name: str, speaker: sc._Speaker, loopback: sc._Microphone) -> Path:
    playback_audio = load_reference(name)
    output_path = OUTPUT_DIR / f"captured_{name}.wav"
    if output_path.exists():
        raise FileExistsError(output_path)

    ready = threading.Event()
    launch = threading.Event()
    errors: list[BaseException] = []
    start_at = 0.0

    def play() -> None:
        try:
            with speaker.player(samplerate=CAPTURE_SAMPLE_RATE, channels=2) as player:
                ready.set()
                if not launch.wait(timeout=10):
                    raise TimeoutError("Playback start timed out")
                time.sleep(max(0.0, start_at - time.perf_counter()))
                player.play(playback_audio)
                while player.currentpadding:
                    time.sleep(0.005)
        except BaseException as exc:
            errors.append(exc)
            ready.set()

    print(f"Playing speaker {name.upper()} alone; recording 8 seconds...", flush=True)
    with loopback.recorder(samplerate=CAPTURE_SAMPLE_RATE) as recorder:
        worker = threading.Thread(target=play, name=f"speaker_{name}")
        worker.start()
        try:
            if not ready.wait(timeout=10):
                raise TimeoutError("Playback stream did not open")
            if errors:
                raise RuntimeError(f"Playback stream failed: {errors[0]}") from errors[0]
            start_at = time.perf_counter() + 0.5
            launch.set()
            captured = recorder.record(numframes=8 * CAPTURE_SAMPLE_RATE)
        finally:
            launch.set()
            worker.join()

    if errors:
        raise RuntimeError(f"Playback stream failed: {errors[0]}") from errors[0]

    audio = _resample(_to_mono(np.asarray(captured, dtype=np.float32)), CAPTURE_SAMPLE_RATE, TARGET_SAMPLE_RATE)
    if not np.isfinite(audio).all():
        raise ValueError("Captured audio contains NaN or Inf")
    sf.write(output_path, audio, TARGET_SAMPLE_RATE, subtype="PCM_16")
    print(f"Saved: {output_path}", flush=True)
    return output_path


def main() -> int:
    if sys.platform != "win32":
        raise RuntimeError("WASAPI loopback requires Windows")
    sys.stdout.reconfigure(encoding="utf-8")
    speaker = sc.default_speaker()
    loopback = sc.get_microphone(speaker.name, include_loopback=True)
    print(f"System audio device: {speaker.name}", flush=True)
    for name in ("a", "b"):
        capture_one(name, speaker, loopback)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

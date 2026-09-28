from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundcard as sc
import soundfile as sf

from playback_capture import load_reference


OUTPUT_PATH = Path(__file__).resolve().parent / "captured_a_8ch_48k.wav"


def main() -> int:
    if sys.platform != "win32":
        raise RuntimeError("WASAPI loopback requires Windows")
    sys.stdout.reconfigure(encoding="utf-8")
    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)

    speaker = sc.default_speaker()
    loopback = sc.get_microphone(speaker.name, include_loopback=True)
    reference = load_reference("a")
    ready = threading.Event()
    launch = threading.Event()
    errors: list[BaseException] = []
    start_at = 0.0

    def play() -> None:
        try:
            with speaker.player(samplerate=48_000, channels=2) as player:
                ready.set()
                if not launch.wait(timeout=10):
                    raise TimeoutError("Playback start timed out")
                time.sleep(max(0.0, start_at - time.perf_counter()))
                player.play(reference)
                while player.currentpadding:
                    time.sleep(0.005)
        except BaseException as exc:
            errors.append(exc)
            ready.set()

    print(f"System audio device: {speaker.name}", flush=True)
    print("Playing A alone and recording the 8-channel WASAPI loopback...", flush=True)
    with loopback.recorder(samplerate=48_000) as recorder:
        worker = threading.Thread(target=play, name="speaker_a")
        worker.start()
        try:
            if not ready.wait(timeout=10):
                raise TimeoutError("Playback stream did not open")
            if errors:
                raise RuntimeError(f"Playback stream failed: {errors[0]}") from errors[0]
            start_at = time.perf_counter() + 0.5
            launch.set()
            captured = recorder.record(numframes=8 * 48_000)
        finally:
            launch.set()
            worker.join()

    if errors:
        raise RuntimeError(f"Playback stream failed: {errors[0]}") from errors[0]
    if captured.shape != (8 * 48_000, 8) or not np.isfinite(captured).all():
        raise ValueError(f"Unexpected loopback shape or non-finite audio: {captured.shape}")
    sf.write(OUTPUT_PATH, captured, 48_000, subtype="FLOAT")
    print(f"Saved: {OUTPUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

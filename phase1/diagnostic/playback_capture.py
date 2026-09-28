from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundcard as sc
import soundfile as sf
from scipy.signal import correlate


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = Path(__file__).resolve().parent / "wasapi_mixed.wav"
REFERENCE_DIR = ROOT / "phase1" / "reference"
sys.path.insert(0, str(ROOT / "phase1"))

from capture import CAPTURE_SAMPLE_RATE, TARGET_SAMPLE_RATE, _resample, _to_mono  # noqa: E402


def load_reference(name: str) -> np.ndarray:
    path = REFERENCE_DIR / f"speaker_{name}.wav"
    audio, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != TARGET_SAMPLE_RATE or audio.ndim != 1 or not audio.size:
        raise ValueError(f"Expected nonempty 16 kHz mono WAV: {path}")
    if not np.isfinite(audio).all():
        raise ValueError(f"Non-finite reference audio: {path}")
    return _resample(audio, TARGET_SAMPLE_RATE, CAPTURE_SAMPLE_RATE)


def estimate_offset(captured: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    if len(reference) > len(captured):
        raise ValueError("Reference is longer than capture")
    reference = reference.astype(np.float64)
    captured = captured.astype(np.float64)
    centered_reference = reference - reference.mean()
    candidates = correlate(captured, centered_reference, mode="valid", method="fft")
    squares = np.concatenate(([0.0], np.cumsum(captured * captured)))
    segment_energy = squares[len(reference):] - squares[:-len(reference)]
    normalized = candidates / np.sqrt(np.maximum(segment_energy * np.dot(centered_reference, centered_reference), 1e-20))
    index = int(np.argmax(normalized))
    return index / TARGET_SAMPLE_RATE, float(normalized[index])


def main() -> int:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    else:
        raise RuntimeError("WASAPI loopback requires Windows")

    references = {name: load_reference(name) for name in ("a", "b")}
    speaker = sc.default_speaker()
    loopback = sc.get_microphone(speaker.name, include_loopback=True)
    ready = threading.Barrier(3)
    launch = threading.Barrier(3)
    errors: list[tuple[str, BaseException]] = []
    playback_starts: dict[str, float] = {}
    start_at = 0.0

    def play_one(name: str) -> None:
        try:
            # Each worker owns its own WASAPI shared-mode render client.
            with speaker.player(samplerate=CAPTURE_SAMPLE_RATE, channels=2) as player:
                ready.wait(timeout=10)
                launch.wait(timeout=10)
                time.sleep(max(0.0, start_at - time.perf_counter()))
                playback_starts[name] = time.perf_counter()
                player.play(references[name])
                while player.currentpadding:
                    time.sleep(0.005)
        except BaseException as exc:
            errors.append((name, exc))
            ready.abort()
            launch.abort()

    threads = [threading.Thread(target=play_one, args=(name,), name=f"speaker_{name}") for name in references]
    print(f"System audio device: {speaker.name}", flush=True)
    print("Playing A and B through two independent streams; recording WASAPI loopback for 8 seconds...", flush=True)
    with loopback.recorder(samplerate=CAPTURE_SAMPLE_RATE) as recorder:
        for thread in threads:
            thread.start()
        try:
            ready.wait(timeout=10)
            start_at = time.perf_counter() + 0.5
            launch.wait(timeout=10)
            captured = recorder.record(numframes=8 * CAPTURE_SAMPLE_RATE)
        finally:
            for thread in threads:
                thread.join()

    if errors:
        name, exc = errors[0]
        raise RuntimeError(f"Playback stream {name} failed: {exc}") from exc

    audio = _resample(_to_mono(np.asarray(captured, dtype=np.float32)), CAPTURE_SAMPLE_RATE, TARGET_SAMPLE_RATE)
    if not np.isfinite(audio).all():
        raise ValueError("Captured audio contains NaN or Inf")
    sf.write(OUTPUT_PATH, audio, TARGET_SAMPLE_RATE, subtype="PCM_16")
    saved, sample_rate = sf.read(OUTPUT_PATH, dtype="float32")
    peak = float(np.max(np.abs(saved)))
    rms = float(np.sqrt(np.mean(saved * saved)))
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Format: {sample_rate} Hz, 1 channel, {len(saved) / sample_rate:.3f} sec")
    print(f"Peak: {peak:.6f}, RMS: {rms:.6f}")
    print(f"Silent: {'yes' if peak == 0 else 'no'}, clipped samples: {int(np.count_nonzero(np.abs(saved) >= 0.999))}")
    print(f"Playback call offset (B - A): {(playback_starts['b'] - playback_starts['a']) * 1000:.2f} ms")
    for name in references:
        reference = _resample(references[name], CAPTURE_SAMPLE_RATE, TARGET_SAMPLE_RATE)
        offset, score = estimate_offset(saved, reference)
        print(f"Reference {name.upper()} alignment: {offset:.3f} sec, normalized correlation {score:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

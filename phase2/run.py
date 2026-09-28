from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundcard as sc
import soundfile as sf


ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "phase2" / "input" / "captured_mixed_16k.wav"
OUTPUT_DIR = ROOT / "phase2" / "output"
sys.path.insert(0, str(ROOT / "phase0"))
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT / "phase1" / "diagnostic"))

from capture import CAPTURE_SAMPLE_RATE, TARGET_SAMPLE_RATE, _resample  # noqa: E402
from playback_capture import estimate_offset, load_reference  # noqa: E402
from separate import MODEL_NAME, separate_wav  # noqa: E402


def describe(path: Path) -> tuple[np.ndarray, float]:
    audio, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != TARGET_SAMPLE_RATE or audio.ndim != 1 or not audio.size:
        raise ValueError(f"Expected nonempty 16 kHz mono WAV: {path}")
    if not np.isfinite(audio).all():
        raise ValueError(f"Non-finite audio: {path}")
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio * audio)))
    print(
        f"{path.name}: {sample_rate} Hz, 1 channel, {len(audio) / sample_rate:.3f} sec, "
        f"peak={peak:.6f}, RMS={rms:.6f}",
        flush=True,
    )
    if peak == 0:
        raise ValueError(f"Silent audio: {path}")
    return audio, peak


def capture_mixture() -> None:
    speaker = sc.default_speaker()
    loopback = sc.get_microphone(speaker.name, include_loopback=True)
    references = {name: load_reference(name) for name in ("a", "b")}
    ready = threading.Barrier(3)
    launch = threading.Barrier(3)
    errors: list[tuple[str, BaseException]] = []
    start_at = 0.0

    def play_one(name: str) -> None:
        try:
            with speaker.player(samplerate=CAPTURE_SAMPLE_RATE, channels=2) as player:
                ready.wait(timeout=10)
                launch.wait(timeout=10)
                time.sleep(max(0.0, start_at - time.perf_counter()))
                player.play(references[name])
                while player.currentpadding:
                    time.sleep(0.005)
        except BaseException as exc:
            errors.append((name, exc))
            ready.abort()
            launch.abort()

    threads = [threading.Thread(target=play_one, args=(name,)) for name in references]
    print(f"System audio device: {speaker.name}", flush=True)
    print("Playing A and B; capturing 10 seconds from WASAPI loopback channel 0...", flush=True)
    with loopback.recorder(samplerate=CAPTURE_SAMPLE_RATE) as recorder:
        for thread in threads:
            thread.start()
        try:
            ready.wait(timeout=10)
            start_at = time.perf_counter() + 0.5
            launch.wait(timeout=10)
            captured = recorder.record(numframes=10 * CAPTURE_SAMPLE_RATE)
        finally:
            for thread in threads:
                thread.join()

    if errors:
        name, exc = errors[0]
        raise RuntimeError(f"Playback stream {name} failed: {exc}") from exc
    if captured.shape != (10 * CAPTURE_SAMPLE_RATE, 8) or not np.isfinite(captured).all():
        raise ValueError(f"Unexpected WASAPI capture: {captured.shape}")

    mono_16k = _resample(np.asarray(captured[:, 0], dtype=np.float32), CAPTURE_SAMPLE_RATE, TARGET_SAMPLE_RATE)
    INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sf.write(INPUT_PATH, mono_16k, TARGET_SAMPLE_RATE, subtype="PCM_16")


def main() -> int:
    if sys.platform != "win32":
        raise RuntimeError("Phase 2-A requires Windows WASAPI")
    sys.stdout.reconfigure(encoding="utf-8")
    output_paths = [OUTPUT_DIR / f"speaker_{number}.wav" for number in (0, 1)]
    for path in (INPUT_PATH, *output_paths):
        if path.exists():
            raise FileExistsError(path)

    capture_mixture()
    describe(INPUT_PATH)

    print(f"Separating with {MODEL_NAME} on CPU...", flush=True)
    raw_dir = OUTPUT_DIR / "raw"
    raw_paths = separate_wav(INPUT_PATH, raw_dir)
    for source, destination in zip(raw_paths, output_paths):
        source.rename(destination)
    raw_dir.rmdir()
    outputs = [describe(path)[0] for path in output_paths]

    references = [sf.read(ROOT / "phase1" / "reference" / f"speaker_{name}.wav", dtype="float32")[0] for name in ("a", "b")]
    scores = np.array([[estimate_offset(output, reference)[1] for reference in references] for output in outputs])
    permutation = "speaker_0=A, speaker_1=B" if scores[0, 0] + scores[1, 1] >= scores[0, 1] + scores[1, 0] else "speaker_0=B, speaker_1=A"
    print(f"Reference correlations: speaker_0 A={scores[0, 0]:.3f}, B={scores[0, 1]:.3f}; speaker_1 A={scores[1, 0]:.3f}, B={scores[1, 1]:.3f}")
    print(f"Likely mapping: {permutation}")
    print("Phase 2-A: success")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

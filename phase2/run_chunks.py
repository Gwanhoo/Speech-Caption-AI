from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundcard as sc
import soundfile as sf


ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "phase2" / "input"
OUTPUT_DIR = ROOT / "phase2" / "output"
CHUNK_SECONDS = 5
CHUNK_COUNT = 3
sys.path.insert(0, str(ROOT / "phase0"))
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT / "phase1" / "diagnostic"))

from capture import CAPTURE_SAMPLE_RATE, TARGET_SAMPLE_RATE, _resample  # noqa: E402
from playback_capture import estimate_offset, load_reference  # noqa: E402
from separate import CHECKPOINT_MARKER, MODEL_NAME, _load_clearvoice  # noqa: E402


def capture_chunks() -> list[np.ndarray]:
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
                for _ in range(3):
                    player.play(references[name])
                while player.currentpadding:
                    time.sleep(0.005)
        except BaseException as exc:
            errors.append((name, exc))
            ready.abort()
            launch.abort()

    threads = [threading.Thread(target=play_one, args=(name,)) for name in references]
    chunks: list[np.ndarray] = []
    print(f"System audio device: {speaker.name}", flush=True)
    print("Capturing three consecutive 5-second WASAPI chunks (channel 0)...", flush=True)
    with loopback.recorder(samplerate=CAPTURE_SAMPLE_RATE) as recorder:
        for thread in threads:
            thread.start()
        try:
            ready.wait(timeout=10)
            start_at = time.perf_counter() + 0.5
            launch.wait(timeout=10)
            for index in range(CHUNK_COUNT):
                captured = recorder.record(numframes=CHUNK_SECONDS * CAPTURE_SAMPLE_RATE)
                if captured.shape != (CHUNK_SECONDS * CAPTURE_SAMPLE_RATE, 8) or not np.isfinite(captured).all():
                    raise ValueError(f"Unexpected WASAPI chunk {index}: {captured.shape}")
                mono_16k = _resample(
                    np.asarray(captured[:, 0], dtype=np.float32),
                    CAPTURE_SAMPLE_RATE,
                    TARGET_SAMPLE_RATE,
                )
                if len(mono_16k) != CHUNK_SECONDS * TARGET_SAMPLE_RATE or not np.isfinite(mono_16k).all():
                    raise ValueError(f"Invalid resampled chunk {index}")
                chunks.append(mono_16k)
                print(f"Captured chunk {index:03d}: {len(mono_16k) / TARGET_SAMPLE_RATE:.3f} sec", flush=True)
        finally:
            for thread in threads:
                thread.join()

    if errors:
        name, exc = errors[0]
        raise RuntimeError(f"Playback stream {name} failed: {exc}") from exc
    return chunks


def reference_score(output: np.ndarray, reference: np.ndarray) -> float:
    # Short known-reference snippets can appear at different offsets after looping playback.
    segment = 2 * TARGET_SAMPLE_RATE
    return max(estimate_offset(output, reference[start : start + segment])[1] for start in (0, 3 * TARGET_SAMPLE_RATE))


def main() -> int:
    if sys.platform != "win32":
        raise RuntimeError("Phase 2-B requires Windows WASAPI")
    sys.stdout.reconfigure(encoding="utf-8")
    paths = [INPUT_DIR / f"chunk_{index:03d}_mixed_16k.wav" for index in range(CHUNK_COUNT)]
    paths += [
        OUTPUT_DIR / f"chunk_{index:03d}_speaker_{source}.wav"
        for index in range(CHUNK_COUNT)
        for source in (0, 1)
    ]
    for path in paths:
        if path.exists():
            raise FileExistsError(path)

    ClearVoice = _load_clearvoice()
    print(f"Loading {MODEL_NAME} on CPU...", flush=True)
    separator = ClearVoice(task="speech_separation", model_names=[MODEL_NAME])
    marker = Path(separator.models[0].args.checkpoint_dir) / CHECKPOINT_MARKER
    if not marker.is_file():
        raise RuntimeError(f"Pretrained checkpoint not found: {marker}")
    print("Model load count: 1", flush=True)

    chunks = capture_chunks()
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    references = [sf.read(ROOT / "phase1" / "reference" / f"speaker_{name}.wav", dtype="float32")[0] for name in ("a", "b")]
    mappings: list[str] = []
    for index, chunk in enumerate(chunks):
        sf.write(INPUT_DIR / f"chunk_{index:03d}_mixed_16k.wav", chunk, TARGET_SAMPLE_RATE, subtype="PCM_16")
        start = time.perf_counter()
        separated = np.asarray(separator(chunk.reshape(1, -1).astype(np.float32), False))
        elapsed = time.perf_counter() - start
        if separated.ndim != 3 or separated.shape[0] < 2 or separated.shape[1] != 1:
            raise RuntimeError(f"Unexpected separation output for chunk {index}: {separated.shape}")

        outputs = []
        for source in (0, 1):
            audio = separated[source, 0, :]
            if len(audio) == 0 or not np.isfinite(audio).all():
                raise ValueError(f"Invalid speaker {source} in chunk {index}")
            path = OUTPUT_DIR / f"chunk_{index:03d}_speaker_{source}.wav"
            sf.write(path, audio, TARGET_SAMPLE_RATE, subtype="PCM_16")
            saved, sample_rate = sf.read(path, dtype="float32")
            peak = float(np.max(np.abs(saved)))
            rms = float(np.sqrt(np.mean(saved * saved)))
            if sample_rate != TARGET_SAMPLE_RATE or saved.ndim != 1 or peak == 0:
                raise ValueError(f"Invalid saved speaker WAV: {path}")
            outputs.append(saved)
            print(f"  speaker_{source}: {len(saved) / sample_rate:.3f} sec, peak={peak:.6f}, RMS={rms:.6f}", flush=True)

        scores = np.array([[reference_score(output, reference) for reference in references] for output in outputs])
        direct = scores[0, 0] + scores[1, 1]
        swapped = scores[0, 1] + scores[1, 0]
        mapping = "0=A, 1=B" if direct >= swapped else "0=B, 1=A"
        mappings.append(mapping)
        print(f"Chunk {index:03d}: separation={elapsed:.3f} sec, RTF={elapsed / CHUNK_SECONDS:.3f}", flush=True)
        print(f"  reference scores: {scores.tolist()}; likely {mapping}", flush=True)

    print(f"Speaker order consistent by reference scores: {len(set(mappings)) == 1}", flush=True)
    print("Phase 2-B: success", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

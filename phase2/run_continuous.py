from __future__ import annotations

import queue
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundcard as sc
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "phase2" / "input" / "continuous"
OUTPUT_DIR = ROOT / "phase2" / "output" / "continuous"
CHUNK_SECONDS = 5
CHUNK_COUNT = 6
CAPTURE_SECONDS = CHUNK_SECONDS * CHUNK_COUNT
PHASE_NAME = "Phase 2-D"
sys.path.insert(0, str(ROOT / "phase0"))
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT / "phase1" / "diagnostic"))

from benchmark_cuda_amp import run_once  # noqa: E402
from capture import CAPTURE_SAMPLE_RATE, TARGET_SAMPLE_RATE, _resample  # noqa: E402
from playback_capture import load_reference  # noqa: E402
from separate import CHECKPOINT_MARKER, MODEL_NAME, _load_clearvoice  # noqa: E402


def main() -> int:
    if sys.platform != "win32" or not torch.cuda.is_available():
        raise RuntimeError("Phase 2-D requires Windows WASAPI and CUDA")
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
    separator = ClearVoice(task="speech_separation", model_names=[MODEL_NAME])
    speech_model = separator.models[0]
    marker = Path(speech_model.args.checkpoint_dir) / CHECKPOINT_MARKER
    if not marker.is_file():
        raise RuntimeError(f"Pretrained checkpoint not found: {marker}")
    parameter = next(speech_model.model.parameters())
    device = parameter.device
    if speech_model.device.type != "cuda" or device.type != "cuda" or parameter.dtype != torch.float32:
        raise RuntimeError(f"Unexpected model device or dtype: {device}, {parameter.dtype}")
    torch.cuda.synchronize(device)
    print(f"Model load count: 1; device: {device}; AMP: CUDA float16 autocast; weights: {parameter.dtype}", flush=True)

    speaker = sc.default_speaker()
    loopback = sc.get_microphone(speaker.name, include_loopback=True)
    references = {name: load_reference(name) for name in ("a", "b")}
    ready = threading.Barrier(3)
    launch = threading.Event()
    chunks: queue.Queue[tuple[int, np.ndarray, float] | None] = queue.Queue()
    playback_errors: list[tuple[str, BaseException]] = []
    capture_state: dict[str, object] = {"max_queue": 0, "read_gaps": []}
    start_at = 0.0

    def play_one(name: str) -> None:
        try:
            with speaker.player(samplerate=CAPTURE_SAMPLE_RATE, channels=2) as player:
                ready.wait(timeout=10)
                if not launch.wait(timeout=10):
                    raise TimeoutError("Playback launch timed out")
                time.sleep(max(0.0, start_at - time.perf_counter()))
                remaining = (CAPTURE_SECONDS + 2) * CAPTURE_SAMPLE_RATE
                while remaining > 0:
                    segment = references[name][: min(remaining, len(references[name]))]
                    player.play(segment)
                    remaining -= len(segment)
                while player.currentpadding:
                    time.sleep(0.005)
        except BaseException as exc:
            playback_errors.append((name, exc))
            ready.abort()
            launch.set()

    playback_threads = [threading.Thread(target=play_one, args=(name,), name=f"playback_{name}") for name in references]

    def capture_worker() -> None:
        previous_read_end = None
        try:
            with loopback.recorder(samplerate=CAPTURE_SAMPLE_RATE) as recorder:
                for thread in playback_threads:
                    thread.start()
                ready.wait(timeout=10)
                nonlocal start_at
                start_at = time.perf_counter() + 0.5
                launch.set()
                capture_state["start"] = time.perf_counter()
                for index in range(CHUNK_COUNT):
                    read_start = time.perf_counter()
                    if previous_read_end is not None:
                        capture_state["read_gaps"].append(read_start - previous_read_end)
                    raw = recorder.record(numframes=CHUNK_SECONDS * CAPTURE_SAMPLE_RATE)
                    read_end = time.perf_counter()
                    if raw.shape != (CHUNK_SECONDS * CAPTURE_SAMPLE_RATE, 8) or not np.isfinite(raw).all():
                        raise ValueError(f"Unexpected WASAPI chunk {index}: {raw.shape}")
                    chunks.put((index, raw, read_end))
                    capture_state["max_queue"] = max(capture_state["max_queue"], chunks.qsize())
                    print(
                        f"Captured {index:03d}: {index * CHUNK_SECONDS}-{(index + 1) * CHUNK_SECONDS} sec, "
                        f"read wall={read_end - read_start:.3f} sec, queue={chunks.qsize()}",
                        flush=True,
                    )
                    previous_read_end = read_end
                capture_state["end"] = time.perf_counter()
        except BaseException as exc:
            capture_state["error"] = exc
            ready.abort()
            launch.set()
        finally:
            chunks.put(None)
            for thread in playback_threads:
                if thread.is_alive():
                    thread.join()

    print(f"System audio device: {speaker.name}", flush=True)
    print(f"Capturing {CAPTURE_SECONDS} seconds through WASAPI loopback and separating each {CHUNK_SECONDS}-second chunk...", flush=True)
    capture_thread = threading.Thread(target=capture_worker, name="wasapi_capture")
    capture_thread.start()
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    times = []
    processed = 0
    try:
        while True:
            item = chunks.get()
            if item is None:
                break
            index, raw, read_end = item
            queue_wait = time.perf_counter() - read_end
            backlog = chunks.qsize()
            audio = _resample(np.asarray(raw[:, 0], dtype=np.float32), CAPTURE_SAMPLE_RATE, TARGET_SAMPLE_RATE)
            if len(audio) != CHUNK_SECONDS * TARGET_SAMPLE_RATE or not np.isfinite(audio).all():
                raise ValueError(f"Invalid resampled chunk {index}")
            sf.write(INPUT_DIR / f"chunk_{index:03d}_mixed_16k.wav", audio, TARGET_SAMPLE_RATE, subtype="PCM_16")

            separated, elapsed = run_once(separator, audio.reshape(1, -1).astype(np.float32), device)
            print(f"Inference {index:03d}: {elapsed:.3f} sec, RTF={elapsed / CHUNK_SECONDS:.3f}", flush=True)
            for source in (0, 1):
                result = separated[source, 0, :]
                if len(result) != CHUNK_SECONDS * TARGET_SAMPLE_RATE:
                    raise ValueError(f"Unexpected output length for chunk {index}, speaker {source}")
                path = OUTPUT_DIR / f"chunk_{index:03d}_speaker_{source}.wav"
                sf.write(path, result.astype(np.float32, copy=False), TARGET_SAMPLE_RATE, subtype="PCM_16")
                saved, sample_rate = sf.read(path, dtype="float32")
                rms = float(np.sqrt(np.mean(saved * saved)))
                if sample_rate != TARGET_SAMPLE_RATE or saved.ndim != 1 or not np.isfinite(saved).all() or rms <= 1e-5:
                    raise ValueError(f"Invalid speaker WAV: {path}")
                print(f"  speaker_{source}: valid, RMS={rms:.6f}", flush=True)
            times.append(elapsed)
            processed += 1
            print(
                f"Chunk {index:03d}: {index * CHUNK_SECONDS}-{(index + 1) * CHUNK_SECONDS} sec, "
                f"separation={elapsed:.3f} sec, RTF={elapsed / CHUNK_SECONDS:.3f}, "
                f"queue_wait={queue_wait:.3f} sec, backlog_after_get={backlog}",
                flush=True,
            )
    finally:
        capture_thread.join()

    if "error" in capture_state:
        raise RuntimeError("WASAPI capture failed") from capture_state["error"]
    if playback_errors:
        name, exc = playback_errors[0]
        raise RuntimeError(f"Playback stream {name} failed") from exc
    if processed != CHUNK_COUNT:
        raise RuntimeError(f"Processed {processed} of {CHUNK_COUNT} chunks")

    elapsed_capture = capture_state["end"] - capture_state["start"]
    print(f"Capture wall time: {elapsed_capture:.3f} sec; chunks: {processed}", flush=True)
    print(f"Max queue size: {capture_state['max_queue']}; max read-call gap: {max(capture_state['read_gaps']):.6f} sec", flush=True)
    print(f"Mean RTF: {statistics.mean(times) / CHUNK_SECONDS:.3f}; median RTF: {statistics.median(times) / CHUNK_SECONDS:.3f}", flush=True)
    print(f"{PHASE_NAME}: success", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

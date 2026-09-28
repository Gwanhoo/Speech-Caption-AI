from __future__ import annotations

import argparse
import json
import queue
import re
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundcard as sc
import soundfile as sf
import torch
from faster_whisper import WhisperModel


ROOT = Path(__file__).resolve().parent.parent
CHUNK_SECONDS = 5
CHUNK_COUNT = 6
CAPTURE_SECONDS = CHUNK_SECONDS * CHUNK_COUNT
QUEUE_MAXSIZE = 2
SAMPLE_RATE = 16_000
DEBUG_OUTPUT_DIR = ROOT / "phase3" / "output" / "realtime_debug"
SEPARATION_WARMUP_INPUT = ROOT / "phase2" / "input" / "chunk_000_mixed_16k.wav"
MODEL_DIR = (
    ROOT
    / "checkpoints"
    / "faster-whisper"
    / "models--Systran--faster-whisper-base"
    / "snapshots"
    / "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66"
)
SENSEVOICE_MODEL = "FunAudioLLM/SenseVoiceSmall"
MIB = 2**20
STOP = object()

sys.path.insert(0, str(ROOT / "phase0"))
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT / "phase1" / "diagnostic"))
sys.path.insert(0, str(ROOT / "phase2"))

from benchmark_cuda_amp import run_once  # noqa: E402
from capture import CAPTURE_SAMPLE_RATE, _resample  # noqa: E402
from playback_capture import estimate_offset, load_reference  # noqa: E402
from separate import CHECKPOINT_MARKER, MODEL_NAME, _load_clearvoice  # noqa: E402
from separation_recovery import infer_separator, separate_with_fp32_fallback  # noqa: E402


@dataclass
class AudioChunk:
    index: int
    audio: np.ndarray
    stream_end_seconds: float
    capture_started: float
    capture_ended: float
    capture_timestamp: str
    audio_queue_size: int


@dataclass
class SeparatedChunk:
    source: AudioChunk
    speakers: tuple[np.ndarray, np.ndarray]
    separation_started: float
    separation_ended: float
    separation_seconds: float
    audio_queue_backlog: int
    separated_queue_size: int
    torch_allocated_mib: float
    torch_reserved_mib: float
    torch_peak_allocated_mib: float
    speaker_rms: tuple[float, float]
    reference_scores: list[list[float]]
    pair_correlation: float
    speaker_mapping: str


class GpuMonitor:
    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.active_stages: set[str] = set()
        self.samples: list[int] = []
        self.stage_peaks: dict[str, int] = {}
        self.thread = threading.Thread(target=self._run, name="gpu_monitor")

    @staticmethod
    def used_mib() -> int | None:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode:
            return None
        try:
            return int(result.stdout.strip().splitlines()[0])
        except (IndexError, ValueError):
            return None

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join()

    def enter(self, stage: str) -> None:
        with self.lock:
            self.active_stages.add(stage)

    def leave(self, stage: str) -> None:
        with self.lock:
            self.active_stages.discard(stage)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            used = self.used_mib()
            if used is not None:
                self.samples.append(used)
                with self.lock:
                    stages = tuple(self.active_stages)
                for stage in stages:
                    self.stage_peaks[stage] = max(self.stage_peaks.get(stage, 0), used)
            self.stop_event.wait(0.25)


def format_stream_time(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": float(np.percentile(values, 95)),
        "max": max(values),
    }


def normalize_transcript(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]+", "", text).lower()


def transcript_quality(results: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(results, key=lambda item: item["chunk"])
    transcripts = [text for item in ordered for text in item["transcripts"]]
    nonempty = [text for text in transcripts if text]
    fragments = sum(not text.rstrip().endswith((".", "?", "!", "。", "？", "！")) for text in nonempty)
    too_short = sum(len(normalize_transcript(text)) < 4 for text in nonempty)
    adjacent_repeats = 0
    for speaker in (0, 1):
        previous = ""
        for item in ordered:
            current = normalize_transcript(item["transcripts"][speaker])
            if previous and current:
                shorter, longer = sorted((previous, current), key=len)
                if len(shorter) >= 6 and shorter in longer:
                    adjacent_repeats += 1
            previous = current
    return {
        "speaker_input_count": len(transcripts),
        "nonempty_count": len(nonempty),
        "empty_count": len(transcripts) - len(nonempty),
        "fragment_count": fragments,
        "adjacent_repeat_count": adjacent_repeats,
        "too_short_count": too_short,
        "reference_transcript_available": False,
    }


def enqueue_or_drop(
    target: queue.Queue[Any],
    item: Any,
    queue_name: str,
    chunk_index: int,
    dropped: list[tuple[str, int]],
    lock: threading.Lock,
) -> bool:
    try:
        target.put_nowait(item)
        return True
    except queue.Full:
        # Keep admitted FIFO work and reject new work so latency stays bounded.
        with lock:
            dropped.append((queue_name, chunk_index))
        print(f"[DROP] {queue_name} full; dropped chunk {chunk_index:03d}", flush=True)
        return False


def put_stop(target: queue.Queue[Any], queue_name: str) -> None:
    try:
        target.put(STOP, timeout=CHUNK_SECONDS * 2)
    except queue.Full as exc:
        raise RuntimeError(f"Timed out stopping {queue_name}") from exc


def data_backlog(target: queue.Queue[Any]) -> int:
    with target.mutex:
        return sum(item is not STOP for item in target.queue)


def transcribe_audio(model: Any, audio: np.ndarray, backend: str = "whisper") -> tuple[str, float]:
    started = time.perf_counter()
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if backend == "whisper":
        segments, _ = model.transcribe(
            audio,
            language="ko",
            beam_size=5,
            condition_on_previous_text=False,
        )
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
    elif backend == "sensevoice":
        result = model.generate(input=audio, language="ko", use_itn=True)
        raw_text = " ".join(str(item.get("text", "")).strip() for item in result).strip()
        transcript = re.sub(r"<\|[^|>]+\|>", "", raw_text).strip()
    else:
        raise ValueError(f"Unsupported STT backend: {backend}")
    return transcript, time.perf_counter() - started


def load_models(monitor: GpuMonitor, stt_backend: str = "whisper"):
    if not (MODEL_DIR / "model.bin").is_file():
        raise FileNotFoundError(MODEL_DIR / "model.bin")

    baseline = monitor.used_mib()
    print(f"GPU VRAM before models: {baseline} MiB", flush=True)

    ClearVoice = _load_clearvoice()
    separator = ClearVoice(task="speech_separation", model_names=[MODEL_NAME])
    speech_model = separator.models[0]
    marker = Path(speech_model.args.checkpoint_dir) / CHECKPOINT_MARKER
    if not marker.is_file():
        raise RuntimeError(f"Pretrained checkpoint not found: {marker}")
    parameter = next(speech_model.model.parameters())
    device = parameter.device
    if speech_model.device.type != "cuda" or device.type != "cuda":
        raise RuntimeError(f"MossFormer2 did not load on CUDA: {speech_model.device}, {device}")
    torch.cuda.synchronize(device)
    after_separator = monitor.used_mib()
    print(
        f"Separation model loaded: {MODEL_NAME}, device={device}, weights={parameter.dtype}; "
        f"GPU VRAM={after_separator} MiB",
        flush=True,
    )

    if stt_backend == "whisper":
        stt_model = WhisperModel(str(MODEL_DIR), device="cuda", compute_type="float16")
        stt_description = (
            f"faster-whisper base, device={stt_model.model.device}, "
            f"compute_type={stt_model.model.compute_type}"
        )
    elif stt_backend == "sensevoice":
        from funasr import AutoModel

        stt_model = AutoModel(
            model=SENSEVOICE_MODEL,
            hub="hf",
            device="cuda",
            disable_update=True,
        )
        stt_description = f"SenseVoiceSmall, model={SENSEVOICE_MODEL}, device=cuda, language=ko"
    else:
        raise ValueError(f"Unsupported STT backend: {stt_backend}")
    after_models = monitor.used_mib()
    print(
        f"STT model loaded: {stt_description}; GPU VRAM={after_models} MiB",
        flush=True,
    )
    return separator, device, stt_model, baseline, after_separator, after_models


def warm_up(
    separator,
    device: torch.device,
    stt_model: Any,
    monitor: GpuMonitor,
    stt_backend: str = "whisper",
) -> None:
    audio, sample_rate = sf.read(SEPARATION_WARMUP_INPUT, dtype="float32")
    if sample_rate != SAMPLE_RATE or audio.ndim != 1 or len(audio) != 5 * SAMPLE_RATE:
        raise ValueError(f"Invalid warm-up input: {SEPARATION_WARMUP_INPUT}")

    separated = None
    for index in range(2):
        monitor.enter("separation")
        try:
            torch.cuda.reset_peak_memory_stats(device)
            recovery = separate_with_fp32_fallback(
                audio,
                amp_inference=lambda value: infer_separator(
                    separator, value.reshape(1, -1), device, True
                ),
                fp32_inference=lambda value: infer_separator(
                    separator, value.reshape(1, -1), device, False
                ),
            )
        finally:
            monitor.leave("separation")
        separated = recovery.output
        print(
            f"Separation warm-up {index + 1}: {recovery.total_seconds:.3f} sec, "
            f"fp32_fallback={recovery.fallback_attempted}",
            flush=True,
        )

    monitor.enter("stt")
    try:
        _, elapsed = transcribe_audio(stt_model, separated[0, 0, :], backend=stt_backend)
    finally:
        monitor.leave("stt")
    warm_up_name = "SenseVoice" if stt_backend == "sensevoice" else "STT"
    print(f"{warm_up_name} warm-up: {elapsed:.3f} sec", flush=True)


def main() -> int:
    global CHUNK_SECONDS, CHUNK_COUNT, CAPTURE_SECONDS

    parser = argparse.ArgumentParser(description="Phase 3-B real-time separation and STT pipeline")
    parser.add_argument("--save-wav", action="store_true", help="Save captured and separated WAV files")
    parser.add_argument("--chunk-seconds", type=int, choices=(1, 2, 3, 5), default=5)
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--analyze-separation-quality", action="store_true")
    args = parser.parse_args()
    if args.duration <= 0 or args.duration % args.chunk_seconds:
        parser.error("--duration must be positive and divisible by --chunk-seconds")
    CHUNK_SECONDS = args.chunk_seconds
    CAPTURE_SECONDS = args.duration
    CHUNK_COUNT = CAPTURE_SECONDS // CHUNK_SECONDS
    debug_output_dir = DEBUG_OUTPUT_DIR / f"{CHUNK_SECONDS}s"

    if sys.platform != "win32" or not torch.cuda.is_available():
        raise RuntimeError("Phase 3-B requires Windows WASAPI and CUDA")
    sys.stdout.reconfigure(encoding="utf-8")

    monitor = GpuMonitor()
    monitor.start()
    errors: list[tuple[str, BaseException]] = []
    dropped: list[tuple[str, int]] = []
    results: list[dict[str, Any]] = []
    state_lock = threading.Lock()
    audio_queue: queue.Queue[AudioChunk | object] = queue.Queue(maxsize=QUEUE_MAXSIZE)
    separated_queue: queue.Queue[SeparatedChunk | object] = queue.Queue(maxsize=QUEUE_MAXSIZE)

    try:
        separator, device, stt_model, baseline_vram, separator_vram, models_vram = load_models(monitor)
        warm_up(separator, device, stt_model, monitor)
        after_warmup_vram = monitor.used_mib()
        print(f"GPU VRAM after warm-up: {after_warmup_vram} MiB", flush=True)

        speaker = sc.default_speaker()
        loopback = sc.get_microphone(speaker.name, include_loopback=True)
        references = {name: load_reference(name) for name in ("a", "b")}
        quality_references = []
        if args.analyze_separation_quality:
            for name in ("a", "b"):
                reference, reference_rate = sf.read(
                    ROOT / "phase1" / "reference" / f"speaker_{name}.wav",
                    dtype="float32",
                )
                if reference_rate != SAMPLE_RATE or reference.ndim != 1:
                    raise ValueError(f"Invalid quality reference: speaker_{name}.wav")
                quality_references.append(np.tile(reference, 4))
        playback_ready = threading.Barrier(3)
        playback_launch = threading.Event()
        playback_errors: list[tuple[str, BaseException]] = []
        capture_state: dict[str, Any] = {"max_queue_depth": 0, "captured": 0}
        separation_state: dict[str, Any] = {
            "max_audio_backlog": 0,
            "max_queue_depth": 0,
            "inferred": 0,
            "delivered": 0,
        }
        stt_state: dict[str, Any] = {"max_separated_backlog": 0, "processed": 0}
        playback_start_at = 0.0

        def play_one(name: str) -> None:
            try:
                with speaker.player(samplerate=CAPTURE_SAMPLE_RATE, channels=2) as player:
                    playback_ready.wait(timeout=10)
                    if not playback_launch.wait(timeout=10):
                        raise TimeoutError("Playback launch timed out")
                    time.sleep(max(0.0, playback_start_at - time.perf_counter()))
                    remaining = (CAPTURE_SECONDS + 2) * CAPTURE_SAMPLE_RATE
                    while remaining > 0:
                        segment = references[name][: min(remaining, len(references[name]))]
                        player.play(segment)
                        remaining -= len(segment)
                    while player.currentpadding:
                        time.sleep(0.005)
            except BaseException as exc:
                playback_errors.append((name, exc))
                playback_launch.set()
                try:
                    playback_ready.abort()
                except threading.BrokenBarrierError:
                    pass

        playback_threads = [
            threading.Thread(target=play_one, args=(name,), name=f"playback_{name}")
            for name in references
        ]

        def capture_worker() -> None:
            nonlocal playback_start_at
            try:
                with loopback.recorder(samplerate=CAPTURE_SAMPLE_RATE) as recorder:
                    for thread in playback_threads:
                        thread.start()
                    playback_ready.wait(timeout=10)
                    playback_start_at = time.perf_counter() + 0.5
                    playback_launch.set()
                    for index in range(CHUNK_COUNT):
                        capture_started = time.perf_counter()
                        raw = recorder.record(numframes=CHUNK_SECONDS * CAPTURE_SAMPLE_RATE)
                        capture_ended = time.perf_counter()
                        if (
                            raw.ndim != 2
                            or raw.shape[0] != CHUNK_SECONDS * CAPTURE_SAMPLE_RATE
                            or raw.shape[1] < 1
                            or not np.isfinite(raw).all()
                        ):
                            raise ValueError(f"Unexpected WASAPI chunk {index}: {raw.shape}")
                        audio = _resample(
                            np.asarray(raw[:, 0], dtype=np.float32),
                            CAPTURE_SAMPLE_RATE,
                            SAMPLE_RATE,
                        )
                        if len(audio) != CHUNK_SECONDS * SAMPLE_RATE or not np.isfinite(audio).all():
                            raise ValueError(f"Invalid resampled chunk {index}")
                        audio = np.ascontiguousarray(audio, dtype=np.float32)
                        chunk = AudioChunk(
                            index=index,
                            audio=audio,
                            stream_end_seconds=(index + 1) * CHUNK_SECONDS,
                            capture_started=capture_started,
                            capture_ended=capture_ended,
                            capture_timestamp=datetime.now().astimezone().isoformat(timespec="milliseconds"),
                            audio_queue_size=audio_queue.qsize(),
                        )
                        if args.save_wav:
                            debug_output_dir.mkdir(parents=True, exist_ok=True)
                            sf.write(
                                debug_output_dir / f"chunk_{index:03d}_mixed.wav",
                                audio,
                                SAMPLE_RATE,
                                subtype="PCM_16",
                            )
                        accepted = enqueue_or_drop(
                            audio_queue, chunk, "audio_queue", index, dropped, state_lock
                        )
                        if accepted:
                            chunk.audio_queue_size = audio_queue.qsize()
                            capture_state["max_queue_depth"] = max(
                                capture_state["max_queue_depth"], audio_queue.qsize()
                            )
                            capture_state["captured"] += 1
                        print(
                            f"Capture {index:03d}: timestamp={chunk.capture_timestamp}, "
                            f"wall={capture_ended - capture_started:.3f} sec, "
                            f"audio_queue={audio_queue.qsize()}/{QUEUE_MAXSIZE}",
                            flush=True,
                        )
            except BaseException as exc:
                with state_lock:
                    errors.append(("capture", exc))
                playback_launch.set()
            finally:
                try:
                    put_stop(audio_queue, "audio_queue")
                except BaseException as exc:
                    with state_lock:
                        errors.append(("capture-stop", exc))
                for thread in playback_threads:
                    if thread.is_alive():
                        thread.join()

        def separation_worker() -> None:
            try:
                while True:
                    item = audio_queue.get()
                    try:
                        if item is STOP:
                            break
                        assert isinstance(item, AudioChunk)
                        backlog = data_backlog(audio_queue)
                        separation_state["max_audio_backlog"] = max(
                            separation_state["max_audio_backlog"], backlog
                        )
                        separation_started = time.perf_counter()
                        monitor.enter("separation")
                        try:
                            separated, elapsed = run_once(
                                separator,
                                item.audio.reshape(1, -1),
                                device,
                            )
                        finally:
                            monitor.leave("separation")
                        separation_ended = time.perf_counter()
                        speakers = tuple(
                            np.ascontiguousarray(separated[source, 0, :], dtype=np.float32)
                            for source in (0, 1)
                        )
                        speaker_rms = tuple(
                            float(np.sqrt(np.mean(audio * audio))) for audio in speakers
                        )
                        for source, audio in enumerate(speakers):
                            rms = speaker_rms[source]
                            if len(audio) != CHUNK_SECONDS * SAMPLE_RATE or not np.isfinite(audio).all() or rms <= 1e-5:
                                raise ValueError(
                                    f"Invalid separated audio: chunk={item.index}, speaker={source}, RMS={rms}"
                                )
                            if args.save_wav:
                                sf.write(
                                    debug_output_dir / f"chunk_{item.index:03d}_speaker_{source}.wav",
                                    audio,
                                    SAMPLE_RATE,
                                    subtype="PCM_16",
                                )
                        pair_correlation = float(np.corrcoef(speakers)[0, 1])
                        if quality_references:
                            scores = np.array(
                                [
                                    [estimate_offset(reference, output)[1] for reference in quality_references]
                                    for output in speakers
                                ]
                            )
                            direct = scores[0, 0] + scores[1, 1]
                            swapped = scores[0, 1] + scores[1, 0]
                            if (
                                abs(direct - swapped) < 0.2
                                or max(
                                    min(scores[0, 0], scores[1, 1]),
                                    min(scores[0, 1], scores[1, 0]),
                                )
                                < 0.3
                            ):
                                mapping = "uncertain"
                            else:
                                mapping = "0=A,1=B" if direct > swapped else "0=B,1=A"
                        else:
                            scores = np.empty((2, 0))
                            mapping = "not_evaluated"
                        separated_item = SeparatedChunk(
                            source=item,
                            speakers=(speakers[0], speakers[1]),
                            separation_started=separation_started,
                            separation_ended=separation_ended,
                            separation_seconds=elapsed,
                            audio_queue_backlog=backlog,
                            separated_queue_size=separated_queue.qsize(),
                            torch_allocated_mib=torch.cuda.memory_allocated(device) / MIB,
                            torch_reserved_mib=torch.cuda.memory_reserved(device) / MIB,
                            torch_peak_allocated_mib=torch.cuda.max_memory_allocated(device) / MIB,
                            speaker_rms=(speaker_rms[0], speaker_rms[1]),
                            reference_scores=scores.tolist(),
                            pair_correlation=pair_correlation,
                            speaker_mapping=mapping,
                        )
                        separation_state["inferred"] += 1
                        accepted = enqueue_or_drop(
                            separated_queue,
                            separated_item,
                            "separated_queue",
                            item.index,
                            dropped,
                            state_lock,
                        )
                        if accepted:
                            separated_item.separated_queue_size = separated_queue.qsize()
                            separation_state["max_queue_depth"] = max(
                                separation_state["max_queue_depth"], separated_queue.qsize()
                            )
                            separation_state["delivered"] += 1
                        print(
                            f"Separation {item.index:03d}: {elapsed:.3f} sec, "
                            f"RTF={elapsed / CHUNK_SECONDS:.3f}, audio_backlog={backlog}, "
                            f"separated_queue={separated_queue.qsize()}/{QUEUE_MAXSIZE}, "
                            f"pair_corr={pair_correlation:.3f}, mapping={mapping}",
                            flush=True,
                        )
                    finally:
                        audio_queue.task_done()
            except BaseException as exc:
                with state_lock:
                    errors.append(("separation", exc))
            finally:
                try:
                    put_stop(separated_queue, "separated_queue")
                except BaseException as exc:
                    with state_lock:
                        errors.append(("separation-stop", exc))

        def stt_worker() -> None:
            try:
                while True:
                    item = separated_queue.get()
                    try:
                        if item is STOP:
                            break
                        assert isinstance(item, SeparatedChunk)
                        separated_backlog = data_backlog(separated_queue)
                        stt_state["max_separated_backlog"] = max(
                            stt_state["max_separated_backlog"], separated_backlog
                        )
                        stt_started = time.perf_counter()
                        speaker_times: list[float] = []
                        transcripts: list[str] = []
                        for speaker, audio in enumerate(item.speakers):
                            monitor.enter("stt")
                            try:
                                transcript, elapsed = transcribe_audio(stt_model, audio)
                            finally:
                                monitor.leave("stt")
                            transcripts.append(transcript)
                            speaker_times.append(elapsed)
                            if transcript:
                                print(
                                    f"[{format_stream_time(item.source.stream_end_seconds)}] "
                                    f"speaker_{speaker}: {transcript}",
                                    flush=True,
                                )
                        stt_ended = time.perf_counter()
                        gpu_vram_mib = monitor.used_mib()
                        metric = {
                            "chunk": item.source.index,
                            "chunk_seconds": CHUNK_SECONDS,
                            "capture_timestamp": item.source.capture_timestamp,
                            "capture_seconds": item.source.capture_ended - item.source.capture_started,
                            "separation_started_after_capture_start": item.separation_started - item.source.capture_started,
                            "separation_ended_after_capture_start": item.separation_ended - item.source.capture_started,
                            "separation_seconds": item.separation_seconds,
                            "separation_rtf": item.separation_seconds / CHUNK_SECONDS,
                            "speaker_0_stt_seconds": speaker_times[0],
                            "speaker_1_stt_seconds": speaker_times[1],
                            "stt_total_seconds": stt_ended - stt_started,
                            "end_to_end_seconds": stt_ended - item.source.capture_started,
                            "post_capture_latency_seconds": stt_ended - item.source.capture_ended,
                            "estimated_end_to_end_seconds": CHUNK_SECONDS + stt_ended - item.source.capture_ended,
                            "audio_queue_depth": item.source.audio_queue_size,
                            "separated_queue_depth": item.separated_queue_size,
                            "audio_queue_backlog": item.audio_queue_backlog,
                            "separated_queue_backlog": separated_backlog,
                            "torch_allocated_mib": item.torch_allocated_mib,
                            "torch_reserved_mib": item.torch_reserved_mib,
                            "torch_peak_allocated_mib": item.torch_peak_allocated_mib,
                            "gpu_vram_mib": gpu_vram_mib,
                            "speaker_rms": item.speaker_rms,
                            "reference_scores": item.reference_scores,
                            "pair_correlation": item.pair_correlation,
                            "speaker_mapping": item.speaker_mapping,
                            "transcripts": transcripts,
                            "dropped": False,
                            "error": None,
                        }
                        with state_lock:
                            results.append(metric)
                        stt_state["processed"] += 1
                        print(
                            f"Chunk {item.source.index:03d} metrics: "
                            f"capture={metric['capture_seconds']:.3f}s, "
                            f"separation={metric['separation_seconds']:.3f}s, "
                            f"stt=[{speaker_times[0]:.3f}, {speaker_times[1]:.3f}]s, "
                            f"stt_total={metric['stt_total_seconds']:.3f}s, "
                            f"end_to_end={metric['end_to_end_seconds']:.3f}s, "
                            f"post_capture={metric['post_capture_latency_seconds']:.3f}s, "
                            f"queues=[{metric['audio_queue_backlog']}, {metric['separated_queue_backlog']}], "
                            f"VRAM={metric['torch_allocated_mib']:.1f}/{metric['torch_reserved_mib']:.1f} MiB",
                            flush=True,
                        )
                    finally:
                        separated_queue.task_done()
            except BaseException as exc:
                with state_lock:
                    errors.append(("stt", exc))

        print(
            f"Starting Phase 3-B: device={speaker.name}, duration={CAPTURE_SECONDS}s, "
            f"chunk={CHUNK_SECONDS}s, queues={QUEUE_MAXSIZE}/{QUEUE_MAXSIZE}",
            flush=True,
        )
        separation_thread = threading.Thread(target=separation_worker, name="separation_worker")
        stt_thread = threading.Thread(target=stt_worker, name="stt_worker")
        capture_thread = threading.Thread(target=capture_worker, name="capture_worker")
        separation_thread.start()
        stt_thread.start()
        capture_thread.start()
        capture_thread.join()
        separation_thread.join()
        stt_thread.join()

        if playback_errors:
            errors.extend((f"playback-{name}", exc) for name, exc in playback_errors)

        results.sort(key=lambda item: item["chunk"])
        performance: dict[str, Any] = {}
        quality: dict[str, Any] = {}
        separation_quality: dict[str, Any] = {}
        if results:
            speaker_stt_times = [
                value
                for item in results
                for value in (item["speaker_0_stt_seconds"], item["speaker_1_stt_seconds"])
            ]
            performance = {
                "separation": distribution([item["separation_seconds"] for item in results]),
                "speaker_stt": distribution(speaker_stt_times),
                "chunk_stt_total": distribution([item["stt_total_seconds"] for item in results]),
                "post_capture_latency": distribution(
                    [item["post_capture_latency_seconds"] for item in results]
                ),
                "estimated_end_to_end": distribution(
                    [item["estimated_end_to_end_seconds"] for item in results]
                ),
            }
            quality = transcript_quality(results)
            mappings = [item["speaker_mapping"] for item in results]
            separation_quality = {
                "silent_output_count": sum(
                    rms <= 1e-5 for item in results for rms in item["speaker_rms"]
                ),
                "collapse_candidate_count": sum(
                    abs(item["pair_correlation"]) >= 0.98 for item in results
                ),
                "pair_correlation_abs_max": max(
                    abs(item["pair_correlation"]) for item in results
                ),
                "uncertain_mapping_count": mappings.count("uncertain"),
                "mapping_counts": {mapping: mappings.count(mapping) for mapping in set(mappings)},
            }
        print("\n========== Phase 3-B summary ==========", flush=True)
        print(f"Capture success: {capture_state['captured']}/{CHUNK_COUNT}")
        print(f"Separation inference success: {separation_state['inferred']}/{CHUNK_COUNT}")
        print(f"Separation queue delivery: {separation_state['delivered']}/{CHUNK_COUNT}")
        print(f"STT success: {stt_state['processed']}/{CHUNK_COUNT}")
        print(f"Dropped: {dropped}")
        print(f"Errors: {[(stage, type(exc).__name__, str(exc)) for stage, exc in errors]}")
        if results:
            print(f"Mean separation: {statistics.mean(item['separation_seconds'] for item in results):.3f} sec")
            print(
                f"Mean speaker STT: {statistics.mean(value for item in results for value in (item['speaker_0_stt_seconds'], item['speaker_1_stt_seconds'])):.3f} sec"
            )
            print(f"Mean chunk STT total: {statistics.mean(item['stt_total_seconds'] for item in results):.3f} sec")
            print(f"Mean end-to-end: {statistics.mean(item['end_to_end_seconds'] for item in results):.3f} sec")
            print(f"Mean post-capture latency: {statistics.mean(item['post_capture_latency_seconds'] for item in results):.3f} sec")
            print(
                f"Post-capture median/p95/max: "
                f"{performance['post_capture_latency']['median']:.3f}/"
                f"{performance['post_capture_latency']['p95']:.3f}/"
                f"{performance['post_capture_latency']['max']:.3f} sec"
            )
            print(f"Max audio queue depth: {capture_state['max_queue_depth']}")
            print(f"Max audio queue backlog: {separation_state['max_audio_backlog']}")
            print(f"Max separated queue depth: {separation_state['max_queue_depth']}")
            print(f"Max separated queue backlog: {stt_state['max_separated_backlog']}")
        print(f"GPU baseline: {baseline_vram} MiB")
        print(f"GPU after separation model: {separator_vram} MiB")
        print(f"GPU after both models: {models_vram} MiB")
        print(f"GPU after warm-up: {after_warmup_vram} MiB")
        print(f"GPU separation-stage peak: {monitor.stage_peaks.get('separation')} MiB")
        print(f"GPU STT-stage peak: {monitor.stage_peaks.get('stt')} MiB")
        print(f"GPU overall peak: {max(monitor.samples) if monitor.samples else None} MiB")

        success = (
            capture_state["captured"] == CHUNK_COUNT
            and separation_state["delivered"] == CHUNK_COUNT
            and stt_state["processed"] == CHUNK_COUNT
            and not dropped
            and not errors
            and all(item["separation_seconds"] < CHUNK_SECONDS for item in results)
            and any(text for item in results for text in item["transcripts"])
        )
        summary = {
            "phase": "3-C",
            "chunk_seconds": CHUNK_SECONDS,
            "duration_seconds": CAPTURE_SECONDS,
            "chunk_count": CHUNK_COUNT,
            "queue_maxsize": QUEUE_MAXSIZE,
            "capture_success": capture_state["captured"],
            "separation_success": separation_state["inferred"],
            "separation_delivered": separation_state["delivered"],
            "stt_success": stt_state["processed"],
            "dropped": dropped,
            "errors": [
                {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
                for stage, exc in errors
            ],
            "max_audio_queue_depth": capture_state["max_queue_depth"],
            "max_audio_queue_backlog": separation_state["max_audio_backlog"],
            "max_separated_queue_depth": separation_state["max_queue_depth"],
            "max_separated_queue_backlog": stt_state["max_separated_backlog"],
            "performance": performance,
            "transcript_quality": quality,
            "separation_quality": separation_quality,
            "gpu": {
                "baseline_mib": baseline_vram,
                "after_separation_model_mib": separator_vram,
                "after_both_models_mib": models_vram,
                "after_warmup_mib": after_warmup_vram,
                "separation_stage_peak_mib": monitor.stage_peaks.get("separation"),
                "stt_stage_peak_mib": monitor.stage_peaks.get("stt"),
                "overall_peak_mib": max(monitor.samples) if monitor.samples else None,
            },
            "chunks": results,
            "success": success,
        }
        if args.json_output:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"JSON: {args.json_output}", flush=True)
        print(f"Phase 3-B: {'PASS' if success else 'FAIL'}", flush=True)
        return 0 if success else 1
    finally:
        monitor.stop()


if __name__ == "__main__":
    raise SystemExit(main())

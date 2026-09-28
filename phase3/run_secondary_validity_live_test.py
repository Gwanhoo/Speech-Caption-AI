"""Orchestrate the fixed A/B WASAPI Live validity-evidence test.

This runner deliberately owns only test orchestration.  The child process is
the unmodified overlap pipeline, so playback never participates in separation,
VAD, STT, subtitle, queue, or validity-classification decisions.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundcard as sc
import soundfile as sf
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parent.parent
PIPELINE_PATH = ROOT / "phase3" / "run_overlap_pipeline.py"
TEST_AUDIO_DIR = ROOT / "phase3" / "test_audio"
OUTPUT_DIR = ROOT / "phase3" / "output"
DEFAULT_A_PATH = TEST_AUDIO_DIR / "A.wav"
DEFAULT_B_PATH = TEST_AUDIO_DIR / "B.wav"
DEFAULT_JSON_OUTPUT = OUTPUT_DIR / "live_secondary_validity_75s.json"
DEFAULT_SCHEDULE_OUTPUT = OUTPUT_DIR / "live_secondary_validity_75s_schedule.json"

CAPTURE_READY_MARKER = "Live mode: test WAV playback disabled; capturing WASAPI loopback only."
CAPTURE_SECONDS = 75.0
PLAYBACK_SAMPLE_RATE = 48_000
MINIMUM_SOURCE_SECONDS = 20.0


@dataclass(frozen=True)
class ScheduleSegment:
    start_seconds: float
    end_seconds: float
    kind: str
    sources: tuple[str, ...]


SCHEDULE = (
    ScheduleSegment(0.0, 5.0, "silence", ()),
    ScheduleSegment(5.0, 25.0, "A_only", ("A",)),
    ScheduleSegment(25.0, 30.0, "silence", ()),
    ScheduleSegment(30.0, 50.0, "B_only", ("B",)),
    ScheduleSegment(50.0, 55.0, "silence", ()),
    ScheduleSegment(55.0, 70.0, "A+B", ("A", "B")),
    ScheduleSegment(70.0, 75.0, "silence", ()),
)


@dataclass(frozen=True)
class AudioInfo:
    path: str
    sample_rate: int
    channels: int
    duration_seconds: float
    frames: int
    subtype: str
    peak: float


@dataclass(frozen=True)
class PlaybackCommand:
    start_at: float
    duration_seconds: float


def utc_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def validate_schedule(schedule: tuple[ScheduleSegment, ...] = SCHEDULE) -> None:
    if not schedule:
        raise ValueError("Playback schedule is empty")
    previous_end = 0.0
    for segment in schedule:
        if segment.start_seconds != previous_end:
            raise ValueError(f"Schedule is not contiguous at {segment.kind}")
        if segment.end_seconds <= segment.start_seconds:
            raise ValueError(f"Schedule has non-positive duration: {segment.kind}")
        previous_end = segment.end_seconds
    if previous_end != CAPTURE_SECONDS:
        raise ValueError(
            f"Schedule ends at {previous_end:.3f}s, expected {CAPTURE_SECONDS:.3f}s"
        )


def required_source_seconds(
    schedule: tuple[ScheduleSegment, ...] = SCHEDULE,
) -> dict[str, float]:
    """A source restarts at each segment, so only the longest segment matters."""
    required: dict[str, float] = {}
    for segment in schedule:
        duration = segment.end_seconds - segment.start_seconds
        for source in segment.sources:
            required[source] = max(required.get(source, 0.0), duration)
    return required


def inspect_wav(path: Path, *, minimum_seconds: float) -> AudioInfo:
    if not path.is_file():
        raise FileNotFoundError(f"Required test WAV does not exist: {path}")
    info = sf.info(path)
    if info.samplerate <= 0 or info.channels <= 0 or info.frames <= 0:
        raise ValueError(f"Invalid or empty WAV: {path}")
    duration = info.frames / info.samplerate
    if duration + 1e-9 < minimum_seconds:
        raise ValueError(
            f"{path.name} is {duration:.3f}s, but this schedule needs at least "
            f"{minimum_seconds:.3f}s for one playback segment"
        )
    audio, _ = sf.read(path, dtype="float32", always_2d=True)
    if not np.isfinite(audio).all():
        raise ValueError(f"Non-finite audio samples: {path}")
    peak = float(np.max(np.abs(audio)))
    if peak == 0.0:
        raise ValueError(f"Silent test WAV: {path}")
    return AudioInfo(
        path=str(path),
        sample_rate=info.samplerate,
        channels=info.channels,
        duration_seconds=duration,
        frames=info.frames,
        subtype=info.subtype,
        peak=peak,
    )


def load_for_playback(path: Path, *, duration_seconds: float) -> np.ndarray:
    """Convert only the in-memory playback copy to 48 kHz stereo float32."""
    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1, dtype=np.float32)
    source_frames = int(math.ceil(duration_seconds * source_rate))
    mono = mono[:source_frames]
    if source_rate != PLAYBACK_SAMPLE_RATE:
        divisor = math.gcd(source_rate, PLAYBACK_SAMPLE_RATE)
        mono = resample_poly(
            mono, PLAYBACK_SAMPLE_RATE // divisor, source_rate // divisor
        ).astype(np.float32, copy=False)
    target_frames = int(round(duration_seconds * PLAYBACK_SAMPLE_RATE))
    if len(mono) < target_frames:
        raise ValueError(f"Converted playback audio is too short: {path}")
    return np.repeat(mono[:target_frames, np.newaxis], 2, axis=1)


class ScheduleLog:
    def __init__(self, path: Path, schedule: tuple[ScheduleSegment, ...]) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._schedule = schedule
        self._events: list[dict[str, Any]] = []
        self._metadata: dict[str, Any] = {}

    def set_metadata(self, **metadata: Any) -> None:
        with self._lock:
            self._metadata.update(metadata)
            self._save_locked()

    def event(self, name: str, *, test_start: float | None = None, **details: Any) -> None:
        elapsed = None if test_start is None else time.perf_counter() - test_start
        record: dict[str, Any] = {"event": name, "timestamp": utc_timestamp()}
        if elapsed is not None:
            record["test_elapsed_seconds"] = round(elapsed, 3)
        record.update(details)
        with self._lock:
            self._events.append(record)
            self._save_locked()
        suffix = "" if elapsed is None else f" t={elapsed:06.3f}s"
        print(f"{record['timestamp']} [{name}]{suffix}", flush=True)

    def _save_locked(self) -> None:
        payload = {
            "capture_duration_seconds": CAPTURE_SECONDS,
            "synchronization_marker": CAPTURE_READY_MARKER,
            "schedule": [asdict(segment) for segment in self._schedule],
            "events": self._events,
            "metadata": self._metadata,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class IndependentPlayback:
    """Two long-lived WASAPI shared-mode render streams, one for each source."""

    def __init__(
        self,
        speaker: Any,
        sources: dict[str, np.ndarray],
        on_stream_start: Callable[[str, float], None],
    ) -> None:
        self._speaker = speaker
        self._sources = sources
        self._on_stream_start = on_stream_start
        self._queues = {name: queue.Queue() for name in sources}
        self._ready = threading.Barrier(len(sources) + 1)
        self._threads: list[threading.Thread] = []
        self._errors: list[tuple[str, BaseException]] = []
        self._errors_lock = threading.Lock()

    def start(self) -> None:
        for name in self._sources:
            thread = threading.Thread(
                target=self._run_stream, args=(name,), name=f"validity_playback_{name}"
            )
            thread.start()
            self._threads.append(thread)
        try:
            self._ready.wait(timeout=4)
        except threading.BrokenBarrierError as exc:
            raise RuntimeError("Could not open independent WASAPI playback streams") from exc
        self.raise_if_failed()

    def schedule(self, name: str, start_at: float, duration_seconds: float) -> None:
        self._queues[name].put(PlaybackCommand(start_at, duration_seconds))

    def close(self) -> None:
        for command_queue in self._queues.values():
            command_queue.put(None)
        for thread in self._threads:
            thread.join(timeout=5)

    def raise_if_failed(self) -> None:
        with self._errors_lock:
            if self._errors:
                name, exc = self._errors[0]
                raise RuntimeError(f"Independent playback stream {name} failed: {exc}") from exc

    def _run_stream(self, name: str) -> None:
        try:
            with self._speaker.player(samplerate=PLAYBACK_SAMPLE_RATE, channels=2) as player:
                self._ready.wait(timeout=4)
                while True:
                    command = self._queues[name].get()
                    if command is None:
                        return
                    time.sleep(max(0.0, command.start_at - time.perf_counter()))
                    frames = int(round(command.duration_seconds * PLAYBACK_SAMPLE_RATE))
                    self._on_stream_start(name, command.start_at)
                    player.play(self._sources[name][:frames])
                    while player.currentpadding:
                        time.sleep(0.005)
        except BaseException as exc:
            with self._errors_lock:
                self._errors.append((name, exc))
            try:
                self._ready.abort()
            except threading.BrokenBarrierError:
                pass


def sleep_until(target: float) -> None:
    time.sleep(max(0.0, target - time.perf_counter()))


def log_schedule_boundaries(log: ScheduleLog, test_start: float) -> None:
    names = (
        "SILENCE_START",
        "A_ONLY_START",
        "SILENCE",
        "B_ONLY_START",
        "SILENCE",
        "AB_OVERLAP_START",
        "FINAL_SILENCE_START",
    )
    for segment, name in zip(SCHEDULE, names, strict=True):
        sleep_until(test_start + segment.start_seconds)
        log.event(name, test_start=test_start, ground_truth=segment.kind)
    sleep_until(test_start + CAPTURE_SECONDS)
    log.event("CAPTURE_FINISHED", test_start=test_start, detail="scheduled_75_second_capture_elapsed")


def pipeline_command(json_output: Path) -> list[str]:
    return [
        sys.executable,
        str(PIPELINE_PATH),
        "--live",
        "--duration", str(int(CAPTURE_SECONDS)),
        "--window-seconds", "3",
        "--stride-seconds", "2",
        "--stt-backend", "whisper",
        "--vad",
        "--assemble",
        "--websocket",
        "--ws-host", "127.0.0.1",
        "--ws-port", "8765",
        "--diagnostic-audio",
        "--json-output", str(json_output),
    ]


def verify_pipeline_json(path: Path, launch_time_ns: int) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_mtime_ns < launch_time_ns:
        raise RuntimeError(f"Pipeline did not create a new JSON result: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("duration_seconds") != int(CAPTURE_SECONDS):
        raise RuntimeError(f"Unexpected JSON duration: {payload.get('duration_seconds')}")
    if not payload.get("diagnostic_audio", {}).get("enabled"):
        raise RuntimeError("Diagnostic audio was not enabled in the pipeline JSON")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Automated WASAPI Live secondary-validity test")
    parser.add_argument("--a-wav", type=Path, default=DEFAULT_A_PATH)
    parser.add_argument("--b-wav", type=Path, default=DEFAULT_B_PATH)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--schedule-output", type=Path, default=DEFAULT_SCHEDULE_OUTPUT)
    args = parser.parse_args()
    if sys.platform != "win32":
        raise RuntimeError("This helper requires Windows WASAPI")
    sys.stdout.reconfigure(encoding="utf-8")

    validate_schedule()
    required = required_source_seconds()
    wav_paths = {"A": args.a_wav.resolve(), "B": args.b_wav.resolve()}
    wav_info = {
        name: inspect_wav(path, minimum_seconds=required[name])
        for name, path in wav_paths.items()
    }
    speaker = sc.default_speaker()
    if not speaker.name:
        raise RuntimeError("Windows default output device is unavailable")
    sources = {
        name: load_for_playback(path, duration_seconds=required[name])
        for name, path in wav_paths.items()
    }
    log = ScheduleLog(args.schedule_output.resolve(), SCHEDULE)
    log.set_metadata(
        helper=str(Path(__file__).resolve()),
        output_device=speaker.name,
        playback_sample_rate=PLAYBACK_SAMPLE_RATE,
        source_wav={name: asdict(info) for name, info in wav_info.items()},
        pipeline_command=pipeline_command(args.json_output.resolve()),
    )
    print(f"Default Windows output device: {speaker.name}", flush=True)
    for name, info in wav_info.items():
        print(
            f"{name}.wav: {info.sample_rate} Hz, {info.channels} ch, "
            f"{info.duration_seconds:.3f}s, {info.subtype}; playback copy=48 kHz stereo",
            flush=True,
        )

    launch_time_ns = time.time_ns()
    process = subprocess.Popen(
        pipeline_command(args.json_output.resolve()),
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    playback: IndependentPlayback | None = None
    schedule_thread: threading.Thread | None = None
    test_start: float | None = None
    try:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            if test_start is None and CAPTURE_READY_MARKER in line:
                # The pipeline exposes no capture callback.  Its flushed marker is
                # emitted inside the recorder context immediately before recording.
                test_start = time.perf_counter()
                log.event("CAPTURE_READY", test_start=test_start)
                playback = IndependentPlayback(
                    speaker,
                    sources,
                    lambda name, target: log.event(
                        f"{name}_STREAM_START",
                        test_start=test_start,
                        scheduled_start_seconds=round(target - test_start, 3),
                    ),
                )
                playback.start()
                playback.schedule("A", test_start + 5.0, 20.0)
                playback.schedule("B", test_start + 30.0, 20.0)
                playback.schedule("A", test_start + 55.0, 15.0)
                playback.schedule("B", test_start + 55.0, 15.0)
                schedule_thread = threading.Thread(
                    target=log_schedule_boundaries,
                    args=(log, test_start),
                    name="validity_schedule_log",
                )
                schedule_thread.start()
        exit_code = process.wait()
    finally:
        if schedule_thread is not None:
            schedule_thread.join()
        if playback is not None:
            playback.close()

    if test_start is None:
        log.event("CAPTURE_READY_MISSING", detail="pipeline exited before Live capture marker")
        raise RuntimeError("Pipeline exited before the Live capture synchronization marker")
    if exit_code != 0:
        log.set_metadata(pipeline_exit_code=exit_code)
        raise RuntimeError(f"Pipeline exited with code {exit_code}")
    assert playback is not None
    playback.raise_if_failed()
    payload = verify_pipeline_json(args.json_output.resolve(), launch_time_ns)
    log.event("JSON_SAVED", test_start=test_start, path=str(args.json_output.resolve()))
    log.set_metadata(
        pipeline_exit_code=exit_code,
        pipeline_success=payload.get("success"),
        result_json=str(args.json_output.resolve()),
    )
    print(f"JSON_SAVED: {args.json_output.resolve()}", flush=True)
    print(f"SCHEDULE_SAVED: {args.schedule_output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

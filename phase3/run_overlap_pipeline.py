from __future__ import annotations

import argparse
import difflib
import itertools
import json
import queue
import re
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import soundcard as sc
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "phase3"))
sys.path.insert(0, str(ROOT / "phase4"))

import run_realtime_pipeline as base  # noqa: E402
from audio_diagnostics import select_capture_channel, capture_audio_diagnostics
from separation_recovery import (  # noqa: E402
    Fp32FallbackError,
    InvalidSeparationInput,
    PRE_SEPARATION_SILENCE_RMS,
    finite_audio_stats,
    infer_separator,
    is_pre_separation_silence,
    separate_with_fp32_fallback,
    silent_separation_output,
)
from secondary_leakage_diagnostics import (  # noqa: E402
    SecondaryValidityTracker,
    admit_subtitle_streams,
    build_summary as build_secondary_leakage_summary,
    build_secondary_validity_summary,
    build_full_window_diagnostic,
    build_window_diagnostic,
    save_candidate_full_window_wavs,
    secondary_transcript_suppression_reason,
    resolve_subtitle_fragments,
    subtitle_overlap_evidence,
)
from speaker_tracking import PersistentSpeakerTracker  # noqa: E402
from remote_gpu_client import (  # noqa: E402
    RemoteGPUClient,
    RemoteGPUError,
    RemoteGPUMonitor,
    RemoteWindowResult,
    queue_wait_seconds,
)
from subtitle_assembler import (  # noqa: E402
    DEFAULT_FINALIZE_SILENCE_MS,
    FUZZY_SIMILARITY_THRESHOLD,
    SpeakerSubtitleState,
    SubtitleAssembler,
    SubtitleStateEvent,
    append_preserving_text,
)


WINDOW_SECONDS = 3
STRIDE_SECONDS = 2
TEST_SECONDS = 30
QUEUE_MAXSIZE = base.QUEUE_MAXSIZE
# A remote request can transiently include proxy/TLS startup work before its
# steady-state cadence reaches the 2-second stride. Retain one additional
# complete window only for that bounded startup burst; local backpressure stays
# at the original base capacity.
REMOTE_AUDIO_QUEUE_MAXSIZE = QUEUE_MAXSIZE + 1
SAMPLE_RATE = base.SAMPLE_RATE
MIB = base.MIB
STOP = base.STOP
DEFAULT_OUTPUT = ROOT / "phase3" / "output" / "overlap_3s_2s.json"
DEBUG_OUTPUT_DIR = ROOT / "phase3" / "output" / "overlap_debug"
DIAGNOSTIC_OUTPUT_DIR = ROOT / "phase3" / "output" / "diagnostic"
DIAGNOSTIC_WINDOW_OUTPUT_DIR = ROOT / "phase3" / "output" / "diagnostic_windows"


@dataclass
class LivePipelineHooks:
    """Optional observers and stop control for non-CLI frontends.

    The hooks only observe the already-produced Phase 4-I data.  Leaving them
    unset preserves the command-line pipeline's original behavior.
    """

    stop_event: threading.Event = field(default_factory=threading.Event)
    on_subtitle: Callable[[dict[str, Any]], None] | None = None
    on_metric: Callable[[dict[str, Any]], None] | None = None


def _notify_runtime_hook(
    callback: Callable[[dict[str, Any]], None] | None,
    payload: dict[str, Any],
    name: str,
) -> None:
    if callback is None:
        return
    try:
        callback(dict(payload))
    except Exception as exc:
        # A presentation-layer callback must never interrupt the validated AI
        # pipeline.  This is intentionally diagnostic-only.
        print(f"[RUNTIME HOOK] {name} callback failed: {exc}", flush=True)


@dataclass
class AudioWindow:
    index: int
    audio: np.ndarray
    stream_start_seconds: float
    stream_end_seconds: float
    capture_started: float
    capture_ended: float
    capture_timestamp: str
    audio_queue_size: int
    capture_audio_stats: dict[str, Any] = field(default_factory=dict)
    latency_timing: dict[str, Any] = field(default_factory=dict)


@dataclass
class SeparatedWindow:
    source: AudioWindow
    raw_speakers: tuple[np.ndarray, np.ndarray]
    speakers: tuple[np.ndarray, np.ndarray]
    speaker_assignment: dict[str, Any]
    separation_started: float
    separation_ended: float
    separation_started_timestamp: str
    separation_ended_timestamp: str
    separation_seconds: float
    amp_inference_seconds: float
    fp32_retry_seconds: float
    used_fp32_fallback: bool
    pre_separation_silence: bool
    separation_input_stats: dict[str, float | bool | None]
    audio_queue_backlog: int
    separated_queue_size: int
    torch_allocated_mib: float
    torch_reserved_mib: float
    torch_peak_allocated_mib: float
    speaker_rms: tuple[float, float]
    speaker_peak: tuple[float, float]
    raw_speaker_rms: tuple[float, float]
    raw_speaker_peak: tuple[float, float]
    reference_scores: list[list[float]]
    pair_correlation: float
    speaker_mapping: str
    remote_result: RemoteWindowResult | None = None
    latency_timing: dict[str, Any] = field(default_factory=dict)


def audio_queue_maxsize(processing_mode: str) -> int:
    if processing_mode == "local":
        return QUEUE_MAXSIZE
    if processing_mode == "remote":
        return REMOTE_AUDIO_QUEUE_MAXSIZE
    raise ValueError(f"Unsupported processing mode: {processing_mode}")


def build_window_latency_diagnostics(
    timeline: dict[str, Any],
    remote_result: RemoteWindowResult | None,
) -> dict[str, Any]:
    client = dict(timeline)
    client["audio_queue_wait_seconds"] = queue_wait_seconds(
        client["audio_queue_enqueued_perf_counter"],
        client["audio_queue_dequeued_perf_counter"],
    )
    client["separated_queue_wait_seconds"] = queue_wait_seconds(
        client["separated_queue_enqueued_perf_counter"],
        client["separated_queue_dequeued_perf_counter"],
    )
    client["assembler_subtitle_seconds"] = queue_wait_seconds(
        client["assembler_subtitle_started_perf_counter"],
        client["assembler_subtitle_ended_perf_counter"],
    )
    client["capture_to_result_seconds"] = queue_wait_seconds(
        client["capture_completed_perf_counter"],
        client["result_created_perf_counter"],
    )
    first_subtitle = client.get("first_subtitle_created_perf_counter")
    client["capture_to_first_subtitle_seconds"] = (
        queue_wait_seconds(client["capture_completed_perf_counter"], first_subtitle)
        if first_subtitle is not None
        else None
    )
    client["capture_complete_to_audio_enqueue_seconds"] = queue_wait_seconds(
        client["capture_completed_perf_counter"],
        client["audio_queue_enqueued_perf_counter"],
    )
    client["audio_dequeue_to_remote_request_seconds"] = (
        queue_wait_seconds(
            client["audio_queue_dequeued_perf_counter"],
            client["remote_worker_before_request_perf_counter"],
        )
        if "remote_worker_before_request_perf_counter" in client
        else None
    )
    client["separated_dequeue_to_assembler_seconds"] = queue_wait_seconds(
        client["separated_queue_dequeued_perf_counter"],
        client["assembler_subtitle_started_perf_counter"],
    )
    if remote_result is not None:
        response_parsed = remote_result.client_timing.get(
            "response_parse_ended_perf_counter"
        )
        client["remote_response_parse_to_separated_enqueue_seconds"] = (
            queue_wait_seconds(
                response_parsed, client["separated_queue_enqueued_perf_counter"]
            )
            if response_parsed is not None
            else None
        )
    return {
        "enabled": True,
        "clock_domains": {
            "client": "Windows time.perf_counter; comparable only with client fields",
            "server": "RunPod time.perf_counter; comparable only with server fields",
        },
        "client_pipeline": client,
        "remote_client": remote_result.client_timing if remote_result is not None else {},
        "server": remote_result.timing if remote_result is not None else {},
    }


def detect_speech_activity(model: Any, audio: np.ndarray, minimum_speech_ms: int) -> dict[str, Any]:
    from silero_vad import get_speech_timestamps

    started = time.perf_counter()
    timestamps = get_speech_timestamps(
        torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)),
        model,
        sampling_rate=SAMPLE_RATE,
        min_speech_duration_ms=minimum_speech_ms,
    )
    speech_samples = sum(timestamp["end"] - timestamp["start"] for timestamp in timestamps)
    duration_ms = len(audio) * 1000 / SAMPLE_RATE
    speech_ms = speech_samples * 1000 / SAMPLE_RATE
    return {
        "speech_detected": speech_ms >= minimum_speech_ms,
        "speech_duration_ms": speech_ms,
        "speech_ratio": speech_samples / len(audio),
        "rms": float(np.sqrt(np.mean(audio * audio))),
        "peak": float(np.max(np.abs(audio))),
        "processing_seconds": time.perf_counter() - started,
        "timestamps": timestamps,
        "audio_duration_ms": duration_ms,
    }


def complete_window_count(duration: int, window: int, stride: int) -> int:
    if duration < window:
        return 0
    return 1 + (duration - window) // stride


def clean_overlap_text(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣 ]+", "", text).strip().lower()


def overlap_candidate(previous: str, current: str) -> tuple[str, int]:
    left = clean_overlap_text(previous)
    right = clean_overlap_text(current)
    if not left or not right:
        return "", 0

    left_words = left.split()
    right_words = right.split()
    for size in range(min(len(left_words), len(right_words)), 0, -1):
        if left_words[-size:] == right_words[:size]:
            phrase = " ".join(right_words[:size])
            if len(base.normalize_transcript(phrase)) >= 2:
                return phrase, len(base.normalize_transcript(phrase))

    compact_left = base.normalize_transcript(left)
    compact_right = base.normalize_transcript(right)
    left_boundary = compact_left[-16:]
    right_boundary = compact_right[:16]
    match = difflib.SequenceMatcher(None, left_boundary, right_boundary).find_longest_match()
    near_previous_end = len(left_boundary) - (match.a + match.size) <= 2
    near_current_start = match.b <= 2
    if match.size >= 3 and near_previous_end and near_current_start:
        phrase = right_boundary[match.b : match.b + match.size]
        return phrase, match.size
    return "", 0


def analyze_overlap_duplicates(results: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(results, key=lambda item: item["window"])
    candidates: list[dict[str, Any]] = []
    pair_hits: set[tuple[int, int]] = set()
    for speaker in (0, 1):
        for previous, current in zip(ordered, ordered[1:]):
            phrase, length = overlap_candidate(
                previous["transcripts"][speaker], current["transcripts"][speaker]
            )
            if not phrase:
                continue
            pair = (previous["window"], current["window"])
            pair_hits.add(pair)
            candidates.append(
                {
                    "speaker": speaker,
                    "previous_window": pair[0],
                    "current_window": pair[1],
                    "phrase": phrase,
                    "normalized_character_count": length,
                    "previous_text": previous["transcripts"][speaker],
                    "current_text": current["transcripts"][speaker],
                }
            )
    pair_count = max(0, len(ordered) - 1)
    lengths = [item["normalized_character_count"] for item in candidates]
    representatives = sorted(candidates, key=lambda item: item["normalized_character_count"], reverse=True)[:10]
    return {
        "adjacent_window_pair_count": pair_count,
        "duplicate_pair_count": len(pair_hits),
        "duplicate_pair_ratio": len(pair_hits) / pair_count if pair_count else 0.0,
        "speaker_duplicate_count": len(candidates),
        "duplicate_length_characters": base.distribution(lengths) if lengths else {},
        "representative_candidates": representatives,
        "method": (
            "exact suffix-prefix words, then a normalized character match (>=3) "
            "within two characters of both transcript boundaries"
        ),
    }


def truncate_assembler_text(text: str, limit: int = 96) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def build_diagnostic_audio_window_record(item: dict[str, Any]) -> dict[str, Any]:
    """Project the persisted window schema into the diagnostic-audio summary."""
    return {
        "window": item["window"],
        "capture_timestamp": item["capture_completed_timestamp"],
        "original_rms": item["separation_input_stats"].get("rms"),
        "original_peak": item["separation_input_stats"].get("peak"),
        "speaker_0_rms": item["speaker_rms"][0],
        "speaker_0_peak": item["speaker_peak"][0],
        "speaker_1_rms": item["speaker_rms"][1],
        "speaker_1_peak": item["speaker_peak"][1],
        "raw_slot_0_rms": item["raw_slot_rms"][0],
        "raw_slot_0_peak": item["raw_slot_peak"][0],
        "raw_slot_1_rms": item["raw_slot_rms"][1],
        "raw_slot_1_peak": item["raw_slot_peak"][1],
        "speaker_assignment": item["speaker_assignment"],
        "vad": item["vad"],
        "pre_separation_silence": item["pre_separation_silence"],
    }


def main(
    argv: Sequence[str] | None = None,
    runtime_hooks: LivePipelineHooks | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 4-I overlap pipeline with partial/final subtitle consensus"
    )
    parser.add_argument("--duration", type=int)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Capture only the current WASAPI loopback output; do not play test WAV files",
    )
    parser.add_argument(
        "--reference-a",
        type=Path,
        help="Override the non-live speaker A playback WAV",
    )
    parser.add_argument(
        "--reference-b",
        type=Path,
        help="Override the non-live speaker B playback WAV",
    )
    parser.add_argument("--window-seconds", type=int, default=WINDOW_SECONDS)
    parser.add_argument("--stride-seconds", type=int, default=STRIDE_SECONDS)
    parser.add_argument("--stt-backend", choices=("whisper", "sensevoice"), default="whisper")
    parser.add_argument("--processing-mode", choices=("local", "remote"), default="local")
    parser.add_argument("--remote-server-url", default="http://127.0.0.1:8787")
    parser.add_argument("--remote-connect-timeout", type=float, default=5)
    parser.add_argument("--remote-read-timeout", type=float, default=30)
    parser.add_argument(
        "--latency-diagnostics",
        action="store_true",
        help="Record per-window client/server monotonic timings without changing pipeline behavior",
    )
    parser.add_argument("--vad", action="store_true", help="Run Silero VAD on each separated speaker")
    parser.add_argument("--vad-min-speech-ms", type=int, default=200)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--save-wav", action="store_true")
    parser.add_argument("--capture-channel", choices=("first", "mean"), default="first",
                        help="Channel selection; compare diagnostics before choosing mean")
    parser.add_argument(
        "--diagnostic-audio",
        action="store_true",
        help="Save live input/separation WAVs and compare their post-run STT results",
    )
    parser.add_argument("--analyze-separation-quality", action="store_true")
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--assembler-min-characters", type=int, default=6)
    parser.add_argument(
        "--subtitle-finalize-silence-ms",
        type=int,
        default=DEFAULT_FINALIZE_SILENCE_MS,
        help="Finalize a speaker partial after this much VAD silence (default: 2000)",
    )
    parser.add_argument("--websocket", action="store_true")
    parser.add_argument("--ws-host", default="127.0.0.1")
    parser.add_argument("--ws-port", type=int, default=8765)
    parser.add_argument("--subtitle-event-queue-size", type=int, default=32)
    args = parser.parse_args(argv)
    remote = args.processing_mode == "remote"
    audio_queue_maxsize_for_run = audio_queue_maxsize(args.processing_mode)
    separated_queue_maxsize = QUEUE_MAXSIZE
    if remote:
        if args.stt_backend != "whisper":
            parser.error("remote mode currently supports faster-whisper only")
        if args.diagnostic_audio:
            parser.error("--diagnostic-audio requires local mode (post-run whole-stream STT)")
        if args.vad_min_speech_ms != 200:
            parser.error("remote server uses --vad-min-speech-ms 200")
        args.vad = True
    if args.remote_connect_timeout <= 0 or args.remote_read_timeout <= 0:
        parser.error("remote timeouts must be positive")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    if args.window_seconds <= 0 or args.stride_seconds <= 0:
        parser.error("duration, window, and stride must be positive")
    if args.stride_seconds >= args.window_seconds:
        parser.error("--stride-seconds must be smaller than --window-seconds")
    if args.assembler_min_characters < 2:
        parser.error("--assembler-min-characters must be at least 2")
    if args.subtitle_finalize_silence_ms < 1:
        parser.error("--subtitle-finalize-silence-ms must be positive")
    if args.websocket and not args.assemble:
        parser.error("--websocket requires --assemble so UI events use incremental NEW text")
    if args.subtitle_event_queue_size < 1:
        parser.error("--subtitle-event-queue-size must be positive")
    if args.vad_min_speech_ms < 1:
        parser.error("--vad-min-speech-ms must be positive")
    for reference_path in (args.reference_a, args.reference_b):
        if reference_path is not None and not reference_path.is_file():
            parser.error(f"Reference WAV does not exist: {reference_path}")
    capture_duration = args.duration if args.duration is not None else (None if args.live else TEST_SECONDS)
    window_count = (
        None
        if capture_duration is None
        else complete_window_count(capture_duration, args.window_seconds, args.stride_seconds)
    )
    if window_count == 0:
        parser.error("--duration must contain at least one complete window")
    if sys.platform != "win32" or (not remote and not torch.cuda.is_available()):
        raise RuntimeError("Capture requires Windows WASAPI; local mode additionally requires CUDA")
    sys.stdout.reconfigure(encoding="utf-8")

    monitor = RemoteGPUMonitor() if remote else base.GpuMonitor()
    remote_client = RemoteGPUClient(args.remote_server_url, args.remote_connect_timeout, args.remote_read_timeout) if remote else None
    monitor.start()
    errors: list[tuple[str, BaseException]] = []
    dropped: list[tuple[str, int]] = []
    results: list[dict[str, Any]] = []
    assembler_events: list[dict[str, Any]] = []
    subtitle_state_events: list[dict[str, Any]] = []
    subtitle_events: list[dict[str, Any]] = []
    published_subtitle_utterances: set[tuple[int, int]] = set()
    websocket_server: Any | None = None
    websocket_stats: dict[str, Any] = {"enabled": False}
    event_sequence = itertools.count(1)
    state_lock = threading.Lock()
    audio_queue: queue.Queue[AudioWindow | object] = queue.Queue(
        maxsize=audio_queue_maxsize_for_run
    )
    separated_queue: queue.Queue[SeparatedWindow | object] = queue.Queue(
        maxsize=separated_queue_maxsize
    )
    diagnostic_original_parts: list[np.ndarray] = []
    diagnostic_raw_slot_windows: list[dict[int, np.ndarray]] = [{}, {}]
    diagnostic_logical_speaker_windows: list[dict[int, np.ndarray]] = [{}, {}]
    diagnostic_state = {"window_count": 0}
    diagnostic_summary: dict[str, Any] = {"enabled": args.diagnostic_audio}
    speaker_tracker = PersistentSpeakerTracker(
        overlap_samples=(args.window_seconds - args.stride_seconds) * SAMPLE_RATE
    )
    # Observation-only tracker. It is not created for ordinary runs, so validity
    # instrumentation cannot affect their timing or pipeline decisions.
    secondary_validity_tracker = (
        SecondaryValidityTracker() if args.diagnostic_audio else None
    )
    vad_model: Any | None = None
    vad_state: dict[str, Any] = {
        "enabled": args.vad,
        "minimum_speech_ms": args.vad_min_speech_ms,
        "checked_speakers": 0,
        "active_speakers": 0,
        "skipped_speakers": 0,
        "processing_seconds": [],
    }
    assemblers = [
        SubtitleAssembler(speaker=speaker, minimum_characters=args.assembler_min_characters)
        for speaker in (0, 1)
    ]
    subtitle_states = [
        SpeakerSubtitleState(
            speaker=speaker,
            minimum_characters=args.assembler_min_characters,
            finalize_silence_ms=args.subtitle_finalize_silence_ms,
            require_final_support=args.vad,
        )
        for speaker in (0, 1)
    ]

    def publish_subtitle_state_event(
        event: SubtitleStateEvent,
        stt_inference_ended: float | None = None,
    ) -> dict[str, Any]:
        if runtime_hooks is not None and runtime_hooks.stop_event.is_set():
            return {}
        state_event = event.to_dict()
        subtitle_state_events.append(state_event)
        print(
            f"[SUBTITLE] window={event.window:03d} speaker={event.speaker} "
            f"utterance={event.utterance_id} status={event.status} action={event.action}"
            + (f" reason={event.finalize_reason}" if event.finalize_reason else "")
            + f"\n  raw=\"{truncate_assembler_text(event.raw_text)}\""
            + f"\n  before=\"{truncate_assembler_text(event.before)}\""
            + f"\n  after=\"{truncate_assembler_text(event.text)}\"",
            flush=True,
        )
        print(
            f"[CONSENSUS] window={event.window:03d} speaker={event.speaker} "
            f"utterance={event.utterance_id} action={event.stability_action} "
            f"history={event.history_size}\n"
            f"  stable=\"{truncate_assembler_text(event.stable_text)}\"\n"
            f"  tentative=\"{truncate_assembler_text(event.tentative_text)}\"",
            flush=True,
        )
        utterance_key = (event.speaker, event.utterance_id)
        needs_retraction = (
            event.action == "discard" and utterance_key in published_subtitle_utterances
        )
        will_publish = bool(event.publication_text or needs_retraction)
        publication_reason = (
            "retraction"
            if needs_retraction
            else "source_supported_text"
            if event.publication_text and event.require_final_support
            else "support_not_required"
            if event.publication_text
            else "discarded_unsupported_tentative"
            if event.action == "discard"
            else "awaiting_source_support"
        )
        print(
            f"[SUBTITLE PUBLICATION] window={event.window:03d} "
            f"speaker={event.speaker} utterance={event.utterance_id} "
            f"status={event.status} action={event.action} "
            f'event_text="{truncate_assembler_text(event.text)}" '
            f'publication_text="{truncate_assembler_text(event.publication_text)}" '
            f'stable_text="{truncate_assembler_text(event.stable_text)}" '
            f'source_supported_text="{truncate_assembler_text(event.source_supported_text)}" '
            f"source_supported={event.source_supported} "
            f"candidate_provenance={event.candidate_provenance} "
            f"candidate_transition={event.candidate_transition} "
            f"require_final_support={event.require_final_support} "
            f"needs_retraction={needs_retraction} will_publish={will_publish} "
            f"reason={publication_reason}",
            flush=True,
        )
        if not will_publish:
            return {}
        state = subtitle_states[event.speaker]
        display_text = ""
        for segment in state.final_segments:
            display_text = append_preserving_text(display_text, segment)
        if event.status == "partial":
            display_text = append_preserving_text(display_text, event.publication_text)
        subtitle_event: dict[str, Any] = {
            "type": "subtitle",
            "status": event.status,
            "action": event.action,
            "utterance_id": event.utterance_id,
            "sequence": next(event_sequence),
            "window_index": event.window,
            "speaker": f"speaker_{event.speaker}",
            "text": event.publication_text,
            "assembled_text": display_text,
            "raw_text": event.raw_text,
            "overlap_text": event.overlap_text,
            "match_type": event.match_type,
            "similarity": event.similarity,
            "stable_text": event.stable_text,
            "tentative_text": event.tentative_text,
            "history_size": event.history_size,
            "stability_action": event.stability_action,
            "finalize_reason": event.finalize_reason,
            "start_ms": round(event.utterance_start_seconds * 1000),
            "end_ms": round(event.stream_time_seconds * 1000),
            "timestamp": int(time.time() * 1000),
            "stt_to_enqueue_ms": (
                (time.perf_counter() - stt_inference_ended) * 1000
                if stt_inference_ended is not None
                else 0.0
            ),
        }
        if websocket_server is not None:
            publication = websocket_server.publish(subtitle_event)
            subtitle_event["websocket_accepted"] = publication.accepted
            subtitle_event["websocket_dropped_oldest"] = publication.dropped_oldest
            subtitle_event["websocket_queue_depth"] = publication.queue_depth
            print(
                f"WebSocket sequence={subtitle_event['sequence']}, "
                f"accepted={publication.accepted}, queue={publication.queue_depth}/"
                f"{args.subtitle_event_queue_size}",
                flush=True,
            )
        subtitle_events.append(subtitle_event)
        _notify_runtime_hook(
            runtime_hooks.on_subtitle if runtime_hooks is not None else None,
            subtitle_event,
            "subtitle",
        )
        if event.status == "partial":
            published_subtitle_utterances.add(utterance_key)
        else:
            published_subtitle_utterances.discard(utterance_key)
        return subtitle_event

    try:
        if remote:
            health = remote_client.health()
            monitor.observe(health["gpu_memory"])
            baseline_vram = separator_vram = models_vram = monitor.used_mib()
            print(f"Remote GPU ready: {args.remote_server_url}", flush=True)
            print(
                f"Remote audio queue capacity: {audio_queue_maxsize_for_run} "
                f"(bounded startup burst; local={QUEUE_MAXSIZE})",
                flush=True,
            )
        else:
            separator, device, stt_model, baseline_vram, separator_vram, models_vram = base.load_models(
                monitor, stt_backend=args.stt_backend
            )
            base.warm_up(separator, device, stt_model, monitor, stt_backend=args.stt_backend)
        after_warmup_vram = monitor.used_mib()
        print(f"GPU VRAM after warm-up: {after_warmup_vram} MiB", flush=True)
        if not remote and (args.vad or args.diagnostic_audio):
            from silero_vad import load_silero_vad

            vad_model = load_silero_vad()
            vad_warm_up = detect_speech_activity(
                vad_model, np.zeros(SAMPLE_RATE, dtype=np.float32), args.vad_min_speech_ms
            )
            print(
                f"Silero VAD loaded: CPU, minimum speech={args.vad_min_speech_ms}ms",
                flush=True,
            )
            print(f"Silero VAD warm-up: {vad_warm_up['processing_seconds']:.3f} sec", flush=True)
        if args.websocket:
            from subtitle_websocket import SubtitleWebSocketServer

            websocket_server = SubtitleWebSocketServer(
                host=args.ws_host,
                port=args.ws_port,
                queue_maxsize=args.subtitle_event_queue_size,
            )
            websocket_server.start()
            print(
                f"WebSocket server: ws://{args.ws_host}:{args.ws_port}, "
                f"event_queue={args.subtitle_event_queue_size}",
                flush=True,
            )

        speaker = sc.default_speaker()
        loopback = sc.get_microphone(speaker.name, include_loopback=True)
        reference_overrides = {"a": args.reference_a, "b": args.reference_b}
        references: dict[str, np.ndarray] = {}
        if not args.live:
            for name in ("a", "b"):
                override = reference_overrides[name]
                if override is None:
                    references[name] = base.load_reference(name)
                    continue
                reference, reference_rate = sf.read(override, dtype="float32")
                if reference_rate != SAMPLE_RATE or reference.ndim != 1:
                    raise ValueError(
                        f"Reference WAV must be {SAMPLE_RATE}Hz mono: {override}"
                    )
                # Match load_reference(): player.play consumes samples at 48 kHz.
                references[name] = np.ascontiguousarray(
                    base._resample(reference, reference_rate, base.CAPTURE_SAMPLE_RATE),
                    dtype=np.float32,
                )
        if args.diagnostic_audio:
            DIAGNOSTIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            print(f"Diagnostic WAV directory: {DIAGNOSTIC_OUTPUT_DIR}", flush=True)
        quality_references: list[np.ndarray] = []
        if args.analyze_separation_quality:
            for name in ("a", "b"):
                reference, reference_rate = sf.read(
                    ROOT / "phase1" / "reference" / f"speaker_{name}.wav", dtype="float32"
                )
                if reference_rate != SAMPLE_RATE or reference.ndim != 1:
                    raise ValueError(f"Invalid quality reference: speaker_{name}.wav")
                quality_references.append(np.tile(reference, 4))

        playback_ready = threading.Barrier(3)
        playback_launch = threading.Event()
        playback_errors: list[tuple[str, BaseException]] = []
        capture_state: dict[str, Any] = {
            "max_queue_depth": 0,
            "captured": 0,
            "stream_started": None,
            "stream_ended": None,
            "invalid_audio_windows": 0,
        }
        separation_state = {
            "max_audio_backlog": 0,
            "max_queue_depth": 0,
            "inferred": 0,
            "delivered": 0,
            "amp_attempts": 0,
            "amp_non_finite_outputs": 0,
            "fp32_fallback_attempts": 0,
            "fp32_fallback_successes": 0,
            "fp32_fallback_failures": 0,
            "invalid_input_windows": 0,
            "windows_skipped_after_failure": 0,
            "pre_separation_silence_checked": 0,
            "pre_separation_silence_skipped": 0,
            "pre_separation_non_silence": 0,
            "amp_inference_seconds": [],
            "fp32_retry_seconds": [],
        }
        stt_state = {
            "max_separated_backlog": 0,
            "processed": 0,
            "speaker_inputs": 0,
            "calls_executed": 0,
            "calls_skipped_by_vad": 0,
        }
        playback_start_at = 0.0

        def play_one(name: str) -> None:
            try:
                with speaker.player(samplerate=base.CAPTURE_SAMPLE_RATE, channels=2) as player:
                    playback_ready.wait(timeout=10)
                    if not playback_launch.wait(timeout=10):
                        raise TimeoutError("Playback launch timed out")
                    time.sleep(max(0.0, playback_start_at - time.perf_counter()))
                    assert capture_duration is not None
                    remaining = (capture_duration + 2) * base.CAPTURE_SAMPLE_RATE
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

        playback_threads = (
            []
            if args.live
            else [
                threading.Thread(target=play_one, args=(name,), name=f"playback_{name}")
                for name in references
            ]
        )
        stop_capture = (
            runtime_hooks.stop_event
            if runtime_hooks is not None
            else threading.Event()
        )

        def stitch_diagnostic_channels(
            channel_windows: list[dict[int, np.ndarray]],
        ) -> tuple[np.ndarray, np.ndarray]:
            """Keep the first 3s window, then only each following window's new stride."""
            window_samples = args.window_seconds * SAMPLE_RATE
            stride_samples = args.stride_seconds * SAMPLE_RATE
            stitched: list[np.ndarray] = []
            for speaker_windows in channel_windows:
                parts: list[np.ndarray] = []
                for index in range(diagnostic_state["window_count"]):
                    audio = speaker_windows.get(index)
                    if audio is None:
                        parts.append(np.zeros(window_samples if index == 0 else stride_samples, dtype=np.float32))
                    elif index == 0:
                        parts.append(audio)
                    else:
                        parts.append(audio[-stride_samples:])
                stitched.append(
                    np.concatenate(parts).astype(np.float32, copy=False)
                    if parts
                    else np.empty(0, dtype=np.float32)
                )
            return stitched[0], stitched[1]

        def run_diagnostic_stt() -> dict[str, Any]:
            original = (
                np.concatenate(diagnostic_original_parts).astype(np.float32, copy=False)
                if diagnostic_original_parts
                else np.empty(0, dtype=np.float32)
            )
            raw_slot_0, raw_slot_1 = stitch_diagnostic_channels(
                diagnostic_raw_slot_windows
            )
            speaker_0, speaker_1 = stitch_diagnostic_channels(
                diagnostic_logical_speaker_windows
            )
            comparison_audio = {
                "raw_slot_0": (DIAGNOSTIC_OUTPUT_DIR / "raw_slot_0.wav", raw_slot_0),
                "raw_slot_1": (DIAGNOSTIC_OUTPUT_DIR / "raw_slot_1.wav", raw_slot_1),
                "logical_speaker_0": (
                    DIAGNOSTIC_OUTPUT_DIR / "logical_speaker_0.wav",
                    speaker_0,
                ),
                "logical_speaker_1": (
                    DIAGNOSTIC_OUTPUT_DIR / "logical_speaker_1.wav",
                    speaker_1,
                ),
            }
            comparison_files: dict[str, Any] = {}
            for name, (path, audio) in comparison_audio.items():
                if not len(audio):
                    raise ValueError(f"Diagnostic audio is empty: {path.name}")
                sf.write(path, audio, SAMPLE_RATE, subtype="FLOAT")
                comparison_files[name] = {
                    "path": str(path),
                    "sample_rate": SAMPLE_RATE,
                    "channels": 1,
                    "duration_seconds": len(audio) / SAMPLE_RATE,
                    "rms": float(np.sqrt(np.mean(audio * audio))),
                    "peak": float(np.max(np.abs(audio))),
                }
            diagnostic_inputs = {
                "ORIGINAL": (DIAGNOSTIC_OUTPUT_DIR / "original.wav", original),
                # Backward-compatible names now contain logical, tracked speakers.
                "SPEAKER 0": (DIAGNOSTIC_OUTPUT_DIR / "speaker_0.wav", speaker_0),
                "SPEAKER 1": (DIAGNOSTIC_OUTPUT_DIR / "speaker_1.wav", speaker_1),
            }
            for path, audio in diagnostic_inputs.values():
                if not len(audio):
                    raise ValueError(f"Diagnostic audio is empty: {path.name}")
                sf.write(path, audio, SAMPLE_RATE, subtype="FLOAT")

            print("\n========== Phase 4-B Diagnostic STT ==========", flush=True)
            report: dict[str, Any] = {
                "enabled": True,
                "original": {},
                "speaker_0": {},
                "speaker_1": {},
                "files": {},
                "speaker_identity_files": comparison_files,
                "legacy_speaker_files_are_logical": True,
                "windows": [],
            }
            for label, (path, _) in diagnostic_inputs.items():
                audio, sample_rate = sf.read(path, dtype="float32")
                if sample_rate != SAMPLE_RATE or audio.ndim != 1:
                    raise ValueError(f"Invalid diagnostic WAV: {path}")
                rms = float(np.sqrt(np.mean(audio * audio)))
                peak = float(np.max(np.abs(audio)))
                monitor.enter("diagnostic_stt")
                try:
                    transcript, elapsed = base.transcribe_audio(
                        stt_model, audio, backend=args.stt_backend
                    )
                finally:
                    monitor.leave("diagnostic_stt")
                duration = len(audio) / SAMPLE_RATE
                rtf = elapsed / duration
                diagnostic_key = label.lower().replace(" ", "_")
                vad = (
                    detect_speech_activity(vad_model, audio, args.vad_min_speech_ms)
                    if vad_model is not None
                    else None
                )
                file_report = {
                    "path": str(path),
                    "sample_rate": sample_rate,
                    "channels": 1,
                    "duration_seconds": duration,
                    "rms": rms,
                    "peak": peak,
                    "stt_processing_seconds": elapsed,
                    "rtf": rtf,
                    "transcript": transcript,
                    "vad_speech_duration_ms": (
                        vad["speech_duration_ms"] if vad is not None else None
                    ),
                    "vad_speech_ratio": vad["speech_ratio"] if vad is not None else None,
                    "vad_speech_detected": vad["speech_detected"] if vad is not None else None,
                }
                report[diagnostic_key] = file_report
                report["files"][diagnostic_key] = file_report
                print(f"\n[{label}]", flush=True)
                print(f"audio duration: {duration:.3f}s", flush=True)
                print(f"RMS: {rms:.6f}", flush=True)
                print(f"peak: {peak:.6f}", flush=True)
                print(f"STT processing time: {elapsed:.3f}s", flush=True)
                print(f"RTF: {rtf:.3f}", flush=True)
                print(f"transcribed text: {transcript}", flush=True)
            report["windows"] = [
                build_diagnostic_audio_window_record(item)
                for item in sorted(results, key=lambda value: value["window"])
            ]
            print("==============================================", flush=True)
            return report

        def capture_worker() -> None:
            nonlocal playback_start_at
            try:
                with loopback.recorder(samplerate=base.CAPTURE_SAMPLE_RATE) as recorder:
                    if args.live:
                        print("Live mode: test WAV playback disabled; capturing WASAPI loopback only.", flush=True)
                    else:
                        for thread in playback_threads:
                            thread.start()
                        playback_ready.wait(timeout=10)
                        playback_start_at = time.perf_counter() + 0.5
                        playback_launch.set()
                    capture_state["stream_started"] = time.perf_counter()

                    overlap_seconds = args.window_seconds - args.stride_seconds
                    overlap_samples = overlap_seconds * SAMPLE_RATE
                    retained_tail: np.ndarray | None = None
                    captured_audio_seconds = 0

                    index = 0
                    while window_count is None or index < window_count:
                        if stop_capture.is_set():
                            break
                        new_seconds = args.window_seconds if index == 0 else args.stride_seconds
                        capture_started = time.perf_counter()
                        raw = recorder.record(numframes=new_seconds * base.CAPTURE_SAMPLE_RATE)
                        capture_ended = time.perf_counter()
                        if (
                            raw.ndim != 2
                            or raw.shape[0] != new_seconds * base.CAPTURE_SAMPLE_RATE
                            or raw.shape[1] < 1
                        ):
                            raise ValueError(f"Unexpected WASAPI capture for window {index}: {raw.shape}")
                        if not np.isfinite(raw).all():
                            stats = finite_audio_stats(np.asarray(raw[:, 0]))
                            capture_state["invalid_audio_windows"] += 1
                            with state_lock:
                                errors.append(
                                    (
                                        f"capture-input-window-{index:03d}",
                                        InvalidSeparationInput(stats),
                                    )
                                )
                            retained_tail = np.zeros(overlap_samples, dtype=np.float32)
                            captured_audio_seconds += new_seconds
                            print(
                                f"[CAPTURE SKIP] window={index:03d} "
                                f"reason=non_finite_wasapi stats={stats.to_dict()}",
                                flush=True,
                            )
                            index += 1
                            continue
                        capture_mono = select_capture_channel(raw, args.capture_channel)
                        capture_stats = capture_audio_diagnostics(raw, capture_mono, args.capture_channel)
                        new_audio = base._resample(
                            capture_mono,
                            base.CAPTURE_SAMPLE_RATE,
                            SAMPLE_RATE,
                        )
                        expected_new_samples = new_seconds * SAMPLE_RATE
                        if len(new_audio) != expected_new_samples:
                            raise ValueError(f"Invalid resampled capture for window {index}")
                        if not np.isfinite(new_audio).all():
                            stats = finite_audio_stats(new_audio)
                            capture_state["invalid_audio_windows"] += 1
                            with state_lock:
                                errors.append(
                                    (
                                        f"capture-resample-window-{index:03d}",
                                        InvalidSeparationInput(stats),
                                    )
                                )
                            retained_tail = np.zeros(overlap_samples, dtype=np.float32)
                            captured_audio_seconds += new_seconds
                            print(
                                f"[CAPTURE SKIP] window={index:03d} "
                                f"reason=non_finite_resample stats={stats.to_dict()}",
                                flush=True,
                            )
                            index += 1
                            continue
                        if args.diagnostic_audio:
                            diagnostic_original_parts.append(new_audio.copy())
                            diagnostic_state["window_count"] = index + 1
                        if index == 0:
                            audio = np.ascontiguousarray(new_audio, dtype=np.float32)
                        else:
                            assert retained_tail is not None
                            audio = np.concatenate((retained_tail, new_audio)).astype(np.float32, copy=False)
                        if len(audio) != args.window_seconds * SAMPLE_RATE:
                            raise ValueError(f"Invalid sliding window {index}: {len(audio)} samples")
                        retained_tail = np.ascontiguousarray(audio[-overlap_samples:], dtype=np.float32)
                        captured_audio_seconds += new_seconds

                        stream_start = index * args.stride_seconds
                        item = AudioWindow(
                            index=index,
                            audio=audio,
                            stream_start_seconds=stream_start,
                            stream_end_seconds=stream_start + args.window_seconds,
                            capture_started=capture_started,
                            capture_ended=capture_ended,
                            capture_timestamp=datetime.now().astimezone().isoformat(timespec="milliseconds"),
                            capture_audio_stats=capture_stats,
                            audio_queue_size=audio_queue.qsize(),
                            latency_timing=(
                                {
                                    "capture_started_perf_counter": capture_started,
                                    "capture_completed_perf_counter": capture_ended,
                                    "capture_window_ready_perf_counter": time.perf_counter(),
                                }
                                if args.latency_diagnostics
                                else {}
                            ),
                        )
                        if args.save_wav:
                            DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                            sf.write(
                                DEBUG_OUTPUT_DIR / f"window_{index:03d}_mixed.wav",
                                audio,
                                SAMPLE_RATE,
                                subtype="FLOAT",
                            )
                        audio_queue_enqueue_started = time.perf_counter()
                        accepted = base.enqueue_or_drop(
                            audio_queue, item, "audio_queue", index, dropped, state_lock
                        )
                        audio_queue_enqueue_ended = time.perf_counter()
                        if accepted:
                            if args.latency_diagnostics:
                                item.latency_timing.update(
                                    {
                                        "audio_queue_enqueue_started_perf_counter": (
                                            audio_queue_enqueue_started
                                        ),
                                        "audio_queue_enqueued_perf_counter": (
                                            audio_queue_enqueue_ended
                                        ),
                                        "audio_queue_enqueue_seconds": (
                                            audio_queue_enqueue_ended
                                            - audio_queue_enqueue_started
                                        ),
                                    }
                                )
                            item.audio_queue_size = audio_queue.qsize()
                            capture_state["max_queue_depth"] = max(
                                capture_state["max_queue_depth"], audio_queue.qsize()
                            )
                            capture_state["captured"] += 1
                        print(
                            f"Capture window {index:03d} [{stream_start:.1f}, {item.stream_end_seconds:.1f}]s: "
                            f"new_audio={new_seconds}s, wall={capture_ended - capture_started:.3f}s, "
                            f"audio_queue={audio_queue.qsize()}/{audio_queue_maxsize_for_run}",
                            flush=True,
                        )
                        index += 1

                    if capture_duration is not None:
                        remainder = capture_duration - captured_audio_seconds
                        if remainder > 0:
                            raw_remainder = recorder.record(numframes=remainder * base.CAPTURE_SAMPLE_RATE)
                            if args.diagnostic_audio:
                                remainder_audio = base._resample(
                                    select_capture_channel(raw_remainder, args.capture_channel),
                                    base.CAPTURE_SAMPLE_RATE,
                                    SAMPLE_RATE,
                                )
                                if len(remainder_audio) != remainder * SAMPLE_RATE:
                                    raise ValueError("Invalid diagnostic capture remainder")
                                diagnostic_original_parts.append(remainder_audio)
                    capture_state["stream_ended"] = time.perf_counter()
            except BaseException as exc:
                with state_lock:
                    errors.append(("capture", exc))
                playback_launch.set()
            finally:
                try:
                    base.put_stop(audio_queue, "audio_queue")
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
                    audio_queue_dequeued = time.perf_counter()
                    try:
                        if item is STOP:
                            break
                        assert isinstance(item, AudioWindow)
                        if args.latency_diagnostics:
                            item.latency_timing["audio_queue_dequeued_perf_counter"] = (
                                audio_queue_dequeued
                            )
                        backlog = base.data_backlog(audio_queue)
                        separation_state["max_audio_backlog"] = max(
                            separation_state["max_audio_backlog"], backlog
                        )
                        separation_started = time.perf_counter()
                        separation_started_timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
                        input_stats = finite_audio_stats(item.audio)
                        if not input_stats.finite:
                            exc = InvalidSeparationInput(input_stats)
                            separation_state["invalid_input_windows"] += 1
                            separation_state["windows_skipped_after_failure"] += 1
                            with state_lock:
                                errors.append((f"separation-input-window-{item.index:03d}", exc))
                            print(
                                f"[SEPARATION SKIP] window={item.index:03d} "
                                f"reason=non_finite_input stats={input_stats.to_dict()}",
                                flush=True,
                            )
                            continue

                        separation_state["pre_separation_silence_checked"] += 1
                        pre_separation_silence = is_pre_separation_silence(input_stats)
                        remote_result = None
                        if remote:
                            try:
                                if args.latency_diagnostics:
                                    item.latency_timing[
                                        "remote_worker_before_request_perf_counter"
                                    ] = time.perf_counter()
                                    print(
                                        f"[REMOTE REQUEST] window={item.index:03d} "
                                        f"status=start audio_duration={len(item.audio) / SAMPLE_RATE:.3f}s "
                                        f"samples={len(item.audio)}",
                                        flush=True,
                                    )
                                remote_result = remote_client.process(
                                    item.audio, item.index, item.capture_timestamp,
                                    item.stream_start_seconds, item.stream_end_seconds,
                                    latency_diagnostics=args.latency_diagnostics,
                                )
                                if args.latency_diagnostics:
                                    item.latency_timing[
                                        "remote_client_returned_perf_counter"
                                    ] = time.perf_counter()
                                    print(
                                        f"[REMOTE REQUEST] window={item.index:03d} "
                                        f"status=complete round_trip={remote_result.request_seconds:.6f}s "
                                        f"server_processing={remote_result.timing['processing_seconds']:.6f}s "
                                        f"separation={remote_result.timing['separation_seconds']:.6f}s "
                                        f"vad={remote_result.timing['vad_seconds']:.6f}s "
                                        f"stt={remote_result.timing['stt_seconds']:.6f}s "
                                        f"silence_gate={remote_result.pre_separation_silence}",
                                        flush=True,
                                    )
                            except RemoteGPUError as exc:
                                separation_state["windows_skipped_after_failure"] += 1
                                with state_lock:
                                    errors.append((f"remote-window-{item.index:03d}", exc))
                                print(f"[REMOTE SKIP] window={item.index:03d} {type(exc).__name__}: {exc}", flush=True)
                                continue
                            monitor.observe(remote_result.gpu_memory)
                            separated = np.stack(remote_result.raw_speakers)[:, None, :]
                            pre_separation_silence = remote_result.pre_separation_silence
                            elapsed = remote_result.timing["separation_seconds"]
                            amp_seconds = remote_result.timing["amp_seconds"]
                            fp32_seconds = remote_result.timing["fp32_seconds"]
                            used_fp32_fallback = remote_result.fp32_fallback
                            separation_state["pre_separation_silence_skipped" if pre_separation_silence else "pre_separation_non_silence"] += 1
                            if not pre_separation_silence:
                                separation_state["amp_attempts"] += 1
                            separation_state["amp_inference_seconds"].append(amp_seconds)
                            if used_fp32_fallback:
                                separation_state["amp_non_finite_outputs"] += 1
                                separation_state["fp32_fallback_attempts"] += 1
                                separation_state["fp32_fallback_successes"] += 1
                                separation_state["fp32_retry_seconds"].append(fp32_seconds)
                        elif pre_separation_silence:
                            separation_state["pre_separation_silence_skipped"] += 1
                            separated = silent_separation_output(len(item.audio))
                            elapsed = 0.0
                            amp_seconds = 0.0
                            fp32_seconds = 0.0
                            used_fp32_fallback = False
                            print(
                                f"[SILENCE GATE] window={item.index:03d} "
                                f"rms={input_stats.rms:.6f} peak={input_stats.peak:.6f} "
                                "action=skip_separation",
                                flush=True,
                            )
                        else:
                            separation_state["pre_separation_non_silence"] += 1
                            separation_state["amp_attempts"] += 1
                            torch.cuda.reset_peak_memory_stats(device)

                            def amp_inference(audio: np.ndarray) -> tuple[np.ndarray, float]:
                                return infer_separator(separator, audio.reshape(1, -1), device, True)

                            def fp32_inference(audio: np.ndarray) -> tuple[np.ndarray, float]:
                                print(
                                    f"[AMP FALLBACK] window={item.index:03d} "
                                    "reason=non_finite_output "
                                    f"input_min={input_stats.minimum:.6f} "
                                    f"input_max={input_stats.maximum:.6f} "
                                    f"input_rms={input_stats.rms:.6f} "
                                    f"input_peak={input_stats.peak:.6f} retry=fp32",
                                    flush=True,
                                )
                                return infer_separator(separator, audio.reshape(1, -1), device, False)

                            monitor.enter("separation")
                            try:
                                recovery = separate_with_fp32_fallback(
                                    item.audio,
                                    amp_inference=amp_inference,
                                    fp32_inference=fp32_inference,
                                )
                            except Fp32FallbackError as exc:
                                separation_state["amp_non_finite_outputs"] += 1
                                separation_state["fp32_fallback_attempts"] += 1
                                separation_state["fp32_fallback_failures"] += 1
                                separation_state["windows_skipped_after_failure"] += 1
                                separation_state["amp_inference_seconds"].append(exc.amp_seconds)
                                separation_state["fp32_retry_seconds"].append(exc.fp32_seconds)
                                with state_lock:
                                    errors.append((f"separation-window-{item.index:03d}", exc))
                                print(
                                    f"[AMP FALLBACK FAILED] window={item.index:03d} "
                                    "reason=non_finite_fp32_output",
                                    flush=True,
                                )
                                continue
                            finally:
                                monitor.leave("separation")
                            separated = recovery.output
                            elapsed = recovery.total_seconds
                            amp_seconds = recovery.amp_seconds
                            fp32_seconds = recovery.fp32_seconds
                            used_fp32_fallback = recovery.fallback_attempted
                            separation_state["amp_inference_seconds"].append(amp_seconds)
                            if recovery.fallback_attempted:
                                separation_state["amp_non_finite_outputs"] += 1
                                separation_state["fp32_fallback_attempts"] += 1
                                separation_state["fp32_fallback_successes"] += 1
                                separation_state["fp32_retry_seconds"].append(fp32_seconds)
                                print(
                                    f"[AMP FALLBACK OK] window={item.index:03d} "
                                    f"amp_time={amp_seconds:.3f}s "
                                    f"fp32_time={fp32_seconds:.3f}s "
                                    "output_finite=True",
                                    flush=True,
                                )
                        separation_ended = time.perf_counter()
                        if args.latency_diagnostics:
                            item.latency_timing[
                                "remote_worker_postprocess_ended_perf_counter"
                            ] = separation_ended
                        separation_ended_timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
                        raw_speakers = tuple(
                            np.ascontiguousarray(separated[source, 0, :], dtype=np.float32)
                            for source in (0, 1)
                        )
                        raw_speaker_rms = tuple(
                            float(np.sqrt(np.mean(audio * audio))) for audio in raw_speakers
                        )
                        raw_speaker_peak = tuple(
                            float(np.max(np.abs(audio))) for audio in raw_speakers
                        )
                        for source, audio in enumerate(raw_speakers):
                            if (
                                len(audio) != args.window_seconds * SAMPLE_RATE
                                or not np.isfinite(audio).all()
                                or (
                                    raw_speaker_rms[source] <= 1e-5
                                    and not pre_separation_silence
                                    and not remote
                                )
                            ):
                                raise ValueError(
                                    f"Invalid separated audio: window={item.index}, "
                                    f"raw_slot={source}, RMS={raw_speaker_rms[source]}"
                                )
                            if args.save_wav:
                                sf.write(
                                    DEBUG_OUTPUT_DIR / f"window_{item.index:03d}_speaker_{source}.wav",
                                    audio,
                                    SAMPLE_RATE,
                                    subtype="FLOAT",
                                )
                                sf.write(
                                    DEBUG_OUTPUT_DIR / f"window_{item.index:03d}_raw_slot_{source}.wav",
                                    audio,
                                    SAMPLE_RATE,
                                    subtype="FLOAT",
                                )
                            if args.diagnostic_audio:
                                diagnostic_raw_slot_windows[source][item.index] = audio.copy()
                        assignment = speaker_tracker.assign(
                            window=item.index,
                            raw_speakers=(raw_speakers[0], raw_speakers[1]),
                            mixture=item.audio,
                            pre_separation_silence=pre_separation_silence,
                        )
                        speakers = assignment.speakers
                        speaker_rms = tuple(
                            float(np.sqrt(np.mean(audio * audio))) for audio in speakers
                        )
                        speaker_peak = tuple(
                            float(np.max(np.abs(audio))) for audio in speakers
                        )
                        for logical_speaker, audio in enumerate(speakers):
                            if args.save_wav:
                                sf.write(
                                    DEBUG_OUTPUT_DIR
                                    / f"window_{item.index:03d}_logical_speaker_{logical_speaker}.wav",
                                    audio,
                                    SAMPLE_RATE,
                                    subtype="FLOAT",
                                )
                            if args.diagnostic_audio:
                                diagnostic_logical_speaker_windows[logical_speaker][
                                    item.index
                                ] = audio.copy()
                        pair_correlation = (
                            0.0
                            if pre_separation_silence or (remote and min(raw_speaker_rms) == 0)
                            else float(np.corrcoef(raw_speakers)[0, 1])
                        )
                        if pre_separation_silence:
                            scores = np.empty((2, 0))
                            mapping = "silence_gate"
                        elif quality_references:
                            scores = np.array(
                                [
                                    [base.estimate_offset(reference, output)[1] for reference in quality_references]
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
                        separated_item = SeparatedWindow(
                            source=item,
                            raw_speakers=(raw_speakers[0], raw_speakers[1]),
                            speakers=(speakers[0], speakers[1]),
                            speaker_assignment=assignment.diagnostic,
                            separation_started=separation_started,
                            separation_ended=separation_ended,
                            separation_started_timestamp=separation_started_timestamp,
                            separation_ended_timestamp=separation_ended_timestamp,
                            separation_seconds=elapsed,
                            amp_inference_seconds=amp_seconds,
                            fp32_retry_seconds=fp32_seconds,
                            used_fp32_fallback=used_fp32_fallback,
                            pre_separation_silence=pre_separation_silence,
                            separation_input_stats=input_stats.to_dict(),
                            audio_queue_backlog=backlog,
                            separated_queue_size=separated_queue.qsize(),
                            torch_allocated_mib=0.0 if remote else torch.cuda.memory_allocated(device) / MIB,
                            torch_reserved_mib=0.0 if remote else torch.cuda.memory_reserved(device) / MIB,
                            torch_peak_allocated_mib=0.0 if remote else torch.cuda.max_memory_allocated(device) / MIB,
                            speaker_rms=(speaker_rms[0], speaker_rms[1]),
                            speaker_peak=(speaker_peak[0], speaker_peak[1]),
                            raw_speaker_rms=(raw_speaker_rms[0], raw_speaker_rms[1]),
                            raw_speaker_peak=(raw_speaker_peak[0], raw_speaker_peak[1]),
                            reference_scores=scores.tolist(),
                            pair_correlation=pair_correlation,
                            speaker_mapping=mapping,
                            remote_result=remote_result,
                            latency_timing=dict(item.latency_timing),
                        )
                        separation_state["inferred"] += 1
                        separated_queue_enqueue_started = time.perf_counter()
                        accepted = base.enqueue_or_drop(
                            separated_queue,
                            separated_item,
                            "separated_queue",
                            item.index,
                            dropped,
                            state_lock,
                        )
                        separated_queue_enqueue_ended = time.perf_counter()
                        if accepted:
                            if args.latency_diagnostics:
                                separated_item.latency_timing.update(
                                    {
                                        "separated_queue_enqueue_started_perf_counter": (
                                            separated_queue_enqueue_started
                                        ),
                                        "separated_queue_enqueued_perf_counter": (
                                            separated_queue_enqueue_ended
                                        ),
                                        "separated_queue_enqueue_seconds": (
                                            separated_queue_enqueue_ended
                                            - separated_queue_enqueue_started
                                        ),
                                    }
                                )
                            separated_item.separated_queue_size = separated_queue.qsize()
                            separation_state["max_queue_depth"] = max(
                                separation_state["max_queue_depth"], separated_queue.qsize()
                            )
                            separation_state["delivered"] += 1
                        print(
                            f"Separation {item.index:03d}: {elapsed:.3f}s, "
                            f"RTF={elapsed / args.window_seconds:.3f}, audio_backlog={backlog}, "
                            f"separated_queue={separated_queue.qsize()}/{separated_queue_maxsize}, "
                            f"pair_corr={pair_correlation:.3f}, mapping={mapping}, "
                            f"tracking={assignment.diagnostic['assignment_method']} "
                            f"raw_to_logical={assignment.diagnostic['raw_to_logical_mapping']} "
                            f"identity_score={assignment.diagnostic['identity_score']} "
                            f"swap_score={assignment.diagnostic['swap_score']} "
                            f"raw_input_similarity={assignment.diagnostic['raw_input_similarity']} "
                            f"overlap_matrix={assignment.diagnostic['overlap_similarity_matrix']} "
                            f"source_continuity={assignment.diagnostic['single_source_continuity']}",
                            flush=True,
                        )
                    finally:
                        audio_queue.task_done()
            except BaseException as exc:
                stop_capture.set()
                with state_lock:
                    errors.append(("separation", exc))
                while True:
                    try:
                        audio_queue.get_nowait()
                        audio_queue.task_done()
                    except queue.Empty:
                        break
            finally:
                try:
                    base.put_stop(separated_queue, "separated_queue")
                except BaseException as exc:
                    with state_lock:
                        errors.append(("separation-stop", exc))

        def stt_worker() -> None:
            previous_subtitle_window: int | None = None
            previous_subtitle_audio: tuple[np.ndarray, np.ndarray] | None = None
            previous_subtitle_vads: list[dict[str, Any]] = []
            try:
                while True:
                    item = separated_queue.get()
                    separated_queue_dequeued = time.perf_counter()
                    try:
                        if item is STOP:
                            break
                        assert isinstance(item, SeparatedWindow)
                        if args.latency_diagnostics:
                            item.latency_timing[
                                "separated_queue_dequeued_perf_counter"
                            ] = separated_queue_dequeued
                        separated_backlog = base.data_backlog(separated_queue)
                        stt_state["max_separated_backlog"] = max(
                            stt_state["max_separated_backlog"], separated_backlog
                        )
                        stt_started = time.perf_counter()
                        if args.latency_diagnostics:
                            item.latency_timing[
                                "local_result_worker_started_perf_counter"
                            ] = stt_started
                        speaker_times: list[float] = []
                        transcripts: list[str] = []
                        vad_results: list[dict[str, Any]] = []
                        logical_remote_slots = item.remote_result.logical_slots(item.speaker_assignment) if item.remote_result is not None else None
                        for speaker_index, audio in enumerate(item.speakers):
                            active = True
                            if args.vad:
                                if logical_remote_slots is not None:
                                    vad_result = logical_remote_slots[speaker_index]["vad"]
                                else:
                                    assert vad_model is not None
                                    vad_result = detect_speech_activity(
                                        vad_model, audio, args.vad_min_speech_ms
                                    )
                                vad_results.append(vad_result)
                                vad_state["checked_speakers"] += 1
                                vad_state["processing_seconds"].append(vad_result["processing_seconds"])
                                active = vad_result["speech_detected"]
                                vad_state["active_speakers" if active else "skipped_speakers"] += 1
                                print(
                                    f"[VAD] window={item.source.index:03d} speaker_{speaker_index} "
                                    f"speech={active} speech_ms={vad_result['speech_duration_ms']:.0f} "
                                    f"ratio={vad_result['speech_ratio']:.3f} rms={vad_result['rms']:.6f} "
                                    f"peak={vad_result['peak']:.6f} vad={vad_result['processing_seconds'] * 1000:.1f}ms",
                                    flush=True,
                                )
                            if logical_remote_slots is not None:
                                slot = logical_remote_slots[speaker_index]
                                transcript, elapsed = slot["raw_transcript"], slot["stt_seconds"]
                                stt_state["calls_executed" if active else "calls_skipped_by_vad"] += 1
                            elif active:
                                monitor.enter("stt")
                                try:
                                    transcript, elapsed = base.transcribe_audio(
                                        stt_model, audio, backend=args.stt_backend
                                    )
                                finally:
                                    monitor.leave("stt")
                                stt_state["calls_executed"] += 1
                            else:
                                transcript, elapsed = "", 0.0
                                stt_state["calls_skipped_by_vad"] += 1
                            transcripts.append(transcript)
                            speaker_times.append(elapsed)
                            stt_state["speaker_inputs"] += 1
                        stt_inference_ended = time.perf_counter()
                        secondary_leakage = build_window_diagnostic(
                            speaker_0=item.speakers[0],
                            speaker_1=item.speakers[1],
                            vad_results=vad_results,
                            transcripts=transcripts,
                        )
                        raw_transcripts = transcripts.copy()
                        if (
                            args.diagnostic_audio
                            and secondary_leakage["secondary_artifact_candidate"]
                        ):
                            full_window_diagnostic = build_full_window_diagnostic(
                                original=item.source.audio,
                                speaker_0=item.speakers[0],
                                speaker_1=item.speakers[1],
                                vad_results=vad_results,
                                raw_transcripts=raw_transcripts,
                                candidate=True,
                                reasons=secondary_leakage["reasons"],
                                sample_rate=SAMPLE_RATE,
                            )
                            full_window_diagnostic["window"] = item.source.index
                            full_window_diagnostic["wav_save"] = save_candidate_full_window_wavs(
                                enabled=True,
                                candidate=True,
                                output_dir=DIAGNOSTIC_WINDOW_OUTPUT_DIR,
                                window=item.source.index,
                                original=item.source.audio,
                                speaker_0=item.speakers[0],
                                speaker_1=item.speakers[1],
                                sample_rate=SAMPLE_RATE,
                            )
                            secondary_leakage["full_window_diagnostic"] = full_window_diagnostic
                            if not full_window_diagnostic["wav_save"]["saved"]:
                                print(
                                    f"[DIAGNOSTIC WINDOW] window={item.source.index:03d} "
                                    f"WAV save failed: {full_window_diagnostic['wav_save'].get('error')}",
                                    flush=True,
                                )
                        if secondary_validity_tracker is not None:
                            secondary_leakage["validity_evidence"] = (
                                secondary_validity_tracker.observe(
                                    window=item.source.index,
                                    diagnostic=secondary_leakage,
                                    raw_stt=raw_transcripts[1],
                                    full_window_diagnostic=secondary_leakage.get(
                                        "full_window_diagnostic"
                                    ),
                                )
                            )
                        suppression_reason = secondary_transcript_suppression_reason(
                            secondary_leakage, raw_transcripts[1]
                        )
                        secondary_leakage["suppression"] = {
                            "applied": suppression_reason is not None,
                            "reason": suppression_reason,
                        }
                        secondary_leakage.update(
                            {
                                "window": item.source.index,
                                "capture_timestamp": item.source.capture_timestamp,
                                "stream_start_seconds": item.source.stream_start_seconds,
                                "stream_end_seconds": item.source.stream_end_seconds,
                                "raw_transcripts": {
                                    "speaker_0": raw_transcripts[0],
                                    "speaker_1": raw_transcripts[1],
                                },
                            }
                        )
                        if suppression_reason is not None:
                            transcripts[1] = ""
                            print(
                                f"[SECONDARY ARTIFACT] window={item.source.index:03d} "
                                f"speaker_1 suppressed reason={suppression_reason} "
                                f'raw="{raw_transcripts[1]}"',
                                flush=True,
                            )
                        assembler_subtitle_started = time.perf_counter()
                        subtitle_vads = [dict(vad) for vad in vad_results]
                        if suppression_reason is not None and len(subtitle_vads) == 2:
                            subtitle_vads[1]["speech_detected"] = False
                            subtitle_vads[1]["timestamps"] = []
                        if args.assemble:
                            transcripts, subtitle_vads, routing = resolve_subtitle_fragments(
                                speakers=item.speakers,
                                transcripts=transcripts,
                                vad_results=subtitle_vads,
                                diagnostic=secondary_leakage,
                                previous_texts=[
                                    assembler.previous
                                    if assembler.last_window == item.source.index - 1 else ""
                                    for assembler in assemblers
                                ],
                                active_hypotheses=[state.partial_text for state in subtitle_states],
                            )
                            secondary_leakage["subtitle_routing"] = routing
                            if routing["applied"]:
                                print(
                                    f"[SUBTITLE ROUTING] window={item.source.index:03d} "
                                    f"source={routing['candidate']} target={routing['owner']} "
                                    f"reason={routing['reason']} "
                                    f"speech_corr={routing['candidate_speech_correlations']}",
                                    flush=True,
                                )
                            transcripts, subtitle_vads, admission = admit_subtitle_streams(
                                mixture=item.source.audio,
                                speakers=item.speakers,
                                transcripts=transcripts,
                                vad_results=subtitle_vads,
                                active_hypotheses=[state.partial_text for state in subtitle_states],
                                speaker_assignment=item.speaker_assignment,
                                candidate_contexts=[state.candidate_context for state in subtitle_states],
                                window=item.source.index,
                            )
                            secondary_leakage["subtitle_admission"] = admission
                            for decision in admission["streams"]:
                                if decision["reason"] != "inactive":
                                    print(
                                        f"[SUBTITLE ADMISSION] window={item.source.index:03d} "
                                        + json.dumps(decision, ensure_ascii=False),
                                        flush=True,
                                    )
                        subtitle_created_times: list[float] = []
                        window_assembly_events: list[dict[str, Any]] = []
                        utterance_hypotheses: list[str] = []
                        window_subtitle_state_events: list[dict[str, Any]] = []
                        if args.assemble:
                            for speaker_index, transcript in enumerate(transcripts):
                                transition = (subtitle_vads[speaker_index].get("candidate_transition", {})
                                              if args.vad else {})
                                if transition.get("restart"):
                                    discarded = subtitle_states[speaker_index].finalize(
                                        item.source.index, item.source.stream_end_seconds,
                                        "secondary_candidate_restart",
                                    )
                                    window_subtitle_state_events.append(discarded.to_dict())
                                    publish_subtitle_state_event(discarded)
                                    assemblers[speaker_index].reset_utterance(final_text=discarded.text)
                                overlap_evidence: dict[str, Any] = {}
                                if (
                                    previous_subtitle_window == item.source.index - 1
                                    and previous_subtitle_audio is not None
                                    and len(previous_subtitle_vads) == len(subtitle_vads) == 2
                                ):
                                    overlap_evidence = subtitle_overlap_evidence(
                                        previous_subtitle_audio[speaker_index],
                                        item.speakers[speaker_index],
                                        previous_subtitle_vads[speaker_index],
                                        subtitle_vads[speaker_index],
                                        (args.window_seconds - args.stride_seconds) * SAMPLE_RATE,
                                    )
                                event = assemblers[speaker_index].process(
                                    item.source.index, transcript,
                                    shared_speech=overlap_evidence.get("shared_speech"),
                                    supported_tail_revision=overlap_evidence.get("supported_tail_revision", False),
                                )
                                event_dict = event.to_dict()
                                event_dict["audio_overlap_evidence"] = overlap_evidence
                                window_assembly_events.append(event_dict)
                                assembler_events.append(event_dict)
                                utterance_hypotheses.append(event.utterance_hypothesis)
                                similarity = (
                                    f" similarity={event.similarity:.3f}"
                                    if event.similarity is not None
                                    else ""
                                )
                                print(
                                    f"[ASSEMBLER] window={item.source.index:03d} speaker={speaker_index} "
                                    f"match={event.match_type}{similarity} "
                                    f"duplicate_only={event.duplicate_only}\n"
                                    f"  previous=\"{truncate_assembler_text(event.previous)}\"\n"
                                    f"  current=\"{truncate_assembler_text(event.raw)}\"\n"
                                    f"  overlap=\"{truncate_assembler_text(event.overlap)}\"\n"
                                    f"  new=\"{truncate_assembler_text(event.new)}\"",
                                    flush=True,
                                )
                                print(
                                    f"[{base.format_stream_time(item.source.stream_end_seconds)}] "
                                    f"speaker_{speaker_index} RAW: {event.raw}",
                                    flush=True,
                                )
                                if event.new:
                                    print(
                                        f"[{base.format_stream_time(item.source.stream_end_seconds)}] "
                                        f"speaker_{speaker_index} NEW: {event.new}",
                                        flush=True,
                                    )
                                elif event.duplicate_only:
                                    print(
                                        f"[{base.format_stream_time(item.source.stream_end_seconds)}] "
                                        f"speaker_{speaker_index} DUPLICATE-ONLY",
                                        flush=True,
                                    )
                                print(
                                    f"[{base.format_stream_time(item.source.stream_end_seconds)}] "
                                    f"speaker_{speaker_index} UTTERANCE: "
                                    f"{event.utterance_hypothesis}",
                                    flush=True,
                                )
                            for speaker_index, hypothesis in enumerate(utterance_hypotheses):
                                speech_detected = (
                                    subtitle_vads[speaker_index]["speech_detected"]
                                    if args.vad
                                    else bool(transcripts[speaker_index])
                                )
                                if speaker_index == 1 and suppression_reason is not None:
                                    speech_detected = False
                                state_events = subtitle_states[speaker_index].process(
                                    window=item.source.index,
                                    hypothesis=hypothesis,
                                    speech_detected=speech_detected,
                                    stream_time_seconds=item.source.stream_end_seconds,
                                    confirmed_prefix_length=window_assembly_events[speaker_index]["confirmed_prefix_length"],
                                    source_supported=(subtitle_vads[speaker_index].get("source_supported", False)
                                                      if args.vad else False),
                                    source_text=transcripts[speaker_index],
                                    candidate_transition=(subtitle_vads[speaker_index].get("candidate_transition")
                                                          if args.vad else None),
                                )
                                for state_event in state_events:
                                    window_subtitle_state_events.append(state_event.to_dict())
                                    print(
                                        f"[SUBTITLE SUPPORT] window={state_event.window:03d} "
                                        f"speaker={state_event.speaker} utterance={state_event.utterance_id} "
                                        + json.dumps(state_event.support_update, ensure_ascii=False),
                                        flush=True,
                                    )
                                    published_event = publish_subtitle_state_event(
                                        state_event,
                                        stt_inference_ended=stt_inference_ended,
                                    )
                                    if published_event:
                                        subtitle_created_times.append(time.perf_counter())
                                    if state_event.status == "final":
                                        assemblers[speaker_index].reset_utterance(final_text=state_event.text)
                            previous_subtitle_window = item.source.index
                            previous_subtitle_audio = item.speakers
                            previous_subtitle_vads = subtitle_vads
                        else:
                            for speaker_index, transcript in enumerate(transcripts):
                                if transcript:
                                    print(
                                        f"[{base.format_stream_time(item.source.stream_end_seconds)}] "
                                        f"speaker_{speaker_index}: {transcript}",
                                        flush=True,
                                    )
                        assembler_subtitle_ended = time.perf_counter()
                        pipeline_ended = time.perf_counter()
                        latency_diagnostic = None
                        if args.latency_diagnostics:
                            item.latency_timing.update(
                                {
                                    "assembler_subtitle_started_perf_counter": (
                                        assembler_subtitle_started
                                    ),
                                    "assembler_subtitle_ended_perf_counter": (
                                        assembler_subtitle_ended
                                    ),
                                    "first_subtitle_created_perf_counter": (
                                        subtitle_created_times[0]
                                        if subtitle_created_times
                                        else None
                                    ),
                                    "result_created_perf_counter": pipeline_ended,
                                }
                            )
                            latency_diagnostic = build_window_latency_diagnostics(
                                item.latency_timing, item.remote_result
                            )
                        assembler_total = sum(
                            event["matching_seconds"] for event in window_assembly_events
                        )
                        post_capture = pipeline_ended - item.source.capture_ended
                        metric = {
                            "window": item.source.index,
                            "processing_mode": args.processing_mode,
                            "remote_request_seconds": item.remote_result.request_seconds if item.remote_result else None,
                            "remote_timing": item.remote_result.timing if item.remote_result else None,
                            "remote_gpu_memory": item.remote_result.gpu_memory if item.remote_result else None,
                            "window_seconds": args.window_seconds,
                            "stride_seconds": args.stride_seconds,
                            "stream_start_seconds": item.source.stream_start_seconds,
                            "stream_end_seconds": item.source.stream_end_seconds,
                            "capture_completed_timestamp": item.source.capture_timestamp,
                            "capture_new_audio_seconds": (
                                args.window_seconds if item.source.index == 0 else args.stride_seconds
                            ),
                            "capture_seconds": item.source.capture_ended - item.source.capture_started,
                            "capture_audio": item.source.capture_audio_stats,
                            "separation_started_timestamp": item.separation_started_timestamp,
                            "separation_ended_timestamp": item.separation_ended_timestamp,
                            "separation_started_after_capture_complete": (
                                item.separation_started - item.source.capture_ended
                            ),
                            "separation_seconds": item.separation_seconds,
                            "separation_rtf": item.separation_seconds / args.window_seconds,
                            "amp_inference_seconds": item.amp_inference_seconds,
                            "fp32_retry_seconds": item.fp32_retry_seconds,
                            "used_fp32_fallback": item.used_fp32_fallback,
                            "pre_separation_silence": item.pre_separation_silence,
                            "separation_input_stats": item.separation_input_stats,
                            "speaker_0_stt_seconds": speaker_times[0],
                            "speaker_1_stt_seconds": speaker_times[1],
                            "stt_total_seconds": item.remote_result.timing["stt_seconds"] if item.remote_result else stt_inference_ended - stt_started,
                            "vad": vad_results,
                            "assembler_total_seconds": assembler_total,
                            "post_capture_latency_seconds": post_capture,
                            "estimated_window_start_latency_seconds": args.window_seconds + post_capture,
                            "estimated_newest_audio_latency_seconds": post_capture,
                            "estimated_oldest_new_audio_latency_seconds": (
                                args.stride_seconds + post_capture
                            ),
                            "result_completed_after_stream_start": (
                                pipeline_ended - capture_state["stream_started"]
                            ),
                            "audio_queue_depth": item.source.audio_queue_size,
                            "separated_queue_depth": item.separated_queue_size,
                            "audio_queue_backlog": item.audio_queue_backlog,
                            "separated_queue_backlog": separated_backlog,
                            "torch_allocated_mib": item.torch_allocated_mib,
                            "torch_reserved_mib": item.torch_reserved_mib,
                            "torch_peak_allocated_mib": item.torch_peak_allocated_mib,
                            "gpu_vram_mib": monitor.used_mib(),
                            "speaker_rms": item.speaker_rms,
                            "speaker_peak": item.speaker_peak,
                            "raw_slot_rms": item.raw_speaker_rms,
                            "raw_slot_peak": item.raw_speaker_peak,
                            "speaker_assignment": item.speaker_assignment,
                            "raw_to_logical_mapping": item.speaker_assignment[
                                "raw_to_logical_mapping"
                            ],
                            "assignment_method": item.speaker_assignment[
                                "assignment_method"
                            ],
                            "reference_scores": item.reference_scores,
                            "pair_correlation": item.pair_correlation,
                            "speaker_mapping": item.speaker_mapping,
                            "transcripts": transcripts,
                            "secondary_leakage_diagnostic": secondary_leakage,
                            "assembly_events": window_assembly_events,
                            "subtitle_state_events": window_subtitle_state_events,
                            "dropped": False,
                            "error": None,
                        }
                        if latency_diagnostic is not None:
                            metric["latency_diagnostics"] = latency_diagnostic
                        with state_lock:
                            results.append(metric)
                        _notify_runtime_hook(
                            runtime_hooks.on_metric
                            if runtime_hooks is not None
                            else None,
                            metric,
                            "metric",
                        )
                        stt_state["processed"] += 1
                        print(
                            f"Window {item.source.index:03d} metrics: separation={item.separation_seconds:.3f}s, "
                            f"stt=[{speaker_times[0]:.3f}, {speaker_times[1]:.3f}]s, "
                            f"assembler={assembler_total * 1000:.3f}ms, "
                            f"post_capture={post_capture:.3f}s, "
                            f"queues=[{item.audio_queue_backlog}, {separated_backlog}], "
                            f"VRAM={item.torch_allocated_mib:.1f}/{item.torch_reserved_mib:.1f} MiB",
                            flush=True,
                        )
                        if latency_diagnostic is not None:
                            client_pipeline = latency_diagnostic["client_pipeline"]
                            remote_client_timing = latency_diagnostic["remote_client"]
                            server_timing = latency_diagnostic["server"]
                            print(
                                f"[LATENCY] window={item.source.index:03d} "
                                f"audio_queue_wait={client_pipeline['audio_queue_wait_seconds']:.6f}s "
                                f"wav_encode={remote_client_timing.get('payload_encode_seconds', 0.0):.6f}s "
                                f"http_round_trip={remote_client_timing.get('http_request_seconds', 0.0):.6f}s "
                                f"connect_tls_proxy={remote_client_timing.get('estimated_connect_tls_proxy_seconds', 0.0):.6f}s "
                                f"response_download={remote_client_timing.get('response_body_download_seconds', 0.0):.6f}s "
                                f"server_body_read={server_timing.get('request_body_read_seconds', 0.0):.6f}s "
                                f"server_queue_wait={server_timing.get('queue_wait_seconds', 0.0):.6f}s "
                                f"separation={server_timing.get('separation_seconds', 0.0):.6f}s "
                                f"vad={server_timing.get('vad_seconds', 0.0):.6f}s "
                                f"stt={server_timing.get('stt_seconds', 0.0):.6f}s "
                                f"gpu_monitor={server_timing.get('gpu_memory_query_seconds', 0.0):.6f}s "
                                f"response_decode={remote_client_timing.get('response_payload_decode_seconds', 0.0):.6f}s "
                                f"separated_queue_wait={client_pipeline['separated_queue_wait_seconds']:.6f}s "
                                f"assembler_subtitle={client_pipeline['assembler_subtitle_seconds']:.6f}s "
                                f"capture_to_result={client_pipeline['capture_to_result_seconds']:.6f}s "
                                f"bytes={remote_client_timing.get('request_body_bytes')}/"
                                f"{remote_client_timing.get('response_wire_bytes')} "
                                f"format={remote_client_timing.get('response_format')} "
                                f"client_connection_reused={remote_client_timing.get('client_connection_reused')} "
                                f"server_connection_reused={remote_client_timing.get('server_connection_reused')} "
                                f"connection_change={remote_client_timing.get('connection_change_reason')} "
                                f"server_connection_requests={remote_client_timing.get('server_connection_request_count')} "
                                f"request_thread={server_timing.get('server_request_thread_id')} "
                                f"inference_thread={server_timing.get('server_inference_thread_id')}",
                                flush=True,
                            )
                    finally:
                        separated_queue.task_done()
            except BaseException as exc:
                with state_lock:
                    errors.append(("stt", exc))

        phase_label = (
            "4-I"
            if args.assemble
            else ("4-D" if args.stt_backend == "sensevoice" else "3-D")
        )
        duration_label = "until Ctrl+C" if capture_duration is None else f"{capture_duration}s"
        window_label = "continuous" if window_count is None else str(window_count)
        print(
            f"Starting Phase {phase_label}: device={speaker.name}, stt_backend={args.stt_backend}, "
            f"duration={duration_label}, "
            f"window={args.window_seconds}s, stride={args.stride_seconds}s, "
            f"windows={window_label}, queues={audio_queue_maxsize_for_run}/{separated_queue_maxsize}",
            flush=True,
        )
        separation_thread = threading.Thread(target=separation_worker, name="separation_worker")
        stt_thread = threading.Thread(target=stt_worker, name="stt_worker")
        capture_thread = threading.Thread(target=capture_worker, name="capture_worker")
        separation_thread.start()
        stt_thread.start()
        capture_thread.start()
        try:
            capture_thread.join()
            separation_thread.join()
            stt_thread.join()
        except KeyboardInterrupt:
            if not args.live:
                raise
            print("\nCtrl+C received; draining the live pipeline before shutdown.", flush=True)
            stop_capture.set()
            capture_thread.join()
            separation_thread.join()
            stt_thread.join()

        if playback_errors:
            errors.extend((f"playback-{name}", exc) for name, exc in playback_errors)
        if args.assemble:
            final_stream_time = max(
                (item["stream_end_seconds"] for item in results),
                default=float(capture_duration or 0),
            )
            for state in subtitle_states:
                final_event = state.flush(
                    window=max(state.last_window, 0),
                    stream_time_seconds=final_stream_time,
                )
                if final_event is not None:
                    publish_subtitle_state_event(final_event)
                    assemblers[state.speaker].reset_utterance(final_text=final_event.text)
        if websocket_server is not None:
            websocket_server.flush()
            time.sleep(0.1)
            websocket_stats = websocket_server.snapshot()
            websocket_stats["enabled"] = True
            if subtitle_events:
                websocket_stats["stt_to_enqueue_ms"] = base.distribution(
                    [event["stt_to_enqueue_ms"] for event in subtitle_events]
                )
            websocket_server.stop()
            websocket_server = None
        if args.diagnostic_audio:
            try:
                diagnostic_summary = run_diagnostic_stt()
            except BaseException as exc:
                diagnostic_summary = {
                    "enabled": True,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
                errors.append(("diagnostic-stt", exc))
        results.sort(key=lambda item: item["window"])

        performance: dict[str, Any] = {}
        quality: dict[str, Any] = {}
        separation_quality: dict[str, Any] = {}
        latency: dict[str, Any] = {}
        duplicates: dict[str, Any] = {}
        assembler_summary: dict[str, Any] = {}
        subtitle_stabilization_summary: dict[str, Any] = {}
        speaker_tracking_summary: dict[str, Any] = {"enabled": True}
        if results:
            speaker_stt_times = [
                value
                for item in results
                for value in (item["speaker_0_stt_seconds"], item["speaker_1_stt_seconds"])
            ]
            completion_times = [item["result_completed_after_stream_start"] for item in results]
            update_intervals = [right - left for left, right in zip(completion_times, completion_times[1:])]
            performance = {
                "separation": base.distribution([item["separation_seconds"] for item in results]),
                "speaker_stt": base.distribution(speaker_stt_times),
                "window_stt_total": base.distribution([item["stt_total_seconds"] for item in results]),
                "post_capture_latency": base.distribution(
                    [item["post_capture_latency_seconds"] for item in results]
                ),
                "estimated_window_start_latency": base.distribution(
                    [item["estimated_window_start_latency_seconds"] for item in results]
                ),
            }
            if vad_state["processing_seconds"]:
                performance["vad"] = base.distribution(vad_state["processing_seconds"])
            quality_rows = [
                {"chunk": item["window"], "transcripts": item["transcripts"]} for item in results
            ]
            quality = base.transcript_quality(quality_rows)
            duplicates = analyze_overlap_duplicates(results)
            mappings = [item["speaker_mapping"] for item in results]
            separation_quality = {
                "silent_output_count": sum(
                    rms <= 1e-5 for item in results for rms in item["speaker_rms"]
                ),
                "collapse_candidate_count": sum(
                    abs(item["pair_correlation"]) >= 0.98 for item in results
                ),
                "pair_correlation_abs_max": max(abs(item["pair_correlation"]) for item in results),
                "speaker_rms": base.distribution(
                    [rms for item in results for rms in item["speaker_rms"]]
                ),
                "uncertain_mapping_count": mappings.count("uncertain"),
                "mapping_counts": {mapping: mappings.count(mapping) for mapping in set(mappings)},
            }
            assignments = [item["speaker_assignment"] for item in results]
            methods = [assignment["assignment_method"] for assignment in assignments]
            logical_mappings = [
                str(assignment["raw_to_logical_mapping"])
                for assignment in assignments
            ]
            speaker_tracking_summary = {
                "enabled": True,
                "method": "adjacent_window_overlap_continuity",
                "overlap_seconds": args.window_seconds - args.stride_seconds,
                "polarity_invariant_correlation": True,
                "speaker_embedding_used": False,
                "assignment_method_counts": {
                    method: methods.count(method) for method in set(methods)
                },
                "mapping_changed_count": sum(
                    assignment["mapping_changed"] for assignment in assignments
                ),
                "raw_to_logical_mapping_counts": {
                    mapping: logical_mappings.count(mapping)
                    for mapping in set(logical_mappings)
                },
                "confidence": base.distribution(
                    [assignment["confidence"] for assignment in assignments]
                ),
                "limitations": (
                    "No long-gap speaker re-identification; speaker embeddings are required "
                    "to reconnect identity after overlap continuity is lost."
                ),
            }
            latency = {
                "initial_subtitle_seconds": completion_times[0],
                "steady_state_result_interval": (
                    base.distribution(update_intervals) if update_intervals else {}
                ),
                "post_capture": performance["post_capture_latency"],
                "estimated_user_perceived_seconds": {
                    "first_result": completion_times[0],
                    "steady_state_new_content_interval_mean": (
                        statistics.mean(update_intervals) if update_intervals else 0.0
                    ),
                    "window_start_mean": performance["estimated_window_start_latency"]["mean"],
                    "newest_audio_mean": performance["post_capture_latency"]["mean"],
                    "new_content_latency_range_mean": [
                        performance["post_capture_latency"]["mean"],
                        args.stride_seconds + performance["post_capture_latency"]["mean"],
                    ],
                },
            }
        if args.assemble and assembler_events:
            assembler_times = [event["matching_seconds"] for event in assembler_events]
            incremental_rows = [
                {
                    "window": item["window"],
                    "transcripts": [event["new"] for event in item["assembly_events"]],
                }
                for item in results
            ]
            assembler_summary = {
                "enabled": True,
                "minimum_characters": args.assembler_min_characters,
                "fuzzy_matching_applied": True,
                "fuzzy_similarity_threshold": FUZZY_SIMILARITY_THRESHOLD,
                "event_count": len(assembler_events),
                "processed_transcript_count": len(assembler_events),
                "overlap_detected_count": sum(
                    event["match_type"] != "none" for event in assembler_events
                ),
                "exact_overlap_count": sum(
                    event["match_type"]
                    in {"token_exact", "raw_exact", "normalized_exact", "boundary_exact"}
                    for event in assembler_events
                ),
                "fuzzy_overlap_count": sum(
                    event["match_type"] in {"fuzzy", "fuzzy_replace"} for event in assembler_events
                ),
                "no_overlap_count": sum(
                    event["match_type"] == "none" for event in assembler_events
                ),
                "removed_character_count": sum(
                    event["overlap_length"] for event in assembler_events
                ),
                "duplicate_only_count": sum(
                    event["duplicate_only"] for event in assembler_events
                ),
                "new_text_count": sum(bool(event["new"]) for event in assembler_events),
                "raw_fragment_count": sum(
                    bool(event["raw_fragment"]) for event in assembler_events
                ),
                "new_fragment_count": sum(
                    bool(event["new_fragment"]) for event in assembler_events
                ),
                "suspicious_deletion_count": sum(
                    event["suspicious_deletion"] for event in assembler_events
                ),
                "matching_time": base.distribution(assembler_times),
                "assembled_by_speaker": [assembler.assembled for assembler in assemblers],
                "current_utterance_by_speaker": [
                    assembler.utterance_hypothesis for assembler in assemblers
                ],
                "by_speaker": [
                    {
                        "speaker": speaker,
                        "processed_transcript_count": sum(
                            event["speaker"] == speaker for event in assembler_events
                        ),
                        "exact_overlap_count": sum(
                            event["speaker"] == speaker
                            and event["match_type"]
                            in {
                                "token_exact",
                                "raw_exact",
                                "normalized_exact",
                                "boundary_exact",
                            }
                            for event in assembler_events
                        ),
                        "fuzzy_overlap_count": sum(
                            event["speaker"] == speaker and event["match_type"] in {"fuzzy", "fuzzy_replace"}
                            for event in assembler_events
                        ),
                        "duplicate_only_count": sum(
                            event["speaker"] == speaker and event["duplicate_only"]
                            for event in assembler_events
                        ),
                        "no_overlap_count": sum(
                            event["speaker"] == speaker and event["match_type"] == "none"
                            for event in assembler_events
                        ),
                    }
                    for speaker in (0, 1)
                ],
                "raw_duplicate_analysis": duplicates,
                "incremental_duplicate_analysis": analyze_overlap_duplicates(incremental_rows),
            }
        else:
            assembler_summary = {"enabled": False}

        if args.assemble:
            subtitle_stabilization_summary = {
                "enabled": True,
                "finalize_silence_ms": args.subtitle_finalize_silence_ms,
                "event_count": len(subtitle_state_events),
                "partial_events": sum(
                    event["status"] == "partial" for event in subtitle_state_events
                ),
                "partial_updates": sum(
                    event["status"] == "partial" and event["action"] != "start"
                    for event in subtitle_state_events
                ),
                "partial_extensions": sum(
                    event["action"] == "extend" for event in subtitle_state_events
                ),
                "hypothesis_extensions": sum(
                    event["match_type"] == "hypothesis_extension"
                    for event in subtitle_state_events
                ),
                "partial_replacements": sum(
                    event["action"] == "replace" for event in subtitle_state_events
                ),
                "partial_retained": sum(
                    event["action"] == "retain" for event in subtitle_state_events
                ),
                "final_events": sum(
                    event["status"] == "final" for event in subtitle_state_events
                ),
                "finalized_by_vad_silence": sum(
                    event["finalize_reason"] == "vad_silence"
                    for event in subtitle_state_events
                ),
                "finalized_at_pipeline_end": sum(
                    event["finalize_reason"] == "pipeline_end"
                    for event in subtitle_state_events
                ),
                "consensus_history_observations": sum(
                    event["status"] == "partial" for event in subtitle_state_events
                ),
                "stable_promotions": sum(
                    event["stability_action"] == "promote_stable"
                    for event in subtitle_state_events
                ),
                "tentative_replacements": sum(
                    event["stability_action"] == "tentative_replace"
                    for event in subtitle_state_events
                ),
                "tentative_retained": sum(
                    event["stability_action"] == "tentative_retained"
                    for event in subtitle_state_events
                ),
                "tentative_discarded_at_finalization": sum(
                    event["stability_action"] == "tentative_discarded_at_final"
                    for event in subtitle_state_events
                ),
                "consensus_corrections": sum(
                    event["stability_action"] == "consensus_correction"
                    for event in subtitle_state_events
                ),
                "hypothesis_corrections": sum(
                    event["match_type"] == "hypothesis_correction"
                    for event in subtitle_state_events
                ),
                "finalized_utterances": sum(
                    event["status"] == "final" for event in subtitle_state_events
                ),
                "final_segments_by_speaker": [state.final_segments for state in subtitle_states],
                "remaining_partial_by_speaker": [state.partial_text for state in subtitle_states],
                "by_speaker": [
                    {
                        "speaker": speaker,
                        "partial_events": sum(
                            event["speaker"] == speaker and event["status"] == "partial"
                            for event in subtitle_state_events
                        ),
                        "partial_extensions": sum(
                            event["speaker"] == speaker and event["action"] == "extend"
                            for event in subtitle_state_events
                        ),
                        "partial_replacements": sum(
                            event["speaker"] == speaker and event["action"] == "replace"
                            for event in subtitle_state_events
                        ),
                        "final_events": sum(
                            event["speaker"] == speaker and event["status"] == "final"
                            for event in subtitle_state_events
                        ),
                        "stable_promotions": sum(
                            event["speaker"] == speaker
                            and event["stability_action"] == "promote_stable"
                            for event in subtitle_state_events
                        ),
                        "tentative_replacements": sum(
                            event["speaker"] == speaker
                            and event["stability_action"] == "tentative_replace"
                            for event in subtitle_state_events
                        ),
                        "tentative_retained": sum(
                            event["speaker"] == speaker
                            and event["stability_action"] == "tentative_retained"
                            for event in subtitle_state_events
                        ),
                        "tentative_discarded_at_finalization": sum(
                            event["speaker"] == speaker
                            and event["stability_action"]
                            == "tentative_discarded_at_final"
                            for event in subtitle_state_events
                        ),
                        "consensus_corrections": sum(
                            event["speaker"] == speaker
                            and event["stability_action"] == "consensus_correction"
                            for event in subtitle_state_events
                        ),
                    }
                    for speaker in (0, 1)
                ],
            }
        else:
            subtitle_stabilization_summary = {"enabled": False}

        separation_recovery_summary = {
            "pre_separation_silence_rms_threshold": PRE_SEPARATION_SILENCE_RMS,
            "pre_separation_silence_checked": separation_state[
                "pre_separation_silence_checked"
            ],
            "pre_separation_silence_skipped": separation_state[
                "pre_separation_silence_skipped"
            ],
            "pre_separation_non_silence": separation_state[
                "pre_separation_non_silence"
            ],
            "pre_separation_silence_ratio": (
                separation_state["pre_separation_silence_skipped"]
                / separation_state["pre_separation_silence_checked"]
                if separation_state["pre_separation_silence_checked"]
                else 0.0
            ),
            "amp_separation_attempts": separation_state["amp_attempts"],
            "amp_non_finite_outputs": separation_state["amp_non_finite_outputs"],
            "fp32_fallback_attempts": separation_state["fp32_fallback_attempts"],
            "fp32_fallback_successes": separation_state["fp32_fallback_successes"],
            "fp32_fallback_failures": separation_state["fp32_fallback_failures"],
            "invalid_input_windows": (
                capture_state["invalid_audio_windows"]
                + separation_state["invalid_input_windows"]
            ),
            "invalid_capture_input_windows": capture_state["invalid_audio_windows"],
            "invalid_separation_input_windows": separation_state["invalid_input_windows"],
            "windows_skipped_after_separation_failure": separation_state[
                "windows_skipped_after_failure"
            ],
            "windows_skipped_total": (
                capture_state["invalid_audio_windows"]
                + separation_state["windows_skipped_after_failure"]
            ),
            "amp_inference_time": (
                base.distribution(separation_state["amp_inference_seconds"])
                if separation_state["amp_inference_seconds"]
                else {}
            ),
            "fp32_retry_time": (
                base.distribution(separation_state["fp32_retry_seconds"])
                if separation_state["fp32_retry_seconds"]
                else {}
            ),
        }

        vad_summary = {
            "enabled": args.vad,
            "minimum_speech_ms": args.vad_min_speech_ms,
            "speakers_checked": vad_state["checked_speakers"],
            "speakers_active": vad_state["active_speakers"],
            "speakers_skipped": vad_state["skipped_speakers"],
            "stt_calls_executed": stt_state["calls_executed"],
            "stt_calls_skipped_by_vad": stt_state["calls_skipped_by_vad"],
            "processing_seconds": (
                base.distribution(vad_state["processing_seconds"])
                if vad_state["processing_seconds"]
                else {}
            ),
        }
        if args.vad:
            active_counts = [
                sum(vad_result["speech_detected"] for vad_result in item["vad"])
                for item in results
            ]
            vad_summary["both_speakers_active_windows"] = active_counts.count(2)
            vad_summary["single_speaker_active_windows"] = active_counts.count(1)
            vad_summary["no_speaker_active_windows"] = active_counts.count(0)

        secondary_leakage_summary = build_secondary_leakage_summary(
            [item["secondary_leakage_diagnostic"] for item in results]
        )
        secondary_validity_summary = build_secondary_validity_summary(
            [item["secondary_leakage_diagnostic"] for item in results]
        )

        expected_window_count = window_count if window_count is not None else capture_state["captured"]
        success = (
            capture_state["captured"] == expected_window_count
            and separation_state["delivered"] == expected_window_count
            and stt_state["processed"] == expected_window_count
            and stt_state["speaker_inputs"] == expected_window_count * 2
            and not dropped
            and not errors
            and (
                all(item["separation_seconds"] < args.stride_seconds for item in results)
                or (
                    args.websocket
                    and performance["separation"]["mean"] < args.stride_seconds
                    and separation_state["max_audio_backlog"] == 0
                    and stt_state["max_separated_backlog"] == 0
                )
            )
            and (
                any(text for item in results for text in item["transcripts"])
                or separation_state["pre_separation_silence_skipped"] == expected_window_count
            )
            and (not args.assemble or len(assembler_events) == expected_window_count * 2)
            and (
                not args.websocket
                or (
                    bool(subtitle_events)
                    and all(event["websocket_accepted"] for event in subtitle_events)
                    and websocket_stats["dropped_oldest"] == 0
                )
            )
        )
        summary = {
            "phase": phase_label,
            "processing_mode": args.processing_mode,
            "remote_server_url": args.remote_server_url if remote else None,
            "stt_backend": args.stt_backend,
            "architecture": (
                "Windows capture -> audio queue -> HTTP GPU separation/VAD/STT -> raw slots -> "
                "logical speaker tracker -> separated queue -> assembler/state -> WebSocket/UI"
                if remote else
                "continuous capture -> sliding buffer -> audio queue -> separation -> separated "
                "queue -> STT -> assembler -> subtitle event queue -> WebSocket -> browser"
                if args.websocket
                else (
                    "continuous capture -> sliding buffer -> audio queue -> separation -> "
                    "separated queue -> STT -> assembler -> console"
                    if args.assemble
                    else "continuous capture -> sliding buffer -> audio queue -> separation -> separated queue -> STT -> console"
                )
            ),
            "duration_seconds": capture_duration,
            "reference_audio": (
                None
                if args.live
                else {
                    "speaker_a": str(args.reference_a or "phase1/reference/speaker_a.wav"),
                    "speaker_b": str(args.reference_b or "phase1/reference/speaker_b.wav"),
                }
            ),
            "window_seconds": args.window_seconds,
            "stride_seconds": args.stride_seconds,
            "overlap_seconds": args.window_seconds - args.stride_seconds,
            "window_count": expected_window_count,
            "queue_maxsize": audio_queue_maxsize_for_run,
            "audio_queue_maxsize": audio_queue_maxsize_for_run,
            "separated_queue_maxsize": separated_queue_maxsize,
            "drop_policy": "drop newest when a bounded queue is full",
            "latency_diagnostics": {
                "enabled": args.latency_diagnostics,
                "per_window_field": (
                    "windows[].latency_diagnostics"
                    if args.latency_diagnostics
                    else None
                ),
            },
            "capture_success": capture_state["captured"],
            "separation_success": separation_state["inferred"],
            "separation_delivered": separation_state["delivered"],
            "stt_window_success": stt_state["processed"],
            "stt_speaker_input_count": stt_state["speaker_inputs"],
            "vad": vad_summary,
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
            "latency": latency,
            "transcript_quality": quality,
            "overlap_duplicates": duplicates,
            "assembler": assembler_summary,
            "subtitle_stabilization": subtitle_stabilization_summary,
            "separation_recovery": separation_recovery_summary,
            "websocket": websocket_stats,
            "diagnostic_audio": diagnostic_summary,
            "subtitle_events": subtitle_events,
            "subtitle_state_events": subtitle_state_events,
            "separation_quality": separation_quality,
            "speaker_tracking": speaker_tracking_summary,
            "secondary_leakage_diagnostic": secondary_leakage_summary,
            "secondary_validity_diagnostic": secondary_validity_summary,
            "gpu": {
                "baseline_mib": baseline_vram,
                "after_separation_model_mib": separator_vram,
                "after_both_models_mib": models_vram,
                "after_warmup_mib": after_warmup_vram,
                "separation_stage_peak_mib": monitor.stage_peaks.get("separation"),
                "stt_stage_peak_mib": monitor.stage_peaks.get("stt"),
                "overall_peak_mib": max(monitor.samples) if monitor.samples else None,
                "cuda_oom": any("out of memory" in str(exc).lower() for _, exc in errors),
            },
            "windows": results,
            "success": success,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        print(f"\n========== Phase {phase_label} summary ==========", flush=True)
        print(f"Capture: {capture_state['captured']}/{expected_window_count}")
        print(f"Separation: {separation_state['inferred']}/{expected_window_count}")
        print(f"STT windows/speakers: {stt_state['processed']}/{stt_state['speaker_inputs']}")
        if args.vad:
            print(
                f"VAD speakers checked/active/skipped: {vad_state['checked_speakers']}/"
                f"{vad_state['active_speakers']}/{vad_state['skipped_speakers']}"
            )
            print(
                f"STT calls executed/skipped by VAD: {stt_state['calls_executed']}/"
                f"{stt_state['calls_skipped_by_vad']}"
            )
            print(
                f"VAD windows both/single/none active: "
                f"{vad_summary['both_speakers_active_windows']}/"
                f"{vad_summary['single_speaker_active_windows']}/"
                f"{vad_summary['no_speaker_active_windows']}"
            )
        print(
            "Secondary leakage diagnostic: "
            f"speaker_1_active={secondary_leakage_summary['speaker_1_active_window_count']}, "
            f"both_active={secondary_leakage_summary['both_active_window_count']}, "
            f"only_speaker_1={secondary_leakage_summary['only_speaker_1_active_window_count']}, "
            f"candidates={secondary_leakage_summary['secondary_artifact_candidate_count']} "
            f"({secondary_leakage_summary['secondary_artifact_candidate_ratio']:.1%}), "
            f"indices={secondary_leakage_summary['candidate_window_indices']}, "
            f"nonlexical_suppressed="
            f"{secondary_leakage_summary['suppressed_nonlexical_window_count']} "
            f"{secondary_leakage_summary['suppressed_nonlexical_window_indices']}",
            flush=True,
        )
        print(f"Dropped: {dropped}")
        print(f"Errors: {[(stage, type(exc).__name__, str(exc)) for stage, exc in errors]}")
        print(
            "Pre-separation silence checked/skipped/non-silence: "
            f"{separation_recovery_summary['pre_separation_silence_checked']}/"
            f"{separation_recovery_summary['pre_separation_silence_skipped']}/"
            f"{separation_recovery_summary['pre_separation_non_silence']} "
            f"(ratio={separation_recovery_summary['pre_separation_silence_ratio']:.3f})"
        )
        print(
            "AMP attempts/non-finite: "
            f"{separation_recovery_summary['amp_separation_attempts']}/"
            f"{separation_recovery_summary['amp_non_finite_outputs']}"
        )
        print(
            "AMP fallback attempts/success/failure: "
            f"{separation_recovery_summary['fp32_fallback_attempts']}/"
            f"{separation_recovery_summary['fp32_fallback_successes']}/"
            f"{separation_recovery_summary['fp32_fallback_failures']}"
        )
        print(
            "Invalid input/skipped windows total: "
            f"{separation_recovery_summary['invalid_input_windows']}/"
            f"{separation_recovery_summary['windows_skipped_total']}"
        )
        if results:
            print(f"Mean separation: {performance['separation']['mean']:.3f}s")
            print(f"Mean window STT: {performance['window_stt_total']['mean']:.3f}s")
            if args.vad and "vad" in performance:
                print(f"Mean VAD: {performance['vad']['mean'] * 1000:.3f}ms")
            print(f"Mean post-capture: {performance['post_capture_latency']['mean']:.3f}s")
            print(f"Initial subtitle: {latency['initial_subtitle_seconds']:.3f}s")
            print(
                f"Steady result interval mean/p95/max: "
                f"{latency['steady_state_result_interval'].get('mean', 0.0):.3f}/"
                f"{latency['steady_state_result_interval'].get('p95', 0.0):.3f}/"
                f"{latency['steady_state_result_interval'].get('max', 0.0):.3f}s"
            )
            print(
                f"Duplicate adjacent pairs: {duplicates['duplicate_pair_count']}/"
                f"{duplicates['adjacent_window_pair_count']} "
                f"({duplicates['duplicate_pair_ratio']:.1%})"
            )
            print(
                f"Queue depth/backlog max: audio={capture_state['max_queue_depth']}/"
                f"{separation_state['max_audio_backlog']}, separated="
                f"{separation_state['max_queue_depth']}/{stt_state['max_separated_backlog']}"
            )
        if args.assemble and assembler_summary.get("enabled"):
            print(
                f"Assembler overlap/removed/duplicate-only: "
                f"{assembler_summary['overlap_detected_count']}/"
                f"{assembler_summary['removed_character_count']}/"
                f"{assembler_summary['duplicate_only_count']}"
            )
            print(
                f"Assembler exact/fuzzy/none: {assembler_summary['exact_overlap_count']}/"
                f"{assembler_summary['fuzzy_overlap_count']}/"
                f"{assembler_summary['no_overlap_count']}"
            )
            print(
                f"Assembler mean/p95/max: "
                f"{assembler_summary['matching_time']['mean'] * 1000:.3f}/"
                f"{assembler_summary['matching_time']['p95'] * 1000:.3f}/"
                f"{assembler_summary['matching_time']['max'] * 1000:.3f}ms"
            )
            print(
                "Subtitle partial/update/replace/final: "
                f"{subtitle_stabilization_summary['partial_events']}/"
                f"{subtitle_stabilization_summary['partial_updates']}/"
                f"{subtitle_stabilization_summary['partial_replacements']}/"
                f"{subtitle_stabilization_summary['final_events']}"
            )
            print(
                "Subtitle finalized by VAD/pipeline-end: "
                f"{subtitle_stabilization_summary['finalized_by_vad_silence']}/"
                f"{subtitle_stabilization_summary['finalized_at_pipeline_end']}"
            )
            print(
                "Consensus observations/promotions/corrections: "
                f"{subtitle_stabilization_summary['consensus_history_observations']}/"
                f"{subtitle_stabilization_summary['stable_promotions']}/"
                f"{subtitle_stabilization_summary['consensus_corrections']}"
            )
            print(
                "Tentative replaced/retained/discarded-at-final: "
                f"{subtitle_stabilization_summary['tentative_replacements']}/"
                f"{subtitle_stabilization_summary['tentative_retained']}/"
                f"{subtitle_stabilization_summary['tentative_discarded_at_finalization']}"
            )
        print(f"GPU after both models: {models_vram} MiB")
        print(f"GPU overall peak: {max(monitor.samples) if monitor.samples else None} MiB")
        print(f"JSON: {args.json_output}")
        print(f"Phase {phase_label}: {'PASS' if success else 'FAIL'}", flush=True)
        return 0 if success else 1
    finally:
        if remote_client is not None:
            remote_client.close()
        if websocket_server is not None:
            websocket_server.stop()
        monitor.stop()


if __name__ == "__main__":
    raise SystemExit(main())

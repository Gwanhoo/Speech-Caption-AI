from __future__ import annotations

import argparse
import base64
import io
import json
import math
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "phase3"))

from separation_recovery import finite_audio_stats, is_pre_separation_silence, silent_separation_output  # noqa: E402
from stt_context import LIVE_WHISPER_OPTIONS, transcribe_base  # noqa: E402
from validate_end_to_end_gpu import (  # noqa: E402
    SAMPLE_RATE,
    VAD_MINIMUM_SPEECH_MS,
    detect_speech_activity,
    load_separator,
    query_gpu_memory,
    separate_amp,
)


MAX_REQUEST_BYTES = 16 * 1024 * 1024
MAX_AUDIO_SECONDS = 30.0
FASTER_WHISPER_MODEL_NAME = "base"
FASTER_WHISPER_CHECKPOINT_ROOT = ROOT / "checkpoints" / "faster-whisper"


def faster_whisper_model_source() -> str:
    """Use a valid project-local base model when present, otherwise use HF's base ID."""
    model_cache = (
        FASTER_WHISPER_CHECKPOINT_ROOT
        / "models--Systran--faster-whisper-base"
    )
    candidates = [FASTER_WHISPER_CHECKPOINT_ROOT, model_cache]
    snapshots = model_cache / "snapshots"
    if snapshots.is_dir():
        candidates.extend(path for path in snapshots.iterdir() if path.is_dir())

    for candidate in candidates:
        if (candidate / "model.bin").is_file():
            return str(candidate)
    return FASTER_WHISPER_MODEL_NAME


@dataclass(frozen=True)
class LoadedModels:
    separator: Any
    separation_device: torch.device
    whisper: WhisperModel
    vad: Any


class PipelineService:
    def __init__(self) -> None:
        self.started_at = time.time()
        self.ready = False
        self.load_error: str | None = None
        self.model_load_count = 0
        self.requests_completed = 0
        self.requests_failed = 0
        self.requests_waiting = 0
        self._stats_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self.load_times: dict[str, float] = {}
        self.memory: dict[str, dict[str, int | None]] = {}
        self.models = self._load_models()

    @staticmethod
    def _memory_record() -> dict[str, int | None]:
        device, process = query_gpu_memory()
        return {"device_used_mib": device, "process_used_mib": process}

    def _load_models(self) -> LoadedModels:
        self.memory["before_models"] = self._memory_record()
        try:
            started = time.perf_counter()
            separator, separation_device = load_separator()
            self.load_times["mossformer2_seconds"] = time.perf_counter() - started
            self.memory["after_mossformer2"] = self._memory_record()

            started = time.perf_counter()
            whisper = WhisperModel(
                faster_whisper_model_source(), device="cuda", compute_type="float16"
            )
            self.load_times["faster_whisper_seconds"] = time.perf_counter() - started
            self.memory["after_faster_whisper"] = self._memory_record()

            started = time.perf_counter()
            vad = load_silero_vad()
            self.load_times["silero_vad_seconds"] = time.perf_counter() - started
            self.model_load_count = 1
            self.ready = True
            return LoadedModels(separator, separation_device, whisper, vad)
        except BaseException as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            raise

    def health(self) -> tuple[HTTPStatus, dict[str, Any]]:
        gpu_device, gpu_process = query_gpu_memory()
        with self._stats_lock:
            payload = {
                "status": "ready" if self.ready else "not_ready",
                "gpu_ready": self.ready and torch.cuda.is_available(),
                "models_loaded": self.ready,
                "model_load_count": self.model_load_count,
                "model_load_seconds": dict(self.load_times),
                "gpu_memory": {
                    "device_used_mib": gpu_device,
                    "process_used_mib": gpu_process,
                },
                "requests": {
                    "completed": self.requests_completed,
                    "failed": self.requests_failed,
                    "waiting": self.requests_waiting,
                },
                "uptime_seconds": time.time() - self.started_at,
                "load_error": self.load_error,
            }
        return (HTTPStatus.OK if self.ready else HTTPStatus.SERVICE_UNAVAILABLE), payload

    def process(
        self,
        request_id: str,
        audio: np.ndarray,
        duration: float,
        request_diagnostics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.ready:
            raise RuntimeError("GPU pipeline is not ready")

        received_at = time.time()
        wait_started = time.perf_counter()
        if request_diagnostics is not None:
            request_diagnostics["server_queue_entered_perf_counter"] = wait_started
        with self._stats_lock:
            self.requests_waiting += 1
        try:
            with self._inference_lock:
                inference_started = time.perf_counter()
                queue_wait_seconds = inference_started - wait_started
                if request_diagnostics is not None:
                    request_diagnostics.update(
                        {
                            "server_inference_started_perf_counter": inference_started,
                            "server_queue_wait_seconds": queue_wait_seconds,
                        }
                    )
                with self._stats_lock:
                    self.requests_waiting -= 1
                response = self._process_serialized(
                    request_id,
                    audio,
                    duration,
                    received_at,
                    queue_wait_seconds,
                    request_diagnostics,
                )
        except BaseException:
            with self._stats_lock:
                self.requests_failed += 1
            raise
        with self._stats_lock:
            self.requests_completed += 1
        return response

    def _process_serialized(
        self,
        request_id: str,
        audio: np.ndarray,
        duration: float,
        received_at: float,
        queue_wait_seconds: float,
        request_diagnostics: dict[str, Any] | None,
    ) -> dict[str, Any]:
        processing_started = time.perf_counter()
        if request_diagnostics is not None:
            request_diagnostics["server_processing_started_perf_counter"] = processing_started
        silence = is_pre_separation_silence(finite_audio_stats(audio))
        separation_started = time.perf_counter()
        separation = None if silence else separate_amp(
            self.models.separator, self.models.separation_device, audio
        )
        separation_ended = time.perf_counter()
        output = silent_separation_output(len(audio)) if silence else separation.output
        separated = [
            np.ascontiguousarray(output[index, 0, :], dtype=np.float32)
            for index in (0, 1)
        ]

        vad_started = time.perf_counter()
        vad_results = []
        vad_stage_ranges: list[tuple[float, float]] = []
        for speaker_audio in separated:
            speaker_vad_started = time.perf_counter()
            vad_results.append(
                detect_speech_activity(
                    self.models.vad, speaker_audio, VAD_MINIMUM_SPEECH_MS
                )
            )
            vad_stage_ranges.append((speaker_vad_started, time.perf_counter()))
        vad_seconds = time.perf_counter() - vad_started

        stt_total = 0.0
        speakers: list[dict[str, Any]] = []
        stt_stage_ranges: list[tuple[float, float]] = []
        waveform_encode_ranges: list[tuple[float, float]] = []
        for speaker, (speaker_audio, vad) in enumerate(zip(separated, vad_results)):
            stt_started = time.perf_counter()
            if vad["speech_detected"]:
                stt_result = transcribe_base(self.models.whisper, speaker_audio)
                raw_text = stt_result.text
                stt_seconds = stt_result.elapsed_seconds
            else:
                raw_text = ""
                stt_seconds = 0.0
            stt_stage_ranges.append((stt_started, time.perf_counter()))
            stt_total += stt_seconds

            waveform_encode_started = time.perf_counter()
            encoded_waveform = base64.b64encode(
                speaker_audio.astype("<f4").tobytes()
            ).decode("ascii")
            waveform_encode_ranges.append(
                (waveform_encode_started, time.perf_counter())
            )
            speakers.append(
                {
                    "raw_slot": speaker,
                    "waveform": {
                        "encoding": "base64-f32le",
                        "data": encoded_waveform,
                    },
                    "sample_count": len(speaker_audio),
                    "rms": float(np.sqrt(np.mean(speaker_audio * speaker_audio))),
                    "peak": float(np.max(np.abs(speaker_audio))),
                    "vad": vad,
                    "stt_seconds": stt_seconds,
                    "raw_transcript": raw_text,
                }
            )

        processing_ended = time.perf_counter()
        processing_seconds = processing_ended - processing_started
        gpu_memory_query_started = time.perf_counter()
        gpu_device, gpu_process = query_gpu_memory()
        gpu_memory_query_ended = time.perf_counter()
        timing: dict[str, Any] = {
            "queue_wait_seconds": queue_wait_seconds,
            "separation_seconds": 0.0 if silence else separation.total_seconds,
            "amp_seconds": 0.0 if silence else separation.amp_seconds,
            "fp32_seconds": 0.0 if silence else separation.fp32_seconds,
            "vad_seconds": vad_seconds,
            "stt_seconds": stt_total,
            "processing_seconds": processing_seconds,
            "end_to_end_rtf": processing_seconds / duration,
        }
        if request_diagnostics is not None:
            request_diagnostics.update(
                {
                    "separation_started_perf_counter": separation_started,
                    "separation_ended_perf_counter": separation_ended,
                    "separation_wall_seconds": separation_ended - separation_started,
                    "vad_started_perf_counter": vad_started,
                    "vad_ended_perf_counter": vad_started + vad_seconds,
                    "vad_speaker_0_started_perf_counter": vad_stage_ranges[0][0],
                    "vad_speaker_0_ended_perf_counter": vad_stage_ranges[0][1],
                    "vad_speaker_0_wall_seconds": vad_stage_ranges[0][1]
                    - vad_stage_ranges[0][0],
                    "vad_speaker_1_started_perf_counter": vad_stage_ranges[1][0],
                    "vad_speaker_1_ended_perf_counter": vad_stage_ranges[1][1],
                    "vad_speaker_1_wall_seconds": vad_stage_ranges[1][1]
                    - vad_stage_ranges[1][0],
                    "stt_speaker_0_started_perf_counter": stt_stage_ranges[0][0],
                    "stt_speaker_0_ended_perf_counter": stt_stage_ranges[0][1],
                    "stt_speaker_0_wall_seconds": stt_stage_ranges[0][1]
                    - stt_stage_ranges[0][0],
                    "stt_speaker_1_started_perf_counter": stt_stage_ranges[1][0],
                    "stt_speaker_1_ended_perf_counter": stt_stage_ranges[1][1],
                    "stt_speaker_1_wall_seconds": stt_stage_ranges[1][1]
                    - stt_stage_ranges[1][0],
                    "waveform_encode_speaker_0_seconds": waveform_encode_ranges[0][1]
                    - waveform_encode_ranges[0][0],
                    "waveform_encode_speaker_1_seconds": waveform_encode_ranges[1][1]
                    - waveform_encode_ranges[1][0],
                    "server_processing_ended_perf_counter": processing_ended,
                    "gpu_memory_query_started_perf_counter": gpu_memory_query_started,
                    "gpu_memory_query_ended_perf_counter": gpu_memory_query_ended,
                    "gpu_memory_query_seconds": gpu_memory_query_ended
                    - gpu_memory_query_started,
                }
            )
            timing.update(request_diagnostics)
        return {
            "type": "processing_result",
            "schema_version": 1,
            "request_id": request_id,
            "received_at_ms": round(received_at * 1000),
            "audio": {
                "duration_seconds": duration,
                "sample_rate": SAMPLE_RATE,
                "channels": 1,
            },
            "models": {
                "mossformer2": {"device": str(self.models.separation_device), "amp": True},
                "faster_whisper": {
                    "model": "base",
                    "device": str(self.models.whisper.model.device),
                    "compute_type": str(self.models.whisper.model.compute_type),
                    "options": LIVE_WHISPER_OPTIONS,
                },
                "silero_vad": {"device": "cpu"},
                "load_count": self.model_load_count,
            },
            "timing": timing,
            "fp32_fallback": False if silence else separation.fallback_attempted,
            "pre_separation_silence": silence,
            "gpu_memory": {
                "device_used_mib": gpu_device,
                "process_used_mib": gpu_process,
            },
            "speakers": speakers,
        }


def read_wav(payload: bytes) -> tuple[np.ndarray, float]:
    try:
        audio, sample_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=False)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid WAV payload: {exc}") from exc
    if sample_rate != SAMPLE_RATE or audio.ndim != 1:
        raise ValueError("Expected mono 16 kHz WAV")
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("WAV must contain finite audio samples")
    duration = len(audio) / sample_rate
    if duration > MAX_AUDIO_SECONDS:
        raise ValueError(f"WAV duration exceeds {MAX_AUDIO_SECONDS:.0f} seconds")
    return np.ascontiguousarray(audio), duration


class ProcessingHandler(BaseHTTPRequestHandler):
    server: "ProcessingHTTPServer"

    def do_GET(self) -> None:
        if self.path != "/healthz":
            self._send_error(HTTPStatus.NOT_FOUND, "not_found", "Endpoint not found")
            return
        status, payload = self.server.service.health()
        self._send_json(status, payload)

    def do_POST(self) -> None:
        request_received = time.perf_counter()
        request_id = self.headers.get("X-Request-ID") or str(uuid.uuid4())
        latency_diagnostics = self.headers.get("X-Latency-Diagnostics") == "1"
        if self.path != "/v1/process":
            self._send_error(
                HTTPStatus.NOT_FOUND, "not_found", "Endpoint not found", request_id
            )
            return
        try:
            content_length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._send_error(
                HTTPStatus.BAD_REQUEST,
                "invalid_content_length",
                "Content-Length must be an integer",
                request_id,
            )
            return
        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            self._send_error(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "invalid_request_size",
                f"WAV body must be between 1 and {MAX_REQUEST_BYTES} bytes",
                request_id,
            )
            return
        try:
            window_index = int(self.headers.get("X-Window-Index", "0"))
            if window_index < 0:
                raise ValueError("Window index must be non-negative")
            stream_start = float(self.headers.get("X-Stream-Start-Seconds", "0"))
            stream_end = float(self.headers.get("X-Stream-End-Seconds", "0"))
            if not math.isfinite(stream_start) or not math.isfinite(stream_end) or stream_start < 0 or stream_end < stream_start:
                raise ValueError("Invalid stream timestamps")
            body_read_started = time.perf_counter()
            body = self.rfile.read(content_length)
            body_read_ended = time.perf_counter()
            decode_started = time.perf_counter()
            audio, duration = read_wav(body)
            decode_ended = time.perf_counter()
            request_diagnostics = None
            if latency_diagnostics:
                request_diagnostics = {
                    "server_request_received_perf_counter": request_received,
                    "request_body_read_started_perf_counter": body_read_started,
                    "request_body_read_ended_perf_counter": body_read_ended,
                    "request_body_read_seconds": body_read_ended - body_read_started,
                    "request_body_bytes": len(body),
                    "request_decode_started_perf_counter": decode_started,
                    "request_decode_ended_perf_counter": decode_ended,
                    "request_decode_seconds": decode_ended - decode_started,
                    "connection_client_port": self.client_address[1],
                }
            result = self.server.service.process(
                request_id, audio, duration, request_diagnostics
            )
            result["window_index"] = window_index
            result["capture_timestamp"] = self.headers.get("X-Capture-Timestamp", "")
            result["stream_start_seconds"] = stream_start
            result["stream_end_seconds"] = stream_end
        except ValueError as exc:
            self._send_error(
                HTTPStatus.BAD_REQUEST, "invalid_audio", str(exc), request_id
            )
            return
        except BaseException as exc:
            self._send_error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "processing_failed",
                f"{type(exc).__name__}: {exc}",
                request_id,
            )
            return
        send_timing = self._send_json(
            HTTPStatus.OK, result, latency_diagnostics=latency_diagnostics
        )
        if latency_diagnostics:
            print(
                f"[LATENCY SERVER] window={window_index:03d} "
                f"response_serialize={send_timing['response_serialization_seconds']:.6f}s "
                f"response_write={send_timing['response_body_write_seconds']:.6f}s "
                f"response_bytes={send_timing['response_body_bytes']}",
                flush=True,
            )

    def log_message(self, format: str, *args: Any) -> None:
        print(
            f"http_client={self.client_address[0]}; {format % args}",
            flush=True,
        )

    def _send_error(
        self,
        status: HTTPStatus,
        code: str,
        message: str,
        request_id: str | None = None,
    ) -> None:
        self._send_json(
            status,
            {
                "type": "error",
                "request_id": request_id,
                "error": {"code": code, "message": message, "retryable": False},
            },
        )

    def _send_json(
        self,
        status: HTTPStatus,
        payload: dict[str, Any],
        latency_diagnostics: bool = False,
    ) -> dict[str, float | int]:
        serialization_started = time.perf_counter()
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        serialization_ended = time.perf_counter()
        serialization_seconds = serialization_ended - serialization_started
        if latency_diagnostics and isinstance(payload.get("timing"), dict):
            payload["timing"]["response_serialization_started_perf_counter"] = (
                serialization_started
            )
            payload["timing"]["response_serialization_ended_perf_counter"] = (
                serialization_ended
            )
            payload["timing"]["response_serialization_seconds"] = serialization_seconds
            payload["timing"]["response_payload_bytes_before_diagnostics"] = len(body)
            body = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        write_started = time.perf_counter()
        self.wfile.write(body)
        write_ended = time.perf_counter()
        return {
            "response_serialization_seconds": serialization_seconds,
            "response_body_write_seconds": write_ended - write_started,
            "response_body_bytes": len(body),
        }


class ProcessingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], service: PipelineService) -> None:
        super().__init__(address, ProcessingHandler)
        self.service = service


def main() -> int:
    parser = argparse.ArgumentParser(description="Local Runpod GPU processing server prototype")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    print("Loading GPU pipeline models once at server startup...", flush=True)
    service = PipelineService()
    server = ProcessingHTTPServer((args.host, args.port), service)
    print(
        f"GPU processing server ready: http://{args.host}:{args.port}; "
        f"model_load_count={service.model_load_count}; load_seconds={service.load_times}; "
        f"memory={service.memory}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

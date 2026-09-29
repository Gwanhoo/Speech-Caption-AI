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
    WHISPER_MODEL_DIR,
    detect_speech_activity,
    load_separator,
    query_gpu_memory,
    separate_amp,
)


MAX_REQUEST_BYTES = 16 * 1024 * 1024
MAX_AUDIO_SECONDS = 30.0


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
                str(WHISPER_MODEL_DIR), device="cuda", compute_type="float16"
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

    def process(self, request_id: str, audio: np.ndarray, duration: float) -> dict[str, Any]:
        if not self.ready:
            raise RuntimeError("GPU pipeline is not ready")

        received_at = time.time()
        wait_started = time.perf_counter()
        with self._stats_lock:
            self.requests_waiting += 1
        try:
            with self._inference_lock:
                queue_wait_seconds = time.perf_counter() - wait_started
                with self._stats_lock:
                    self.requests_waiting -= 1
                response = self._process_serialized(
                    request_id, audio, duration, received_at, queue_wait_seconds
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
    ) -> dict[str, Any]:
        processing_started = time.perf_counter()
        silence = is_pre_separation_silence(finite_audio_stats(audio))
        separation = None if silence else separate_amp(
            self.models.separator, self.models.separation_device, audio
        )
        output = silent_separation_output(len(audio)) if silence else separation.output
        separated = [
            np.ascontiguousarray(output[index, 0, :], dtype=np.float32)
            for index in (0, 1)
        ]

        vad_started = time.perf_counter()
        vad_results = [
            detect_speech_activity(
                self.models.vad, speaker_audio, VAD_MINIMUM_SPEECH_MS
            )
            for speaker_audio in separated
        ]
        vad_seconds = time.perf_counter() - vad_started

        stt_total = 0.0
        speakers: list[dict[str, Any]] = []
        for speaker, (speaker_audio, vad) in enumerate(zip(separated, vad_results)):
            if vad["speech_detected"]:
                stt_result = transcribe_base(self.models.whisper, speaker_audio)
                raw_text = stt_result.text
                stt_seconds = stt_result.elapsed_seconds
            else:
                raw_text = ""
                stt_seconds = 0.0
            stt_total += stt_seconds

            speakers.append(
                {
                    "raw_slot": speaker,
                    "waveform": {
                        "encoding": "base64-f32le",
                        "data": base64.b64encode(speaker_audio.astype("<f4").tobytes()).decode("ascii"),
                    },
                    "sample_count": len(speaker_audio),
                    "rms": float(np.sqrt(np.mean(speaker_audio * speaker_audio))),
                    "peak": float(np.max(np.abs(speaker_audio))),
                    "vad": vad,
                    "stt_seconds": stt_seconds,
                    "raw_transcript": raw_text,
                }
            )

        processing_seconds = time.perf_counter() - processing_started
        gpu_device, gpu_process = query_gpu_memory()
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
            "timing": {
                "queue_wait_seconds": queue_wait_seconds,
                "separation_seconds": 0.0 if silence else separation.total_seconds,
                "amp_seconds": 0.0 if silence else separation.amp_seconds,
                "fp32_seconds": 0.0 if silence else separation.fp32_seconds,
                "vad_seconds": vad_seconds,
                "stt_seconds": stt_total,
                "processing_seconds": processing_seconds,
                "end_to_end_rtf": processing_seconds / duration,
            },
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
        request_id = self.headers.get("X-Request-ID") or str(uuid.uuid4())
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
            audio, duration = read_wav(self.rfile.read(content_length))
            result = self.server.service.process(request_id, audio, duration)
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
        self._send_json(HTTPStatus.OK, result)

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

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


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

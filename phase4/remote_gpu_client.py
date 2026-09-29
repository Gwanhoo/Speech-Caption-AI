"""HTTP inference transport. Waveforms and metadata are indexed by raw slot.

No CUDA or capture dependencies; logical identity and subtitle state stay local.
"""
from __future__ import annotations

import base64
import io
import math
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import requests
import soundfile as sf


class RemoteGPUError(RuntimeError):
    """A failed window may be logged and skipped without stopping capture."""


class RemoteConnectionError(RemoteGPUError):
    pass


class RemoteTimeoutError(RemoteGPUError):
    pass


class RemoteHTTPError(RemoteGPUError):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(f"HTTP {status}: {message}")


class RemoteProtocolError(RemoteGPUError):
    pass


@dataclass(frozen=True)
class RemoteWindowResult:
    request_id: str
    window_index: int
    raw_speakers: tuple[np.ndarray, np.ndarray]
    slots: tuple[dict[str, Any], dict[str, Any]]
    timing: dict[str, float]
    gpu_memory: dict[str, Any]
    fp32_fallback: bool
    pre_separation_silence: bool
    request_seconds: float
    client_timing: dict[str, Any]

    def logical_slots(self, assignment: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        logical = [None, None]
        for raw, slot in enumerate(self.slots):
            logical[assignment["raw_to_logical_mapping"][str(raw)]] = slot
        return tuple(logical)


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def monotonic_duration(started: float, ended: float) -> float:
    if not _number(started) or not _number(ended) or ended < started:
        raise ValueError("Invalid monotonic timing range")
    return ended - started


def queue_wait_seconds(enqueued: float, dequeued: float) -> float:
    return monotonic_duration(enqueued, dequeued)


def parse_result(payload: Any, request_id: str, window_index: int, samples: int,
                 request_seconds: float, client_timing: dict[str, Any] | None = None
                 ) -> RemoteWindowResult:
    try:
        if not isinstance(payload, dict) or payload["type"] != "processing_result" or payload["schema_version"] != 1:
            raise ValueError("unsupported response type/version")
        if payload["request_id"] != request_id or payload["window_index"] != window_index:
            raise ValueError("request/window identity mismatch")
        if payload["audio"]["sample_rate"] != 16000 or payload["audio"]["channels"] != 1:
            raise ValueError("audio format mismatch")
        for flag in ("fp32_fallback", "pre_separation_silence"):
            if type(payload[flag]) is not bool:
                raise ValueError(f"invalid {flag}")
        slots = payload["speakers"]
        if not isinstance(slots, list) or len(slots) != 2:
            raise ValueError("expected two raw slots")
        ordered = sorted(slots, key=lambda slot: slot["raw_slot"])
        waveforms = []
        for raw, slot in enumerate(ordered):
            if type(slot["raw_slot"]) is not int or slot["raw_slot"] != raw or slot["sample_count"] != samples:
                raise ValueError("invalid raw slot/sample count")
            encoded = slot["waveform"]
            if encoded["encoding"] != "base64-f32le" or not isinstance(encoded["data"], str):
                raise ValueError("unsupported waveform encoding")
            if len(encoded["data"]) > ((samples * 4 + 2) // 3) * 4:
                raise ValueError("oversized waveform")
            binary = base64.b64decode(encoded["data"], validate=True)
            if len(binary) != samples * 4:
                raise ValueError("waveform length mismatch")
            audio = np.frombuffer(binary, dtype="<f4").astype(np.float32, copy=True)
            if not np.isfinite(audio).all():
                raise ValueError("non-finite waveform")
            waveforms.append(audio)
            if not isinstance(slot["raw_transcript"], str) or not _number(slot["stt_seconds"]):
                raise ValueError("invalid transcript/timing")
            vad = slot["vad"]
            if type(vad["speech_detected"]) is not bool:
                raise ValueError("invalid VAD flag")
            for key in ("speech_duration_ms", "speech_ratio", "processing_seconds", "rms", "peak", "audio_duration_ms"):
                if not _number(vad[key]):
                    raise ValueError(f"invalid VAD {key}")
            if vad["speech_ratio"] > 1 or vad["speech_duration_ms"] > samples / 16:
                raise ValueError("invalid VAD range")
        timing = payload["timing"]
        for key in ("queue_wait_seconds", "separation_seconds", "vad_seconds", "stt_seconds", "processing_seconds", "amp_seconds", "fp32_seconds"):
            if not _number(timing[key]):
                raise ValueError(f"invalid timing {key}")
        memory = payload["gpu_memory"]
        for key in ("device_used_mib", "process_used_mib"):
            if memory[key] is not None and not _number(memory[key]):
                raise ValueError("invalid GPU memory")
        return RemoteWindowResult(request_id, window_index, tuple(waveforms), tuple(ordered),
                                  timing, memory, payload["fp32_fallback"],
                                  payload["pre_separation_silence"], request_seconds,
                                  {} if client_timing is None else client_timing)
    except (KeyError, ValueError, TypeError, OverflowError) as exc:
        raise RemoteProtocolError(f"Malformed GPU response: {exc}") from exc


class RemoteGPUClient:
    def __init__(self, server_url: str, connect_timeout: float = 5, read_timeout: float = 30):
        parsed = urlsplit(server_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("Expected an http(s) server URL")
        if connect_timeout <= 0 or read_timeout <= 0:
            raise ValueError("Timeouts must be positive")
        if not math.isfinite(connect_timeout) or not math.isfinite(read_timeout):
            raise ValueError("Timeouts must be finite")
        self.url = server_url.rstrip("/")
        self.timeout = (connect_timeout, read_timeout)
        self.session = requests.Session()
        self.session.trust_env = False  # direct server access; no implicit proxy

    def close(self) -> None:
        self.session.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        payload, _ = self._request_with_timing(method, path, **kwargs)
        return payload

    def _request_with_timing(
        self, method: str, path: str, **kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        request_started = time.perf_counter()
        try:
            response = self.session.request(method, self.url + path, timeout=self.timeout, **kwargs)
        except requests.Timeout as exc:
            raise RemoteTimeoutError(str(exc)) from exc
        except requests.RequestException as exc:
            raise RemoteConnectionError(str(exc)) from exc
        response_received = time.perf_counter()
        if response.status_code != 200:
            raise RemoteHTTPError(response.status_code, response.text[:1000])
        json_decode_started = time.perf_counter()
        try:
            payload = response.json()
        except ValueError as exc:
            raise RemoteProtocolError("Response is not JSON") from exc
        json_decode_ended = time.perf_counter()
        if not isinstance(payload, dict) or payload.get("type") == "error":
            raise RemoteProtocolError(f"Server error/invalid object: {payload}")
        try:
            response_body_bytes = len(response.content)
        except TypeError:
            response_body_bytes = None
        elapsed = getattr(response, "elapsed", None)
        response_elapsed_seconds = (
            elapsed.total_seconds()
            if elapsed is not None and callable(getattr(elapsed, "total_seconds", None))
            else None
        )
        raw = getattr(response, "raw", None)
        return payload, {
            "http_request_started_perf_counter": request_started,
            "http_response_received_perf_counter": response_received,
            "http_request_seconds": monotonic_duration(request_started, response_received),
            "requests_response_elapsed_seconds": response_elapsed_seconds,
            "response_json_decode_started_perf_counter": json_decode_started,
            "response_json_decode_ended_perf_counter": json_decode_ended,
            "response_json_decode_seconds": monotonic_duration(
                json_decode_started, json_decode_ended
            ),
            "response_body_bytes": response_body_bytes,
            "response_http_version": getattr(raw, "version", None),
            "response_connection_header": response.headers.get("Connection"),
        }

    def health(self) -> dict[str, Any]:
        result = self._request("GET", "/healthz")
        if result.get("status") != "ready" or result.get("gpu_ready") is not True or result.get("models_loaded") is not True:
            raise RemoteProtocolError("GPU server is not ready")
        return result

    def process(self, audio: np.ndarray, window_index: int, capture_timestamp: str = "",
                stream_start_seconds: float = 0, stream_end_seconds: float = 0,
                latency_diagnostics: bool = False) -> RemoteWindowResult:
        if type(window_index) is not int or window_index < 0:
            raise ValueError("Window index must be a non-negative integer")
        if not _number(stream_start_seconds) or not _number(stream_end_seconds) or stream_end_seconds < stream_start_seconds:
            raise ValueError("Invalid stream timestamps")
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not audio.size or audio.size > 30 * 16000 or not np.isfinite(audio).all():
            raise ValueError("Expected finite mono audio, up to 30 seconds at 16 kHz")
        process_started = time.perf_counter()
        request_id = f"window-{window_index}-{time.time_ns()}"
        encode_started = time.perf_counter()
        buffer = io.BytesIO()
        sf.write(buffer, audio, 16000, format="WAV", subtype="FLOAT")
        request_body = buffer.getvalue()
        encode_ended = time.perf_counter()
        headers = {"Content-Type": "audio/wav", "X-Request-ID": request_id,
                   "X-Window-Index": str(window_index), "X-Capture-Timestamp": capture_timestamp,
                   "X-Stream-Start-Seconds": str(stream_start_seconds),
                   "X-Stream-End-Seconds": str(stream_end_seconds)}
        if latency_diagnostics:
            headers["X-Latency-Diagnostics"] = "1"
        request_ready = time.perf_counter()
        payload, transport_timing = self._request_with_timing(
            "POST", "/v1/process", data=request_body, headers=headers
        )
        parse_started = time.perf_counter()
        client_timing = {
            "process_started_perf_counter": process_started,
            "payload_encode_started_perf_counter": encode_started,
            "payload_encode_ended_perf_counter": encode_ended,
            "payload_encode_seconds": monotonic_duration(encode_started, encode_ended),
            "remote_request_ready_perf_counter": request_ready,
            "request_body_bytes": len(request_body),
            **transport_timing,
            "response_parse_started_perf_counter": parse_started,
        }
        request_seconds = monotonic_duration(
            transport_timing["http_request_started_perf_counter"],
            transport_timing["response_json_decode_ended_perf_counter"],
        )
        result = parse_result(
            payload,
            request_id,
            window_index,
            len(audio),
            request_seconds,
            client_timing,
        )
        parse_ended = time.perf_counter()
        client_timing["response_parse_ended_perf_counter"] = parse_ended
        client_timing["response_parse_seconds"] = monotonic_duration(
            parse_started, parse_ended
        )
        client_timing["client_total_seconds"] = monotonic_duration(
            process_started, parse_ended
        )
        return result


class RemoteGPUMonitor:
    """Server measurements only; never probes the Windows client's CUDA device."""
    def __init__(self):
        self.samples = []
        self.stage_peaks = {}

    def start(self):
        pass

    def stop(self):
        pass

    def used_mib(self):
        return self.samples[-1] if self.samples else None

    def observe(self, memory):
        used = memory.get("device_used_mib")
        if used is not None:
            self.samples.append(used)

    def enter(self, stage):
        pass

    def leave(self, stage):
        pass

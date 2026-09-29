from __future__ import annotations

import base64
import io
import json
import sys
import threading
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import Mock, patch

import numpy as np
import requests
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase4"))
sys.path.insert(0, str(ROOT / "phase3"))
from remote_gpu_client import (RemoteGPUClient, RemoteConnectionError, RemoteTimeoutError,
                               RemoteHTTPError, RemoteProtocolError, monotonic_duration,
                               parse_result, queue_wait_seconds)
from remote_gpu_protocol import encode_binary_result, decode_binary_result
from speaker_tracking import PersistentSpeakerTracker
from subtitle_assembler import SubtitleAssembler, SpeakerSubtitleState


def payload(request_id="r", window=0, samples=48000):
    rng = np.random.default_rng(12)
    audio = rng.normal(0, .1, samples).astype(np.float32)
    return {
        "type": "processing_result", "schema_version": 1, "request_id": request_id,
        "window_index": window, "audio": {"sample_rate": 16000, "channels": 1},
        "fp32_fallback": False, "pre_separation_silence": False,
        "gpu_memory": {"device_used_mib": 1023, "process_used_mib": 1014},
        "timing": {key: .01 for key in ("queue_wait_seconds", "separation_seconds", "vad_seconds", "stt_seconds", "processing_seconds", "amp_seconds", "fp32_seconds")},
        "speakers": [
            {"raw_slot": slot, "sample_count": samples,
             "waveform": {"encoding": "base64-f32le", "data": base64.b64encode((audio if slot == 0 else -audio).astype("<f4").tobytes()).decode()},
             "raw_transcript": f"화자 {slot} 테스트 문장입니다", "stt_seconds": .01,
             "vad": {"speech_detected": True, "speech_duration_ms": samples / 16,
                     "speech_ratio": 1., "processing_seconds": .01, "rms": .1, "peak": .5,
                     "audio_duration_ms": samples / 16, "timestamps": []}}
            for slot in (0, 1)]}


class ClientTests(TestCase):
    def test_binary_response_roundtrip_preserves_float32_waveforms(self):
        p = payload(samples=16000)
        for slot in p["speakers"]:
            slot["waveform"] = {
                "encoding": "binary-f32le",
                "data": base64.b64decode(slot["waveform"]["data"]),
            }
        body = encode_binary_result(p)
        decoded = decode_binary_result(body)
        result = parse_result(decoded, "r", 0, 16000, 0.1)
        expected = np.frombuffer(
            p["speakers"][0]["waveform"]["data"], dtype="<f4"
        )
        np.testing.assert_array_equal(result.raw_speakers[0], expected)
        self.assertLess(len(body), 16000 * 4 * 2 * 4 // 3)

    def test_monotonic_timing_and_queue_wait_calculation(self):
        self.assertAlmostEqual(monotonic_duration(10.25, 12.5), 2.25)
        self.assertAlmostEqual(queue_wait_seconds(100.0, 100.125), 0.125)
        with self.assertRaises(ValueError):
            queue_wait_seconds(2.0, 1.0)

    def test_remote_client_records_transport_timing(self):
        response = Mock(status_code=200)
        response.json.return_value = payload("window-0-123", samples=16000)
        response.content = b"response-body"
        response.headers = {"Connection": "close"}
        response.elapsed = timedelta(seconds=0.25)
        response.raw.version = 10
        client = RemoteGPUClient("http://127.0.0.1:8787")
        client.session.request = Mock(return_value=response)
        try:
            with patch("remote_gpu_client.time.time_ns", return_value=123):
                result = client.process(
                    np.zeros(16000, dtype=np.float32),
                    0,
                    stream_end_seconds=1,
                    latency_diagnostics=True,
                )
            timing = result.client_timing
            self.assertGreater(timing["request_body_bytes"], 16000 * 4)
            self.assertEqual(timing["response_body_bytes"], len(response.content))
            self.assertGreaterEqual(timing["payload_encode_seconds"], 0)
            self.assertGreaterEqual(timing["http_request_seconds"], 0)
            self.assertGreaterEqual(timing["response_json_decode_seconds"], 0)
            self.assertGreaterEqual(timing["response_parse_seconds"], 0)
            self.assertEqual(timing["requests_response_elapsed_seconds"], 0.25)
            self.assertEqual(timing["response_http_version"], 10)
            headers = client.session.request.call_args.kwargs["headers"]
            self.assertEqual(headers["X-Latency-Diagnostics"], "1")
            self.assertIn("processing-result", headers["Accept"])
        finally:
            client.close()

    def test_diagnostic_header_is_absent_by_default(self):
        response = Mock(status_code=200)
        response.json.return_value = payload("window-0-123", samples=16000)
        response.content = b"response-body"
        response.headers = {}
        response.elapsed = timedelta(0)
        response.raw.version = 10
        client = RemoteGPUClient("http://127.0.0.1:8787")
        client.session.request = Mock(return_value=response)
        try:
            with patch("remote_gpu_client.time.time_ns", return_value=123):
                client.process(
                    np.zeros(16000, dtype=np.float32), 0, stream_end_seconds=1
                )
            headers = client.session.request.call_args.kwargs["headers"]
            self.assertNotIn("X-Latency-Diagnostics", headers)
        finally:
            client.close()

    def test_waveform_roundtrip_and_mapping(self):
        p = payload()
        p["speakers"].reverse()
        result = parse_result(p, "r", 0, 48000, .1)
        self.assertEqual(result.raw_speakers[0].shape, (48000,))
        self.assertTrue(np.isfinite(result.raw_speakers[0]).all())
        mapped = result.logical_slots({"raw_to_logical_mapping": {"0": 1, "1": 0}})
        self.assertEqual(mapped[0]["raw_transcript"], "화자 1 테스트 문장입니다")

    def test_waveform_preserves_peak_above_one(self):
        p = payload()
        audio = np.full(48000, 1.6, dtype="<f4")
        p["speakers"][0]["waveform"]["data"] = base64.b64encode(audio.tobytes()).decode()
        result = parse_result(p, "r", 0, 48000, .1)
        np.testing.assert_array_equal(result.raw_speakers[0], audio)

    def test_malformed_contracts(self):
        variants = []
        for key, value in (("schema_version", 2), ("request_id", "other"), ("window_index", 9), ("speakers", []), ("fp32_fallback", "false")):
            p = payload(); p[key] = value; variants.append(p)
        p = payload(); p["speakers"][0]["waveform"]["data"] = "!"; variants.append(p)
        p = payload(); del p["timing"]; variants.append(p)
        p = payload(); p["speakers"][0]["vad"]["speech_ratio"] = float("nan"); variants.append(p)
        p = payload(); p["speakers"][0]["raw_slot"] = 1; variants.append(p)
        p = payload(); p["speakers"][0]["waveform"]["data"] = base64.b64encode(np.full(48000, np.inf, dtype="<f4").tobytes()).decode(); variants.append(p)
        for p in variants:
            with self.subTest(payload=list(p)), self.assertRaises(RemoteProtocolError):
                parse_result(p, "r", 0, 48000, .1)

    def test_transport_errors(self):
        client = RemoteGPUClient("http://127.0.0.1:8787")
        try:
            for error, expected in ((requests.ConnectTimeout("timeout"), RemoteTimeoutError),
                                    (requests.ReadTimeout("timeout"), RemoteTimeoutError),
                                    (requests.ConnectionError("refused"), RemoteConnectionError)):
                client.session.request = Mock(side_effect=error)
                with self.assertRaises(expected):
                    client.health()
            response = Mock(status_code=500, text="GPU failure")
            client.session.request = Mock(return_value=response)
            with self.assertRaises(RemoteHTTPError): client.health()
            response.status_code = 200
            response.json.side_effect = ValueError("invalid json")
            with self.assertRaises(RemoteProtocolError): client.health()
        finally:
            client.close()

    def test_unavailable_real_socket(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
        port = server.server_port
        server.server_close()
        client = RemoteGPUClient(f"http://127.0.0.1:{port}", .2, .2)
        try:
            with self.assertRaises(RemoteConnectionError): client.health()
        finally:
            client.close()

    def test_client_recovers_after_server_closes_a_successful_response(self):
        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            request_count = 0

            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                type(self).request_count += 1
                response = payload(
                    self.headers["X-Request-ID"],
                    int(self.headers["X-Window-Index"]),
                    16000,
                )
                body = json.dumps(response).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header(
                    "X-Server-Connection-ID",
                    f"{self.client_address[0]}:{self.client_address[1]}",
                )
                close = type(self).request_count == 1
                self.send_header("X-Server-Close-Connection", str(close).lower())
                self.send_header("Connection", "close" if close else "keep-alive")
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()
                if close:
                    self.close_connection = True

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = RemoteGPUClient(f"http://127.0.0.1:{server.server_port}")
        try:
            first = client.process(
                np.zeros(16000, dtype=np.float32), 0, stream_end_seconds=1
            )
            second = client.process(
                np.zeros(16000, dtype=np.float32),
                1,
                stream_start_seconds=1,
                stream_end_seconds=2,
            )
            self.assertEqual(
                first.client_timing["connection_change_reason"],
                "server_or_proxy_requested_close",
            )
            self.assertFalse(second.client_timing["client_connection_reused"])
            self.assertEqual(
                second.client_timing["connection_change_reason"],
                "client_transport_connection_replaced",
            )
            self.assertEqual(Handler.request_count, 2)
        finally:
            client.close()
            server.shutdown()
            server.server_close()
            thread.join()

    def test_fixed_wav_http_and_subtitle_continuity(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_GET(self):
                self.send_response(200); self.end_headers()
                self.wfile.write(json.dumps({"status": "ready", "gpu_ready": True, "models_loaded": True}).encode())
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                audio, sr = sf.read(io.BytesIO(body), dtype="float32")
                assert sr == 16000 and audio.ndim == 1
                response = payload(self.headers["X-Request-ID"], int(self.headers["X-Window-Index"]), len(audio))
                self.send_response(200); self.end_headers()
                self.wfile.write(json.dumps(response).encode())
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        client = RemoteGPUClient(f"http://127.0.0.1:{server.server_port}")
        tracker = PersistentSpeakerTracker(overlap_samples=16000)
        assemblers = [SubtitleAssembler(i) for i in (0, 1)]
        states = [SpeakerSubtitleState(i) for i in (0, 1)]
        try:
            client.health()
            audio, _ = sf.read(ROOT / "phase2/input/chunk_000_mixed_16k.wav", dtype="float32")
            for window in (0, 1):
                mixture = audio[window * 32000:window * 32000 + 48000]
                result = client.process(mixture, window)
                assignment = tracker.assign(window=window, raw_speakers=result.raw_speakers, mixture=mixture)
                for speaker, slot in enumerate(result.logical_slots(assignment.diagnostic)):
                    assembled = assemblers[speaker].process(window, slot["raw_transcript"])
                    states[speaker].process(window, assembled.utterance_hypothesis, slot["vad"]["speech_detected"], 3 + 2 * window)
            for state in states:
                final = state.flush(1, 5)
                self.assertIsNotNone(final)
                self.assertEqual(final.utterance_id, 1)
                self.assertEqual(final.status, "final")
        finally:
            client.close(); server.shutdown(); server.server_close(); thread.join()


if __name__ == "__main__":
    main()

from __future__ import annotations

import base64
import io
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import Mock

import numpy as np
import requests
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase4"))
sys.path.insert(0, str(ROOT / "phase3"))
from remote_gpu_client import (RemoteGPUClient, RemoteConnectionError, RemoteTimeoutError,
                               RemoteHTTPError, RemoteProtocolError, parse_result)
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

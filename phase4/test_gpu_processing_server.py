from __future__ import annotations

import base64
import sys
import tempfile
import threading
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase4"))

import gpu_processing_server as server
from remote_gpu_client import RemoteGPUClient


class FasterWhisperModelSourceTests(TestCase):
    def test_uses_base_model_name_when_project_checkpoint_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(server, "FASTER_WHISPER_CHECKPOINT_ROOT", Path(directory)):
                self.assertEqual(server.faster_whisper_model_source(), "base")

    def test_uses_valid_local_snapshot_without_assuming_snapshot_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_root = Path(directory)
            local_model = (
                checkpoint_root
                / "models--Systran--faster-whisper-base"
                / "snapshots"
                / "new-container-cache-id"
            )
            local_model.mkdir(parents=True)
            (local_model / "model.bin").touch()
            with patch.object(server, "FASTER_WHISPER_CHECKPOINT_ROOT", checkpoint_root):
                self.assertEqual(server.faster_whisper_model_source(), str(local_model))


class ServerLatencyDiagnosticsTests(TestCase):
    def test_http_handler_returns_server_diagnostics_without_loading_models(self):
        class FakeService:
            def process(self, request_id, audio, duration, request_diagnostics=None):
                encoded = base64.b64encode(audio.astype("<f4").tobytes()).decode()
                timing = {
                    "queue_wait_seconds": 0.0,
                    "separation_seconds": 0.0,
                    "vad_seconds": 0.0,
                    "stt_seconds": 0.0,
                    "processing_seconds": 0.0,
                    "amp_seconds": 0.0,
                    "fp32_seconds": 0.0,
                    **(request_diagnostics or {}),
                }
                return {
                    "type": "processing_result",
                    "schema_version": 1,
                    "request_id": request_id,
                    "audio": {"sample_rate": 16000, "channels": 1},
                    "fp32_fallback": False,
                    "pre_separation_silence": False,
                    "gpu_memory": {
                        "device_used_mib": None,
                        "process_used_mib": None,
                    },
                    "timing": timing,
                    "speakers": [
                        {
                            "raw_slot": slot,
                            "sample_count": len(audio),
                            "waveform": {
                                "encoding": "base64-f32le",
                                "data": encoded,
                            },
                            "raw_transcript": "",
                            "stt_seconds": 0.0,
                            "vad": {
                                "speech_detected": False,
                                "speech_duration_ms": 0.0,
                                "speech_ratio": 0.0,
                                "processing_seconds": 0.0,
                                "rms": 0.0,
                                "peak": 0.0,
                                "audio_duration_ms": duration * 1000,
                                "timestamps": [],
                            },
                        }
                        for slot in (0, 1)
                    ],
                }

        http_server = server.ProcessingHTTPServer(("127.0.0.1", 0), FakeService())
        thread = threading.Thread(target=http_server.serve_forever, daemon=True)
        thread.start()
        client = RemoteGPUClient(f"http://127.0.0.1:{http_server.server_port}")
        try:
            result = client.process(
                np.zeros(1600, dtype=np.float32),
                0,
                stream_end_seconds=0.1,
                latency_diagnostics=True,
            )
            self.assertIn("server_request_received_perf_counter", result.timing)
            self.assertIn("request_body_read_seconds", result.timing)
            self.assertIn("request_decode_seconds", result.timing)
            self.assertIn("response_serialization_seconds", result.timing)
            self.assertGreater(result.client_timing["response_body_bytes"], 0)
            self.assertEqual(result.client_timing["response_http_version"], 10)
        finally:
            client.close()
            http_server.shutdown()
            http_server.server_close()
            thread.join()


if __name__ == "__main__":
    main()

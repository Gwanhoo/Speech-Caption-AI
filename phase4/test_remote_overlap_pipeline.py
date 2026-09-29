"""Real pipeline workers, synthetic capture only. Never opens WASAPI on Linux."""
import io
import json
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import Mock, MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase3"))
sys.path.insert(0, str(ROOT / "phase4"))
from remote_gpu_client import (RemoteGPUMonitor, RemoteTimeoutError, RemoteConnectionError,
                               RemoteHTTPError, RemoteProtocolError, parse_result)
from separation_recovery import SeparationRecoveryResult, finite_audio_stats
from test_remote_gpu_client import payload
capture_module = sys.modules.get("soundcard")
sys.modules["soundcard"] = MagicMock()
try:
    import run_overlap_pipeline as pipeline
finally:
    if capture_module is None:
        sys.modules.pop("soundcard", None)
    else:
        sys.modules["soundcard"] = capture_module


class PipelineTests(TestCase):
    def run_pipeline(self, remote, fail_first=False, failure=None, fail_all=False,
                     latency_diagnostics=False):
        capture = MagicMock()
        capture.__enter__.return_value = capture
        def record(numframes):
            time.sleep(.04)
            return np.random.default_rng(numframes).normal(0, .1, (numframes, 1)).astype(np.float32)
        capture.record.side_effect = record
        loopback = Mock()
        loopback.recorder.return_value = capture
        client = Mock()
        client.health.return_value = {"gpu_memory": {"device_used_mib": 1023}}
        def process(audio, index, *args, **kwargs):
            if fail_all or (fail_first and index == 0):
                raise failure or RemoteTimeoutError("injected timeout")
            return parse_result(payload("r", index, len(audio)), "r", index, len(audio), .02)
        client.process.side_effect = process
        def separation(audio, **kwargs):
            return SeparationRecoveryResult(np.stack([audio, -audio])[:, None, :], finite_audio_stats(audio), .01, 0., False, False, False)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            output = Path(directory) / "result.json"
            argv = ["run_overlap_pipeline", "--live", "--duration", "5", "--assemble", "--json-output", str(output)]
            if remote: argv += ["--processing-mode", "remote", "--websocket", "--ws-port", "0"]
            if latency_diagnostics: argv += ["--latency-diagnostics"]
            stack.enter_context(patch.object(pipeline.sys, "argv", argv))
            stack.enter_context(patch.object(pipeline.sys, "platform", "win32"))
            stdout_buffer = io.StringIO()
            stdout = Mock(wraps=stdout_buffer); stdout.reconfigure = Mock()
            stack.enter_context(patch.object(pipeline.sys, "stdout", stdout))
            stack.enter_context(patch.object(pipeline.sc, "default_speaker", return_value=Mock(name="speaker")))
            stack.enter_context(patch.object(pipeline.sc, "get_microphone", return_value=loopback))
            stack.enter_context(patch.object(pipeline, "RemoteGPUClient", return_value=client))
            load = stack.enter_context(patch.object(pipeline.base, "load_models", return_value=(Mock(), "cuda:0", Mock(), 1, 1, 1)))
            stack.enter_context(patch.object(pipeline.base, "warm_up"))
            stack.enter_context(patch.object(pipeline.base, "GpuMonitor", RemoteGPUMonitor))
            stt = stack.enter_context(patch.object(pipeline.base, "transcribe_audio", return_value=("한국어 테스트 자막입니다", .01)))
            stack.enter_context(patch.object(pipeline, "separate_with_fp32_fallback", side_effect=separation))
            cuda_calls = []
            for name in ("is_available", "reset_peak_memory_stats", "memory_allocated", "memory_reserved", "max_memory_allocated"):
                cuda_calls.append(stack.enter_context(patch.object(pipeline.torch.cuda, name, return_value=True if name == "is_available" else 0)))
            code = pipeline.main()
            result = json.loads(output.read_text())
            if latency_diagnostics:
                self.assertTrue(result["latency_diagnostics"]["enabled"])
                self.assertTrue(all("latency_diagnostics" in row for row in result["windows"]))
                self.assertIn("[LATENCY]", stdout_buffer.getvalue())
            else:
                self.assertFalse(result["latency_diagnostics"]["enabled"])
                self.assertTrue(all("latency_diagnostics" not in row for row in result["windows"]))
                self.assertNotIn("[LATENCY]", stdout_buffer.getvalue())
            if remote:
                load.assert_not_called(); stt.assert_not_called()
                for call in cuda_calls: call.assert_not_called()
            else:
                load.assert_called_once(); self.assertEqual(stt.call_count, 4)
        return code, result

    def test_remote_workers(self):
        code, result = self.run_pipeline(True)
        self.assertEqual(code, 0)
        self.assertEqual(result["stt_window_success"], 2)
        self.assertTrue(result["subtitle_events"])
        self.assertEqual(result["errors"], [])
        self.assertGreater(result["websocket"]["published"], 0)
        self.assertEqual(result["websocket"]["dropped_oldest"], 0)

    def test_remote_latency_diagnostics(self):
        code, result = self.run_pipeline(True, latency_diagnostics=True)
        self.assertEqual(code, 0)
        timing = result["windows"][0]["latency_diagnostics"]
        self.assertGreaterEqual(
            timing["client_pipeline"]["audio_queue_wait_seconds"], 0
        )
        self.assertGreaterEqual(
            timing["client_pipeline"]["separated_queue_wait_seconds"], 0
        )

    def test_failed_remote_window_continues(self):
        code, result = self.run_pipeline(True, True)
        self.assertEqual(code, 1)  # degraded run is reported, not silently marked successful
        self.assertEqual(result["stt_window_success"], 1)
        self.assertEqual(result["windows"][0]["window"], 1)
        self.assertEqual(result["errors"][0]["stage"], "remote-window-000")

    def test_local_default_path(self):
        code, result = self.run_pipeline(False)
        self.assertEqual(code, 0)
        self.assertEqual(result["processing_mode"], "local")
        self.assertEqual(result["stt_window_success"], 2)

    def test_http_protocol_and_connection_failure_continue(self):
        for failure in (RemoteConnectionError("refused"), RemoteHTTPError(500, "GPU error"), RemoteProtocolError("bad JSON")):
            with self.subTest(failure=type(failure).__name__):
                code, result = self.run_pipeline(True, True, failure)
                self.assertEqual(code, 1)
                self.assertEqual(result["windows"][0]["window"], 1)

    def test_all_remote_windows_fail_without_summary_crash(self):
        code, result = self.run_pipeline(True, fail_all=True)
        self.assertEqual(code, 1)
        self.assertEqual(len(result["errors"]), 2)
        self.assertEqual(result["windows"], [])


if __name__ == "__main__": main()

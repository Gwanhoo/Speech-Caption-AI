from __future__ import annotations

import io
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import Mock, patch

import numpy as np
import requests
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase4"))

import gpu_processing_server as server
from remote_gpu_client import RemoteGPUClient
from remote_gpu_protocol import BINARY_RESPONSE_MEDIA_TYPE, decode_binary_result


class FasterWhisperModelSourceTests(TestCase):
    def test_production_default_is_large_v3_turbo(self):
        self.assertEqual(server.DEFAULT_STT_MODEL, "large-v3-turbo")
        args = server.build_argument_parser().parse_args([])
        self.assertEqual(args.whisper_model, "large-v3-turbo")
        self.assertIs(args.processing_profile, server.SpeakerMode.SINGLE)

    def test_whisper_model_cli_override_is_preserved(self):
        args = server.build_argument_parser().parse_args(["--whisper-model", "small"])
        self.assertEqual(args.whisper_model, "small")

    def test_overlap_profile_cli_override_is_preserved(self):
        args = server.build_argument_parser().parse_args(
            ["--processing-profile", "overlap"]
        )
        self.assertIs(args.processing_profile, server.SpeakerMode.OVERLAP)

    def test_single_profile_does_not_load_mossformer2(self):
        whisper = Mock()
        with patch.object(server, "query_gpu_memory", return_value=(700, 600)), \
                patch.object(server, "load_separator") as load_separator, \
                patch.object(server, "WhisperModel", return_value=whisper), \
                patch.object(server, "load_silero_vad", return_value=Mock()):
            service = server.PipelineService(warm_up=False)
            try:
                status, health = service.health()
            finally:
                service.close()
        load_separator.assert_not_called()
        self.assertIsNone(service.models.separator)
        self.assertIsNone(service.models.separation_device)
        self.assertEqual(status, server.HTTPStatus.OK)
        self.assertEqual(health["processing_profile"], "single")
        self.assertFalse(health["model_status"]["mossformer2_loaded"])
        self.assertEqual(health["model_load_seconds"]["mossformer2_seconds"], 0.0)

    def test_overlap_profile_still_loads_mossformer2(self):
        separator, device, whisper, vad = Mock(), Mock(), Mock(), Mock()
        with patch.object(server, "query_gpu_memory", return_value=(700, 600)), \
                patch.object(
                    server, "load_separator", return_value=(separator, device)
                ) as load_separator, \
                patch.object(server, "WhisperModel", return_value=whisper), \
                patch.object(server, "load_silero_vad", return_value=vad):
            service = server.PipelineService(
                processing_profile=server.SpeakerMode.OVERLAP,
                warm_up=False,
            )
            try:
                _, health = service.health()
            finally:
                service.close()
        load_separator.assert_called_once_with()
        self.assertIs(service.models.separator, separator)
        self.assertIs(service.models.separation_device, device)
        self.assertEqual(health["processing_profile"], "overlap")
        self.assertTrue(health["model_status"]["mossformer2_loaded"])

    def test_single_warmup_skips_separator(self):
        models = server.LoadedModels(None, None, Mock(), Mock())
        vad = {
            "speech_detected": False,
            "speech_duration_ms": 0.0,
            "speech_ratio": 0.0,
            "rms": 0.0,
            "peak": 0.0,
            "processing_seconds": 0.0,
            "timestamps": [],
            "audio_duration_ms": 3000.0,
        }
        with patch.object(server.PipelineService, "_load_models", return_value=models), \
                patch.object(server, "separate_amp") as separate, \
                patch.object(server, "detect_speech_activity", return_value=vad) as detect, \
                patch.object(server, "transcribe_base", return_value=Mock()):
            service = server.PipelineService()
            service.close()
        separate.assert_not_called()
        detect.assert_called_once()

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


    def test_selected_model_does_not_reuse_legacy_base_snapshot(self):
        self.assertEqual(server.faster_whisper_model_source("small"), "small")
        self.assertEqual(server.faster_whisper_model_source("large-v3-turbo"), "large-v3-turbo")

    def test_invalid_model_is_rejected_before_loading_gpu(self):
        with patch.object(server.PipelineService, "_load_models") as load:
            with self.assertRaises(ValueError):
                server.PipelineService("invented-model")
            load.assert_not_called()

    def test_selected_model_is_reported_by_health(self):
        with patch.object(server.PipelineService, "_load_models", return_value=Mock()), \
                patch.object(server, "query_gpu_memory", return_value=(1024, 512)):
            service = server.PipelineService("medium", warm_up=False)
            try:
                self.assertEqual(service.health()[1]["stt_model"], "medium")
            finally:
                service.close()


    def test_warmup_finishes_on_inference_worker_before_ready(self):
        thread_ids = []
        with patch.object(server.PipelineService, "_load_models", return_value=Mock()), \
                patch.object(server.PipelineService, "_warm_up_models", side_effect=lambda: thread_ids.append(threading.get_ident())):
            service = server.PipelineService("small")
            try:
                self.assertTrue(service.ready)
                self.assertNotEqual(thread_ids, [threading.get_ident()])
                self.assertEqual(service._inference_executor.submit(threading.get_ident).result(), thread_ids[0])
                self.assertIn("warm_up_seconds", service.load_times)
            finally:
                service.close()


class ServerLatencyDiagnosticsTests(TestCase):
    def test_input_residual_evidence_is_computed_and_survives_binary_raw_slot_mapping(self):
        from test_input_residual_speech_admission import load_fixture
        from remote_gpu_client import parse_result
        from separation_recovery import SeparationRecoveryResult, finite_audio_stats
        import copy
        (mixture, owner, candidate), metadata = load_fixture()
        raw_vads = copy.deepcopy(metadata["vad_results"])
        for vad in raw_vads:
            vad.pop("input_residual_speech", None)
        seen = []
        def detect(model, audio, minimum_ms):
            self.assertEqual(minimum_ms, 200)
            seen.append(audio)
            if len(seen) <= 2:
                return raw_vads[len(seen)-1]
            self.assertAlmostEqual(float(np.sqrt(np.mean(audio**2))), .1, places=6)
            return dict(raw_vads[1], speech_detected=False, speech_duration_ms=0.,
                        speech_ratio=0., timestamps=[])
        separation = SeparationRecoveryResult(
            np.stack((owner, candidate))[:, None, :], finite_audio_stats(mixture),
            .01, 0., False, False, False)
        models = Mock()
        models.whisper.model.device = "cuda"
        models.whisper.model.compute_type = "float16"
        with patch.object(server.PipelineService, "_load_models", return_value=models), \
                patch.object(server, "separate_amp", return_value=separation), \
                patch.object(server, "detect_speech_activity", side_effect=detect), \
                patch.object(server, "transcribe_base", side_effect=[
                    Mock(text=text, elapsed_seconds=.01) for text in metadata["raw_transcripts"]]):
            service = server.PipelineService(
                processing_profile=server.SpeakerMode.OVERLAP,
                warm_up=False,
            )
            try:
                result = service.process("fixture", mixture, 3.)
            finally:
                service.close()
        self.assertEqual(len(seen), 3)  # Two output VADs, one input residual VAD.
        result["window_index"] = 12
        encoded = server.encode_binary_result(result)
        parsed = parse_result(decode_binary_result(encoded), "fixture", 12, 48000, .02)
        slots = parsed.logical_slots({"raw_to_logical_mapping": {"0": 1, "1": 0}})
        evidence = slots[0]["vad"]["input_residual_speech"]
        self.assertTrue(evidence["available"])
        self.assertFalse(evidence["speech_detected"])
        self.assertEqual(evidence["timestamps"], [])
        self.assertEqual(slots[0]["raw_transcript"], metadata["raw_transcripts"][1])

    def test_process_uses_cached_gpu_memory_without_nvidia_smi(self):
        service = object.__new__(server.PipelineService)
        service.started_at = 0.0
        service.ready = True
        service.processing_profile = server.SpeakerMode.SINGLE
        service.load_error = None
        service.model_load_count = 1
        service.requests_completed = 0
        service.requests_failed = 0
        service.requests_waiting = 0
        service.requests_rejected = 0
        service._request_slots = threading.BoundedSemaphore(server.MAX_INFLIGHT_REQUESTS)
        service._stats_lock = threading.Lock()
        service._inference_lock = threading.Lock()
        service._inference_executor = server.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="test-gpu-inference"
        )
        service.load_times = {}
        service.memory = {}
        service._cached_memory = {
            "device_used_mib": 700,
            "process_used_mib": 650,
        }
        whisper = Mock()
        whisper.model.device = "cuda"
        whisper.model.compute_type = "float16"
        service.models = server.LoadedModels(Mock(), Mock(), whisper, Mock())
        vad = {
            "speech_detected": False,
            "speech_duration_ms": 0.0,
            "speech_ratio": 0.0,
            "rms": 0.0,
            "peak": 0.0,
            "processing_seconds": 0.0,
            "timestamps": [],
            "audio_duration_ms": 100.0,
        }
        try:
            with patch.object(server, "detect_speech_activity", return_value=vad), patch.object(
                server, "query_gpu_memory"
            ) as query, patch.object(server, "transcribe_base") as transcribe:
                result = service.process(
                    "request-off", np.zeros(1600, dtype=np.float32), 0.1
                )
                diagnostic_result = service.process(
                    "request", np.zeros(1600, dtype=np.float32), 0.1, {}
                )
                second_diagnostic_result = service.process(
                    "request-2", np.zeros(1600, dtype=np.float32), 0.1, {}
                )
        finally:
            service.close()
        query.assert_not_called()
        transcribe.assert_not_called()
        self.assertTrue(result["pre_separation_silence"])
        self.assertTrue(
            all(slot["raw_transcript"] == "" for slot in result["speakers"])
        )
        self.assertEqual(result["gpu_memory"]["device_used_mib"], 700)
        self.assertFalse(
            diagnostic_result["timing"]["gpu_memory_query_performed"]
        )
        self.assertEqual(
            diagnostic_result["timing"]["server_inference_thread_id"],
            second_diagnostic_result["timing"]["server_inference_thread_id"],
        )
        self.assertNotEqual(
            diagnostic_result["timing"]["server_inference_thread_id"],
            threading.get_ident(),
        )

    def test_single_process_uses_original_audio_and_preserves_two_slot_contract(self):
        audio = np.linspace(-0.2, 0.2, 16000, dtype=np.float32)
        whisper = Mock()
        whisper.model.device = "cuda"
        whisper.model.compute_type = "float16"
        models = server.LoadedModels(None, None, whisper, Mock())
        primary_vad = {
            "speech_detected": True,
            "speech_duration_ms": 1000.0,
            "speech_ratio": 1.0,
            "rms": float(np.sqrt(np.mean(audio * audio))),
            "peak": float(np.max(np.abs(audio))),
            "processing_seconds": 0.001,
            "timestamps": [{"start": 0, "end": len(audio)}],
            "audio_duration_ms": 1000.0,
        }
        with patch.object(server.PipelineService, "_load_models", return_value=models), \
                patch.object(server, "separate_amp") as separate, \
                patch.object(
                    server, "detect_speech_activity", return_value=primary_vad
                ) as detect, \
                patch.object(
                    server,
                    "transcribe_base",
                    return_value=Mock(text="단일 음성 자막", elapsed_seconds=0.02),
                ) as transcribe:
            service = server.PipelineService(warm_up=False)
            try:
                result = service.process("single", audio, 1.0)
            finally:
                service.close()
        separate.assert_not_called()
        detect.assert_called_once()
        transcribe.assert_called_once()
        np.testing.assert_array_equal(transcribe.call_args.args[1], audio)
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["processing_profile"], "single")
        self.assertEqual(len(result["speakers"]), 2)
        self.assertEqual(result["speakers"][0]["raw_transcript"], "단일 음성 자막")
        self.assertEqual(result["speakers"][1]["raw_transcript"], "")
        self.assertFalse(result["speakers"][1]["vad"]["speech_detected"])
        self.assertFalse(result["models"]["mossformer2"]["loaded"])
        primary = np.frombuffer(
            result["speakers"][0]["waveform"]["data"], dtype="<f4"
        )
        secondary = np.frombuffer(
            result["speakers"][1]["waveform"]["data"], dtype="<f4"
        )
        np.testing.assert_array_equal(primary, audio)
        np.testing.assert_array_equal(secondary, np.zeros_like(audio))

        result["window_index"] = 0
        parsed = decode_binary_result(server.encode_binary_result(result))
        from remote_gpu_client import parse_result

        remote = parse_result(parsed, "single", 0, len(audio), 0.1)
        np.testing.assert_array_equal(remote.raw_speakers[0], audio)
        np.testing.assert_array_equal(remote.raw_speakers[1], np.zeros_like(audio))

    def test_http_handler_returns_server_diagnostics_without_loading_models(self):
        class FakeService:
            def process(self, request_id, audio, duration, request_diagnostics=None):
                encoded = audio.astype("<f4").tobytes()
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
                                "encoding": "binary-f32le",
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
            self.assertEqual(result.client_timing["response_http_version"], 11)
            self.assertEqual(result.client_timing["response_format"], "binary-f32le")
            self.assertLess(result.client_timing["response_wire_bytes"], 18000)
            self.assertTrue(result.client_timing["response_body_consumed"])
            self.assertEqual(
                result.client_timing["response_connection_header"], "keep-alive"
            )
            self.assertEqual(
                result.client_timing["server_close_connection_header"], "false"
            )
            self.assertIn(
                "estimated_connect_tls_proxy_seconds", result.client_timing
            )
            second = client.process(
                np.zeros(1600, dtype=np.float32),
                1,
                stream_start_seconds=0.1,
                stream_end_seconds=0.2,
                latency_diagnostics=True,
            )
            self.assertTrue(second.client_timing["client_connection_reused"])
            self.assertTrue(second.client_timing["server_connection_reused"])
            self.assertEqual(
                second.client_timing["connection_change_reason"],
                "connection_reused_end_to_end",
            )
            self.assertEqual(
                second.client_timing["server_connection_request_count"], "2"
            )
            wav = io.BytesIO()
            sf.write(
                wav,
                np.zeros(1600, dtype=np.float32),
                16000,
                format="WAV",
                subtype="FLOAT",
            )
            with requests.Session() as fallback_session:
                common_headers = {
                    "Content-Type": "audio/wav",
                    "X-Window-Index": "2",
                    "X-Stream-Start-Seconds": "0.2",
                    "X-Stream-End-Seconds": "0.3",
                }
                binary = fallback_session.post(
                    f"http://127.0.0.1:{http_server.server_port}/v1/process",
                    data=wav.getvalue(),
                    headers={
                        **common_headers,
                        "X-Request-ID": "binary-before-fallback",
                        "Accept": BINARY_RESPONSE_MEDIA_TYPE,
                    },
                    timeout=2,
                )
                binary.raise_for_status()
                decoded = decode_binary_result(binary.content)
                self.assertEqual(decoded["request_id"], "binary-before-fallback")
                fallback = fallback_session.post(
                    f"http://127.0.0.1:{http_server.server_port}/v1/process",
                    data=wav.getvalue(),
                    headers={
                        **common_headers,
                        "X-Request-ID": "json-fallback",
                    },
                    timeout=2,
                )
                fallback.raise_for_status()
                self.assertEqual(
                    fallback.json()["speakers"][0]["waveform"]["encoding"],
                    "base64-f32le",
                )
                self.assertEqual(
                    binary.headers["X-Server-Connection-ID"],
                    fallback.headers["X-Server-Connection-ID"],
                )
                self.assertEqual(
                    fallback.headers["X-Server-Connection-Request-Count"], "2"
                )
        finally:
            client.close()
            http_server.shutdown()
            http_server.server_close()
            thread.join()


class ServerBackpressureTests(TestCase):
    def test_delayed_gpu_is_serialized_bounded_and_queue_wait_is_measured(self):
        entered = threading.Event()
        release = threading.Event()
        ids = []
        def infer(request_id, audio, duration, received, queue_wait, diagnostics):
            ids.append(threading.get_ident())
            entered.set()
            if not release.wait(3):
                raise RuntimeError("test failed to release simulated GPU")
            return {"queue_wait_seconds": queue_wait}
        with patch.object(server.PipelineService, "_load_models", return_value=Mock()), patch.object(
                server.PipelineService, "_process_serialized", side_effect=infer):
            service = server.PipelineService(warm_up=False)
            callers = ThreadPoolExecutor(max_workers=4)
            try:
                futures = [callers.submit(service.process, str(i), np.zeros(16), .001, {})
                           for i in range(4)]
                self.assertTrue(entered.wait(1))
                deadline = time.perf_counter() + 2
                while service.requests_waiting < 3 and time.perf_counter() < deadline:
                    time.sleep(.005)
                self.assertEqual(service.requests_waiting, 3)
                with self.assertRaises(server.PipelineBusyError):
                    service.process("excess", np.zeros(16), .001)
                self.assertEqual(service.requests_rejected, 1)
                time.sleep(.08)
                release.set()
                results = [future.result(timeout=2) for future in futures]
                self.assertTrue(all(wait >= .08 for wait in sorted(row["queue_wait_seconds"] for row in results)[1:]))
                self.assertEqual(len(set(ids)), 1)
                self.assertEqual(service.requests_waiting, 0)
                service.process("recovered", np.zeros(16), .001)
                self.assertEqual(service.requests_completed, 5)
                with patch.object(service._inference_executor, "submit", side_effect=RuntimeError("closed")):
                    with self.assertRaises(RuntimeError):
                        service.process("submit-failed", np.zeros(16), .001)
                self.assertEqual(service.requests_waiting, 0)
                with patch.object(service, "_process_serialized", side_effect=RuntimeError("GPU failed")):
                    with self.assertRaises(RuntimeError):
                        service.process("inference-failed", np.zeros(16), .001)
                service.process("after-failures", np.zeros(16), .001)
                self.assertEqual(service.requests_failed, 2)
            finally:
                release.set()
                callers.shutdown(wait=True)
                service.close()

    def test_busy_http_response_preserves_error_contract(self):
        service = Mock()
        service.process.side_effect = server.PipelineBusyError("full")
        http = server.ProcessingHTTPServer(("127.0.0.1", 0), service)
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        try:
            wav = io.BytesIO()
            sf.write(wav, np.zeros(160), 16000, format="WAV", subtype="FLOAT")
            response = requests.post(f"http://127.0.0.1:{http.server_port}/v1/process",
                                     data=wav.getvalue(), timeout=2)
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["type"], "error")
            self.assertEqual(response.json()["error"]["code"], "server_busy")
            self.assertFalse(response.json()["error"]["retryable"])
        finally:
            http.shutdown()
            http.server_close()
            thread.join()


if __name__ == "__main__":
    main()

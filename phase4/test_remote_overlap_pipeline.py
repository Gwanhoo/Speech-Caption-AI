"""Real pipeline workers, synthetic capture only. Never opens WASAPI on Linux."""
import io
import base64
import json
import queue
import sys
import tempfile
import threading
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
                     latency_diagnostics=False, runtime_hooks=None,
                     response_factory=None, duration=5, websocket=True, capture_factory=None):
        capture = MagicMock()
        capture.__enter__.return_value = capture
        def record(numframes):
            time.sleep(.04)
            audio = np.random.default_rng(numframes).normal(0, .1, (numframes, 1)).astype(np.float32)
            return capture_factory(audio) if capture_factory else audio
        capture.record.side_effect = record
        loopback = Mock()
        loopback.recorder.return_value = capture
        client = Mock()
        client.health.return_value = {"gpu_memory": {"device_used_mib": 1023}}
        def process(audio, index, *args, **kwargs):
            if fail_all or (fail_first and index == 0):
                raise failure or RemoteTimeoutError("injected timeout")
            response = (response_factory(audio, index) if response_factory is not None
                        else payload("r", index, len(audio)))
            return parse_result(response, "r", index, len(audio), .02)
        client.process.side_effect = process
        def separation(audio, **kwargs):
            return SeparationRecoveryResult(np.stack([audio, -audio])[:, None, :], finite_audio_stats(audio), .01, 0., False, False, False)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            output = Path(directory) / "result.json"
            argv = ["run_overlap_pipeline", "--live", "--duration", str(duration), "--assemble", "--json-output", str(output)]
            if remote:
                argv += ["--processing-mode", "remote"]
                if websocket:
                    argv += ["--websocket", "--ws-port", "0"]
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
            code = pipeline.main(runtime_hooks=runtime_hooks)
            if remote:
                client.close.assert_called_once()
            result = json.loads(output.read_text())
            if latency_diagnostics:
                self.assertTrue(result["latency_diagnostics"]["enabled"])
                self.assertTrue(all("latency_diagnostics" in row for row in result["windows"]))
                self.assertIn("[LATENCY]", stdout_buffer.getvalue())
                self.assertIn("[REMOTE REQUEST] window=000 status=start", stdout_buffer.getvalue())
                self.assertIn("[REMOTE REQUEST] window=000 status=complete", stdout_buffer.getvalue())
            else:
                self.assertFalse(result["latency_diagnostics"]["enabled"])
                self.assertTrue(all("latency_diagnostics" not in row for row in result["windows"]))
                self.assertNotIn("[LATENCY]", stdout_buffer.getvalue())
                self.assertNotIn("[REMOTE REQUEST]", stdout_buffer.getvalue())
            if remote:
                load.assert_not_called(); stt.assert_not_called()
                for call in cuda_calls: call.assert_not_called()
                self.assertEqual(result["audio_queue_maxsize"], 3)
                self.assertEqual(result["separated_queue_maxsize"], 2)
            else:
                load.assert_called_once(); self.assertEqual(stt.call_count, 4)
                self.assertEqual(result["audio_queue_maxsize"], 2)
                self.assertEqual(result["separated_queue_maxsize"], 2)
        return code, result

    def test_remote_workers(self):
        code, result = self.run_pipeline(True)
        self.assertEqual(code, 0)
        self.assertEqual(result["stt_window_success"], 2)
        self.assertTrue(result["subtitle_events"])
        self.assertEqual(result["errors"], [])
        self.assertGreater(result["websocket"]["published"], 0)
        self.assertEqual(result["websocket"]["dropped_oldest"], 0)

    def test_fragment_routing_permutation_and_revision_in_real_workers(self):
        """Exercise actual worker wiring/remote parsing with synthetic VAD/STT."""
        def response(audio, index):
            result = payload("r", index, len(audio))
            unrelated = np.random.default_rng(90 + index).normal(0, .1, len(audio)).astype(np.float32)
            if index == 0:
                waves, texts, spans = (audio, unrelated), ["지금 오신 분들 위해서 랜덤 챔피언", ""], [[(0, 48000)], []]
            elif index == 1:
                residual = audio.copy()
                residual[:32000] *= .2
                residual *= np.sqrt(np.mean(audio ** 2) / np.mean(residual ** 2))
                waves, texts, spans = (audio, residual), ["랜덤 챔피언 혀당인데요", "룰렛 돌려서"], [[(0, 32000)], [(32000, 48000)]]
            elif index == 2:
                waves, texts, spans = (unrelated, audio), ["", "제가 돌려서 AB 챔피언 3개 AP 챔피언 3개"], [[], [(0, 48000)]]
            else:
                waves, texts, spans = (audio * 0, audio * 0), ["", ""], [[], []]
            for i, slot in enumerate(result["speakers"]):
                slot["waveform"]["data"] = base64.b64encode(waves[i].astype("<f4").tobytes()).decode()
                slot["raw_transcript"] = texts[i]
                duration = sum(end - start for start, end in spans[i])
                slot["vad"].update({
                    "speech_detected": bool(spans[i]), "speech_duration_ms": duration / 16,
                    "speech_ratio": duration / 48000,
                    "timestamps": [{"start": start, "end": end} for start, end in spans[i]],
                })
            return result
        subtitles, metrics = [], []
        hooks = pipeline.LivePipelineHooks(on_subtitle=subtitles.append, on_metric=metrics.append)
        code, result = self.run_pipeline(
            True, response_factory=response, duration=9, websocket=False,
            latency_diagnostics=True, runtime_hooks=hooks,
        )
        self.assertEqual(code, 0)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["stt_window_success"], 4)
        self.assertEqual(len(metrics), 4)
        self.assertEqual(subtitles, result["subtitle_events"])
        self.assertEqual({e["speaker"] for e in subtitles}, {"speaker_0"})
        self.assertEqual(subtitles[-1]["status"], "final")
        self.assertEqual(subtitles[-1]["text"], "지금 오신 분들 위해서 랜덤 챔피언 혀당인데요 제가 돌려서 AB 챔피언 3개 AP 챔피언 3개")
        rows = result["windows"]
        self.assertTrue(rows[1]["secondary_leakage_diagnostic"]["subtitle_routing"]["applied"])
        self.assertEqual(rows[1]["secondary_leakage_diagnostic"]["raw_transcripts"]["speaker_1"], "룰렛 돌려서")
        self.assertTrue(rows[1]["vad"][1]["speech_detected"])
        self.assertEqual(rows[2]["speaker_assignment"]["raw_to_logical_mapping"], {"0": 1, "1": 0})

    def test_new_stream_admission_in_real_workers_with_permutation(self):
        for kind in ("residual", "short_quiet_speaker", "simultaneous_speakers"):
            with self.subTest(kind=kind):
                def response(audio, index):
                    result = payload("r", index, len(audio))
                    secondary = np.random.default_rng(90 + index).normal(0, .1, len(audio)).astype(np.float32)
                    primary = audio.copy()
                    texts = ["실제 본문 계속", ""]
                    spans = [[(0, len(audio))], []]
                    if index in (1, 2):
                        spans[1] = [(16000, 27200)]
                        texts[1] = "렁쇼" if index == 1 else "챔피언 롤러쯩"
                        if kind != "residual":
                            # Construct two input-supported components, including
                            # one very short, quiet response with no persistence.
                            mask = np.zeros(len(audio), dtype=np.float32)
                            if kind == "short_quiet_speaker":
                                spans[1] = [(16000, 17600)] if index == 1 else []
                                mask[16000:17600:2] = .001
                                texts[1] = "네" if index == 1 else ""
                            else:
                                spans[1] = [(0, len(audio))]
                                mask[::2] = 1.
                            secondary = audio * mask
                            primary = audio - secondary
                    elif index == 3:
                        primary = secondary = audio * 0
                        texts, spans = ["", ""], [[], []]
                    waves = [primary, secondary]
                    if index == 2:
                        waves.reverse()
                        texts.reverse()
                        spans.reverse()
                    for i, slot in enumerate(result["speakers"]):
                        slot["waveform"]["data"] = base64.b64encode(waves[i].astype("<f4").tobytes()).decode()
                        slot["raw_transcript"] = texts[i]
                        duration = sum(end - start for start, end in spans[i])
                        slot["vad"].update({
                            "speech_detected": bool(spans[i]), "speech_duration_ms": duration / 16,
                            "speech_ratio": duration / len(audio),
                            "timestamps": [{"start": start, "end": end} for start, end in spans[i]],
                        })
                    return result
                subtitles = []
                code, result = self.run_pipeline(
                    True, response_factory=response, duration=9, websocket=False,
                    runtime_hooks=pipeline.LivePipelineHooks(on_subtitle=subtitles.append),
                )
                self.assertEqual(code, 0)
                self.assertEqual(result["errors"], [])
                self.assertEqual(result["stt_window_success"], 4)
                self.assertEqual(subtitles, result["subtitle_events"])
                rows = result["windows"]
                self.assertEqual(rows[2]["speaker_assignment"]["raw_to_logical_mapping"], {"0": 1, "1": 0})
                for row in rows[1:3]:
                    admission = row["secondary_leakage_diagnostic"]["subtitle_admission"]
                    self.assertEqual(admission["applied"], kind == "residual")
                if kind == "residual":
                    self.assertEqual({e["speaker"] for e in subtitles}, {"speaker_0"})
                    self.assertEqual(rows[1]["secondary_leakage_diagnostic"]["raw_transcripts"]["speaker_1"], "렁쇼")
                    self.assertTrue(rows[1]["vad"][1]["speech_detected"])
                else:
                    self.assertEqual({e["speaker"] for e in subtitles}, {"speaker_0", "speaker_1"})
                    if kind == "short_quiet_speaker":
                        self.assertTrue(any(e["speaker"] == "speaker_1" and e["text"] == "네"
                                            and e["status"] == "final" for e in subtitles))

    def test_weak_noise_and_genuine_short_speech_finalization_in_workers(self):
        for genuine in (False, True):
            with self.subTest(genuine=genuine):
                def capture(audio):
                    audio = audio * .001
                    if genuine:
                        audio[1600:] = 0
                    return audio

                def response(audio, index):
                    result = payload("r", index, len(audio))
                    for i, slot in enumerate(result["speakers"]):
                        active = index == 0 and i == 0
                        wave = audio * 1000 if active else audio * 0
                        slot["waveform"]["data"] = base64.b64encode(wave.astype("<f4").tobytes()).decode()
                        slot["raw_transcript"] = "감사합니다." if active else ""
                        slot["vad"].update({
                            "speech_detected": active, "speech_duration_ms": 100 if active else 0,
                            "speech_ratio": 1600 / len(audio) if active else 0,
                            "timestamps": [{"start": 0, "end": 1600}] if active else [],
                        })
                    return result

                code, result = self.run_pipeline(True, response_factory=response,
                    capture_factory=capture, websocket=False)
                # Existing aggregate success requires some transcript output.
                # An entirely suppressed noise run has code 1 without worker errors.
                self.assertEqual(code, 0 if genuine else 1, result["errors"])
                self.assertEqual(result["errors"], [])
                self.assertEqual(result["stt_window_success"], 2)
                final = [e for e in result["subtitle_events"] if e["status"] == "final"]
                self.assertEqual([e["text"] for e in final], ["감사합니다."] if genuine else [])
                partial = [e for e in result["subtitle_events"] if e["status"] == "partial"]
                self.assertEqual([e["text"] for e in partial], ["감사합니다."] if genuine else [])
                evidence = result["windows"][0]["secondary_leakage_diagnostic"]
                self.assertEqual(evidence["subtitle_admission"]["applied"], not genuine)
                self.assertEqual(evidence["raw_transcripts"]["speaker_0"], "감사합니다.")

    def test_ambiguous_tentative_is_withheld_and_discarded_internally(self):
        from phase5.subtitle_presentation import SubtitlePresentationState
        for duration in (3, 5):
            with self.subTest(duration=duration):
                def response(audio, index):
                    result = payload("r", index, len(audio))
                    rng = np.random.default_rng(129)
                    noise = [rng.normal(size=len(audio)).astype(np.float32) for _ in range(2)]
                    for wave in noise:
                        wave *= np.sqrt(np.mean(audio ** 2) / np.mean(wave ** 2))
                    waves = (audio + .1 * noise[0], .1 * audio - .3 * noise[0] + noise[1])
                    for i, slot in enumerate(result["speakers"]):
                        active = index == 0
                        wave = waves[i] if active else audio * 0
                        span = len(audio) if i == 0 else 764 * 16
                        slot["waveform"]["data"] = base64.b64encode(wave.astype("<f4").tobytes()).decode()
                        slot["raw_transcript"] = ("정상 발화" if i == 0 else "미확인 후보") if active else ""
                        slot["vad"].update({"speech_detected": active,
                            "speech_duration_ms": span / 16 if active else 0,
                            "speech_ratio": span / len(audio) if active else 0,
                            "timestamps": [{"start": 0, "end": span}] if active else []})
                    return result
                code, result = self.run_pipeline(True, response_factory=response, duration=duration)
                self.assertEqual(code, 0)
                self.assertEqual(result["errors"], [])
                admission = result["windows"][0]["secondary_leakage_diagnostic"]["subtitle_admission"]["streams"][1]
                self.assertFalse(admission["suppressed"])
                self.assertFalse(admission["source_supported"])
                events = [e for e in result["subtitle_events"] if e["speaker"] == "speaker_1"]
                self.assertEqual(events, [])
                state_events = [e for e in result["subtitle_state_events"] if e["speaker"] == 1]
                self.assertEqual([e["action"] for e in state_events], ["start", "discard"])
                self.assertEqual([e["publication_text"] for e in state_events], ["", ""])
                self.assertEqual(state_events[-1]["history_size"], 1)
                self.assertEqual(result["websocket"]["published"], len(result["subtitle_events"]))
                presentation = SubtitlePresentationState()
                for event in result["subtitle_events"]:
                    presentation.apply(event)
                self.assertEqual([entry.text for entry in presentation.entries], ["정상 발화"])

    def test_withheld_tentative_is_published_after_next_window_support(self):
        text = "다음 창에서 확인되는 실제 발화"

        def response(audio, index):
            result = payload("r", index, len(audio))
            waves = (audio, audio)
            for i, slot in enumerate(result["speakers"]):
                active = index in (0, 1)
                wave = waves[i] if active else audio * 0
                if i == 0:
                    start, end = 0, len(audio)
                elif index == 0:
                    start, end = len(audio) - 800 * 16, len(audio)
                else:
                    start, end = 0, 800 * 16
                slot["waveform"]["data"] = base64.b64encode(wave.astype("<f4").tobytes()).decode()
                slot["raw_transcript"] = ("정상 발화" if i == 0 else text) if active else ""
                slot["vad"].update({
                    "speech_detected": active,
                    "speech_duration_ms": (end - start) / 16 if active else 0,
                    "speech_ratio": (end - start) / len(audio) if active else 0,
                    "timestamps": [{"start": start, "end": end}] if active else [],
                })
            return result

        delivered = []
        hooks = pipeline.LivePipelineHooks(on_subtitle=delivered.append)
        evidence_calls = 0

        def source_evidence(*_args):
            nonlocal evidence_calls
            evidence_calls += 1
            supported = evidence_calls > 2
            return {
                "candidate_input_correlation": .9 if supported else .1044,
                "owner_input_correlation": .1 if supported else .9847,
                "independent_input_correlation": .9 if supported else .2842,
                "pair_correlation": .05,
            }

        with patch(
            "secondary_leakage_diagnostics.source_input_evidence",
            side_effect=source_evidence,
        ):
            code, result = self.run_pipeline(
                True, response_factory=response, duration=7, runtime_hooks=hooks
            )
        self.assertEqual(code, 0)
        self.assertEqual(result["errors"], [])
        state_events = [e for e in result["subtitle_state_events"] if e["speaker"] == 1]
        self.assertEqual(state_events[0]["stability_action"], "awaiting_independent_support")
        self.assertEqual(state_events[0]["publication_text"], "")
        admissions = [
            row["secondary_leakage_diagnostic"]["subtitle_admission"]["streams"][1]
            for row in result["windows"][:2]
        ]
        self.assertEqual([row["source_supported"] for row in admissions], [False, True])
        self.assertEqual(state_events[1]["publication_text"], text)
        published = [e for e in result["subtitle_events"] if e["speaker"] == "speaker_1"]
        self.assertEqual([e["status"] for e in published], ["partial", "final"])
        self.assertEqual([e["text"] for e in published], [text, text])
        self.assertEqual(delivered, result["subtitle_events"])
        self.assertEqual(result["websocket"]["published"], len(result["subtitle_events"]))

    def test_stop_during_remote_response_prevents_late_publication(self):
        delivered = []
        hooks = pipeline.LivePipelineHooks(on_subtitle=delivered.append)
        def response(audio, index):
            hooks.stop_event.set()  # STOP while the remote request is in flight.
            return payload("r", index, len(audio))
        _, result = self.run_pipeline(True, response_factory=response, runtime_hooks=hooks)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["subtitle_events"], [])
        self.assertEqual(delivered, [])

    def test_vad_positive_artifact_bootstrap_never_publishes_or_finalizes(self):
        for duration_ms, owner_ms, start_ms in ((508, 1718, 1400), (1144, 3000, 1000), (764, 0, 1200)):
            with self.subTest(duration_ms=duration_ms):
                def response(audio, index):
                    result = payload("r", index, len(audio))
                    noise = np.random.default_rng(1002).normal(0, .1, len(audio)).astype(np.float32)
                    waves = [audio, noise] if index == 0 else [audio * 0, audio * 0]
                    texts = ["본문을 설명합니다" if owner_ms else "", "별도의 짧은 발화"] if index == 0 else ["", ""]
                    spans = ([[(0, owner_ms * 16)] if owner_ms else [],
                              [(start_ms * 16, (start_ms + duration_ms) * 16)]]
                             if index == 0 else [[], []])
                    for i, slot in enumerate(result["speakers"]):
                        slot["waveform"]["data"] = base64.b64encode(waves[i].astype("<f4").tobytes()).decode()
                        slot["raw_transcript"] = texts[i]
                        duration = sum(end - start for start, end in spans[i])
                        slot["vad"].update({
                            "speech_detected": bool(spans[i]), "speech_duration_ms": duration / 16,
                            "speech_ratio": duration / len(audio),
                            "timestamps": [{"start": start, "end": end} for start, end in spans[i]],
                        })
                    return result
                code, result = self.run_pipeline(True, response_factory=response)
                self.assertEqual(code, 0 if owner_ms else 1)
                self.assertEqual(result["errors"], [])
                self.assertFalse(any(e["speaker"] == "speaker_1" for e in result["subtitle_events"]))
                self.assertFalse(any(e["speaker"] == 1 for e in result["subtitle_state_events"]))
                first = result["windows"][0]
                self.assertTrue(first["vad"][1]["speech_detected"])
                evidence = first["secondary_leakage_diagnostic"]["subtitle_admission"]["streams"][1]
                self.assertEqual(evidence["reason"], "unsupported_source_on_speech_intervals")
                self.assertFalse(evidence["weak_speech_evidence"]["suppressed"])

    def test_overlap_artifact_and_next_raw_permutation_keep_one_stream_in_workers(self):
        def response(audio, index):
            result = payload("r", index, len(audio))
            rng = np.random.default_rng(1002 + index)
            residual = rng.normal(0, .1, len(audio)).astype(np.float32)
            residual *= np.sqrt(np.mean(audio ** 2) / np.mean(residual ** 2))
            dominant = audio.copy()
            texts, spans = ["연속 설명을 시작합니다", ""], [[(0, len(audio))], []]
            if index == 2:
                dominant[:16000] = .02 * (audio[:16000] + 1.5 * residual[:16000])
                residual[:16000] = 3 * audio[:16000]
                # Same RMS on both outputs, as in ClearVoice; RMS is not a vote.
                for wave in (dominant, residual):
                    wave *= np.sqrt(np.mean(audio ** 2) / np.mean(wave ** 2))
                texts = ["설명을 시작합니다 이어서 진행합니다", "별도 후보입니다"]
                spans = [[(0, 2486 * 16)], [(16000, 16000 + 894 * 16)]]
            elif index == 3:
                texts = ["이어서 진행합니다 마지막 내용입니다", ""]
            elif index == 4:
                dominant = residual = audio * 0
                texts, spans = ["", ""], [[], []]
            waves = [dominant, residual]
            if index == 3:
                waves.reverse()
                texts.reverse()
                spans.reverse()
            for i, slot in enumerate(result["speakers"]):
                slot["waveform"]["data"] = base64.b64encode(waves[i].astype("<f4").tobytes()).decode()
                slot["raw_transcript"] = texts[i]
                duration = sum(end - start for start, end in spans[i])
                slot["vad"].update({
                    "speech_detected": bool(spans[i]), "speech_duration_ms": duration / 16,
                    "speech_ratio": duration / len(audio),
                    "timestamps": [{"start": start, "end": end} for start, end in spans[i]],
                })
            return result
        code, result = self.run_pipeline(True, response_factory=response, duration=11)
        self.assertEqual(code, 0)
        self.assertEqual(result["errors"], [])
        rows = result["windows"]
        self.assertEqual(rows[1]["speaker_assignment"]["assignment_method"], "single_active_hold")
        fifth = rows[2]["speaker_assignment"]
        self.assertGreater(fifth["swap_score"], fifth["identity_score"] + .12)
        self.assertEqual(fifth["assignment_method"], "single_source_continuity")
        self.assertEqual(fifth["raw_to_logical_mapping"], {"0": 0, "1": 1})
        self.assertEqual(rows[3]["speaker_assignment"]["raw_to_logical_mapping"], {"0": 1, "1": 0})
        events = result["subtitle_events"]
        self.assertEqual({e["speaker"] for e in events}, {"speaker_0"})
        self.assertEqual({e["utterance_id"] for e in events}, {1})
        self.assertTrue(all(e["websocket_accepted"] for e in events))
        self.assertEqual(events[-1]["status"], "final")
        self.assertIn("마지막 내용입니다", events[-1]["text"])
        self.assertNotIn("별도 후보입니다", events[-1]["text"])

    def test_structured_runtime_hooks_receive_subtitles_and_metrics(self):
        subtitle_events = []
        metrics = []
        hooks = pipeline.LivePipelineHooks(
            on_subtitle=subtitle_events.append,
            on_metric=metrics.append,
        )
        code, result = self.run_pipeline(True, runtime_hooks=hooks)
        self.assertEqual(code, 0)
        self.assertEqual(subtitle_events, result["subtitle_events"])
        self.assertEqual(len(metrics), result["stt_window_success"])
        self.assertEqual({event["speaker"] for event in subtitle_events}, {"speaker_0", "speaker_1"})
        self.assertTrue(all(event["status"] in {"partial", "final"} for event in subtitle_events))
        self.assertTrue(all("sequence" in event and "timestamp" in event for event in subtitle_events))
        self.assertTrue(all("post_capture_latency_seconds" in metric for metric in metrics))

    def test_remote_audio_queue_absorbs_one_bounded_startup_burst(self):
        self.assertEqual(pipeline.audio_queue_maxsize("local"), 2)
        self.assertEqual(pipeline.audio_queue_maxsize("remote"), 3)

        admitted = queue.Queue(maxsize=pipeline.audio_queue_maxsize("remote"))
        dropped = []
        lock = threading.Lock()
        for index in range(3):
            self.assertTrue(
                pipeline.base.enqueue_or_drop(
                    admitted, index, "audio_queue", index, dropped, lock
                )
            )
        self.assertEqual(admitted.maxsize, 3)
        self.assertEqual(dropped, [])

        # A sustained overload remains bounded and preserves the original
        # newest-window drop policy after the one-window burst allowance.
        self.assertFalse(
            pipeline.base.enqueue_or_drop(
                admitted, 3, "audio_queue", 3, dropped, lock
            )
        )
        self.assertEqual(dropped, [("audio_queue", 3)])

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

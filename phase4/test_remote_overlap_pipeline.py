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
                     response_factory=None, duration=5, websocket=True, capture_factory=None,
                     reference_paths=None, speaker_mode="overlap"):
        speaker = Mock(name="speaker")
        players = []
        def open_player(**kwargs):
            player = MagicMock()
            player.__enter__.return_value = player
            player.currentpadding = 0
            players.append((kwargs["samplerate"], player))
            return player
        speaker.player.side_effect = open_player
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
            if response_factory is not None:
                response = response_factory(audio, index)
            else:
                response = payload("r", index, len(audio))
                mask = np.zeros(len(audio), dtype=np.float32)
                mask[::2] = 1
                waves = (audio * mask, audio * (1 - mask))
                for slot, wave in zip(response["speakers"], waves):
                    slot["waveform"]["data"] = base64.b64encode(
                        wave.astype("<f4").tobytes()
                    ).decode()
                    slot["vad"]["timestamps"] = [{"start": 0, "end": len(audio)}]
            return parse_result(response, "r", index, len(audio), .02)
        client.process.side_effect = process
        def separation(audio, **kwargs):
            return SeparationRecoveryResult(np.stack([audio, -audio])[:, None, :], finite_audio_stats(audio), .01, 0., False, False, False)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            output = Path(directory) / "result.json"
            argv = ["run_overlap_pipeline", "--duration", str(duration), "--assemble", "--json-output", str(output)]
            argv += ["--speaker-mode", speaker_mode]
            if reference_paths is None:
                argv += ["--live"]
            else:
                for name, path in reference_paths.items():
                    argv += [f"--reference-{name}", str(path)]
            if remote:
                argv += ["--processing-mode", "remote"]
                if websocket:
                    argv += ["--websocket", "--ws-port", "0"]
            if latency_diagnostics: argv += ["--latency-diagnostics"]
            stack.enter_context(patch.object(pipeline.sys, "argv", argv))
            stdout_buffer = io.StringIO()
            stdout = Mock(wraps=stdout_buffer); stdout.reconfigure = Mock()
            stack.enter_context(patch.object(pipeline.sys, "stdout", stdout))
            # Spoof only the pipeline guard; real WAV I/O must use the host platform.
            stack.enter_context(patch.object(pipeline, "sys", Mock(wraps=sys, platform="win32")))
            stack.enter_context(patch.object(pipeline.sc, "default_speaker", return_value=speaker))
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
        self.last_stdout = stdout_buffer.getvalue()
        self.last_players = players
        if reference_paths is None:
            speaker.player.assert_not_called()
        return code, result

    def test_reference_override_preserves_playback_duration_and_pitch(self):
        self.check_reference_playback(use_overrides=True)

    def test_default_reference_is_not_resampled_twice(self):
        self.check_reference_playback(use_overrides=False)

    def test_playback_resampler_preserves_sample_count_duration(self):
        for count in (48000, 48001, 480000):
            with self.subTest(source_samples=count):
                source = np.linspace(-.1, .1, count, dtype=np.float32)
                playback = pipeline.base._resample(
                    source, pipeline.SAMPLE_RATE, pipeline.base.CAPTURE_SAMPLE_RATE,
                )
                self.assertEqual(len(playback), count * 3)
                self.assertAlmostEqual(
                    len(source) / pipeline.SAMPLE_RATE,
                    len(playback) / pipeline.base.CAPTURE_SAMPLE_RATE,
                )

    def check_reference_playback(self, *, use_overrides):
        # Real WAV loading and scipy resampling; audio devices and GPU HTTP are mocked.
        # Different source durations exercise both repeat and final slice handling.
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for name, seconds, frequency in (("a", 3, 440), ("b", 5, 660)):
                source = (.1 * np.sin(2 * np.pi * frequency *
                                     np.arange(seconds * pipeline.SAMPLE_RATE) /
                                     pipeline.SAMPLE_RATE)).astype(np.float32)
                paths[name] = Path(directory) / f"speaker_{name}.wav"
                pipeline.sf.write(paths[name], source, pipeline.SAMPLE_RATE, subtype="FLOAT")
            with patch("playback_capture.REFERENCE_DIR", Path(directory)):
                code, result = self.run_pipeline(
                    True, duration=5, websocket=False,
                    reference_paths=paths if use_overrides else {},
                )
        self.assertEqual(code, 0)
        self.assertEqual(result["errors"], [])
        self.assertEqual(len(self.last_players), 2)
        observed_frequencies = []
        for rate, player in self.last_players:
            self.assertEqual(rate, 48000)
            segments = [call.args[0] for call in player.play.call_args_list]
            first = segments[0]
            # Identify the source by pitch, independently of thread scheduling.
            spectrum = np.abs(np.fft.rfft(first))
            frequency = np.argmax(spectrum) * rate / len(first)
            self.assertTrue(any(abs(frequency - expected) < 1 for expected in (440, 660)),
                            f"Playback pitch changed to {frequency} Hz")
            seconds = 3 if abs(frequency - 440) < 1 else 5
            observed_frequencies.append(round(frequency))
            self.assertEqual(len(first), seconds * rate)
            self.assertEqual(len(first) / rate, seconds)
            self.assertEqual(first.dtype, np.float32)
            self.assertEqual(first.ndim, 1)
            self.assertTrue(np.isfinite(first).all())
            # Existing playback budget includes two seconds of capture headroom.
            self.assertEqual(sum(len(segment) for segment in segments), 7 * rate)
            self.assertEqual(len(segments[-1]), (7 % seconds) * rate)
        self.assertCountEqual(observed_frequencies, [440, 660])

    def test_remote_workers(self):
        delivered = []
        code, result = self.run_pipeline(
            True, runtime_hooks=pipeline.LivePipelineHooks(on_subtitle=delivered.append))
        self.assertEqual(code, 0)
        self.assertEqual(result["stt_window_success"], 2)
        self.assertTrue(result["subtitle_events"])
        self.assertEqual(result["errors"], [])
        self.assertGreater(result["websocket"]["published"], 0)
        self.assertEqual(result["websocket"]["dropped_oldest"], 0)
        self.assertEqual(delivered, result["subtitle_events"])
        self.assertEqual(result["websocket"]["published"], len(delivered))
        logged_admissions = [json.loads(line[line.index("{"):])
                             for line in self.last_stdout.splitlines()
                             if line.startswith("[SUBTITLE ADMISSION]")]
        decisions = [decision for row in result["windows"]
                     for decision in row["secondary_leakage_diagnostic"]["subtitle_admission"]["streams"]
                     if decision["reason"] != "inactive"]
        self.assertEqual(logged_admissions, decisions)
        self.assertTrue(any(row["existing_partial"] for row in logged_admissions))
        for decision in decisions:
            self.assertTrue(decision["source_support_checks"])
            self.assertIn("input_speech_rms", decision["weak_speech_evidence"])
        logged_updates = [json.loads(line[line.index("{"):])
                          for line in self.last_stdout.splitlines()
                          if line.startswith("[SUBTITLE SUPPORT]")]
        partials = [event for event in result["subtitle_state_events"] if event["status"] == "partial"]
        self.assertEqual(logged_updates, [event["support_update"] for event in partials])
        self.assertTrue(all(update["reason"] == "current_source_covers_unretained_text"
                            for update in logged_updates))

    def test_single_mode_routes_only_mapped_primary_with_raw_hallucinations(self):
        hallucinations = ["어서 오세요", "An arbitrary English hallucination"]

        def response(audio, index):
            result = payload("r", index, len(audio))
            primary = np.asarray(audio, dtype=np.float32)
            artifact = (
                .5 * primary
                + np.random.default_rng(9000 + index).normal(0, .05, len(audio))
            ).astype(np.float32)
            slots = [artifact, primary] if index == 0 else [primary, artifact]
            texts = [hallucinations[index], "정상 주 음성 자막입니다"] if index == 0 else [
                "정상 주 음성 자막입니다", hallucinations[index]
            ]
            for raw_slot, (slot, wave, text) in enumerate(
                zip(result["speakers"], slots, texts)
            ):
                slot["waveform"]["data"] = base64.b64encode(
                    wave.astype("<f4").tobytes()
                ).decode()
                slot["raw_transcript"] = text
                speech_samples = len(audio) if wave is primary else min(1600, len(audio))
                slot["vad"].update(
                    speech_detected=True,
                    speech_duration_ms=speech_samples / 16,
                    speech_ratio=speech_samples / len(audio),
                    timestamps=[{"start": 0, "end": speech_samples}],
                )
            return result

        delivered = []
        code, result = self.run_pipeline(
            True,
            response_factory=response,
            runtime_hooks=pipeline.LivePipelineHooks(on_subtitle=delivered.append),
            speaker_mode="single",
        )
        self.assertEqual(code, 0)
        self.assertEqual(result["speaker_mode"], "single")
        self.assertEqual(result["windows"][0]["raw_to_logical_mapping"], {"0": 0, "1": 1})
        self.assertTrue(delivered)
        self.assertTrue(all(event["speaker"] == "speaker_1" for event in delivered))
        self.assertIn("정상 주 음성 자막입니다", delivered[-1]["text"])
        published_by_window = {}
        for event in delivered:
            published_by_window.setdefault(event["window_index"], set()).add(event["speaker"])
        self.assertTrue(all(len(speakers) <= 1 for speakers in published_by_window.values()))
        for index, row in enumerate(result["windows"]):
            self.assertEqual(row["speaker_mode"], "single")
            self.assertEqual(row["single_mode_primary"], 1)
            self.assertEqual(len(row["assembly_events"]), 1)
            streams = row["single_mode"]["streams"]
            self.assertEqual(streams[1]["assembler_transcript"], "정상 주 음성 자막입니다")
            self.assertFalse(streams[1]["suppressed"])
            self.assertEqual(streams[0]["raw_transcript"], hallucinations[index])
            self.assertEqual(streams[0]["assembler_transcript"], "")
            self.assertTrue(streams[0]["suppressed"])
            self.assertEqual(streams[0]["reason"], "non_primary_stream")
            self.assertFalse(streams[0]["assembler_forwarded"])
            self.assertFalse(streams[0]["subtitle_state_forwarded"])
            self.assertFalse(streams[0]["published"])

    def test_source_artifact_admission_reaches_real_publishers_without_ghosts(self):
        # Real parsing/tracking/admission/state/WebSocket/hooks; supplied remote
        # output, VAD and STT. This does not claim to run speech recognition.
        for amplitude, text in ((None, "고맙습니다"), (None, "내일 기차가 도착합니다"),
                                (1., "고맙습니다"), (.001, "작지만 실제 두 번째 발화입니다")):
            with self.subTest(genuine_amplitude=amplitude, text=text):
                rng = np.random.default_rng(41213)
                raw_sources = rng.normal(0, .1, (2, 5 * 48000)).astype(np.float32)
                source_windows = []
                offset = 0
                previous = None
                def capture_sources(placeholder):
                    nonlocal offset, previous
                    count = len(placeholder)
                    raw = raw_sources[:, offset:offset + count]
                    converted = [pipeline.base._resample(wave, 48000, 16000) for wave in raw]
                    window = (converted if previous is None else
                              [np.concatenate((old[-16000:], new))
                               for old, new in zip(previous, converted)])
                    source_windows.append(window)
                    previous = window
                    offset += count
                    return (raw[0] + (amplitude or 0) * raw[1])[:, None]

                def response(audio, index):
                    result = payload("r", index, len(audio))
                    rng = np.random.default_rng(41213 + index)
                    primary, other = source_windows[index]
                    error = rng.normal(0, float(np.std(audio)), len(audio)).astype(np.float32)
                    if amplitude is None:
                        waves = [audio + .0094 * error,
                                 .65 * audio + np.sqrt(1 - .65**2) * other]
                    else:
                        # Capture = primary + amplitude * independent secondary.
                        waves = [primary, other]
                    texts = ["건강관리에 주의할 것을 당부했습니다", text]
                    spans = [[(0, len(audio))], [(16000, 16000 + 1052 * 16)]]
                    if index % 2:
                        waves.reverse(); texts.reverse(); spans.reverse()
                    for slot, wave, raw, intervals in zip(result["speakers"], waves, texts, spans):
                        slot["waveform"]["data"] = base64.b64encode(wave.astype("<f4").tobytes()).decode()
                        slot["raw_transcript"] = raw
                        duration = sum(end - start for start, end in intervals)
                        slot["vad"].update(
                            speech_detected=True, speech_duration_ms=duration / 16,
                            speech_ratio=duration / len(audio),
                            timestamps=[{"start": start, "end": end} for start, end in intervals],
                        )
                    return result

                delivered = []
                code, result = self.run_pipeline(
                    True, response_factory=response, capture_factory=capture_sources,
                    runtime_hooks=pipeline.LivePipelineHooks(on_subtitle=delivered.append),
                )
                self.assertEqual(code, 0)
                self.assertEqual(result["errors"], [])
                self.assertEqual(delivered, result["subtitle_events"])
                self.assertEqual(result["websocket"]["published"], len(delivered))
                self.assertTrue(any(e["speaker"] == "speaker_0" for e in delivered))
                for row in result["windows"]:
                    diagnostic = row["secondary_leakage_diagnostic"]
                    self.assertEqual(diagnostic["raw_transcripts"]["speaker_1"], text)
                    decision = diagnostic["subtitle_admission"]["streams"][1]
                    self.assertFalse(decision["stt_skipped_before_inference"])
                    self.assertEqual(decision["raw_transcript"], text)
                    self.assertEqual(decision["transcript_rejected_after_inference"], amplitude is None)
                    self.assertEqual(decision["admitted_transcript"], "" if amplitude is None else text)
                secondary_events = [e for e in delivered if e["speaker"] == "speaker_1"]
                if amplitude is None:
                    self.assertEqual(secondary_events, [])
                    self.assertFalse(any(e["speaker"] == 1 for e in result["subtitle_state_events"]))
                else:
                    self.assertTrue(secondary_events)
                    self.assertEqual(secondary_events[-1]["status"], "final")
                    # Each window supplied the same STT on distinct speech;
                    # assembly may retain both occurrences. Admission must
                    # preserve the supplied text through final publication.
                    self.assertIn(text, secondary_events[-1]["text"])

    def test_recorded_mossformer_artifact_is_rejected_before_assembly_and_publishers(self):
        from test_input_residual_speech_admission import load_fixture
        from phase5.subtitle_presentation import SubtitlePresentationState
        import copy
        (mixture, owner, candidate), metadata = load_fixture()
        # Actual recorded separation/VAD/STT, transported through real parsers
        # and workers. Capture and resampling are injected for exact input replay;
        # this is not WASAPI, fresh GPU inference, or the Windows w12 recording.
        for reverse in (False, True):
            for text in (metadata["raw_transcripts"][1], "내일 기차가 도착합니다",
                         "An arbitrary English sentence"):
                with self.subTest(reverse=reverse, text=text):
                    def response(audio, index):
                        np.testing.assert_array_equal(audio, mixture)
                        result = payload("r", index, len(audio))
                        signals = [owner, candidate]
                        vads = copy.deepcopy(metadata["vad_results"])
                        texts = [metadata["raw_transcripts"][0], text]
                        if reverse:
                            signals.reverse(); vads.reverse(); texts.reverse()
                        for slot, signal, vad, raw in zip(result["speakers"], signals, vads, texts):
                            slot["waveform"]["data"] = base64.b64encode(signal.astype("<f4").tobytes()).decode()
                            slot["vad"] = vad
                            slot["raw_transcript"] = raw
                        return result
                    delivered = []
                    with patch.object(pipeline.base, "_resample", return_value=mixture.copy()):
                        code, result = self.run_pipeline(True, response_factory=response, duration=3,
                            runtime_hooks=pipeline.LivePipelineHooks(on_subtitle=delivered.append))
                    self.assertEqual(code, 0, result["errors"])
                    self.assertEqual(result["errors"], [])
                    row = result["windows"][0]
                    mapping = row["speaker_assignment"]["raw_to_logical_mapping"]
                    primary_id, secondary_id = mapping[str(int(reverse))], mapping[str(int(not reverse))]
                    decision = row["secondary_leakage_diagnostic"]["subtitle_admission"]["streams"][secondary_id]
                    self.assertEqual(decision["raw_transcript"], text)
                    self.assertFalse(decision["stt_skipped_before_inference"])
                    self.assertTrue(decision["transcript_rejected_after_inference"])
                    self.assertEqual(decision["admitted_transcript"], "")
                    self.assertEqual(decision["reason"], "owner_residual_contains_no_candidate_speech")
                    self.assertFalse(any(e["speaker"] == secondary_id for e in result["subtitle_state_events"]))
                    secondary_assembly = [e for e in row["assembly_events"] if e["speaker"] == secondary_id]
                    self.assertTrue(secondary_assembly)
                    self.assertTrue(all(e["raw"] == e["new"] == "" for e in secondary_assembly))
                    self.assertEqual(delivered, result["subtitle_events"])
                    self.assertEqual(result["websocket"]["published"], len(delivered))
                    self.assertTrue(delivered)
                    self.assertTrue(all(e["speaker"] == f"speaker_{primary_id}" for e in delivered))
                    self.assertTrue(all(e["text"] == metadata["raw_transcripts"][0] for e in delivered))
                    presentation = SubtitlePresentationState()
                    for event in delivered:
                        presentation.apply(event)
                    self.assertEqual([e.text for e in presentation.entries], [metadata["raw_transcripts"][0]])

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

    def test_supported_overlap_revises_wrong_primary_partial_in_real_workers(self):
        def response(audio, index):
            result = payload("r", index, len(audio))
            if index == 0:
                text, spans = "제 손은 그냥과와 높아지는", [(32000, 48000)]
                primary = audio
            elif index == 1:
                text = "분양가와 높아지는 청약문턱에 지친 2030세대가"
                spans, primary = [(0, 48000)], audio
            else:
                text, spans, primary = "", [], np.zeros_like(audio)
            waves = (primary, np.zeros_like(audio))
            for slot, wave, slot_text, slot_spans in zip(
                result["speakers"], waves, (text, ""), (spans, []),
            ):
                slot["waveform"]["data"] = base64.b64encode(
                    wave.astype("<f4").tobytes()
                ).decode()
                slot["raw_transcript"] = slot_text
                duration = sum(end - start for start, end in slot_spans)
                slot["vad"].update({
                    "speech_detected": bool(slot_spans),
                    "speech_duration_ms": duration / 16,
                    "speech_ratio": duration / len(audio),
                    "timestamps": [
                        {"start": start, "end": end} for start, end in slot_spans
                    ],
                })
            return result

        delivered = []
        code, result = self.run_pipeline(
            True, response_factory=response, duration=7,
            runtime_hooks=pipeline.LivePipelineHooks(on_subtitle=delivered.append),
        )
        self.assertEqual(code, 0)
        self.assertEqual(result["errors"], [])
        primary = [event for event in delivered if event["speaker"] == "speaker_0"]
        self.assertEqual([event["status"] for event in primary], ["partial", "partial", "final"])
        self.assertEqual(primary[0]["text"], "제 손은 그냥과와 높아지는")
        self.assertEqual(
            primary[1]["text"],
            "분양가와 높아지는 청약문턱에 지친 2030세대가",
        )
        self.assertEqual(primary[2]["text"], primary[1]["text"])
        self.assertNotIn("제 손은", primary[2]["text"])
        rows = result["windows"]
        overlap = rows[1]["assembly_events"][0]["audio_overlap_evidence"]
        self.assertTrue(overlap["shared_speech"])
        self.assertTrue(overlap["supported_tail_revision"])
        self.assertEqual(rows[1]["assembly_events"][0]["match_type"],
                         "supported_tail_replace")
        self.assertEqual(result["websocket"]["published"], len(delivered))

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

    def test_repeated_unsupported_tentative_never_reaches_external_publishers(self):
        text = "반복되는 미지원 후보"

        def response(audio, index):
            result = payload("r", index, len(audio))
            rng = np.random.default_rng(129)
            noise = [rng.normal(size=len(audio)).astype(np.float32) for _ in range(2)]
            for wave in noise:
                wave *= np.sqrt(np.mean(audio ** 2) / np.mean(wave ** 2))
            waves = (audio + .1 * noise[0], .1 * audio - .3 * noise[0] + noise[1])
            for i, slot in enumerate(result["speakers"]):
                active = index in (0, 1)
                wave = waves[i] if active else audio * 0
                span = len(audio) if i == 0 else 764 * 16
                slot["waveform"]["data"] = base64.b64encode(wave.astype("<f4").tobytes()).decode()
                slot["raw_transcript"] = ("정상 발화" if i == 0 else text) if active else ""
                slot["vad"].update({
                    "speech_detected": active,
                    "speech_duration_ms": span / 16 if active else 0,
                    "speech_ratio": span / len(audio) if active else 0,
                    "timestamps": [{"start": 0, "end": span}] if active else [],
                })
            return result

        evidence_calls = 0

        def source_evidence(*_args):
            nonlocal evidence_calls
            supported = evidence_calls % 2 == 0
            evidence_calls += 1
            return {
                "candidate_input_correlation": .9 if supported else .1044,
                "owner_input_correlation": .1 if supported else .9847,
                "independent_input_correlation": .9 if supported else .2842,
                "pair_correlation": .05,
            }

        delivered = []
        hooks = pipeline.LivePipelineHooks(on_subtitle=delivered.append)
        with patch(
            "secondary_leakage_diagnostics.source_input_evidence",
            side_effect=source_evidence,
        ):
            code, result = self.run_pipeline(
                True, response_factory=response, duration=7, runtime_hooks=hooks
            )
        self.assertEqual(code, 0)
        self.assertEqual(result["errors"], [])
        admissions = [
            row["secondary_leakage_diagnostic"]["subtitle_admission"]["streams"][1]
            for row in result["windows"][:2]
        ]
        self.assertEqual([row["source_supported"] for row in admissions], [False, False])
        state_events = [e for e in result["subtitle_state_events"] if e["speaker"] == 1]
        self.assertEqual(len(state_events), 3)
        self.assertEqual(state_events[0]["action"], "start")
        self.assertIn(state_events[1]["action"], {"retain", "extend"})
        self.assertEqual(state_events[2]["action"], "discard")
        self.assertEqual(state_events[1]["history_size"], 2)
        self.assertEqual([e["publication_text"] for e in state_events], ["", "", ""])
        self.assertFalse(any(e["speaker"] == "speaker_1" for e in result["subtitle_events"]))
        self.assertFalse(any(e["speaker"] == "speaker_1" for e in delivered))
        self.assertEqual(delivered, result["subtitle_events"])
        self.assertEqual(result["websocket"]["published"], len(result["subtitle_events"]))
        self.assertIn('publication_text=""', self.last_stdout)
        self.assertIn("source_supported=False", self.last_stdout)
        self.assertIn("will_publish=False", self.last_stdout)

    def test_withheld_tentative_is_published_after_genuine_source_support(self):
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
        self.assertFalse(state_events[0]["source_supported"])
        self.assertTrue(state_events[1]["source_supported"])
        self.assertEqual(state_events[1]["publication_text"], text)
        published = [e for e in result["subtitle_events"] if e["speaker"] == "speaker_1"]
        self.assertEqual([e["status"] for e in published], ["partial", "final"])
        self.assertEqual([e["text"] for e in published], [text, text])
        self.assertEqual(delivered, result["subtitle_events"])
        self.assertEqual(result["websocket"]["published"], len(result["subtitle_events"]))

    def test_latest_secondary_lifecycle_reaches_real_publishers_with_raw_swap(self):
        """Synthetic remote results exercise the full state and publication path."""
        for leakage, restart in ((True, False), (False, False), (False, True)):
            for swap in (False, True):
                with self.subTest(leakage=leakage, raw_swap=swap, restart=restart):
                    silence_window = 4 if restart else 3
                    def response(audio, index):
                        result = payload("r", index, len(audio))
                        mask = np.zeros(len(audio), dtype=np.float32)
                        mask[::2] = 1
                        waves = (audio * mask, audio * (1 - mask))
                        for raw, slot in enumerate(result["speakers"]):
                            logical = 1 - raw if swap and index >= 2 else raw
                            active = index < silence_window and (logical == 0 or index > 0)
                            wave = waves[logical] if active else np.zeros_like(audio)
                            slot["waveform"]["data"] = base64.b64encode(wave.astype("<f4").tobytes()).decode()
                            texts = ["우크라이나 외무장관은 대통령실에서 회담했습니다",
                                     "이천삼십 세대의 청약 통장 예치금은" if index == (2 if restart else 1)
                                     else "청약 통장 예치금은 십만 원입니다"]
                            if restart and index == 1:
                                texts[1] = "미지원 초기 후보"
                            slot["raw_transcript"] = texts[logical] if active else ""
                            start = 486 * 16 if logical == 1 and not leakage else 0
                            duration = (2900 if logical == 0 else 1800) if leakage else (1200 if logical == 0 else 2514)
                            slot["vad"].update(
                                speech_detected=active, speech_duration_ms=duration if active else 0,
                                speech_ratio=duration / 3000 if active else 0,
                                timestamps=[{"start": start, "end": start + duration * 16}] if active else [],
                            )
                        return result

                    primary = {
                        "owner_input_correlation": .60, "candidate_input_correlation": .98,
                        "independent_input_correlation": .95, "input_residual_energy_fraction": .64,
                        "candidate_residual_energy_fraction": .98, "pair_correlation": .30,
                    }
                    candidate = {
                        "owner_input_correlation": .9874 if leakage else .60,
                        "candidate_input_correlation": .9144 if leakage else .80,
                        "independent_input_correlation": .9002 if leakage else .999,
                        "input_residual_energy_fraction": .02498 if leakage else .64,
                        "candidate_residual_energy_fraction": .27718 if leakage else .90,
                        "pair_correlation": .85 if leakage else .20,
                    }
                    delivered = []
                    evidence_rows = [primary, primary, candidate, primary, candidate]
                    if restart:
                        evidence_rows[2] = dict(candidate, candidate_input_correlation=.10,
                                                independent_input_correlation=.28)
                        evidence_rows.extend([primary, candidate])
                    with patch("secondary_leakage_diagnostics.source_input_evidence",
                               side_effect=evidence_rows):
                        code, result = self.run_pipeline(
                            True, response_factory=response, duration=3 + 2 * silence_window,
                            runtime_hooks=pipeline.LivePipelineHooks(on_subtitle=delivered.append),
                        )
                    self.assertEqual(code, 0)
                    self.assertEqual(result["errors"], [])
                    self.assertEqual(result["windows"][2]["raw_to_logical_mapping"],
                                     {"0": int(swap), "1": int(not swap)})
                    decisions = [row["secondary_leakage_diagnostic"]["subtitle_admission"]["streams"][1]
                                 for row in result["windows"][1:silence_window]]
                    self.assertFalse(decisions[0]["cross_stream_lexical_overlap"]["duplicate_structure"])
                    self.assertEqual([row["candidate_transition"] for row in decisions],
                                     ["hold", "hold", "validate"] if restart
                                     else ["hold", "discard" if leakage else "validate"])
                    state_events = [e for e in result["subtitle_state_events"] if e["speaker"] == 1]
                    published = [e for e in result["subtitle_events"] if e["speaker"] == "speaker_1"]
                    self.assertEqual(state_events[0]["candidate_provenance"], "TENTATIVE")
                    self.assertEqual(state_events[0]["source_supported_text"], "")
                    self.assertEqual(state_events[0]["publication_text"], "")
                    if leakage:
                        self.assertEqual(published, [])
                        self.assertEqual([e["action"] for e in state_events], ["start", "discard"])
                        self.assertTrue(all(e["source_supported_text"] == "" for e in state_events))
                        self.assertFalse(any(e["speaker"] == "speaker_1" for e in delivered))
                    else:
                        self.assertEqual(state_events[-2]["candidate_provenance"], "VALIDATED")
                        self.assertEqual([e["status"] for e in published], ["partial", "final"])
                        self.assertEqual([e["window_index"] for e in published],
                                         [silence_window - 1, silence_window])
                        self.assertIn("이천삼십", published[0]["text"])
                        self.assertIn("십만 원", published[0]["text"])
                        self.assertEqual(published[0]["text"], published[1]["text"])
                        if restart:
                            self.assertEqual(state_events[1]["action"], "discard")
                            self.assertEqual(published[0]["utterance_id"], 2)
                            self.assertNotIn("미지원", published[0]["text"])
                    self.assertTrue(any(e["speaker"] == "speaker_0" and e["status"] == "final"
                                        for e in result["subtitle_events"]))
                    self.assertEqual(delivered, result["subtitle_events"])
                    self.assertEqual(result["websocket"]["published"], len(delivered))

    def test_previously_published_utterance_still_retracts_after_support_is_lost(self):
        original_process = pipeline.SpeakerSubtitleState.process

        def correction(state, *args, **kwargs):
            if state.speaker == 0 and kwargs.get("window") == 1:
                # Inject a hypothesis correction without source support, then
                # let the real pipeline-end FINAL and publisher retract it.
                kwargs.update(hypothesis="수정된 미지원 가설", source_supported=False,
                              source_text="수정된 미지원 가설")
            return original_process(state, *args, **kwargs)

        delivered = []
        with patch.object(pipeline.SpeakerSubtitleState, "process", new=correction):
            code, result = self.run_pipeline(
                True, runtime_hooks=pipeline.LivePipelineHooks(on_subtitle=delivered.append))
        self.assertEqual(code, 0)
        events = [event for event in delivered if event["speaker"] == "speaker_0"]
        self.assertEqual([event["action"] for event in events], ["start", "discard"])
        self.assertTrue(events[0]["text"])
        self.assertEqual(events[1]["text"], "")
        self.assertEqual(events[0]["utterance_id"], events[1]["utterance_id"])
        self.assertIn("needs_retraction=True will_publish=True reason=retraction", self.last_stdout)
        self.assertEqual(result["websocket"]["published"], len(delivered))

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

    def test_low_energy_540ms_false_vad_emits_no_websocket_subtitle(self):
        start, end = 34336, 42976

        def response(audio, index):
            result = payload("r", index, len(audio))
            for raw, slot in enumerate(result["speakers"]):
                active = index == 0 and raw == 1
                wave = audio if active else np.zeros_like(audio)
                slot["waveform"]["data"] = base64.b64encode(
                    wave.astype("<f4").tobytes()
                ).decode()
                slot["raw_transcript"] = "무음에서 생성된 임의의 문장" if active else ""
                slot["stt_seconds"] = 0.01 if active else 0.0
                slot["vad"].update(
                    speech_detected=active,
                    speech_duration_ms=540 if active else 0,
                    speech_ratio=0.18 if active else 0,
                    rms=float(np.sqrt(np.mean(wave * wave))),
                    peak=float(np.max(np.abs(wave))),
                    timestamps=[{"start": start, "end": end}] if active else [],
                )
            return result

        correlation_only = {
            "owner_input_correlation": 0.4366949,
            "candidate_input_correlation": 0.7228728,
            "independent_input_correlation": 0.7711960,
            "input_residual_energy_fraction": 0.80,
            "candidate_residual_energy_fraction": 0.90,
            "pair_correlation": 0.10,
        }
        delivered = []
        with patch(
            "secondary_leakage_diagnostics.source_input_evidence",
            return_value=correlation_only,
        ):
            _, result = self.run_pipeline(
                True,
                response_factory=response,
                runtime_hooks=pipeline.LivePipelineHooks(on_subtitle=delivered.append),
                capture_factory=lambda audio: audio * 0.003,
            )

        self.assertEqual(result["errors"], [])
        decisions = [
            decision
            for row in result["windows"]
            for decision in row["secondary_leakage_diagnostic"]["subtitle_admission"]["streams"]
            if decision["reason"] != "inactive"
        ]
        self.assertEqual(len(decisions), 1)
        self.assertFalse(decisions[0]["weak_speech_evidence"]["weak_vad"])
        self.assertTrue(decisions[0]["weak_speech_evidence"]["suppressed"])
        self.assertTrue(decisions[0]["source_support_checks"][0]["raw_direct_pass"])
        self.assertTrue(decisions[0]["source_support_checks"][0]["raw_independent_pass"])
        self.assertFalse(decisions[0]["source_supported"])
        self.assertEqual(result["subtitle_state_events"], [])
        self.assertEqual(result["subtitle_events"], [])
        self.assertEqual(delivered, [])
        self.assertEqual(result["websocket"]["published"], 0)

    def test_strong_316ms_transient_is_tentative_and_emits_no_websocket_subtitle(self):
        start, end = 16000, 16000 + 5056
        capture_calls = 0

        def capture(audio):
            nonlocal capture_calls
            capture_calls += 1
            if capture_calls > 1:
                return np.zeros_like(audio)
            samples = np.arange(len(audio))
            value = 0.06776 * np.sqrt(2) * np.sin(2 * np.pi * samples / 291)
            burst_start, burst_end = 48000, 48000 + 15168
            value[burst_start:burst_end] = (
                0.15226
                * np.sqrt(2)
                * np.sin(2 * np.pi * np.arange(burst_end - burst_start) / 123)
            )
            value[burst_start] = 0.52701
            return value.astype(np.float32).reshape(audio.shape)

        def response(audio, index):
            result = payload("r", index, len(audio))
            for raw, slot in enumerate(result["speakers"]):
                active = index == 0 and raw == 1
                wave = audio if active else np.zeros_like(audio)
                slot["waveform"]["data"] = base64.b64encode(
                    wave.astype("<f4").tobytes()
                ).decode()
                slot["raw_transcript"] = "실제 발화가 아닌 임의의 추론 문장" if active else ""
                slot["stt_seconds"] = 0.01 if active else 0.0
                slot["vad"].update(
                    speech_detected=active,
                    speech_duration_ms=316 if active else 0,
                    speech_ratio=316 / 3000 if active else 0,
                    rms=float(np.sqrt(np.mean(wave * wave))),
                    peak=float(np.max(np.abs(wave))),
                    timestamps=[{"start": start, "end": end}] if active else [],
                    input_residual_speech=(
                        {
                            "version": 1,
                            "available": True,
                            "sample_count": len(audio),
                            "reason": "measured",
                            "speech_detected": False,
                            "timestamps": [],
                        }
                        if active
                        else {"version": 1, "available": False, "reason": "not_required"}
                    ),
                )
            return result

        source_only = {
            "owner_input_correlation": 0.30,
            "candidate_input_correlation": 0.94443,
            "independent_input_correlation": 0.95578,
            "input_residual_energy_fraction": 0.80,
            "candidate_residual_energy_fraction": 0.90,
            "pair_correlation": 0.10,
        }
        delivered = []
        with patch(
            "secondary_leakage_diagnostics.source_input_evidence",
            return_value=source_only,
        ):
            _, result = self.run_pipeline(
                True,
                response_factory=response,
                capture_factory=capture,
                runtime_hooks=pipeline.LivePipelineHooks(on_subtitle=delivered.append),
            )

        self.assertEqual(result["errors"], [])
        decisions = [
            decision
            for row in result["windows"]
            for decision in row["secondary_leakage_diagnostic"]["subtitle_admission"]["streams"]
            if decision["reason"] != "inactive"
        ]
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0]["candidate_transition"], "hold")
        self.assertFalse(decisions[0]["source_supported"])
        state_events = result["subtitle_state_events"]
        self.assertEqual([event["action"] for event in state_events], ["start", "discard"])
        self.assertTrue(all(event["publication_text"] == "" for event in state_events))
        self.assertEqual(result["subtitle_events"], [])
        self.assertEqual(delivered, [])
        self.assertEqual(result["websocket"]["published"], 0)

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

"""Text fixtures test reconciliation, not acoustic recognition accuracy."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import TestCase

import numpy as np
import soundfile as sf

from audio_diagnostics import capture_audio_diagnostics, select_capture_channel
from benchmark_accuracy import text_scores
from secondary_leakage_diagnostics import (
    build_window_diagnostic,
    secondary_transcript_suppression_reason,
    save_candidate_full_window_wavs,
)
from subtitle_assembler import SubtitleAssembler, SpeakerSubtitleState

RECONCILIATION_CASES = [
    (["안뇽하세요", "안녕하세여 오늘은 아침에"], "안녕하세여 오늘은 아침에"),
    (
        ["오늘은 익스트림 모두", "익스트림 모드 실행합니다"],
        "오늘은 익스트림 모드 실행합니다",
    ),
    (
        ["안녕하세요. 오늘은 아침", "오늘은 아침에 일어나서"],
        "안녕하세요. 오늘은 아침에 일어나서",
    ),
    (
        ["집에서 토마토 파스타를 만들어봤는데", "봤는데 처음 시도한 요리입니다"],
        "집에서 토마토 파스타를 만들어봤는데 처음 시도한 요리입니다",
    ),
    (
        ["오늘은 밥을 먹었습니다", "내일은 밥을 먹겠습니다"],
        "오늘은 밥을 먹었습니다 내일은 밥을 먹겠습니다",
    ),
    (["사람입니다", "바람입니다"], "사람입니다 바람입니다"),
    (["진짜 진짜 맛있다"], "진짜 진짜 맛있다"),
    (
        [
            "공원에서 천천히 걷기 좋았고 공원에는 운동을 하거든요.",
            "공원에는 운동을 하거나 강아지와 산책하는 사람들이",
        ],
        "공원에서 천천히 걷기 좋았고 공원에는 운동을 하거나 강아지와 산책하는 사람들이",
    ),
]


class ReconciliationTests(TestCase):
    def test_expected_text_fixtures(self):
        for fragments, expected in RECONCILIATION_CASES:
            with self.subTest(fragments=fragments):
                assembler = SubtitleAssembler(0)
                for i, fragment in enumerate(fragments):
                    event = assembler.process(i, fragment)
                self.assertEqual(event.utterance_hypothesis, expected)

    def test_correction_only_is_not_labeled_duplicate_only(self):
        assembler = SubtitleAssembler(0)
        assembler.process(0, "안뇽하세요")
        event = assembler.process(1, "안녕하세요")
        self.assertEqual(event.utterance_hypothesis, "안녕하세요")
        self.assertFalse(event.duplicate_only)

    def test_gap_does_not_deduplicate_later_real_repetition(self):
        assembler = SubtitleAssembler(0)
        assembler.process(0, "다시 한번 시작하겠습니다")
        event = assembler.process(2, "다시 한번 시작하겠습니다")
        self.assertEqual(event.utterance_hypothesis.count("다시 한번"), 2)

    def test_correction_reaches_partial_and_final_without_rewriting_old_final(self):
        assembler, state = SubtitleAssembler(0), SpeakerSubtitleState(0)
        partials = []
        for i, fragment in enumerate(
            [
                "안뇽하세요",
                "안녕하세여 오늘은 아침에",
                "안녕하세요 오늘은 아침에 일어납니다",
            ]
        ):
            assembly = assembler.process(i, fragment)
            event = state.process(
                i,
                assembly.utterance_hypothesis,
                True,
                3 + 2 * i,
                confirmed_prefix_length=assembly.confirmed_prefix_length,
            )[0]
            partials.append(event)
        self.assertEqual(partials[1].action, "replace")
        final = state.process(3, "", False, 9)[0]
        self.assertEqual(final.text, "안녕하세요 오늘은 아침에 일어납니다")
        assembler.reset_utterance()
        assembler.process(4, "익스트림 모두")
        assembly = assembler.process(5, "익스트림 모드 실행")
        self.assertEqual(assembly.session_text, final.text + " 익스트림 모드 실행")
        self.assertEqual(state.final_segments, [final.text])

    def test_inherited_prefix_is_not_independent_consensus(self):
        assembler, state = SubtitleAssembler(0), SpeakerSubtitleState(0)
        for i, text in enumerate(["오늘은 아침에", "아침에 일어나서"]):
            assembly = assembler.process(i, text)
            event = state.process(
                i,
                assembly.utterance_hypothesis,
                True,
                3 + i * 2,
                confirmed_prefix_length=assembly.confirmed_prefix_length,
            )[0]
        self.assertEqual(event.stable_text, "")
        self.assertEqual(event.text, "오늘은 아침에 일어나서")
        repeated = SubtitleAssembler(0)
        repeated.process(0, "오늘은 아침에 일어나서")
        assembly = repeated.process(1, "오늘은 아침에 일어나서")
        # A full independent repeat can support the prefix.
        self.assertGreater(assembly.confirmed_prefix_length, 0)

    def test_metrics_keep_reference_free_quality_unknown(self):
        self.assertEqual(text_scores("안녕하세요", "안뇽하세요")["cer"], 0.2)
        self.assertEqual(text_scores("학교", "학교학교")["insertions"], 2)
        self.assertEqual(text_scores("학교에", "학교")["deletions"], 1)
        self.assertIsNone(text_scores("", "안녕")["cer"])
        self.assertEqual(text_scores("", "안녕")["silence_output_characters"], 2)


class CaptureAndLeakageTests(TestCase):
    def test_channel_loss_and_cancellation_are_visible(self):
        x = np.linspace(-0.5, 0.5, 480, dtype=np.float32)
        right_only = np.stack([np.zeros_like(x), x], axis=1)
        stats = capture_audio_diagnostics(
            right_only, select_capture_channel(right_only), "first"
        )
        self.assertTrue(stats["first_channel_silent_other_active"])
        self.assertGreater(np.max(select_capture_channel(right_only, "mean")), 0)
        antiphase = np.stack([x, -x], axis=1)
        stats = capture_audio_diagnostics(
            antiphase, select_capture_channel(antiphase), "first"
        )
        self.assertTrue(stats["downmix_cancellation_candidate"])
        self.assertGreater(stats["selected_rms"], 0)

    def test_only_joint_waveform_and_text_evidence_suppresses_duplicate(self):
        x = np.random.default_rng(7).normal(size=1000).astype(np.float32)
        y = np.random.default_rng(8).normal(size=1000).astype(np.float32)
        vad = [{"speech_detected": True, "speech_ratio": 0.9}] * 2
        for other, texts, expected in [
            (
                -x,
                ["같은 발화입니다", "같은 발화입니다"],
                "duplicate_waveform_and_transcript",
            ),
            (y, ["같은 발화입니다", "같은 발화입니다"], None),
            (x, ["오늘의 이야기", "다른 이야기"], None),
            (x, ["네", "네"], None),
        ]:
            diagnostic = build_window_diagnostic(
                speaker_0=x, speaker_1=other, vad_results=vad, transcripts=texts
            )
            self.assertEqual(
                secondary_transcript_suppression_reason(diagnostic, texts[1]), expected
            )

    def test_float_diagnostic_preserves_separator_peaks(self):
        x = np.array([0.0, 1.5, -1.6, 0.1], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            result = save_candidate_full_window_wavs(
                enabled=True,
                candidate=True,
                output_dir=Path(tmp),
                window=0,
                original=x,
                speaker_0=x,
                speaker_1=x,
                sample_rate=16000,
            )
            self.assertTrue(result["saved"])
            for path in Path(tmp).glob("*.wav"):
                actual, sr = sf.read(path, dtype="float32")
                np.testing.assert_array_equal(actual, x)
                self.assertEqual(sf.info(path).subtype, "FLOAT")

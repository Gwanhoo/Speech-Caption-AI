"""Synthetic input/separated audio + supplied STT; no GPU or recognizer claims."""
from __future__ import annotations

import copy
import unittest

import numpy as np

from secondary_leakage_diagnostics import admit_subtitle_streams
from speaker_tracking import PersistentSpeakerTracker
from subtitle_assembler import SpeakerSubtitleState, SubtitleAssembler

SR = 1600
SIZE = 3 * SR


def vad(*spans):
    duration = sum(end - start for start, end in spans)
    return {
        "speech_detected": bool(spans), "speech_ratio": duration / SIZE,
        "speech_duration_ms": duration * 1000 / SR,
        "timestamps": [{"start": start, "end": end} for start, end in spans],
    }


class SubtitleAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(429)
        self.primary = self.rng.normal(0, .1, SIZE).astype(np.float32)
        self.secondary = self.rng.normal(0, .1, SIZE).astype(np.float32)

    def admit(self, *, secondary=None, mixture=None, texts=None, vads=None,
              active=None, owner=0, assignment=None):
        waves = [self.primary, self.secondary if secondary is None else secondary]
        texts = texts or ["실제 본문 계속", "렁쇼"]
        vads = vads or [vad((0, SIZE)), vad((SR, SR + 700))]
        active = ["실제 본문", ""] if active is None else active
        if owner == 1:
            waves.reverse()
            texts = texts[::-1]
            vads = vads[::-1]
            active = active[::-1]
        return admit_subtitle_streams(
            mixture=self.primary if mixture is None else mixture,
            speakers=tuple(waves), transcripts=texts, vad_results=vads,
            active_hypotheses=active,
            speaker_assignment=assignment or {
                "raw_to_logical_mapping": {"0": owner, "1": 1 - owner},
                "active_raw_slots": [0], "assignment_method": "single_active_hold",
            },
        )

    def test_a_one_window_garbage_never_starts_or_finalizes_either_logical_id(self):
        for owner in (0, 1):
            for secondary in (self.secondary, -self.primary):
                with self.subTest(owner=owner, duplicated=secondary is not self.secondary):
                    texts, vads, evidence = self.admit(owner=owner, secondary=secondary)
                    self.assertTrue(evidence["streams"][1 - owner]["suppressed"])
                    assembler, state = SubtitleAssembler(1 - owner), SpeakerSubtitleState(1 - owner)
                    assembly = assembler.process(10, texts[1 - owner])
                    self.assertEqual(state.process(10, assembly.utterance_hypothesis,
                                                  vads[1 - owner]["speech_detected"], 23), [])
                    self.assertEqual(state.process(11, "", False, 25), [])
                    self.assertIsNone(state.flush(11, 25))
                    self.assertEqual(assembly.session_text, "")

    def test_b_two_or_more_windows_do_not_promote_even_stable_text(self):
        for fragments in (("렁쇼", "챔피언 롤러쯩"), ("같은 말입니다",) * 4):
            assembler, state = SubtitleAssembler(1), SpeakerSubtitleState(1)
            for window, text in enumerate(fragments):
                texts, vads, evidence = self.admit(texts=["실제 본문 계속", text],
                                                  active=["실제 본문", state.partial_text])
                self.assertTrue(evidence["streams"][1]["suppressed"])
                assembly = assembler.process(window, texts[1])
                self.assertEqual(state.process(window, assembly.utterance_hypothesis,
                                              vads[1]["speech_detected"], 3 + 2 * window), [])
            self.assertIsNone(state.flush(len(fragments), 20))
            self.assertEqual(state.utterance_id, 0)

    def test_c_real_simultaneous_speech_and_correlated_crosstalk_are_preserved(self):
        for amplitude in (1., .005):
            mixture = self.primary + amplitude * self.secondary
            for secondary in (self.secondary, -mixture):
                texts, vads, evidence = self.admit(
                    mixture=mixture, secondary=secondary,
                    vads=[vad((0, SIZE)), vad((0, SIZE))], active=["", ""],
                )
                self.assertEqual(texts, ["실제 본문 계속", "렁쇼"])
                self.assertFalse(evidence["applied"])
                self.assertTrue(all(row["speech_detected"] for row in vads))
                self.assertEqual(evidence["streams"][1]["reason"], "independent_input_support")

    def test_d_short_quiet_real_speaker_passes_immediately(self):
        quiet = np.zeros(SIZE, dtype=np.float32)
        quiet[SR:SR + 160] = self.secondary[SR:SR + 160] * .001
        for owner in (0, 1):
            texts, _, evidence = self.admit(
                secondary=quiet * 1000, mixture=self.primary + quiet,
                texts=["실제 본문 계속", "네"],
                vads=[vad((0, SIZE)), vad((SR, SR + 160))], owner=owner,
            )
            self.assertEqual(texts[1 - owner], "네")
            self.assertEqual(evidence["streams"][1 - owner]["reason"], "independent_input_support")
        # Even a waveform identical to the owner cannot justify removal when
        # the candidate speaks outside the owner's VAD intervals.
        texts, _, evidence = self.admit(secondary=self.primary,
            vads=[vad((0, SR)), vad((SR, SR + 160))])
        self.assertEqual(texts[1], "렁쇼")
        self.assertFalse(evidence["applied"])

    def test_e_permutation_preserves_logical_primary_without_reviving_residual(self):
        tracker = PersistentSpeakerTracker(overlap_samples=SR)
        states = [SpeakerSubtitleState(i) for i in (0, 1)]
        assemblers = [SubtitleAssembler(i) for i in (0, 1)]
        primary = self.primary
        for window in range(4):
            if window:
                primary = np.concatenate((primary[-SR:], self.rng.normal(0, .1, 2 * SR))).astype(np.float32)
            residual = self.rng.normal(0, .1, SIZE).astype(np.float32)
            swapped = window % 2 == 1
            assignment = tracker.assign(window=window,
                raw_speakers=(residual, primary) if swapped else (primary, residual), mixture=primary)
            self.assertEqual(assignment.diagnostic["raw_to_logical_mapping"],
                             {"0": int(swapped), "1": int(not swapped)})
            texts, vads, evidence = admit_subtitle_streams(
                mixture=primary, speakers=assignment.speakers,
                transcripts=["실제 본문 계속", "렁쇼" if window % 2 else "챔피언 롤러쯩"],
                vad_results=[vad((0, SIZE)), vad((SR, SR + 700))],
                active_hypotheses=[s.partial_text for s in states], speaker_assignment=assignment.diagnostic,
            )
            self.assertTrue(evidence["streams"][1]["suppressed"])
            for i in (0, 1):
                assembly = assemblers[i].process(window, texts[i])
                events = states[i].process(window, assembly.utterance_hypothesis,
                                           vads[i]["speech_detected"], 3 + 2 * window)
                self.assertEqual(bool(events), i == 0)

    def test_f_existing_boundary_and_fuzzy_corrections(self):
        for old, new, expected, match in (
            ("룰렛 돌려서", "룰렛 돌려서 AB 챔피언 3개, AP 챔피언 3개",
             "룰렛 돌려서 AB 챔피언 3개, AP 챔피언 3개", "boundary_exact"),
            ("챔피언 뽑아서 한 번쯤 챔피언쇠", "한 번쯤 챔피언 제외하고 다시 뽑고",
             "챔피언 뽑아서 한 번쯤 챔피언 제외하고 다시 뽑고", "fuzzy_replace"),
            ("챔피언 170", "챔피언 173개 중에서", "챔피언 173개 중에서", "fuzzy_replace"),
        ):
            with self.subTest(old=old):
                assembler = SubtitleAssembler(0)
                assembler.process(0, old)
                texts, _, _ = self.admit(texts=[new, "렁쇼"], active=[old, ""])
                event = assembler.process(1, texts[0])
                self.assertEqual(event.utterance_hypothesis, expected)
                self.assertEqual(event.match_type, match)

    def test_g_repetition_and_same_words_by_independent_speakers(self):
        for text in ("진짜 진짜", "가 가", "123 123", "같은 말입니다"):
            texts, _, evidence = self.admit(mixture=self.primary + self.secondary,
                                           texts=[text, text], active=["", ""])
            self.assertFalse(evidence["applied"])
            for i in (0, 1):
                assembler = SubtitleAssembler(i)
                assembler.process(0, texts[i])
                repeated = assembler.process(1, texts[i], shared_speech=False)
                self.assertEqual(repeated.utterance_hypothesis, text + " " + text)

    def test_nonlexical_bootstrap_is_symmetric_and_raw_inputs_unchanged(self):
        for owner in (0, 1):
            texts = ["실제 본문", "ㄱㄱㄱㄱ"]
            vads = [vad((0, SR)), vad((0, SR))]
            originals = copy.deepcopy((texts, vads))
            resolved, _, evidence = self.admit(texts=texts, vads=vads, owner=owner, active=["", ""])
            self.assertEqual(resolved[1 - owner], "")
            self.assertEqual(evidence["streams"][1 - owner]["reason"], "nonlexical_new_stream")
            self.assertEqual((texts, vads), originals)

    def test_vad_disabled_preserves_inputs(self):
        texts, vads, evidence = admit_subtitle_streams(
            mixture=self.primary, speakers=(self.primary, self.secondary),
            transcripts=["실제 본문", "두 번째 발화"], vad_results=[],
            active_hypotheses=["", ""], speaker_assignment={},
        )
        self.assertEqual(texts, ["실제 본문", "두 번째 발화"])
        self.assertEqual(vads, [])
        self.assertFalse(evidence["applied"])
        self.assertEqual(evidence["reason"], "vad_unavailable")

    def test_missing_or_invalid_evidence_and_existing_partial_fail_open(self):
        for overrides in (
            {"vads": [vad((0, SIZE)), dict(vad((0, SR)), timestamps=[])]},
            {"vads": [vad((0, SIZE)), dict(vad((0, SR)), timestamps=[{"start": -1, "end": SR}])]},
            {"secondary": np.full(SIZE, np.nan)},
            {"active": ["실제 본문", "실제 두 번째 화자"]},
            {"assignment": {"raw_to_logical_mapping": {"0": 0, "1": 1}, "active_raw_slots": [0, 1]}},
        ):
            texts, _, evidence = self.admit(**overrides)
            self.assertFalse(evidence["applied"])
            self.assertEqual(texts[1], "렁쇼")

    def test_independent_interval_protects_whole_utterance_and_next_valid_speech_starts_cleanly(self):
        mix = self.primary.copy()
        mix[SR:2 * SR] += self.secondary[SR:2 * SR] * .01
        texts, _, evidence = self.admit(mixture=mix,
            vads=[vad((0, SIZE)), vad((0, SR), (SR, 2 * SR))])
        self.assertFalse(evidence["applied"])
        self.assertEqual(texts[1], "렁쇼")
        assembler, state = SubtitleAssembler(1), SpeakerSubtitleState(1)
        rejected, effective, _ = self.admit()
        assembly = assembler.process(0, rejected[1])
        self.assertEqual(state.process(0, assembly.utterance_hypothesis, effective[1]["speech_detected"], 3), [])
        accepted, effective, _ = self.admit(mixture=self.primary + self.secondary,
                                            texts=["실제 본문", "네 맞아요"])
        assembly = assembler.process(1, accepted[1])
        event = state.process(1, assembly.utterance_hypothesis, effective[1]["speech_detected"], 5)[0]
        self.assertEqual(event.text, "네 맞아요")
        self.assertEqual(event.action, "start")


if __name__ == "__main__":
    unittest.main()

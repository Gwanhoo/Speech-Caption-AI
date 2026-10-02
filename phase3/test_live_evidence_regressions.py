"""Deterministic audio/evidence regressions, not recordings or STT accuracy tests."""
from __future__ import annotations

import unittest

import numpy as np

from secondary_leakage_diagnostics import admit_subtitle_streams, weak_speech_evidence
from speaker_tracking import PersistentSpeakerTracker, safe_absolute_correlation
from subtitle_assembler import SpeakerSubtitleState, SubtitleAssembler


SR = 1600
SIZE = 3 * SR


def vad(start_ms=0, duration_ms=0):
    return {
        "speech_detected": duration_ms > 0,
        "speech_duration_ms": duration_ms,
        "speech_ratio": duration_ms / 3000,
        "timestamps": ([{"start": round(start_ms * SR / 1000),
                        "end": round((start_ms + duration_ms) * SR / 1000)}]
                       if duration_ms else []),
    }


def normalize(audio, rms=.088232):
    return (audio * (rms / np.sqrt(np.mean(audio ** 2)))).astype(np.float32)


class LiveEvidenceRegressions(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1002)
        self.primary = normalize(self.rng.normal(size=SIZE))
        self.artifact = normalize(self.rng.normal(size=SIZE))

    def admission(self, *, duration, primary_duration, start=0, owner=0,
                  primary_text="안내를 계속합니다", genuine=False):
        local = vad(start, duration)
        candidate = self.primary.copy()
        for span in local["timestamps"]:
            candidate[span["start"]:span["end"]] = self.artifact[span["start"]:span["end"]]
        # Outside the VAD span the channel leaks the primary. Thus it really
        # qualifies as active by full-window tracking, despite local artifacts.
        candidate = normalize(candidate)
        mixture = self.primary.copy()
        if genuine:
            # A quiet independent source may exist for just one short interval.
            for span in local["timestamps"]:
                mixture[span["start"]:span["end"]] += .005 * candidate[span["start"]:span["end"]]
        waves = [self.primary, candidate]
        texts = [primary_text, "별도의 짧은 발화"]
        vads = [vad(0, primary_duration), local]
        if owner == 1:
            waves.reverse()
            texts.reverse()
            vads.reverse()
        assignment = PersistentSpeakerTracker(overlap_samples=SR).assign(
            window=2, raw_speakers=tuple(waves), mixture=mixture)
        self.assertEqual(assignment.diagnostic["active_raw_slots"], [0, 1])
        return admit_subtitle_streams(
            mixture=mixture, speakers=tuple(waves), transcripts=texts,
            vad_results=vads, active_hypotheses=["", ""],
            speaker_assignment=assignment.diagnostic,
        )

    def test_vad_positive_artifact_cannot_reach_partial_silence_final_or_flush(self):
        # Live durations/ratios, with explicit synthetic timing/acoustic evidence.
        # The real logs do not contain timestamps or local source correlations.
        for duration, primary_duration, start, primary_text in (
            (508, 1718, 1400, "안내를 계속합니다"),  # not contained in owner VAD
            (1144, 3000, 1000, "안내를 계속합니다"),
            (764, 0, 1200, ""),  # owner waveform still explains input, VAD/STT absent
        ):
            for owner in (0, 1):
                with self.subTest(duration=duration, owner=owner):
                    self.assertFalse(weak_speech_evidence(self.primary, vad(start, duration))["suppressed"])
                    texts, effective, diagnostic = self.admission(
                        duration=duration, primary_duration=primary_duration,
                        start=start, owner=owner, primary_text=primary_text)
                    candidate = 1 - owner
                    assembler, state = SubtitleAssembler(candidate), SpeakerSubtitleState(candidate)
                    assembly = assembler.process(2, texts[candidate])
                    partials = state.process(2, assembly.utterance_hypothesis,
                        effective[candidate]["speech_detected"], 7,
                        confirmed_prefix_length=assembly.confirmed_prefix_length)
                    finals = state.process(3, "", False, 9)
                    self.assertEqual(finals, [])
                    self.assertEqual(partials, [])
                    self.assertIsNone(state.flush(3, 9))
                    self.assertEqual(state.final_segments, [])
                    self.assertEqual(assembly.session_text, "")
                    self.assertTrue(diagnostic["streams"][candidate]["suppressed"])

    def test_one_window_genuine_short_speech_with_same_vad_survives_final_and_flush(self):
        for duration in (100, 508, 764, 1144):
            for owner in (0, 1):
                for flush in (False, True):
                    with self.subTest(duration=duration, owner=owner, flush=flush):
                        texts, effective, diagnostic = self.admission(
                            duration=duration, primary_duration=0, start=1200,
                            owner=owner, primary_text="", genuine=True)
                        candidate = 1 - owner
                        self.assertFalse(diagnostic["streams"][candidate]["suppressed"])
                        assembler, state = SubtitleAssembler(candidate), SpeakerSubtitleState(candidate)
                        assembly = assembler.process(2, texts[candidate])
                        partial = state.process(2, assembly.utterance_hypothesis,
                            effective[candidate]["speech_detected"], 7,
                            confirmed_prefix_length=assembly.confirmed_prefix_length)[0]
                        self.assertEqual(partial.stable_text, "")  # no second window required
                        final = state.flush(2, 7) if flush else state.process(3, "", False, 9)[0]
                        self.assertEqual(final.text, texts[candidate])
                        self.assertEqual(final.status, "final")

    def test_window_004_to_005_overlap_artifact_cannot_take_continuous_stream(self):
        for permuted in (False, True):
            with self.subTest(permuted=permuted):
                tracker = PersistentSpeakerTracker(overlap_samples=SR)
                tracker.assign(window=3, raw_speakers=(self.primary, self.artifact), mixture=self.primary)
                fourth = np.concatenate((self.primary[-SR:], normalize(self.rng.normal(size=2 * SR))))
                held = tracker.assign(window=4, raw_speakers=(fourth, self.artifact), mixture=fourth)
                self.assertEqual(held.diagnostic["assignment_method"], "single_active_hold")
                mixture = np.concatenate((fourth[-SR:], normalize(self.rng.normal(size=2 * SR))))
                dominant = mixture.copy()
                dominant[:SR] = .02 * (mixture[:SR] + 1.5 * normalize(self.rng.normal(size=SR)))
                residual = normalize(self.rng.normal(size=SIZE))
                residual[:SR] = 3 * mixture[:SR]  # perfect old speech, unrelated new audio
                dominant, residual = normalize(dominant), normalize(residual)
                self.assertLess(safe_absolute_correlation(dominant, residual), .05)
                raw = (residual, dominant) if permuted else (dominant, residual)
                result = tracker.assign(window=5, raw_speakers=raw, mixture=mixture)
                old_winner = "identity_score" if permuted else "swap_score"
                old_loser = "swap_score" if permuted else "identity_score"
                self.assertGreater(result.diagnostic[old_winner], result.diagnostic[old_loser] + .12)
                self.assertEqual(result.diagnostic["raw_to_logical_mapping"],
                                 {"0": int(permuted), "1": int(not permuted)})
                self.assertTrue(result.diagnostic["single_source_continuity"]["applied"])
                self.assertEqual(result.diagnostic["excluded_residual_tail_reference"], 1)
                np.testing.assert_array_equal(result.speakers[0], dominant)
                # Next actual permutation must continue to follow the audio.
                sixth = np.concatenate((mixture[-SR:], normalize(self.rng.normal(size=2 * SR))))
                following = tracker.assign(window=6, raw_speakers=(self.artifact, sixth), mixture=sixth)
                self.assertEqual(following.diagnostic["raw_to_logical_mapping"], {"0": 1, "1": 0})

    def test_independent_new_audio_prevents_override_even_with_shared_speech_heads(self):
        tracker = PersistentSpeakerTracker(overlap_samples=SR)
        tracker.assign(window=4, raw_speakers=(self.primary, self.artifact), mixture=self.primary)
        original = np.concatenate((self.primary[-SR:], normalize(self.rng.normal(size=2 * SR))))
        primary = original.copy()
        primary[:SR] = .02 * (original[:SR] + 1.5 * self.artifact[:SR])
        secondary = self.artifact.copy()
        secondary[:SR] = 3 * original[:SR]
        mixture = original.copy()
        mixture[SR:] += .005 * secondary[SR:]  # real quiet second source in new audio
        result = tracker.assign(window=5, raw_speakers=(primary, secondary), mixture=mixture)
        self.assertGreater(result.diagnostic["swap_score"], result.diagnostic["identity_score"] + .12)
        guard = result.diagnostic["single_source_continuity"]
        self.assertGreater(guard["new_region_source_evidence"]["0"]["independent_input_correlation"], .99)
        self.assertFalse(guard["applied"])
        self.assertIsNone(guard["selected_raw"])
        self.assertIsNone(result.diagnostic["excluded_residual_tail_reference"])
        self.assertTrue(all(value >= .45 for value in result.diagnostic["next_reference_confidence"]))

    def test_single_to_genuine_simultaneous_sources_keeps_both_identities(self):
        tracker = PersistentSpeakerTracker(overlap_samples=SR)
        tracker.assign(window=4, raw_speakers=(self.primary, self.artifact), mixture=self.primary)
        a = np.concatenate((self.primary[-SR:], normalize(self.rng.normal(size=2 * SR))))
        b = normalize(self.rng.normal(size=SIZE))
        result = tracker.assign(window=5, raw_speakers=(b, a), mixture=a + b)
        self.assertEqual(result.diagnostic["raw_to_logical_mapping"], {"0": 1, "1": 0})
        np.testing.assert_array_equal(result.speakers[0], a)
        np.testing.assert_array_equal(result.speakers[1], b)
        texts, effective, diagnostic = admit_subtitle_streams(
            mixture=a + b, speakers=result.speakers,
            transcripts=["첫 번째 이야기", "동시에 다른 이야기"],
            vad_results=[vad(0, 3000), vad(0, 3000)],
            active_hypotheses=["", ""], speaker_assignment=result.diagnostic)
        self.assertFalse(diagnostic["applied"])
        for speaker in (0, 1):
            state = SpeakerSubtitleState(speaker)
            self.assertEqual(state.process(5, texts[speaker], effective[speaker]["speech_detected"], 13)[0].status, "partial")


if __name__ == "__main__":
    unittest.main()

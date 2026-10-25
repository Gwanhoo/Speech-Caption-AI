"""Geometry, provenance and recorded production-regression checks for R2-B6."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np
import soundfile as sf

from compare_windows import CONFIGS, EXPECTED_OPTIONS, ROOT, SR, Replay, starts, validate_health


class ExperimentChecks(unittest.TestCase):
    def test_production_full_window_geometry_and_tail(self):
        expected = [(19, 39, 1), (19, 40, 0), (13, 40, 0), (12, 38, 2)]
        for (_, window, stride), (count, end, tail) in zip(CONFIGS, expected):
            positions = list(starts(40 * SR, window, stride))
            self.assertEqual(len(positions), count)
            self.assertEqual(positions[-1] + window * SR, end * SR)
            self.assertEqual(40 * SR - positions[-1] - window * SR, tail * SR)
            self.assertTrue(all(b - a == stride * SR for a, b in zip(positions, positions[1:])))
        self.assertEqual(list(starts(SR, 3, 2)), [])
        with self.assertRaises(ValueError):
            starts(40 * SR, 3, 3)

    def test_mismatched_model_or_decoding_fails_before_inference(self):
        health = dict(gpu_ready=True, status='ready', processing_profile='single',
                      stt_model='large-v3-turbo', stt_options=EXPECTED_OPTIONS,
                      requests={'waiting': 0})
        validate_health(health)
        for key, value in [('stt_model', 'base'), ('processing_profile', 'overlap'),
                           ('stt_options', {**EXPECTED_OPTIONS, 'beam_size': 1}),
                           ('requests', {'waiting': 1})]:
            bad = copy.deepcopy(health)
            bad[key] = value
            with self.assertRaises(RuntimeError):
                validate_health(bad)

    def test_real_evidence_through_complete_local_pipeline_preserves_known_fix(self):
        evidence = ROOT / 'phase3/output/evidence/news_20261022_160827'
        if not evidence.exists():
            self.skipTest('Original evidence not available')
        audio, sr = sf.read(evidence / 'original.wav', dtype='float32')
        self.assertEqual(sr, SR)
        timeline = [json.loads(x) for x in (evidence / 'timeline.jsonl').read_text().splitlines()]
        replay = Replay(3, 2)
        rows = []
        for row in timeline:
            if row['record_type'] != 'window':
                continue
            start, end = row['audio']['sample_start'], row['audio']['sample_end']
            chunk = audio[start:end]
            slots = [{'raw_transcript': text, 'vad': vad} for text, vad in
                     zip(row['raw_stt']['remote_raw_slots'], row['vad'])]
            result = replay.process(row['window_id'], end / SR, chunk,
                                    (chunk, np.zeros_like(chunk)), slots, not np.any(chunk))
            self.assertEqual(result['primary'], 0)
            self.assertEqual(result['raw'], row['raw_stt']['logical_speakers'])
            self.assertEqual(result['admitted'], row['admitted_stt'])
            rows.append(result)
        # Existing news fixture regression: fuzzy overlap must retain '하거나'.
        self.assertIn('실행하거나 시스템을 장악할 수', rows[9]['assembly']['utterance_hypothesis'])
        self.assertNotIn('실행... 시스템', rows[9]['assembly']['utterance_hypothesis'])
        # Acoustic overlap is calculated from original samples, not fabricated.
        self.assertEqual(rows[9]['overlap_evidence']['overlap_correlation'], 1.0)
        self.assertTrue(rows[9]['overlap_evidence']['shared_speech'])
        final = replay.flush(18, 39)
        self.assertTrue(final)
        self.assertEqual(final[-1]['finalize_reason'], 'pipeline_end')
        self.assertEqual(replay.flush(18, 39), [])


if __name__ == '__main__':
    unittest.main()

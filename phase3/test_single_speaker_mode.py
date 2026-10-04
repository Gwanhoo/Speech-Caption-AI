from __future__ import annotations

import unittest

try:
    from .speaker_mode import SpeakerMode, select_single_mode_primary
except ImportError:
    from speaker_mode import SpeakerMode, select_single_mode_primary


class SingleSpeakerModeTests(unittest.TestCase):
    def test_mode_values_are_stable_for_config_and_json(self) -> None:
        self.assertEqual(SpeakerMode.SINGLE.value, "single")
        self.assertEqual(SpeakerMode.OVERLAP.value, "overlap")

    def test_primary_is_selected_after_raw_to_logical_mapping(self) -> None:
        primary, evidence = select_single_mode_primary(
            {
                "raw_to_logical_mapping": {"0": 1, "1": 0},
                "raw_input_similarity": {"0": 0.98, "1": 0.31},
                "active_raw_slots": [0, 1],
                "next_reference_confidence": [0.31, 0.98],
            },
            [
                {"speech_detected": True, "speech_duration_ms": 800, "speech_ratio": .3},
                {"speech_detected": True, "speech_duration_ms": 2200, "speech_ratio": .8},
            ],
        )
        self.assertEqual(primary, 1)
        self.assertEqual(evidence["mapped_input_similarity"], [0.31, 0.98])


if __name__ == "__main__":
    unittest.main()

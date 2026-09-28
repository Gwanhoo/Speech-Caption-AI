from __future__ import annotations

from run_overlap_pipeline import build_diagnostic_audio_window_record


def main() -> None:
    persisted_window = {
        "window": 7,
        "capture_completed_timestamp": "2026-09-28T00:00:00+09:00",
        "separation_input_stats": {"rms": 0.25, "peak": 0.75},
        "speaker_rms": [0.2, 0.1],
        "speaker_peak": [0.8, 0.6],
        # These are the canonical persisted names produced in the window metric.
        "raw_slot_rms": [0.21, 0.11],
        "raw_slot_peak": [0.81, 0.61],
        "speaker_assignment": {
            "raw_to_logical_mapping": {"0": 1, "1": 0},
            "assignment_method": "overlap_continuity",
        },
        "vad": [{"speech_detected": True}, {"speech_detected": False}],
        "pre_separation_silence": False,
    }
    record = build_diagnostic_audio_window_record(persisted_window)
    assert record["raw_slot_0_rms"] == 0.21
    assert record["raw_slot_1_rms"] == 0.11
    assert record["speaker_0_rms"] == 0.2
    assert record["speaker_assignment"]["raw_to_logical_mapping"] == {"0": 1, "1": 0}
    print("overlap diagnostic schema regression: PASS")


if __name__ == "__main__":
    main()

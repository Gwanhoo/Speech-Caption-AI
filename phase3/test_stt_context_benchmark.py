from __future__ import annotations

from benchmark_stt_context import (
    edit_distance,
    evaluation_category,
    expected_active,
    score_text,
    window_starts,
)
from analyze_stt_context_benchmark import best_substring_edit_distance


def main() -> int:
    assert window_starts(20.0, 3.0) == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 17.0]
    assert window_starts(15.0, 5.0) == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    assert window_starts(2.0, 3.0) == []

    assert edit_distance("토마토", "토마트") == 1
    assert best_substring_edit_distance("오늘은 토마토 파스타", "토마토") == 0
    assert best_substring_edit_distance("오늘은 토마토 파스타", "토마트") == 1
    score = score_text("토마토 파스타", "토마트 파스타")
    assert score["character_edits"] == 1
    assert score["reference_characters"] == 6

    assert expected_active("original", "A_only")
    assert expected_active("logical_speaker_0", "B_only")
    assert expected_active("logical_speaker_1", "A+B")
    assert not expected_active("logical_speaker_1", "A_only")
    assert not expected_active("original", "silence_0")
    assert evaluation_category("logical_speaker_0", "A_only") == "separated_single"
    assert evaluation_category("logical_speaker_1", "A_only") == "separated_residual"
    assert evaluation_category("logical_speaker_1", "A+B") == "separated_overlap"

    print("STT context benchmark regression: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

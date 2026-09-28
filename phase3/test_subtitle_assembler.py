from __future__ import annotations

from subtitle_assembler import SubtitleAssembler, normalize_for_matching


def process_pair(previous: str, current: str):
    assembler = SubtitleAssembler(speaker=0)
    assembler.process(0, previous)
    return assembler.process(1, current)


def main() -> int:
    exact = process_pair(
        "오늘은 전국적으로 맑은 날씨가 이어지겠습니다",
        "맑은 날씨가 이어지겠습니다 서울의 낮 기온은",
    )
    assert exact.new == "서울의 낮 기온은", exact

    duplicate = process_pair(
        "데이터베이스는 여러 사용자가 필요한 정보를 관리합니다",
        "필요한 정보를 관리합니다",
    )
    assert duplicate.new == "" and duplicate.duplicate_only, duplicate

    spacing = process_pair(
        "데이터베이스는 여러 사용자가 필요한 정보를",
        "데이터 베이스는 여러 사용자가 필요한 정보를 효율적으로",
    )
    assert spacing.new == "효율적으로", spacing
    assert spacing.match_type in {"normalized_exact", "fuzzy"}, spacing

    unrelated = process_pair(
        "오늘은 전국적으로 맑은 날씨입니다",
        "데이터베이스는 정보를 저장하는 시스템입니다",
    )
    assert unrelated.match_type == "none" and unrelated.new == unrelated.raw, unrelated

    short_utterance = process_pair("오늘 저녁 먹었어?", "아니")
    assert short_utterance.new == "아니", short_utterance

    korean_suffix = process_pair(
        "안녕하세요. 오늘은 아침",
        "오늘은 아침에 일어나서 간단하게 식사를 하고",
    )
    assert normalize_for_matching(korean_suffix.utterance_hypothesis).count(
        "오늘은아침"
    ) == 1, korean_suffix

    punctuation_boundary = process_pair(
        "다 새로운 음식을 만들어보는 것을 좋아합니다",
        "좋아합니다. 지난주에는 집에서 토마트",
    )
    assert normalize_for_matching(punctuation_boundary.utterance_hypothesis).count(
        "좋아합니다"
    ) == 1, punctuation_boundary

    repeated_inside_fragment = SubtitleAssembler(speaker=0).process(
        0, "진짜 진짜 맛있다"
    )
    assert repeated_inside_fragment.utterance_hypothesis == "진짜 진짜 맛있다"

    speaker_0 = SubtitleAssembler(speaker=0)
    speaker_1 = SubtitleAssembler(speaker=1)
    speaker_0.process(0, "오늘은 전국적으로 맑은 날씨가 이어지겠습니다")
    speaker_1.process(0, "데이터베이스는 정보를 저장하는 시스템입니다")
    assert speaker_0.process(1, "맑은 날씨가 이어지겠습니다").duplicate_only
    assert speaker_1.process(1, "정보를 저장하는 시스템입니다").duplicate_only

    print("subtitle assembler regression: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

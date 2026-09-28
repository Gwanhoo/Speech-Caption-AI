from __future__ import annotations

import argparse
import difflib
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from subtitle_assembler import SubtitleAssembler, normalize_for_matching


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "phase3" / "output" / "overlap_3s_2s.json"
DEFAULT_OUTPUT = ROOT / "phase3" / "output" / "assembler_replay.json"


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": float(np.percentile(values, 95)),
        "max": max(values),
    }


def fuzzy_boundary_candidate(previous: str, current: str) -> dict[str, Any] | None:
    left = normalize_for_matching(previous)[-16:]
    right = normalize_for_matching(current)[:16]
    if not left or not right:
        return None
    match = difflib.SequenceMatcher(None, left, right).find_longest_match()
    near_left_end = len(left) - (match.a + match.size) <= 2
    near_right_start = match.b <= 2
    if match.size < 4 or not near_left_end or not near_right_start:
        return None
    return {
        "candidate": right[match.b : match.b + match.size],
        "length": match.size,
        "previous": previous,
        "current": current,
    }


def replay(raw: dict[str, Any], minimum_characters: int) -> dict[str, Any]:
    assemblers = [
        SubtitleAssembler(speaker=speaker, minimum_characters=minimum_characters)
        for speaker in (0, 1)
    ]
    events: list[dict[str, Any]] = []
    missed_fuzzy_candidates: list[dict[str, Any]] = []
    for window in sorted(raw["windows"], key=lambda item: item["window"]):
        for speaker, transcript in enumerate(window["transcripts"]):
            event = assemblers[speaker].process(window["window"], transcript)
            event_dict = event.to_dict()
            events.append(event_dict)
            if event.match_type == "none" and event.previous and event.raw:
                candidate = fuzzy_boundary_candidate(event.previous, event.raw)
                if candidate:
                    missed_fuzzy_candidates.append(
                        {
                            "window": event.window,
                            "speaker": speaker,
                            **candidate,
                        }
                    )

    matches = [event for event in events if event["match_type"] != "none"]
    timings = [event["matching_seconds"] for event in events]
    return {
        "minimum_characters": minimum_characters,
        "event_count": len(events),
        "overlap_detected_count": len(matches),
        "removed_character_count": sum(event["overlap_length"] for event in matches),
        "duplicate_only_count": sum(event["duplicate_only"] for event in events),
        "new_text_count": sum(bool(event["new"]) for event in events),
        "suspicious_deletion_count": sum(event["suspicious_deletion"] for event in events),
        "matching_time": distribution(timings),
        "match_type_counts": {
            match_type: sum(event["match_type"] == match_type for event in events)
            for match_type in ("raw_exact", "normalized_exact", "none")
        },
        "missed_fuzzy_candidates": missed_fuzzy_candidates,
        "assembled_by_speaker": [assembler.assembled for assembler in assemblers],
        "events": events,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline replay for Phase 3-E assembler")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-characters", type=int, default=6)
    parser.add_argument("--thresholds", type=int, nargs="*", default=list(range(2, 9)))
    args = parser.parse_args()

    raw = json.loads(args.input.read_text(encoding="utf-8"))
    threshold_comparison = []
    for threshold in args.thresholds:
        result = replay(raw, threshold)
        threshold_comparison.append(
            {
                key: result[key]
                for key in (
                    "minimum_characters",
                    "event_count",
                    "overlap_detected_count",
                    "removed_character_count",
                    "duplicate_only_count",
                    "new_text_count",
                    "suspicious_deletion_count",
                    "match_type_counts",
                )
            }
        )
    selected = replay(raw, args.minimum_characters)
    report = {
        "phase": "3-E-offline-replay",
        "source": str(args.input),
        "selected_threshold": args.minimum_characters,
        "fuzzy_matching_applied": False,
        "threshold_comparison": threshold_comparison,
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(threshold_comparison, ensure_ascii=False, indent=2))
    print(
        f"Selected threshold={args.minimum_characters}: matches="
        f"{selected['overlap_detected_count']}/{selected['event_count']}, "
        f"removed={selected['removed_character_count']} chars, "
        f"duplicate-only={selected['duplicate_only_count']}, "
        f"suspicious={selected['suspicious_deletion_count']}, "
        f"fuzzy candidates not applied={len(selected['missed_fuzzy_candidates'])}"
    )
    print(f"JSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

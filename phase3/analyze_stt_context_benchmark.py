from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from subtitle_assembler import normalize_for_matching


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "phase3" / "output" / "stt_context_benchmark.json"
DEFAULT_OUTPUT = ROOT / "phase3" / "output" / "stt_context_benchmark_analysis.json"


def best_substring_edit_distance(reference: str, hypothesis: str) -> int:
    """Levenshtein distance from hypothesis to its best substring in reference."""
    if not hypothesis:
        return 0
    if not reference:
        return len(hypothesis)
    previous = [0] * (len(reference) + 1)
    for hypothesis_index, hypothesis_character in enumerate(hypothesis, 1):
        current = [hypothesis_index]
        for reference_index, reference_character in enumerate(reference, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[reference_index] + 1,
                    previous[reference_index - 1]
                    + (hypothesis_character != reference_character),
                )
            )
        previous = current
    return min(previous)


def distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": float(np.percentile(values, 95)),
        "max": max(values),
    }


def analyze_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    active = [sample for sample in samples if sample["expected_active"]]
    inactive = [sample for sample in samples if not sample["expected_active"]]
    calls = [call for sample in active for call in sample["calls"]]
    elapsed = [call["elapsed_seconds"] for call in calls]
    rtfs = [call["rtf"] for call in calls]

    substring_edits = 0
    substring_hypothesis_characters = 0
    output_length_ratios: list[float] = []
    boundary_events: list[dict[str, Any]] = []
    for sample in active:
        reference = normalize_for_matching(sample["pseudo_reference"] or "")
        hypothesis = normalize_for_matching(sample["assembled_text"])
        if reference:
            output_length_ratios.append(len(hypothesis) / len(reference))
        for call in sample["calls"]:
            call_text = normalize_for_matching(call["transcript"])
            if call_text and reference:
                substring_edits += best_substring_edit_distance(reference, call_text)
                substring_hypothesis_characters += len(call_text)
        boundary_events.extend(sample["assembly_events"][1:])

    suspicious_examples = []
    for sample in active:
        for call in sample["calls"]:
            text = call["transcript"]
            if re.search(r"(?:\.{2,}|…)$", text.strip()):
                suspicious_examples.append(
                    {
                        "track": sample["track"],
                        "region": sample["region"],
                        "start_seconds": call["start_seconds"],
                        "text": text,
                    }
                )

    return {
        "active_call_count": len(calls),
        "active_stt_seconds": distribution(elapsed),
        "active_rtf": distribution(rtfs),
        "calls_over_two_second_stride": sum(value > 2.0 for value in elapsed),
        "calls_with_rtf_at_least_one": sum(value >= 1.0 for value in rtfs),
        "pseudo_reference_best_substring_character_error_ratio": (
            substring_edits / substring_hypothesis_characters
            if substring_hypothesis_characters
            else None
        ),
        "assembled_to_pseudo_reference_length_ratio": distribution(output_length_ratios),
        "boundary_event_count_excluding_first_windows": len(boundary_events),
        "boundary_overlap_detected_count": sum(
            event["match_type"] != "none" for event in boundary_events
        ),
        "boundary_no_match_count": sum(
            event["match_type"] == "none" for event in boundary_events
        ),
        "boundary_no_match_rate": (
            sum(event["match_type"] == "none" for event in boundary_events)
            / len(boundary_events)
            if boundary_events
            else None
        ),
        "obvious_truncation_marker_count": len(suspicious_examples),
        "obvious_truncation_examples": suspicious_examples[:10],
        "inactive_sample_count": len(inactive),
        "inactive_nonempty_transcript_count": sum(
            bool(sample["assembled_text"]) for sample in inactive
        ),
        "prompt_applied_to_inactive_count": sum(
            call["prompt_applied"]
            for sample in inactive
            for call in sample["calls"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze saved P2-A STT context results")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    configs = []
    for config in raw["configs"]:
        categories = sorted({sample["category"] for sample in config["samples"]})
        configs.append(
            {
                "name": config["name"],
                "overall": analyze_samples(config["samples"]),
                "by_category": {
                    category: analyze_samples(
                        [sample for sample in config["samples"] if sample["category"] == category]
                    )
                    for category in categories
                },
            }
        )
    report = {
        "phase": "P2-A-analysis",
        "source": str(args.input),
        "metric_warning": "All text error metrics use model-generated pseudo-references, not ground truth.",
        "configs": configs,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"JSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

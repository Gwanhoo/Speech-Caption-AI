from __future__ import annotations

from enum import Enum
from typing import Any, Sequence


class SpeakerMode(str, Enum):
    """User-selected subtitle source policy."""

    SINGLE = "single"
    OVERLAP = "overlap"

    def __str__(self) -> str:
        return self.value


def select_single_mode_primary(
    speaker_assignment: dict[str, Any],
    vad_results: Sequence[dict[str, Any]],
) -> tuple[int | None, dict[str, Any]]:
    """Choose one logical stream using the tracker's mapped source evidence."""

    mapping = speaker_assignment.get("raw_to_logical_mapping", {})
    raw_similarity = speaker_assignment.get("raw_input_similarity", {})
    mapped_similarity = [0.0, 0.0]
    active_logical: set[int] = set()
    for raw_slot in (0, 1):
        logical = mapping.get(str(raw_slot))
        if logical not in (0, 1):
            continue
        value = raw_similarity.get(str(raw_slot))
        if isinstance(value, (int, float)):
            mapped_similarity[logical] = float(value)
        if raw_slot in speaker_assignment.get("active_raw_slots", []):
            active_logical.add(logical)

    reference_confidence = speaker_assignment.get("next_reference_confidence", [0.0, 0.0])
    if not isinstance(reference_confidence, (list, tuple)) or len(reference_confidence) != 2:
        reference_confidence = [0.0, 0.0]
    reference_confidence = [
        float(value) if isinstance(value, (int, float)) else 0.0
        for value in reference_confidence
    ]
    vad_active = {
        speaker for speaker in (0, 1)
        if speaker < len(vad_results) and vad_results[speaker].get("speech_detected") is True
    }
    candidates = vad_active or active_logical
    if not candidates:
        candidates = {
            speaker for speaker in (0, 1)
            if max(reference_confidence[speaker], mapped_similarity[speaker]) > 0.0
        }
    if not candidates:
        return None, {
            "method": "no_active_logical_stream",
            "active_logical_streams": [],
            "vad_active_logical_streams": [],
            "next_reference_confidence": reference_confidence,
            "mapped_input_similarity": mapped_similarity,
        }

    def rank(speaker: int) -> tuple[float, float, float, float, int]:
        vad = vad_results[speaker] if speaker < len(vad_results) else {}
        duration = vad.get("speech_duration_ms", 0.0)
        ratio = vad.get("speech_ratio", 0.0)
        return (
            reference_confidence[speaker],
            mapped_similarity[speaker],
            float(duration) if isinstance(duration, (int, float)) else 0.0,
            float(ratio) if isinstance(ratio, (int, float)) else 0.0,
            -speaker,
        )

    primary = max(candidates, key=rank)
    return primary, {
        "method": "logical_tracking_source_evidence",
        "active_logical_streams": sorted(active_logical),
        "vad_active_logical_streams": sorted(vad_active),
        "next_reference_confidence": reference_confidence,
        "mapped_input_similarity": mapped_similarity,
        "selected_rank": list(rank(primary)[:-1]),
    }

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

from speaker_tracking import (
    DEFAULT_SILENCE_RMS, DEFAULT_SOURCE_CONFIDENCE, HIGH_SOURCE_CORRELATION,
    MAX_UNSUPPORTED_INPUT_CORRELATION, MIN_DOMINANT_INPUT_CORRELATION,
    safe_absolute_correlation,
    source_input_evidence, unsupported_source,
)
from subtitle_assembler import append_preserving_text


ENERGY_EPSILON = 1e-12
SECONDARY_ARTIFACT_CONFIG = {
    # Diagnostic labels, also reused by subtitle routing ONLY with independent
    # waveform, lexical, timing and existing-utterance evidence below.
    "primary_vad_ratio_min": 0.60,
    "secondary_vad_ratio_max": 0.50,
    "secondary_fragment_max_normalized_characters": 12,
    "high_abs_waveform_correlation": HIGH_SOURCE_CORRELATION,
}
RMS_NORMALIZATION_NOTE = (
    "ClearVoice normalizes every separated output to the input RMS; "
    "speaker RMS ratios are recorded but are not reliable artifact features."
)
_NORMALIZE_PATTERN = re.compile(r"[^0-9A-Za-z가-힣]+")
TEMPORAL_CONTINUITY_CONFIG = {
    # These are positive-evidence thresholds only. They never reject a lexical window.
    "exact_overlap_min_characters": 2,
    "fuzzy_overlap_min_characters": 4,
    "fuzzy_overlap_similarity": 0.72,
    "strong_secondary_min_consecutive_windows": 3,
    "strong_secondary_min_vad_ratio": 0.75,
}
# A correlation direction is not useful source evidence when an established
# owner already explains virtually all of the input.  The live ghost fixture
# left 0.008117 of normalized input energy after owner projection while still
# producing 0.857164 residual correlation.  Keep a round, auditable 1% energy
# boundary above that observation.  This gate is only applied when tracking or
# an active hypothesis has already identified the other stream as the owner;
# isolated quiet speech and unowned first speech therefore keep the fail-open
# path.
MIN_INDEPENDENT_INPUT_ENERGY_FRACTION = 0.01
WEAK_SPEECH_MAX_DURATION_MS = 500


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def safe_waveform_correlation(
    speaker_0: np.ndarray, speaker_1: np.ndarray, epsilon: float = ENERGY_EPSILON
) -> tuple[float | None, float | None]:
    """Return Pearson correlation and its absolute value, or null when undefined."""
    left = np.asarray(speaker_0, dtype=np.float64).reshape(-1)
    right = np.asarray(speaker_1, dtype=np.float64).reshape(-1)
    length = min(left.size, right.size)
    if length < 2:
        return None, None
    left = left[:length]
    right = right[:length]
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 2:
        return None, None
    left = left[finite]
    right = right[finite]
    if np.std(left) <= epsilon or np.std(right) <= epsilon:
        return None, None
    correlation = float(np.corrcoef(left, right)[0, 1])
    if not np.isfinite(correlation):
        return None, None
    return correlation, abs(correlation)


def safe_energy_ratio(numerator_rms: Any, denominator_rms: Any) -> float | None:
    numerator = _finite_float(numerator_rms)
    denominator = _finite_float(denominator_rms)
    if numerator is None or denominator is None:
        return None
    return numerator / max(abs(denominator), ENERGY_EPSILON)


def safe_audio_levels(audio: np.ndarray) -> tuple[float | None, float | None]:
    samples = np.asarray(audio, dtype=np.float64).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if not samples.size:
        return None, None
    return float(np.sqrt(np.mean(samples * samples))), float(np.max(np.abs(samples)))


def activity_relationship(vad_0: dict[str, Any] | None, vad_1: dict[str, Any] | None) -> str:
    active_0 = bool((vad_0 or {}).get("speech_detected", False))
    active_1 = bool((vad_1 or {}).get("speech_detected", False))
    if active_0 and active_1:
        return "both_active"
    if active_0:
        return "only_speaker_0"
    if active_1:
        return "only_speaker_1"
    return "none_active"


def text_similarity(text_0: str, text_1: str) -> dict[str, float | None]:
    if not text_0 or not text_1:
        return {"raw": None, "normalized": None}
    normalized_0 = _NORMALIZE_PATTERN.sub("", text_0).lower()
    normalized_1 = _NORMALIZE_PATTERN.sub("", text_1).lower()
    return {
        "raw": SequenceMatcher(None, text_0, text_1).ratio(),
        "normalized": (
            SequenceMatcher(None, normalized_0, normalized_1).ratio()
            if normalized_0 and normalized_1
            else None
        ),
    }


def normalize_lexical_text(text: str) -> str:
    """Normalize text for diagnostic comparison without changing STT output."""
    return _NORMALIZE_PATTERN.sub("", text or "").lower()


def cross_stream_lexical_overlap(owner_text: str, candidate_text: str) -> dict[str, Any]:
    """Describe exact structural overlap between simultaneous stream transcripts.

    This is only a corroborating leakage signal.  It never suppresses a stream
    without owner-dominant acoustic evidence and temporal containment.
    """
    owner = normalize_lexical_text(owner_text)
    candidate = normalize_lexical_text(candidate_text)
    minimum = int(TEMPORAL_CONTINUITY_CONFIG["fuzzy_overlap_min_characters"])
    bounded_length = int(
        SECONDARY_ARTIFACT_CONFIG["secondary_fragment_max_normalized_characters"]
    )
    prefix = 0
    for left, right in zip(owner, candidate):
        if left != right:
            break
        prefix += 1
    suffix = 0
    for left, right in zip(reversed(owner), reversed(candidate)):
        if left != right:
            break
        suffix += 1
    candidate_contained = bool(
        len(candidate) >= minimum and candidate in owner
    )
    owner_contained = bool(len(owner) >= minimum and owner in candidate)
    bounded_boundary_overlap = bool(
        candidate
        and len(candidate) <= bounded_length
        and max(prefix, suffix) >= minimum
    )
    return {
        "available": bool(owner and candidate),
        "owner_normalized_characters": len(owner),
        "candidate_normalized_characters": len(candidate),
        "common_prefix_characters": prefix,
        "common_suffix_characters": suffix,
        "candidate_contained_in_owner_text": candidate_contained,
        "owner_contained_in_candidate_text": owner_contained,
        "bounded_boundary_overlap": bounded_boundary_overlap,
        "duplicate_structure": bool(
            candidate_contained or owner_contained or bounded_boundary_overlap
        ),
    }


def _crest_factor(rms: Any, peak: Any) -> float | None:
    finite_rms = _finite_float(rms)
    finite_peak = _finite_float(peak)
    if finite_rms is None or finite_peak is None or finite_rms <= ENERGY_EPSILON:
        return None
    return finite_peak / finite_rms


def _suffix_prefix_continuity(previous: str, current: str) -> dict[str, Any]:
    """Find overlap-style continuity using only the immediately preceding window."""
    if not previous or not current:
        return {
            "detected": False,
            "kind": None,
            "exact_overlap_characters": 0,
            "fuzzy_similarity": None,
        }

    maximum = min(len(previous), len(current))
    exact_overlap = 0
    for size in range(maximum, 0, -1):
        if previous[-size:] == current[:size]:
            exact_overlap = size
            break

    fuzzy_similarity: float | None = None
    fuzzy_size = min(maximum, 24)
    if fuzzy_size >= TEMPORAL_CONTINUITY_CONFIG["fuzzy_overlap_min_characters"]:
        fuzzy_similarity = SequenceMatcher(
            None, previous[-fuzzy_size:], current[:fuzzy_size]
        ).ratio()

    if exact_overlap >= TEMPORAL_CONTINUITY_CONFIG["exact_overlap_min_characters"]:
        kind = "exact_suffix_prefix"
        detected = True
    elif (
        fuzzy_similarity is not None
        and fuzzy_similarity >= TEMPORAL_CONTINUITY_CONFIG["fuzzy_overlap_similarity"]
    ):
        kind = "fuzzy_suffix_prefix"
        detected = True
    else:
        kind = None
        detected = False
    return {
        "detected": detected,
        "kind": kind,
        "exact_overlap_characters": exact_overlap,
        "fuzzy_similarity": fuzzy_similarity,
    }


def _validity_acoustic_evidence(
    diagnostic: dict[str, Any], full_window_diagnostic: dict[str, Any] | None
) -> dict[str, Any]:
    """Expose available acoustics; ClearVoice-normalized RMS is intentionally excluded."""
    speaker_metrics = diagnostic.get("speaker_metrics", [{}, {}])
    speaker_1_metrics = speaker_metrics[1] if len(speaker_metrics) > 1 else {}
    full = full_window_diagnostic or {}
    speaker_0 = full.get("speaker_0", {})
    speaker_1 = full.get("speaker_1", {})
    pair = full.get("speaker_pair", {})
    speaker_1_rms = speaker_1.get("rms", speaker_1_metrics.get("rms"))
    speaker_1_peak = speaker_1.get("peak", speaker_1_metrics.get("peak"))
    return {
        "full_window_available": bool(full_window_diagnostic),
        "original_speaker_0_correlation": speaker_0.get("original_correlation"),
        "original_speaker_0_abs_correlation": speaker_0.get("original_abs_correlation"),
        "original_speaker_1_correlation": speaker_1.get("original_correlation"),
        "original_speaker_1_abs_correlation": speaker_1.get("original_abs_correlation"),
        "speaker_pair_correlation": pair.get(
            "correlation", diagnostic.get("waveform_pearson_correlation")
        ),
        "speaker_pair_abs_correlation": pair.get(
            "abs_correlation", diagnostic.get("abs_waveform_pearson_correlation")
        ),
        "speaker_1_peak": _finite_float(speaker_1_peak),
        "speaker_1_crest_factor": _crest_factor(speaker_1_rms, speaker_1_peak),
        "rms_ratio_excluded_reason": RMS_NORMALIZATION_NOTE,
    }


class SecondaryValidityTracker:
    """Past-only speaker_1 evidence recorder; it never influences pipeline decisions."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._previous_window: int | None = None
        self._previous_active = False
        self._previous_normalized_text = ""
        self._previous_meaningful = False
        self._consecutive_active = 0
        self._consecutive_meaningful = 0

    def observe(
        self,
        *,
        window: int,
        diagnostic: dict[str, Any],
        raw_stt: str,
        full_window_diagnostic: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        speaker_metrics = diagnostic.get("speaker_metrics", [{}, {}])
        speaker_1 = speaker_metrics[1] if len(speaker_metrics) > 1 else {}
        active = bool(speaker_1.get("vad_speech_detected", False))
        speech_duration_ms = _finite_float(speaker_1.get("vad_speech_duration_ms"))
        speech_ratio = _finite_float(speaker_1.get("vad_speech_ratio"))
        normalized = normalize_lexical_text(raw_stt)
        meaningful = bool(normalized)
        consecutive = self._previous_window == window - 1
        previous_normalized = self._previous_normalized_text if consecutive else ""
        continuity = _suffix_prefix_continuity(previous_normalized, normalized)

        self._consecutive_active = (
            self._consecutive_active + 1 if active and consecutive and self._previous_active else int(active)
        )
        self._consecutive_meaningful = (
            self._consecutive_meaningful + 1
            if meaningful and consecutive and self._previous_meaningful
            else int(meaningful)
        )

        positive_reasons: list[str] = []
        negative_reasons: list[str] = []
        if meaningful:
            positive_reasons.append("lexical_text_present")
        if continuity["detected"]:
            positive_reasons.append("temporal_text_continuity")
        if self._consecutive_active >= 2:
            positive_reasons.append("consecutive_vad_activity")
        if self._consecutive_meaningful >= 2:
            positive_reasons.append("consecutive_meaningful_text")
        if speech_ratio is not None and speech_ratio >= TEMPORAL_CONTINUITY_CONFIG[
            "strong_secondary_min_vad_ratio"
        ]:
            positive_reasons.append("high_vad_speech_ratio")

        acoustic = _validity_acoustic_evidence(diagnostic, full_window_diagnostic)
        original_0 = acoustic["original_speaker_0_abs_correlation"]
        original_1 = acoustic["original_speaker_1_abs_correlation"]
        if (
            original_0 is not None
            and original_1 is not None
            and original_0 >= 0.95
            and original_1 <= 0.20
        ):
            # Context only: this does not alter classification of lexical text.
            negative_reasons.append("primary_dominant_acoustic_pattern")

        if not active and not raw_stt.strip():
            classification = "inactive"
        elif not normalized:
            classification = "nonlexical"
        elif (
            continuity["detected"]
            and self._consecutive_active
            >= TEMPORAL_CONTINUITY_CONFIG["strong_secondary_min_consecutive_windows"]
            and self._consecutive_meaningful
            >= TEMPORAL_CONTINUITY_CONFIG["strong_secondary_min_consecutive_windows"]
            and speech_ratio is not None
            and speech_ratio >= TEMPORAL_CONTINUITY_CONFIG["strong_secondary_min_vad_ratio"]
        ):
            classification = "strong_secondary"
        elif continuity["detected"] and self._consecutive_meaningful >= 2:
            classification = "likely_secondary"
        else:
            # A short, low-VAD, isolated, or low-correlation lexical response remains unknown.
            classification = "unknown_lexical"

        evidence = {
            "window": window,
            "classification": classification,
            "vad": {
                "speech_duration_ms": speech_duration_ms,
                "speech_ratio": speech_ratio,
                "active": active,
                "consecutive_active_count": self._consecutive_active,
            },
            "lexical": {
                "raw_stt": raw_stt,
                "normalized_text": normalized,
                "meaningful_character_count": len(normalized),
                "nonlexical": bool(raw_stt.strip()) and not meaningful,
            },
            "temporal_text": {
                "previous_normalized_text": previous_normalized,
                "continuity": continuity,
                "consecutive_meaningful_window_count": self._consecutive_meaningful,
            },
            "acoustic": acoustic,
            "positive_evidence_count": len(positive_reasons),
            "negative_evidence_count": len(negative_reasons),
            "positive_evidence_reasons": positive_reasons,
            "negative_evidence_reasons": negative_reasons,
            "classification_note": (
                "Diagnostic observation only; this classification never controls suppression, "
                "VAD, STT, subtitle assembly, WebSocket, or queue behavior."
            ),
        }
        self._previous_window = window
        self._previous_active = active
        self._previous_normalized_text = normalized
        self._previous_meaningful = meaningful
        return evidence


def build_secondary_validity_summary(window_records: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_records = [
        record.get("validity_evidence") for record in window_records if record.get("validity_evidence")
    ]
    classifications = ("inactive", "nonlexical", "unknown_lexical", "likely_secondary", "strong_secondary")
    indices_by_classification = {
        classification: [
            record.get("window")
            for record in evidence_records
            if record["classification"] == classification
        ]
        for classification in classifications
    }
    return {
        "enabled": bool(evidence_records),
        "config": TEMPORAL_CONTINUITY_CONFIG,
        "rms_ratio_excluded_reason": RMS_NORMALIZATION_NOTE,
        "classification_window_counts": {
            classification: len(indices) for classification, indices in indices_by_classification.items()
        },
        "nonlexical_window_indices": indices_by_classification["nonlexical"],
        "unknown_lexical_window_indices": indices_by_classification["unknown_lexical"],
        "likely_secondary_window_indices": indices_by_classification["likely_secondary"],
        "strong_secondary_window_indices": indices_by_classification["strong_secondary"],
        "continuity_detected_count": sum(
            record["temporal_text"]["continuity"]["detected"] for record in evidence_records
        ),
        "max_consecutive_active_count": max(
            (record["vad"]["consecutive_active_count"] for record in evidence_records), default=0
        ),
        "max_consecutive_meaningful_count": max(
            (
                record["temporal_text"]["consecutive_meaningful_window_count"]
                for record in evidence_records
            ),
            default=0,
        ),
    }


def build_window_diagnostic(
    *,
    speaker_0: np.ndarray,
    speaker_1: np.ndarray,
    vad_results: list[dict[str, Any]],
    transcripts: list[str],
) -> dict[str, Any]:
    vad_0 = vad_results[0] if len(vad_results) > 0 else None
    vad_1 = vad_results[1] if len(vad_results) > 1 else None
    text_0 = transcripts[0] if len(transcripts) > 0 else ""
    text_1 = transcripts[1] if len(transcripts) > 1 else ""
    rms_0, peak_0 = safe_audio_levels(speaker_0)
    rms_1, peak_1 = safe_audio_levels(speaker_1)
    correlation, absolute_correlation = safe_waveform_correlation(speaker_0, speaker_1)
    energy_ratio = safe_energy_ratio(rms_1, rms_0)
    reverse_energy_ratio = safe_energy_ratio(rms_0, rms_1)
    relationship = activity_relationship(vad_0, vad_1)
    similarity = text_similarity(text_0, text_1)
    primary_ratio = _finite_float((vad_0 or {}).get("speech_ratio"))
    secondary_ratio = _finite_float((vad_1 or {}).get("speech_ratio"))
    normalized_secondary_text = _NORMALIZE_PATTERN.sub("", text_1)
    reasons: list[str] = []
    primary_dominant = (
        relationship == "both_active"
        and primary_ratio is not None
        and primary_ratio >= SECONDARY_ARTIFACT_CONFIG["primary_vad_ratio_min"]
    )
    secondary_sparse = (
        secondary_ratio is not None
        and secondary_ratio <= SECONDARY_ARTIFACT_CONFIG["secondary_vad_ratio_max"]
    )
    if primary_dominant:
        reasons.append("primary_vad_dominant")
    if secondary_sparse:
        reasons.append("secondary_vad_sparse")
    if text_1 and len(normalized_secondary_text) <= SECONDARY_ARTIFACT_CONFIG[
        "secondary_fragment_max_normalized_characters"
    ]:
        reasons.append("secondary_transcript_fragment")
    if (
        absolute_correlation is not None
        and absolute_correlation >= SECONDARY_ARTIFACT_CONFIG["high_abs_waveform_correlation"]
    ):
        reasons.append("high_cross_channel_correlation")
    candidate = primary_dominant and secondary_sparse
    return {
        "speaker_metrics": [
            {
                "rms": rms_0,
                "peak": peak_0,
                "vad_speech_duration_ms": _finite_float((vad_0 or {}).get("speech_duration_ms")),
                "vad_speech_ratio": primary_ratio,
                "vad_speech_detected": bool((vad_0 or {}).get("speech_detected", False)),
            },
            {
                "rms": rms_1,
                "peak": peak_1,
                "vad_speech_duration_ms": _finite_float((vad_1 or {}).get("speech_duration_ms")),
                "vad_speech_ratio": secondary_ratio,
                "vad_speech_detected": bool((vad_1 or {}).get("speech_detected", False)),
            },
        ],
        "activity_relationship": relationship,
        "waveform_pearson_correlation": correlation,
        "abs_waveform_pearson_correlation": absolute_correlation,
        "speaker_1_to_speaker_0_energy_ratio": energy_ratio,
        "speaker_0_to_speaker_1_energy_ratio": reverse_energy_ratio,
        "text_similarity": similarity,
        "secondary_artifact_candidate": candidate,
        "reasons": reasons,
    }


def secondary_transcript_suppression_reason(
    diagnostic: dict[str, Any], transcript: str
) -> str | None:
    """Require both waveform and text agreement before rejecting lexical output."""
    secondary_active = diagnostic["speaker_metrics"][1]["vad_speech_detected"]
    if secondary_active and transcript.strip() and not _NORMALIZE_PATTERN.sub("", transcript):
        return "secondary_nonlexical_transcript"
    # Similar words alone are not leakage evidence: two real speakers can say
    # the same thing. Near-identical waveforms in the same window plus an exact
    # normalized transcript indicate a duplicated separator output.
    correlation = diagnostic.get("abs_waveform_pearson_correlation")
    if (
        diagnostic.get("activity_relationship") == "both_active"
        and len(normalize_lexical_text(transcript)) >= 4
        and correlation is not None and correlation >= 0.98
        and diagnostic.get("text_similarity", {}).get("normalized") == 1.0
    ):
        return "duplicate_waveform_and_transcript"
    return None


def speech_intervals(vad: dict[str, Any], samples: int) -> list[tuple[int, int]]:
    """Read existing Silero sample timestamps; missing/invalid evidence fails open."""
    intervals = []
    stamps = vad.get("timestamps", [])
    if not isinstance(stamps, (list, tuple)):
        return []
    for stamp in stamps:
        if not isinstance(stamp, dict):
            return []
        start, end = stamp.get("start"), stamp.get("end")
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= samples
                or (intervals and start < intervals[-1][1])):
            return []
        intervals.append((start, end))
    return intervals


def subtitle_overlap_evidence(
    previous_audio: np.ndarray, current_audio: np.ndarray,
    previous_vad: dict[str, Any], current_vad: dict[str, Any],
    overlap_samples: int,
) -> dict[str, Any]:
    """Local evidence for text matching; no word timestamps are assumed."""
    evidence: dict[str, Any] = {"shared_speech": None, "supported_tail_revision": False}
    previous = speech_intervals(previous_vad, len(previous_audio))
    current = speech_intervals(current_vad, len(current_audio))
    if not previous or not current or not 0 < overlap_samples <= min(len(previous_audio), len(current_audio)):
        return evidence
    stride = len(previous_audio) - overlap_samples
    shared = [
        (max(0, start - stride, head_start), min(overlap_samples, end - stride, head_end))
        for start, end in previous for head_start, head_end in current
        if max(0, start - stride, head_start) < min(overlap_samples, end - stride, head_end)
    ]
    evidence["shared_speech"] = bool(shared)
    if not shared:
        return evidence
    correlation = safe_absolute_correlation(
        previous_audio[-overlap_samples:], current_audio[:overlap_samples]
    )
    evidence["overlap_correlation"] = correlation
    evidence["previous_tail_in_overlap"] = previous[-1][0] >= stride
    evidence["tail_covered_by_current_prefix"] = (
        current[0][0] <= previous[-1][0] - stride
        and previous[-1][1] - stride <= current[0][1]
    )
    evidence["supported_tail_revision"] = bool(
        previous[-1][0] >= stride
        and evidence["tail_covered_by_current_prefix"]
        and correlation is not None
        and correlation >= SECONDARY_ARTIFACT_CONFIG["high_abs_waveform_correlation"]
    )
    return evidence


def resolve_subtitle_fragments(
    *,
    speakers: tuple[np.ndarray, np.ndarray],
    transcripts: list[str],
    vad_results: list[dict[str, Any]],
    diagnostic: dict[str, Any],
    previous_texts: list[str],
    active_hypotheses: list[str],
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    """Keep a new, acoustically duplicated fragment in an established utterance.

    This is subtitle routing, not a change to separator/tracker identities. Never
    discard novel lexical content: append only with ordered VAD intervals. A
    short/quiet secondary alone is NOT grounds for routing. Independent or
    already established speakers, missing timestamps and ambiguous order all
    retain their original streams. RMS ratios are deliberately not used.
    """
    texts = list(transcripts)
    vads = [dict(vad) for vad in vad_results]
    evidence: dict[str, Any] = {"applied": False, "reason": "insufficient_evidence"}
    established = [i for i in (0, 1) if active_hypotheses[i] and previous_texts[i]]
    if len(vads) != 2 or len(established) != 1:
        return texts, vads, evidence
    owner = established[0]
    candidate = 1 - owner
    evidence.update({"owner": owner, "candidate": candidate})
    if active_hypotheses[candidate]:
        evidence["reason"] = "established_secondary"
        return texts, vads, evidence
    primary_text = normalize_lexical_text(texts[owner])
    fragment = normalize_lexical_text(texts[candidate])
    metrics = diagnostic["speaker_metrics"]
    primary_ratio = metrics[owner].get("vad_speech_ratio")
    secondary_ratio = metrics[candidate].get("vad_speech_ratio")
    primary_ms = metrics[owner].get("vad_speech_duration_ms")
    secondary_ms = metrics[candidate].get("vad_speech_duration_ms")
    continuity = _suffix_prefix_continuity(
        normalize_lexical_text(previous_texts[owner]), primary_text
    )
    evidence["owner_text_continuity"] = continuity
    if not (
        all(vad.get("speech_detected") for vad in vads)
        and primary_ratio is not None and secondary_ratio is not None
        and primary_ms is not None and secondary_ms is not None
        and primary_ratio >= SECONDARY_ARTIFACT_CONFIG["primary_vad_ratio_min"]
        and secondary_ratio <= SECONDARY_ARTIFACT_CONFIG["secondary_vad_ratio_max"]
        and primary_ms > secondary_ms > 0
        and TEMPORAL_CONTINUITY_CONFIG["fuzzy_overlap_min_characters"] <= len(fragment)
        <= SECONDARY_ARTIFACT_CONFIG["secondary_fragment_max_normalized_characters"]
        and continuity["detected"]
    ):
        return texts, vads, evidence
    intervals = [speech_intervals(vad, len(speakers[0])) for vad in vads]
    if not all(intervals):
        evidence["reason"] = "missing_speech_intervals"
        return texts, vads, evidence
    # Compare every candidate speech interval, not whole-window RMS or a
    # correlation dominated by the primary speaker elsewhere in the window.
    correlations = [
        safe_absolute_correlation(speakers[owner][start:end], speakers[candidate][start:end])
        for start, end in intervals[candidate]
    ]
    evidence["candidate_speech_correlations"] = correlations
    if not all(value is not None and value >= SECONDARY_ARTIFACT_CONFIG[
        "high_abs_waveform_correlation"
    ] for value in correlations):
        evidence["reason"] = "independent_or_uncertain_waveform"
        return texts, vads, evidence
    if fragment in primary_text:
        reason = "duplicate_fragment_on_same_waveform"
    elif intervals[candidate][0][0] >= intervals[owner][-1][1]:
        reason = "ordered_continuation_on_same_waveform"
        texts[owner] = append_preserving_text(texts[owner], texts[candidate])
        vads[owner]["timestamps"] = [
            {"start": start, "end": end}
            for start, end in intervals[owner] + intervals[candidate]
        ]
    else:
        evidence["reason"] = "ambiguous_fragment_order"
        return texts, vads, evidence
    texts[candidate] = ""
    # Effective subtitle activity only: the original VAD/STT records stay intact.
    vads[candidate]["speech_detected"] = False
    vads[candidate]["timestamps"] = []
    evidence.update({"applied": True, "reason": reason})
    return texts, vads, evidence


def weak_speech_evidence(mixture: np.ndarray, vad: dict[str, Any]) -> dict[str, Any]:
    """Reject a new hypothesis only with joint localized input evidence.

    Use original input, since separator RMS normalization can amplify noise.
    A short/quiet but localized speech burst passes without a second observation.
    Independently measured speech in the input residual also passes. Missing
    evidence fails open; this is not a general noise/speech classifier.
    """
    evidence: dict[str, Any] = {"suppressed": False, "available": False}
    intervals = speech_intervals(vad, len(mixture))
    duration = _finite_float(vad.get("speech_duration_ms"))
    ratio = _finite_float(vad.get("speech_ratio"))
    if not intervals or duration is None or ratio is None or not np.isfinite(mixture).all():
        return evidence
    mask = np.zeros(len(mixture), dtype=bool)
    for start, end in intervals:
        mask[start:end] = True
    speech_rms, speech_peak = safe_audio_levels(mixture[mask])
    background_rms, _ = safe_audio_levels(mixture[~mask])
    evidence.update(input_speech_rms=speech_rms, input_speech_peak=speech_peak,
                    input_background_rms=background_rms)
    if speech_rms is None or speech_peak is None or background_rms is None:
        return evidence
    # Bounded low-level noise AND no localized energy increase must agree.
    # VAD duration is diagnostic only: separator noise does not become speech
    # merely by remaining VAD-positive for more than 500 ms. Correlation is
    # intentionally absent here because it establishes source similarity, not
    # the presence of speech in that source.
    weak_vad = 0 < duration <= WEAK_SPEECH_MAX_DURATION_MS and 0 < ratio <= 0.20
    no_burst = speech_rms <= 1.5 * max(background_rms, ENERGY_EPSILON)
    residual_speech = [
        _residual_speech_overlap(vad, start, end, len(mixture))
        for start, end in intervals
    ]
    independent_speech = any(value is True for value in residual_speech)
    rms_within_noise_bound = speech_rms <= DEFAULT_SILENCE_RMS
    peak_within_noise_bound = speech_peak <= 5 * DEFAULT_SILENCE_RMS
    evidence.update(
        available=True, weak_vad=weak_vad, no_localized_burst=no_burst,
        rms_within_noise_bound=rms_within_noise_bound,
        peak_within_noise_bound=peak_within_noise_bound,
        input_residual_speech_overlap=residual_speech,
        independent_speech_evidence=independent_speech,
    )
    evidence["suppressed"] = bool(
        no_burst and rms_within_noise_bound and peak_within_noise_bound
        and not independent_speech
    )
    return evidence


def attach_input_residual_speech_evidence(
    mixture: np.ndarray,
    speakers: tuple[np.ndarray, np.ndarray] | list[np.ndarray],
    vad_results: list[dict[str, Any]],
    detect_speech: Callable[[np.ndarray], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Check speech in the input left by the OTHER output, before slot mapping.

    Residual correlation is not speech evidence: correlated separator errors
    can pass it. Use the existing VAD on an independently constructed input
    residual, never on the candidate waveform again. Normalize to a fixed RMS
    (a gain, NOT a silence threshold), giving quiet residuals the same VAD scale.
    Inspect the full window to preserve VAD context; admission intersects its
    timestamps with each candidate interval. This is optional evidence: old
    servers, degenerate projections and detector failures remain fail-open.
    """
    vads = [dict(vad) for vad in vad_results]
    if len(vads) != 2 or len(speakers) != 2:
        return vads
    x = np.asarray(mixture, dtype=np.float64)
    if x.ndim != 1 or x.size < 3 or not np.isfinite(x).all():
        return vads
    x = x - x.mean()
    input_energy = float(x @ x)
    if input_energy <= 0:
        return vads
    for candidate, vad in enumerate(vads):
        diagnostic: dict[str, Any] = {
            "version": 1, "available": False, "sample_count": len(x),
            "reason": "not_required", "normalization_rms": 0.1,
        }
        vad["input_residual_speech"] = diagnostic
        intervals = speech_intervals(vad, len(x))
        if not vad.get("speech_detected") or not intervals:
            continue
        owner = np.asarray(speakers[1 - candidate], dtype=np.float64)
        if owner.shape != x.shape or not np.isfinite(owner).all():
            diagnostic["reason"] = "invalid_owner"
            continue
        # Only spend another VAD inference on an owner-dominated interval.
        # A genuine interval elsewhere still protects the entire candidate.
        if not any((safe_absolute_correlation(x[start:end], owner[start:end]) or 0)
                   >= MIN_DOMINANT_INPUT_CORRELATION for start, end in intervals):
            continue
        owner = owner - owner.mean()
        owner_energy = float(owner @ owner)
        if owner_energy <= 0:
            diagnostic["reason"] = "degenerate_owner"
            continue
        residual = x - float(x @ owner) / owner_energy * owner
        energy_fraction = float(residual @ residual) / input_energy
        diagnostic["input_residual_energy_fraction"] = energy_fraction
        if not np.isfinite(energy_fraction) or energy_fraction <= 1e-14:
            diagnostic["reason"] = "degenerate_input_residual"
            continue
        gain = 0.1 / float(np.sqrt(np.mean(residual * residual)))
        diagnostic["normalization_gain"] = gain
        try:
            result = detect_speech(np.ascontiguousarray(residual * gain, dtype=np.float32))
            stamps = result.get("timestamps")
            spans = speech_intervals(result, len(x))
            if (not isinstance(stamps, list) or (stamps and not spans)
                    or type(result.get("speech_detected")) is not bool
                    or bool(spans) != result["speech_detected"]):
                diagnostic["reason"] = "invalid_residual_vad"
                continue
            diagnostic.update(
                available=True, reason="measured", timestamps=stamps,
                speech_detected=result["speech_detected"],
                processing_seconds=result.get("processing_seconds"),
            )
        except Exception as exc:
            diagnostic.update(reason="residual_vad_error", error=type(exc).__name__)
    return vads


def _residual_speech_overlap(vad: dict[str, Any], start: int, end: int,
                             samples: int) -> bool | None:
    evidence = vad.get("input_residual_speech", {})
    if not isinstance(evidence, dict) or not (
        evidence.get("version") == 1 and evidence.get("available") is True
        and evidence.get("sample_count") == samples
        and evidence.get("reason") == "measured"
    ):
        return None
    stamps = evidence.get("timestamps")
    spans = speech_intervals(evidence, samples)
    if (not isinstance(stamps, list) or (stamps and not spans)
            or type(evidence.get("speech_detected")) is not bool
            or bool(spans) != evidence["speech_detected"]):
        return None
    return any(left < end and start < right for left, right in spans)


def admit_subtitle_streams(
    *,
    mixture: np.ndarray,
    speakers: tuple[np.ndarray, np.ndarray],
    transcripts: list[str],
    vad_results: list[dict[str, Any]],
    active_hypotheses: list[str],
    speaker_assignment: dict[str, Any],
    candidate_contexts: list[dict[str, Any]] | None = None,
    window: int | None = None,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    """Reject evidenced residuals before a new logical utterance can start.

    Fresh acoustics drive decisions; text persistence and tracker confirmation
    are never validation evidence. Subtitle state owns candidate provenance and
    receives explicit hold/validate/discard decisions. Held text may be assembled
    internally, but remains unpublished until acoustic confirmation.
    Missing/ambiguous acoustics fail open. Rejected continuations leave the
    previously admitted utterance prefix untouched.
    Generic temporal consensus is not an admission test: a genuine short
    utterance may have only one observation. The narrow exception is a strong,
    localized, short source with no independent speech confirmation; keep that
    candidate internal until speech evidence replaces it or silence discards it.
    Block acoustically unsupported starts here, before PARTIAL publication, so
    silence/flush cannot retain them as FINAL.
    """
    texts = list(transcripts)
    vads = [dict(vad) for vad in vad_results]
    if len(vads) != 2:
        return texts, vads, {"applied": False, "reason": "vad_unavailable", "streams": []}
    decisions = []
    mapping = speaker_assignment.get("raw_to_logical_mapping", {})
    active_logical = {
        mapping.get(str(raw)) for raw in speaker_assignment.get("active_raw_slots", [])
    }
    intervals = [speech_intervals(vad, len(mixture)) for vad in vad_results]
    for candidate in (0, 1):
        owner = 1 - candidate
        context = candidate_contexts[candidate] if candidate_contexts else {}
        provenance = context.get("provenance", "NONE")
        consecutive = window is not None and context.get("window") == window - 1
        evidence: dict[str, Any] = {
            "speaker": candidate, "owner": owner, "suppressed": False,
            "reason": "insufficient_evidence", "existing_partial": bool(active_hypotheses[candidate]),
            "vad_ratio": vad_results[candidate].get("speech_ratio"),
            "vad_duration_ms": vad_results[candidate].get("speech_duration_ms"),
            "tracking_method": speaker_assignment.get("assignment_method"),
            "tracking_active_logical": sorted(i for i in active_logical if i in (0, 1)),
            "speech_intervals": intervals[candidate],
            "owner_speech_intervals": intervals[owner],
            "speech_local_evidence": [],
            "candidate_provenance": provenance,
            "candidate_transition": "keep",
            "previous_candidate_window": context.get("window"),
            "candidate_confirmation_consecutive": consecutive,
            "source_support_deferred": False,
            "input_residual_speech": vad_results[candidate].get("input_residual_speech"),
        }
        decisions.append(evidence)
        if not transcripts[candidate].strip() or not vad_results[candidate].get("speech_detected"):
            evidence["reason"] = "inactive"
            continue
        local = [
            source_input_evidence(
                mixture[start:end], speakers[owner][start:end], speakers[candidate][start:end],
            ) for start, end in intervals[candidate]
        ]
        evidence["speech_local_evidence"] = local
        # Positive support is separate from fail-open PARTIAL admission. Reuse
        # the tracker/admission source confidence, including quiet independent
        # components. A missing/zero owner can still leave direct input support.
        # Record the existing OR decision, including the direct-correlation
        # fallback that otherwise appears only as a null local evidence row.
        owner_only = (
            active_logical == {owner}
            and not speaker_assignment.get("input_silence", False)
            and not speaker_assignment.get("low_energy_tail", False)
        )
        owner_established = owner_only or bool(
            active_hypotheses[owner] and (
                not candidate_contexts or candidate_contexts[owner].get("provenance") != "TENTATIVE"
            )
        )
        candidate_duration = _finite_float(
            vad_results[candidate].get("speech_duration_ms")
        )
        support_checks = []
        for (start, end), row in zip(intervals[candidate], local):
            direct = (row["candidate_input_correlation"] if row is not None else
                      safe_absolute_correlation(mixture[start:end], speakers[candidate][start:end]))
            independent = row["independent_input_correlation"] if row is not None else None
            owner_correlation = row.get("owner_input_correlation") if row is not None else None
            residual_energy = row.get("input_residual_energy_fraction") if row is not None else None
            candidate_residual_energy = (
                row.get("candidate_residual_energy_fraction") if row is not None else None
            )
            owner_dominates = bool(
                owner_correlation is not None
                and owner_correlation >= MIN_DOMINANT_INPUT_CORRELATION
            )
            residual_energy_pass = bool(
                residual_energy is None
                or residual_energy >= MIN_INDEPENDENT_INPUT_ENERGY_FRACTION
            )
            raw_direct_pass = (direct or 0) >= DEFAULT_SOURCE_CONFIDENCE
            raw_independent_pass = (independent or 0) >= DEFAULT_SOURCE_CONFIDENCE
            residual_speech_overlap = _residual_speech_overlap(
                vad_results[candidate], start, end, len(mixture)
            )
            # An error direction shared by the separator outputs is not an
            # independent speech source. Require positive owner evidence AND
            # measured absence of speech in the gain-normalized input residual.
            # No duration, loudness, transcript or prior consensus is a vote.
            residual_nonspeech = bool(
                owner_dominates and raw_independent_pass
                and residual_speech_overlap is False
                and residual_energy is not None and residual_energy > 1e-14
                and candidate_residual_energy is not None
                and candidate_residual_energy > residual_energy
            )
            # Mixture correlation can come entirely from leaked owner speech.
            # A normalized invented component then passes the direct OR route
            # despite contradicting the input residual. Require measured
            # contradiction, not simply absence of positive source evidence.
            # Quiet genuine sources have strong residual correlation regardless
            # of their energy; missing/ambiguous evidence remains fail-open.
            contradicted_by_owner = bool(
                owner_dominates
                and raw_direct_pass
                and independent is not None
                and independent <= MAX_UNSUPPORTED_INPUT_CORRELATION
                and direct < owner_correlation
                and candidate_residual_energy is not None
                and residual_energy is not None
                # source_input_evidence uses this numerical floor to mark a
                # degenerate residual. An output equal to the whole mixture
                # cannot disprove the other source through residual direction.
                and residual_energy > 1e-14
                and candidate_residual_energy > residual_energy
            )
            # The live false source passed both correlation routes even though
            # the owner left <1% input energy.  A genuinely quiet separated
            # source instead has weak direct correlation and is protected by
            # the independent route.  Also avoid blocking a candidate that is
            # merely a polarity-inverted copy of the mixture: its residual
            # energy is the same as the input residual, not a normalized large
            # component projected onto a tiny owner error.
            blocked_by_owner_residual = bool(
                owner_established
                and owner_dominates
                and candidate_duration is not None
                and candidate_duration > WEAK_SPEECH_MAX_DURATION_MS
                and not residual_energy_pass
                and raw_direct_pass
                and raw_independent_pass
                and candidate_residual_energy is not None
                and residual_energy is not None
                and candidate_residual_energy > residual_energy
            )
            # The residual fraction alone is insufficient: the latest live
            # leakage left 2.3--3.9% residual energy.  Record the broader
            # geometric conflict without moving that calibrated 1% boundary.
            # A new contained candidate is held regardless of lexical overlap.
            owner_candidate_conflict = bool(
                owner_established
                and owner_dominates
                and candidate_duration is not None
                and candidate_duration > WEAK_SPEECH_MAX_DURATION_MS
                and raw_direct_pass
                and raw_independent_pass
                and candidate_residual_energy is not None
                and residual_energy is not None
                and candidate_residual_energy > residual_energy
            )
            support_checks.append({
                "interval": [start, end],
                "direct_correlation": direct,
                "independent_correlation": independent,
                "owner_input_correlation": owner_correlation,
                "input_residual_energy_fraction": residual_energy,
                "candidate_residual_energy_fraction": candidate_residual_energy,
                "owner_established": owner_established,
                "owner_dominates": owner_dominates,
                "residual_gate_duration_pass": bool(
                    candidate_duration is not None
                    and candidate_duration > WEAK_SPEECH_MAX_DURATION_MS
                ),
                "residual_energy_pass": residual_energy_pass,
                "blocked_by_owner_residual": blocked_by_owner_residual,
                "owner_candidate_conflict": owner_candidate_conflict,
                "raw_direct_pass": raw_direct_pass,
                "raw_independent_pass": raw_independent_pass,
                "input_residual_speech_overlap": residual_speech_overlap,
                "residual_nonspeech": residual_nonspeech,
                "contradicted_by_owner": contradicted_by_owner,
                "direct_pass": (
                    raw_direct_pass and not blocked_by_owner_residual
                    and not contradicted_by_owner and not residual_nonspeech
                ),
                "independent_pass": (
                    raw_independent_pass and not blocked_by_owner_residual and not residual_nonspeech
                ),
                "used_direct_fallback": row is None,
            })
        supported = any(row["direct_pass"] or row["independent_pass"] for row in support_checks)
        evidence["source_support_checks"] = support_checks
        evidence["source_support_threshold"] = DEFAULT_SOURCE_CONFIDENCE
        evidence["minimum_independent_input_energy_fraction"] = (
            MIN_INDEPENDENT_INPUT_ENERGY_FRACTION
        )
        evidence["source_supported"] = supported
        vads[candidate]["source_supported"] = supported
        # Apply the same localized low-energy input test to starts and continuations.
        # A continuation keeps its previously supported prefix in subtitle state;
        # only the unsupported current fragment is removed here.
        weak_speech = weak_speech_evidence(mixture, vad_results[candidate])
        evidence["weak_speech_evidence"] = weak_speech
        evidence["weak_speech_check_applied"] = True
        short_localized_source_without_speech_confirmation = bool(
            weak_speech.get("available")
            and weak_speech.get("weak_vad")
            and not weak_speech.get("no_localized_burst")
            and not weak_speech.get("rms_within_noise_bound")
            and not weak_speech.get("peak_within_noise_bound")
            and not weak_speech.get("independent_speech_evidence")
        )
        evidence["short_localized_source_without_speech_confirmation"] = (
            short_localized_source_without_speech_confirmation
        )
        weak_existing_tail = bool(
            active_hypotheses[candidate]
            and weak_speech.get("available")
            and weak_speech.get("weak_vad")
            and weak_speech.get("no_localized_burst")
            and not weak_speech.get("independent_speech_evidence")
        )
        evidence["weak_existing_tail_suppressed"] = weak_existing_tail
        if weak_speech["suppressed"] or weak_existing_tail:
            if weak_existing_tail:
                reason = "existing_partial_weak_unlocalized_tail"
            elif weak_speech.get("weak_vad"):
                reason = "weak_vad_and_unlocalized_low_energy_input"
            else:
                reason = "unlocalized_low_energy_input_without_speech_evidence"
            evidence.update(
                suppressed=True,
                source_supported=False,
                reason=reason,
            )
            vads[candidate]["source_supported"] = False
            texts[candidate] = ""
            vads[candidate]["speech_detected"] = False
            vads[candidate]["timestamps"] = []
            continue

        contained: bool | None = None
        if (
            transcripts[owner].strip()
            and vad_results[owner].get("speech_detected")
            and intervals[candidate]
            and intervals[owner]
        ):
            contained = all(
                any(left <= start and end <= right for left, right in intervals[owner])
                for start, end in intervals[candidate]
            )
        evidence["speech_contained_in_owner"] = contained
        lexical_overlap = cross_stream_lexical_overlap(
            transcripts[owner], transcripts[candidate]
        )
        evidence["cross_stream_lexical_overlap"] = lexical_overlap
        owner_conflict = bool(
            support_checks
            and all(row["owner_candidate_conflict"] for row in support_checks)
        )
        evidence["owner_candidate_conflict"] = owner_conflict
        independent_observation = bool(normalize_lexical_text(transcripts[candidate])) and any(
            row["independent_pass"] and not row["owner_candidate_conflict"]
            for row in support_checks
        )
        leakage_like = bool(contained and owner_conflict)
        evidence["candidate_independent_support"] = independent_observation

        def candidate_decision(
            action: str,
            reason: str,
            *,
            independent_support: bool | None = None,
            speech_confirmation_required: bool = False,
            restart: bool = False,
        ) -> None:
            evidence.update(candidate_transition=action, reason=reason)
            vads[candidate]["candidate_transition"] = {
                "action": action,
                "independent_support": (
                    independent_observation
                    if independent_support is None
                    else independent_support
                ),
                "owner_contained_conflict": leakage_like,
                "speech_confirmation_required": speech_confirmation_required,
            }
            if restart:
                evidence["candidate_restart"] = True
                vads[candidate]["candidate_transition"]["restart"] = True
            if action in {"hold", "discard"}:
                evidence["source_support_deferred"] = supported if action == "hold" else False
                evidence["source_supported"] = False
                vads[candidate]["source_supported"] = False
            if action == "discard":
                evidence["suppressed"] = True
                texts[candidate] = ""
                vads[candidate]["speech_detected"] = False
                vads[candidate]["timestamps"] = []

        # A partial can be merely withheld text. Re-evaluate it before the
        # existing-utterance fast path, even when both STT strings differ.
        # Reject only when EVERY candidate VAD interval has contradictory
        # evidence. One independent interval protects a genuine short response.
        # Neither a prior partial nor repeated text can validate today's audio.
        if (
            support_checks
            and any(check["residual_nonspeech"] for check in support_checks)
            and all(check["residual_nonspeech"] or check["contradicted_by_owner"]
                    or unsupported_source(row)
                    for check, row in zip(support_checks, local))
        ):
            candidate_decision("discard", "owner_residual_contains_no_candidate_speech")
            continue
        if (
            support_checks
            and any(row["contradicted_by_owner"] for row in support_checks)
            and all(check["contradicted_by_owner"] or unsupported_source(row)
                    for check, row in zip(support_checks, local))
        ):
            candidate_decision("discard", "shared_owner_component_without_independent_source")
            continue
        if provenance == "TENTATIVE":
            if context.get("speech_confirmation_required"):
                if short_localized_source_without_speech_confirmation:
                    candidate_decision(
                        "hold",
                        "short_localized_source_awaiting_speech_confirmation",
                        independent_support=False,
                        speech_confirmation_required=True,
                    )
                else:
                    candidate_decision(
                        "keep",
                        "speech_evidence_restarts_deferred_transient",
                        restart=True,
                    )
                continue
            if (consecutive and leakage_like and context.get("owner_contained_conflict")) or (
                local and all(unsupported_source(row) for row in local)
            ):
                candidate_decision("discard", "secondary_candidate_repeated_leakage")
            elif (consecutive and independent_observation and context.get("independent_support")
                  and context.get("pending_text_supported")):
                candidate_decision("validate", "secondary_candidate_confirmed")
            else:
                candidate_decision("hold", "secondary_candidate_awaiting_confirmation")
                # An interrupted/contradictory history cannot be validated by a
                # later supported suffix. Start a fresh withheld observation;
                # the pipeline discards the old unvalidated utterance first.
                if independent_observation:
                    evidence["candidate_restart"] = True
                    vads[candidate]["candidate_transition"]["restart"] = True
            continue

        if (
            not active_hypotheses[candidate]
            and short_localized_source_without_speech_confirmation
        ):
            candidate_decision(
                "hold",
                "short_localized_source_awaiting_speech_confirmation",
                independent_support=False,
                speech_confirmation_required=True,
            )
            continue

        if (
            not active_hypotheses[candidate]
            and normalize_lexical_text(transcripts[candidate])
            and support_checks
            and all(row["blocked_by_owner_residual"] for row in support_checks)
        ):
            evidence.update(
                suppressed=True,
                source_supported=False,
                reason="owner_explains_input_with_insufficient_residual_energy",
            )
            vads[candidate]["source_supported"] = False
            texts[candidate] = ""
            vads[candidate]["speech_detected"] = False
            vads[candidate]["timestamps"] = []
            continue
        if (
            active_hypotheses[candidate]
            and contained
            and owner_conflict
            and lexical_overlap["duplicate_structure"]
        ):
            evidence.update(
                suppressed=True,
                source_supported=False,
                reason="contained_cross_stream_leakage",
                tentative_retained=True,
            )
            vads[candidate]["source_supported"] = False
            texts[candidate] = ""
            # Preserve the raw VAD activity so subtitle state keeps the already
            # withheld tentative alive until a real silence/weak-tail decision.
            # Only this window's duplicate text is removed; it can neither
            # extend source_supported_text nor become user-visible.
            continue
        if (
            not active_hypotheses[candidate]
            and contained
            and owner_conflict
        ):
            candidate_decision("hold", "contained_owner_dominant_awaiting_confirmation")
            continue
        # VAD on a normalized separated artifact can be positive even when
        # the owner VAD/STT is missing or its timestamps do not contain it.
        # Inspect ALL candidate speech intervals before those metadata gates.
        # A single independently supported interval protects a short response.
        if (normalize_lexical_text(transcripts[candidate]) and local
                and all(unsupported_source(row) for row in local)):
            evidence.update(suppressed=True, source_supported=False,
                            reason="unsupported_source_on_speech_intervals")
            vads[candidate]["source_supported"] = False
            texts[candidate] = ""
            vads[candidate]["speech_detected"] = False
            vads[candidate]["timestamps"] = []
            continue
        if active_hypotheses[candidate]:
            evidence["reason"] = "existing_utterance"
            continue
        # Apply the existing nonlexical rule symmetrically, including bootstrap
        # before either logical stream has acquired a PARTIAL.
        if not normalize_lexical_text(transcripts[candidate]):
            evidence.update(suppressed=True, reason="nonlexical_new_stream")
        elif (
            transcripts[owner].strip() and vad_results[owner].get("speech_detected")
            and intervals[candidate] and intervals[owner]
        ):
            evidence["reason"] = "insufficient_owner_evidence" if contained else "speech_outside_owner"
            short_independent_reply = bool(
                independent_observation and candidate_duration is not None
                and candidate_duration <= WEAK_SPEECH_MAX_DURATION_MS
            )
            if not contained and owner_established and not short_independent_reply:
                # A first observation outside an established owner's VAD can be
                # a separator tail just as easily as a new speaker.  Keep its
                # text as an internal tentative, but require the next overlap
                # window to provide fresh acoustic support before publication.
                # Existing candidate utterances took the continuation path
                # above, so this adds no global delay.
                candidate_decision("hold", "speech_outside_owner_awaiting_confirmation")
            if contained and (active_hypotheses[owner] or owner_only):
                evidence["reason"] = "ambiguous_acoustics" if all(row is not None for row in local) else "invalid_speech_audio"
                if all(row is not None for row in local):
                    # Any independently input-supported interval protects the
                    # entire short utterance, even if other intervals leak.
                    if independent_observation:
                        evidence["reason"] = "independent_input_support"
                    # Duplicate leakage still needs the timing/owner gates;
                    # absence of independent input support alone is not enough.
                    elif all(row["independent_input_correlation"] is not None and
                             row["independent_input_correlation"] <= MAX_UNSUPPORTED_INPUT_CORRELATION for row in local):
                        duplicate = all(row["pair_correlation"] is not None and
                                        row["pair_correlation"] >= HIGH_SOURCE_CORRELATION for row in local)
                        if duplicate:
                            evidence.update(suppressed=True,
                                            reason="duplicated_speech_without_independent_support")
        elif not intervals[candidate] or not intervals[owner]:
            evidence["reason"] = "missing_speech_intervals"
        if evidence["suppressed"]:
            evidence["source_supported"] = False
            vads[candidate]["source_supported"] = False
            texts[candidate] = ""
            vads[candidate]["speech_detected"] = False
            vads[candidate]["timestamps"] = []
    return texts, vads, {
        "applied": any(row["suppressed"] for row in decisions),
        "raw_to_logical_mapping": dict(mapping), "streams": decisions,
    }


def build_full_window_diagnostic(
    *,
    original: np.ndarray,
    speaker_0: np.ndarray,
    speaker_1: np.ndarray,
    vad_results: list[dict[str, Any]],
    raw_transcripts: list[str],
    candidate: bool,
    reasons: list[str],
    sample_rate: int,
) -> dict[str, Any]:
    """Describe the exact audio arrays used for one separation inference."""
    original_rms, original_peak = safe_audio_levels(original)
    speaker_0_rms, speaker_0_peak = safe_audio_levels(speaker_0)
    speaker_1_rms, speaker_1_peak = safe_audio_levels(speaker_1)
    original_0_corr, original_0_abs_corr = safe_waveform_correlation(original, speaker_0)
    original_1_corr, original_1_abs_corr = safe_waveform_correlation(original, speaker_1)
    pair_corr, pair_abs_corr = safe_waveform_correlation(speaker_0, speaker_1)
    vad_0 = vad_results[0] if len(vad_results) > 0 else {}
    vad_1 = vad_results[1] if len(vad_results) > 1 else {}
    return {
        "candidate": candidate,
        "candidate_reasons": list(reasons),
        "sample_rate": sample_rate,
        "sample_count": len(original),
        "duration_seconds": len(original) / sample_rate if sample_rate else None,
        "rms_normalization_note": RMS_NORMALIZATION_NOTE,
        "original": {"rms": original_rms, "peak": original_peak},
        "speaker_0": {
            "rms": speaker_0_rms,
            "peak": speaker_0_peak,
            "original_correlation": original_0_corr,
            "original_abs_correlation": original_0_abs_corr,
            "vad_speech_duration_ms": _finite_float(vad_0.get("speech_duration_ms")),
            "vad_speech_ratio": _finite_float(vad_0.get("speech_ratio")),
            "raw_stt": raw_transcripts[0] if raw_transcripts else "",
        },
        "speaker_1": {
            "rms": speaker_1_rms,
            "peak": speaker_1_peak,
            "original_correlation": original_1_corr,
            "original_abs_correlation": original_1_abs_corr,
            "vad_speech_duration_ms": _finite_float(vad_1.get("speech_duration_ms")),
            "vad_speech_ratio": _finite_float(vad_1.get("speech_ratio")),
            "raw_stt": raw_transcripts[1] if len(raw_transcripts) > 1 else "",
        },
        "speaker_pair": {
            "correlation": pair_corr,
            "abs_correlation": pair_abs_corr,
        },
    }


def save_candidate_full_window_wavs(
    *,
    enabled: bool,
    candidate: bool,
    output_dir: Path,
    window: int,
    original: np.ndarray,
    speaker_0: np.ndarray,
    speaker_1: np.ndarray,
    sample_rate: int,
) -> dict[str, Any]:
    """Save full inference inputs only for diagnostic candidate windows."""
    if not enabled:
        return {"attempted": False, "saved": False, "reason": "diagnostic_disabled"}
    if not candidate:
        return {"attempted": False, "saved": False, "reason": "not_candidate"}
    try:
        audio_by_name = {
            "original": np.ascontiguousarray(original, dtype=np.float32),
            "speaker_0": np.ascontiguousarray(speaker_0, dtype=np.float32),
            "speaker_1": np.ascontiguousarray(speaker_1, dtype=np.float32),
        }
        if not all(np.isfinite(audio).all() for audio in audio_by_name.values()):
            raise ValueError("full-window diagnostic audio must be finite")
        sample_counts = {len(audio) for audio in audio_by_name.values()}
        if len(sample_counts) != 1 or not sample_counts or next(iter(sample_counts)) == 0:
            raise ValueError("full-window diagnostic audio must have equal non-zero lengths")
        output_dir.mkdir(parents=True, exist_ok=True)
        files: dict[str, str] = {}
        for name, audio in audio_by_name.items():
            path = output_dir / f"window_{window:03d}_{name}.wav"
            # FLOAT preserves separator peaks above 1.0 that PCM_16 would clip.
            sf.write(path, audio, sample_rate, subtype="FLOAT")
            info = sf.info(path)
            if info.samplerate != sample_rate or info.frames != len(audio) or info.channels != 1:
                raise ValueError(f"invalid full-window diagnostic WAV: {path.name}")
            files[name] = str(path)
        return {
            "attempted": True,
            "saved": True,
            "subtype": "FLOAT",
            "sample_count": next(iter(sample_counts)),
            "files": files,
        }
    except Exception as exc:
        return {
            "attempted": True,
            "saved": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }


def _mean(records: list[dict[str, Any]], key: str) -> float | None:
    values = [_finite_float(record.get(key)) for record in records]
    finite = [value for value in values if value is not None]
    return float(np.mean(finite)) if finite else None


def build_summary(window_records: list[dict[str, Any]]) -> dict[str, Any]:
    active = [
        record
        for record in window_records
        if record["speaker_metrics"][1]["vad_speech_detected"]
    ]
    candidates = [record for record in active if record["secondary_artifact_candidate"]]
    non_candidates = [record for record in active if not record["secondary_artifact_candidate"]]
    relationship_counts = {
        relationship: sum(record["activity_relationship"] == relationship for record in window_records)
        for relationship in ("both_active", "only_speaker_0", "only_speaker_1", "none_active")
    }
    suppressed = [record for record in window_records if record.get("suppression", {}).get("applied")]
    full_window_records = [
        record["full_window_diagnostic"]
        for record in window_records
        if "full_window_diagnostic" in record
    ]
    saved_full_windows = [
        record for record in full_window_records if record.get("wav_save", {}).get("saved")
    ]
    failed_full_windows = [
        record for record in full_window_records if record.get("wav_save", {}).get("error")
    ]
    return {
        "config": SECONDARY_ARTIFACT_CONFIG,
        "speaker_1_active_window_count": len(active),
        "both_active_window_count": relationship_counts["both_active"],
        "only_speaker_1_active_window_count": relationship_counts["only_speaker_1"],
        "activity_relationship_counts": relationship_counts,
        "secondary_artifact_candidate_count": len(candidates),
        "secondary_artifact_candidate_ratio": len(candidates) / len(active) if active else 0.0,
        "candidate_window_indices": [record["window"] for record in candidates],
        "speaker_1_active_windows": active,
        "candidate_windows": candidates,
        "suppressed_nonlexical_window_count": len(suppressed),
        "suppressed_nonlexical_window_indices": [record["window"] for record in suppressed],
        "full_window_diagnostic": {
            "candidate_window_count": len(full_window_records),
            "saved_window_count": len(saved_full_windows),
            "failed_window_count": len(failed_full_windows),
            "saved_window_indices": [record["window"] for record in saved_full_windows],
            "failed_window_indices": [record["window"] for record in failed_full_windows],
            "rms_normalization_note": RMS_NORMALIZATION_NOTE,
        },
        "mean_speaker_1_to_speaker_0_energy_ratio": _mean(
            active, "speaker_1_to_speaker_0_energy_ratio"
        ),
        "mean_abs_waveform_pearson_correlation": _mean(
            active, "abs_waveform_pearson_correlation"
        ),
        "mean_text_similarity_normalized": (
            float(
                np.mean(
                    [
                        record["text_similarity"]["normalized"]
                        for record in active
                        if record["text_similarity"]["normalized"] is not None
                    ]
                )
            )
            if any(record["text_similarity"]["normalized"] is not None for record in active)
            else None
        ),
        "candidate_means": {
            "energy_ratio": _mean(candidates, "speaker_1_to_speaker_0_energy_ratio"),
            "abs_correlation": _mean(candidates, "abs_waveform_pearson_correlation"),
        },
        "non_candidate_means": {
            "energy_ratio": _mean(non_candidates, "speaker_1_to_speaker_0_energy_ratio"),
            "abs_correlation": _mean(non_candidates, "abs_waveform_pearson_correlation"),
        },
    }

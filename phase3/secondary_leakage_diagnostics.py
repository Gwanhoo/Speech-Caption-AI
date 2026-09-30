from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


ENERGY_EPSILON = 1e-12
SECONDARY_ARTIFACT_CONFIG = {
    # These label suspicious observations only. They never control the live pipeline.
    "primary_vad_ratio_min": 0.60,
    "secondary_vad_ratio_max": 0.50,
    "secondary_fragment_max_normalized_characters": 12,
    "high_abs_waveform_correlation": 0.85,
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

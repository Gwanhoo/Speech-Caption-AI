"""Window-to-window permutation alignment for two-source separation output.

Raw separator slots are intentionally kept distinct from downstream logical
speaker IDs.  The tracker uses only the audio shared by adjacent sliding
windows; it is not a long-term speaker recognizer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


DEFAULT_SILENCE_RMS = 1e-3
DEFAULT_SOURCE_CONFIDENCE = 0.45
DEFAULT_CONTINUITY_SCORE = 0.45
DEFAULT_SCORE_MARGIN = 0.12
DEFAULT_LOW_ENERGY_TAIL_RATIO = 0.25
HIGH_SOURCE_CORRELATION = 0.85
# Existing admission/validity dominance bounds, shared with tracking. These
# describe contradictory source evidence, not speech duration or loudness.
MIN_DOMINANT_INPUT_CORRELATION = 0.95
MAX_UNSUPPORTED_INPUT_CORRELATION = 0.20


def _finite_rms(audio: np.ndarray) -> float | None:
    values = np.asarray(audio, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        return None
    return float(np.sqrt(np.mean(values * values)))


def safe_absolute_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    """Return polarity-invariant normalized correlation, or None if unsafe."""
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if (
        left_values.ndim != 1
        or right_values.ndim != 1
        or left_values.shape != right_values.shape
        or not left_values.size
        or not np.isfinite(left_values).all()
        or not np.isfinite(right_values).all()
    ):
        return None
    left_centered = left_values - float(np.mean(left_values))
    right_centered = right_values - float(np.mean(right_values))
    denominator = math.sqrt(
        float(np.dot(left_centered, left_centered))
        * float(np.dot(right_centered, right_centered))
    )
    if not math.isfinite(denominator) or denominator <= 1e-12:
        return None
    value = abs(float(np.dot(left_centered, right_centered)) / denominator)
    return min(1.0, value) if math.isfinite(value) else None


def source_input_evidence(
    mixture: np.ndarray, owner: np.ndarray, candidate: np.ndarray,
) -> dict[str, Any] | None:
    """Compare input support after removing shared audio, independent of level.

    A quiet real source may have little full-window mixture correlation. Its
    component orthogonal to the owner can still explain the mixture residual.
    This is evidence about separated signals, not speaker recognition.
    """
    values = [np.asarray(audio, dtype=np.float64) for audio in (mixture, owner, candidate)]
    if any(value.ndim != 1 or value.size < 3 or not np.isfinite(value).all() for value in values):
        return None
    if len({value.shape for value in values}) != 1:
        return None
    values = [value - value.mean() for value in values]
    norms = [float(np.linalg.norm(value)) for value in values]
    if any(norm == 0 for norm in norms):
        return None
    source, primary, secondary = [value / norm for value, norm in zip(values, norms)]
    input_residual = source - np.dot(source, primary) * primary
    candidate_residual = secondary - np.dot(secondary, primary) * primary
    input_energy = float(np.dot(input_residual, input_residual))
    candidate_energy = float(np.dot(candidate_residual, candidate_residual))
    # Numerical degeneracy only, not a minimum speaker volume. This continues
    # to protect very quiet independent sources after per-output normalization.
    independent = (
        safe_absolute_correlation(
            input_residual / np.sqrt(input_energy),
            candidate_residual / np.sqrt(candidate_energy),
        ) if min(input_energy, candidate_energy) > 1e-14 else 0.0
    )
    return {
        "pair_correlation": safe_absolute_correlation(primary, secondary),
        "owner_input_correlation": safe_absolute_correlation(source, primary),
        "candidate_input_correlation": safe_absolute_correlation(source, secondary),
        "independent_input_correlation": independent,
        "input_residual_energy_fraction": input_energy,
        "candidate_residual_energy_fraction": candidate_energy,
    }


def unsupported_source(evidence: dict[str, Any] | None) -> bool:
    """Require positive owner evidence AND negative candidate/residual evidence."""
    return bool(
        evidence is not None
        and evidence["owner_input_correlation"] is not None
        and evidence["owner_input_correlation"] >= MIN_DOMINANT_INPUT_CORRELATION
        and evidence["candidate_input_correlation"] is not None
        and evidence["candidate_input_correlation"] <= MAX_UNSUPPORTED_INPUT_CORRELATION
        and evidence["independent_input_correlation"] is not None
        and evidence["independent_input_correlation"] <= MAX_UNSUPPORTED_INPUT_CORRELATION
    )


@dataclass(frozen=True)
class SpeakerAssignment:
    speakers: tuple[np.ndarray, np.ndarray]
    diagnostic: dict[str, Any]


class PersistentSpeakerTracker:
    """Align two raw separator slots to two logical IDs across adjacent windows."""

    IDENTITY = (0, 1)  # raw slot -> logical speaker
    SWAP = (1, 0)

    def __init__(
        self,
        *,
        overlap_samples: int,
        silence_rms: float = DEFAULT_SILENCE_RMS,
        source_confidence: float = DEFAULT_SOURCE_CONFIDENCE,
        continuity_score: float = DEFAULT_CONTINUITY_SCORE,
        score_margin: float = DEFAULT_SCORE_MARGIN,
        low_energy_tail_ratio: float = DEFAULT_LOW_ENERGY_TAIL_RATIO,
    ) -> None:
        if overlap_samples < 1:
            raise ValueError("overlap_samples must be positive")
        if min(
            silence_rms,
            source_confidence,
            continuity_score,
            score_margin,
            low_energy_tail_ratio,
        ) < 0:
            raise ValueError("speaker tracking thresholds must be non-negative")
        self.overlap_samples = overlap_samples
        self.silence_rms = silence_rms
        self.source_confidence = source_confidence
        self.continuity_score = continuity_score
        self.score_margin = score_margin
        self.low_energy_tail_ratio = low_energy_tail_ratio
        self.reset()

    def reset(self) -> None:
        self._mapping = self.IDENTITY
        self._previous_window: int | None = None
        self._previous_logical_tails: tuple[np.ndarray, np.ndarray] | None = None
        self._previous_logical_confidence = (0.0, 0.0)
        self._previous_input_rms: float | None = None
        self._initialized = False
        self._confirmed_logical_speakers: set[int] = set()

    @staticmethod
    def _logical_speakers(
        raw_speakers: tuple[np.ndarray, np.ndarray], mapping: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        logical: list[np.ndarray | None] = [None, None]
        for raw_slot, logical_speaker in enumerate(mapping):
            logical[logical_speaker] = raw_speakers[raw_slot]
        assert logical[0] is not None and logical[1] is not None
        return logical[0], logical[1]

    def _score_mapping(
        self,
        mapping: tuple[int, int],
        matrix: list[list[float | None]],
        current_confidence: tuple[float, float],
    ) -> float | None:
        weighted_score = 0.0
        weight_total = 0.0
        for raw_slot, logical_speaker in enumerate(mapping):
            previous_reliability = self._previous_logical_confidence[logical_speaker]
            current_reliability = current_confidence[raw_slot]
            similarity = matrix[logical_speaker][raw_slot]
            if (
                previous_reliability < self.source_confidence
                or current_reliability < self.source_confidence
                or similarity is None
            ):
                continue
            weight = math.sqrt(previous_reliability * current_reliability)
            weighted_score += weight * similarity
            weight_total += weight
        return weighted_score / weight_total if weight_total else None

    def assign(
        self,
        *,
        window: int,
        raw_speakers: tuple[np.ndarray, np.ndarray],
        mixture: np.ndarray,
        pre_separation_silence: bool = False,
    ) -> SpeakerAssignment:
        raw = tuple(np.ascontiguousarray(audio, dtype=np.float32) for audio in raw_speakers)
        if len(raw) != 2 or raw[0].ndim != 1 or raw[1].ndim != 1:
            raise ValueError("PersistentSpeakerTracker requires two 1-D raw slots")
        if raw[0].shape != raw[1].shape or len(raw[0]) < self.overlap_samples:
            raise ValueError("Raw slots must share a shape and contain the configured overlap")
        mixture_values = np.ascontiguousarray(mixture, dtype=np.float32)
        if mixture_values.shape != raw[0].shape:
            raise ValueError("Mixture and raw speaker windows must share a shape")

        input_rms = _finite_rms(mixture_values)
        is_silence = (
            pre_separation_silence
            or input_rms is None
            or input_rms <= self.silence_rms
        )
        current_confidence = tuple(
            safe_absolute_correlation(mixture_values, audio) or 0.0 for audio in raw
        )
        active_raw_slots = [
            index
            for index, confidence in enumerate(current_confidence)
            if confidence >= self.source_confidence
        ]
        input_rms_ratio = (
            input_rms / self._previous_input_rms
            if input_rms is not None
            and self._previous_input_rms is not None
            and self._previous_input_rms > self.silence_rms
            else None
        )
        low_energy_tail = bool(
            not is_silence
            and input_rms_ratio is not None
            and input_rms_ratio <= self.low_energy_tail_ratio
            and len(active_raw_slots) <= 1
        )
        consecutive = self._previous_window is not None and window == self._previous_window + 1
        matrix: list[list[float | None]] = [[None, None], [None, None]]
        if consecutive and self._previous_logical_tails is not None:
            current_heads = (
                raw[0][: self.overlap_samples],
                raw[1][: self.overlap_samples],
            )
            for logical_speaker in (0, 1):
                for raw_slot in (0, 1):
                    matrix[logical_speaker][raw_slot] = safe_absolute_correlation(
                        self._previous_logical_tails[logical_speaker],
                        current_heads[raw_slot],
                    )

        identity_score = self._score_mapping(self.IDENTITY, matrix, current_confidence)
        swap_score = self._score_mapping(self.SWAP, matrix, current_confidence)
        previous_mapping = self._mapping
        mapping = previous_mapping
        method = "uncertain_hold"

        if is_silence:
            method = "silence_hold"
        elif not self._initialized:
            if len(active_raw_slots) == 1 and active_raw_slots[0] == 1:
                mapping = self.SWAP
            else:
                mapping = self.IDENTITY
            method = "bootstrap"
            self._initialized = True
        elif not consecutive:
            method = "uncertain_hold"
        else:
            candidates = [
                (self.IDENTITY, identity_score),
                (self.SWAP, swap_score),
            ]
            valid = [(candidate, score) for candidate, score in candidates if score is not None]
            if len(valid) == 1 and valid[0][1] >= self.continuity_score:
                mapping = valid[0][0]
                method = (
                    "single_active_hold"
                    if len(active_raw_slots) <= 1 and mapping == previous_mapping
                    else "overlap_continuity"
                )
            elif len(valid) == 2:
                best_mapping, best_score = max(valid, key=lambda value: value[1])
                other_score = min(score for _, score in valid)
                if (
                    best_score >= self.continuity_score
                    and best_score - other_score >= self.score_margin
                ):
                    mapping = best_mapping
                    method = (
                        "single_active_hold"
                        if len(active_raw_slots) <= 1 and mapping == previous_mapping
                        else "overlap_continuity"
                    )
                else:
                    method = "uncertain_hold"

        # A separator can preserve the old speech best in a slot which becomes
        # artifact in the NEW part of the window. Full-window input correlation
        # still qualifies that slot to win the overlap vote. When BOTH heads
        # follow the sole previous source, use independently checked new audio
        # to keep its logical identity on the actual continuation. Do not hold
        # raw indices: the continuing source itself may have changed raw slot.
        previous_active = [i for i, value in enumerate(self._previous_logical_confidence)
                           if value >= self.source_confidence]
        continuity_evidence: dict[str, Any] = {
            "previous_active_logical": previous_active,
            "overlap_selected_mapping": list(mapping),
            "new_region_source_evidence": {},
            "selected_raw": None,
            "applied": False,
        }
        continuation_raw = None
        if (consecutive and not is_silence and not low_energy_tail
                and len(previous_active) == 1):
            owner = previous_active[0]
            if all(value is not None and value >= self.continuity_score
                   for value in matrix[owner]):
                for raw_slot in active_raw_slots:
                    evidence = source_input_evidence(
                        mixture_values[self.overlap_samples:],
                        raw[raw_slot][self.overlap_samples:],
                        raw[1 - raw_slot][self.overlap_samples:],
                    )
                    continuity_evidence["new_region_source_evidence"][str(raw_slot)] = evidence
                    if unsupported_source(evidence):
                        continuation_raw = raw_slot
                        selected = self.IDENTITY if raw_slot == owner else self.SWAP
                        continuity_evidence["selected_raw"] = raw_slot
                        continuity_evidence["applied"] = selected != mapping
                        if selected != mapping:
                            mapping = selected
                            method = "single_source_continuity"
                        break

        # A correlation coefficient ignores absolute level.  At speech end a
        # low-energy device/separator tail can therefore look continuous enough
        # to swap identities even though it is not usable speech.  Only veto a
        # mapping change; never use this guard to suppress or relabel audio.
        if low_energy_tail and mapping != previous_mapping:
            mapping = previous_mapping
            method = "low_energy_tail_hold"

        logical_speakers = self._logical_speakers(raw, mapping)
        logical_confidence = [0.0, 0.0]
        if not is_silence and not low_energy_tail:
            for raw_slot in active_raw_slots:
                logical_speaker = mapping[raw_slot]
                logical_confidence[logical_speaker] = current_confidence[raw_slot]
                self._confirmed_logical_speakers.add(logical_speaker)

        # Two references containing the same overlap audio cannot vote as two
        # independent identities in the next window. Prefer the reference with
        # clearly stronger full-window mixture support; leave tied/independent
        # sources alone. This does not suppress either source or veto a swap.
        tail_correlation = safe_absolute_correlation(
            logical_speakers[0][-self.overlap_samples:],
            logical_speakers[1][-self.overlap_samples:],
        )
        excluded_reference = None
        if (tail_correlation is not None and tail_correlation >= HIGH_SOURCE_CORRELATION
                and abs(logical_confidence[0] - logical_confidence[1]) >= self.score_margin
                and max(logical_confidence) >= self.source_confidence):
            excluded_reference = min((0, 1), key=lambda speaker: logical_confidence[speaker])
            logical_confidence[excluded_reference] = 0.0

        # The old speech in the competing head must not establish a second
        # identity for the next window; its tail is the unsupported new audio.
        excluded_residual_reference = None
        if continuation_raw is not None:
            excluded_residual_reference = mapping[1 - continuation_raw]
            logical_confidence[excluded_residual_reference] = 0.0

        self._mapping = mapping
        self._previous_window = window
        self._previous_logical_tails = (
            logical_speakers[0][-self.overlap_samples :].copy(),
            logical_speakers[1][-self.overlap_samples :].copy(),
        )
        self._previous_logical_confidence = (
            logical_confidence[0],
            logical_confidence[1],
        )
        self._previous_input_rms = input_rms

        if identity_score is not None and swap_score is not None:
            confidence = abs(identity_score - swap_score)
        else:
            confidence = next(
                (score for score in (identity_score, swap_score) if score is not None),
                0.0,
            )
        diagnostic = {
            "raw_to_logical_mapping": {
                str(raw_slot): logical_speaker
                for raw_slot, logical_speaker in enumerate(mapping)
            },
            "assignment_method": method,
            "identity_score": identity_score,
            "swap_score": swap_score,
            "confidence": confidence,
            "mapping_changed": mapping != previous_mapping,
            "overlap_similarity_matrix": {
                f"logical_{logical_speaker}": {
                    f"raw_{raw_slot}": matrix[logical_speaker][raw_slot]
                    for raw_slot in (0, 1)
                }
                for logical_speaker in (0, 1)
            },
            "raw_input_similarity": {
                str(raw_slot): current_confidence[raw_slot] for raw_slot in (0, 1)
            },
            "active_raw_slots": active_raw_slots,
            "tail_source_correlation": tail_correlation,
            "excluded_duplicate_tail_reference": excluded_reference,
            "single_source_continuity": continuity_evidence,
            "excluded_residual_tail_reference": excluded_residual_reference,
            "next_reference_confidence": list(logical_confidence),
            "confirmed_logical_speakers": sorted(self._confirmed_logical_speakers),
            "input_rms": input_rms,
            "input_rms_ratio_to_previous": input_rms_ratio,
            "input_silence": is_silence,
            "low_energy_tail": low_energy_tail,
            "low_energy_tail_ratio_threshold": self.low_energy_tail_ratio,
            "window_consecutive": consecutive,
            "overlap_samples": self.overlap_samples,
            "polarity_invariant_correlation": True,
            "limitations": (
                "Adjacent-window permutation alignment only; long-gap speaker "
                "re-identification requires speaker embeddings."
            ),
        }
        return SpeakerAssignment(speakers=logical_speakers, diagnostic=diagnostic)

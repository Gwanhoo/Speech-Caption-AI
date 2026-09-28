from __future__ import annotations

import numpy as np

from speaker_tracking import PersistentSpeakerTracker, safe_absolute_correlation


OVERLAP = 800
NEW = 1600
WINDOW = OVERLAP + NEW


def continuation(tail: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.concatenate((tail, rng.normal(size=NEW))).astype(np.float32)


def assert_mapping(result, expected: dict[str, int]) -> None:
    assert result.diagnostic["raw_to_logical_mapping"] == expected


def test_stable_identity() -> None:
    rng = np.random.default_rng(1)
    a0 = rng.normal(size=WINDOW).astype(np.float32)
    b0 = rng.normal(size=WINDOW).astype(np.float32)
    a1 = continuation(a0[-OVERLAP:], rng)
    b1 = continuation(b0[-OVERLAP:], rng)
    tracker = PersistentSpeakerTracker(overlap_samples=OVERLAP)
    tracker.assign(window=0, raw_speakers=(a0, b0), mixture=a0 + b0)
    result = tracker.assign(window=1, raw_speakers=(a1, b1), mixture=a1 + b1)
    assert_mapping(result, {"0": 0, "1": 1})
    assert result.diagnostic["assignment_method"] == "overlap_continuity"


def test_swapped_raw_outputs() -> None:
    rng = np.random.default_rng(2)
    a0 = rng.normal(size=WINDOW).astype(np.float32)
    b0 = rng.normal(size=WINDOW).astype(np.float32)
    a1 = continuation(a0[-OVERLAP:], rng)
    b1 = continuation(b0[-OVERLAP:], rng)
    tracker = PersistentSpeakerTracker(overlap_samples=OVERLAP)
    tracker.assign(window=0, raw_speakers=(a0, b0), mixture=a0 + b0)
    # Separation polarity can flip without changing speaker identity.
    result = tracker.assign(window=1, raw_speakers=(-b1, -a1), mixture=a1 + b1)
    assert_mapping(result, {"0": 1, "1": 0})
    assert result.diagnostic["mapping_changed"]


def test_single_to_overlap() -> None:
    rng = np.random.default_rng(3)
    a0 = rng.normal(size=WINDOW).astype(np.float32)
    residual = rng.normal(size=WINDOW).astype(np.float32) * 0.01
    a1 = continuation(a0[-OVERLAP:], rng)
    b1 = rng.normal(size=WINDOW).astype(np.float32)
    tracker = PersistentSpeakerTracker(overlap_samples=OVERLAP)
    first = tracker.assign(window=0, raw_speakers=(a0, residual), mixture=a0)
    assert first.diagnostic["confirmed_logical_speakers"] == [0]
    result = tracker.assign(window=1, raw_speakers=(b1, a1), mixture=a1 + b1)
    assert_mapping(result, {"0": 1, "1": 0})
    assert safe_absolute_correlation(result.speakers[0][:OVERLAP], a0[-OVERLAP:]) > 0.99
    assert result.diagnostic["confirmed_logical_speakers"] == [0, 1]


def test_single_active_residual() -> None:
    rng = np.random.default_rng(4)
    a0 = rng.normal(size=WINDOW).astype(np.float32)
    residual0 = rng.normal(size=WINDOW).astype(np.float32) * 0.01
    a1 = continuation(a0[-OVERLAP:], rng)
    residual1 = rng.normal(size=WINDOW).astype(np.float32) * 0.01
    tracker = PersistentSpeakerTracker(overlap_samples=OVERLAP)
    tracker.assign(window=0, raw_speakers=(a0, residual0), mixture=a0)
    result = tracker.assign(window=1, raw_speakers=(a1, residual1), mixture=a1)
    assert_mapping(result, {"0": 0, "1": 1})
    assert result.diagnostic["confirmed_logical_speakers"] == [0]
    assert result.diagnostic["assignment_method"] == "single_active_hold"


def test_silence_holds_mapping() -> None:
    rng = np.random.default_rng(5)
    a0 = rng.normal(size=WINDOW).astype(np.float32)
    b0 = rng.normal(size=WINDOW).astype(np.float32)
    tracker = PersistentSpeakerTracker(overlap_samples=OVERLAP)
    tracker.assign(window=0, raw_speakers=(a0, b0), mixture=a0 + b0)
    a1 = continuation(a0[-OVERLAP:], rng)
    b1 = continuation(b0[-OVERLAP:], rng)
    swapped = tracker.assign(window=1, raw_speakers=(b1, a1), mixture=a1 + b1)
    assert_mapping(swapped, {"0": 1, "1": 0})
    zeros = np.zeros(WINDOW, dtype=np.float32)
    result = tracker.assign(window=2, raw_speakers=(zeros, zeros), mixture=zeros)
    assert_mapping(result, {"0": 1, "1": 0})
    assert result.diagnostic["assignment_method"] == "silence_hold"
    assert not result.diagnostic["mapping_changed"]


def test_low_confidence_holds_previous_mapping() -> None:
    rng = np.random.default_rng(6)
    shared0 = rng.normal(size=WINDOW).astype(np.float32)
    shared1 = continuation(shared0[-OVERLAP:], rng)
    tracker = PersistentSpeakerTracker(overlap_samples=OVERLAP)
    tracker.assign(window=0, raw_speakers=(shared0, shared0), mixture=shared0)
    result = tracker.assign(window=1, raw_speakers=(shared1, shared1), mixture=shared1)
    assert_mapping(result, {"0": 0, "1": 1})
    assert result.diagnostic["assignment_method"] == "uncertain_hold"


def test_non_finite_and_zero_do_not_crash() -> None:
    tracker = PersistentSpeakerTracker(overlap_samples=OVERLAP)
    zeros = np.zeros(WINDOW, dtype=np.float32)
    bad = zeros.copy()
    bad[0] = np.nan
    other = zeros.copy()
    other[1] = np.inf
    result = tracker.assign(window=0, raw_speakers=(bad, other), mixture=bad)
    assert result.diagnostic["assignment_method"] == "silence_hold"
    assert result.diagnostic["identity_score"] is None
    assert result.diagnostic["swap_score"] is None


def test_low_energy_speech_tail_cannot_swap_mapping() -> None:
    rng = np.random.default_rng(7)
    a0 = rng.normal(size=WINDOW).astype(np.float32)
    b0 = rng.normal(size=WINDOW).astype(np.float32)
    a1 = continuation(a0[-OVERLAP:], rng)
    b1 = continuation(b0[-OVERLAP:], rng)
    tracker = PersistentSpeakerTracker(overlap_samples=OVERLAP)
    tracker.assign(window=0, raw_speakers=(a0, b0), mixture=a0 + b0)
    tracker.assign(window=1, raw_speakers=(a1, b1), mixture=a1 + b1)

    # A decaying separator/device tail appears in raw1, which would otherwise
    # look like a strong swap because correlation is scale-invariant.
    tail_a = continuation(a1[-OVERLAP:], rng) * 0.1
    tail_residual = rng.normal(size=WINDOW).astype(np.float32) * 0.1
    result = tracker.assign(
        window=2,
        raw_speakers=(tail_residual, tail_a),
        mixture=tail_a,
    )
    assert result.diagnostic["swap_score"] > result.diagnostic["identity_score"]
    assert result.diagnostic["low_energy_tail"]
    assert result.diagnostic["assignment_method"] == "low_energy_tail_hold"
    assert_mapping(result, {"0": 0, "1": 1})
    assert not result.diagnostic["mapping_changed"]


def main() -> None:
    test_stable_identity()
    test_swapped_raw_outputs()
    test_single_to_overlap()
    test_single_active_residual()
    test_silence_holds_mapping()
    test_low_confidence_holds_previous_mapping()
    test_non_finite_and_zero_do_not_crash()
    test_low_energy_speech_tail_cannot_swap_mapping()
    print("persistent speaker tracking tests: PASS")


if __name__ == "__main__":
    main()

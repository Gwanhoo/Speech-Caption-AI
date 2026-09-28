from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from secondary_leakage_diagnostics import (
    SecondaryValidityTracker,
    activity_relationship,
    build_secondary_validity_summary,
    build_full_window_diagnostic,
    build_window_diagnostic,
    safe_audio_levels,
    safe_energy_ratio,
    safe_waveform_correlation,
    save_candidate_full_window_wavs,
    secondary_transcript_suppression_reason,
)


def vad(active: bool, ratio: float) -> dict[str, float | bool]:
    return {
        "speech_detected": active,
        "speech_ratio": ratio,
        "speech_duration_ms": ratio * 3000,
    }


def diagnostic(left: np.ndarray, right: np.ndarray, vads: list[dict], texts: list[str]) -> dict:
    return build_window_diagnostic(
        speaker_0=left,
        speaker_1=right,
        vad_results=vads,
        transcripts=texts,
    )


def main() -> None:
    zeros = np.zeros(16, dtype=np.float32)
    correlation, absolute = safe_waveform_correlation(zeros, zeros)
    assert correlation is None and absolute is None
    assert safe_energy_ratio(0.0, 0.0) == 0.0
    assert safe_energy_ratio(float("nan"), 1.0) is None
    zero_record = diagnostic(zeros, zeros, [vad(False, 0.0), vad(False, 0.0)], ["", ""])
    assert zero_record["activity_relationship"] == "none_active"
    assert not zero_record["secondary_artifact_candidate"]

    identical = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    identical_record = diagnostic(
        identical, identical, [vad(True, 0.9), vad(True, 0.9)], ["같은 문장", "같은 문장"]
    )
    assert identical_record["abs_waveform_pearson_correlation"] is not None
    assert identical_record["abs_waveform_pearson_correlation"] > 0.999
    assert identical_record["text_similarity"]["normalized"] == 1.0
    assert not identical_record["secondary_artifact_candidate"]

    sine = np.sin(np.linspace(0, 2 * np.pi, 256, dtype=np.float32))
    cosine = np.cos(np.linspace(0, 2 * np.pi, 256, dtype=np.float32))
    different_record = diagnostic(
        sine, cosine, [vad(True, 0.8), vad(False, 0.0)], ["첫 번째", ""]
    )
    assert different_record["abs_waveform_pearson_correlation"] is not None
    assert different_record["abs_waveform_pearson_correlation"] < 0.02
    assert activity_relationship(vad(True, 0.8), vad(False, 0.0)) == "only_speaker_0"

    quiet = identical * 0.1
    levels_record = diagnostic(
        identical, quiet, [vad(True, 0.9), vad(True, 0.9)], ["긴 실제 발화", "네"]
    )
    expected_0 = safe_audio_levels(identical)
    expected_1 = safe_audio_levels(quiet)
    assert np.isclose(levels_record["speaker_metrics"][0]["rms"], expected_0[0])
    assert np.isclose(levels_record["speaker_metrics"][1]["rms"], expected_1[0])
    assert not np.isclose(
        levels_record["speaker_metrics"][0]["rms"],
        levels_record["speaker_metrics"][1]["rms"],
    )
    assert secondary_transcript_suppression_reason(levels_record, "네") is None
    assert secondary_transcript_suppression_reason(levels_record, "응!") is None
    assert secondary_transcript_suppression_reason(levels_record, ".") == (
        "secondary_nonlexical_transcript"
    )

    candidate_record = diagnostic(
        sine, cosine * 0.5, [vad(True, 0.9), vad(True, 0.2)], ["긴 실제 발화", "네"]
    )
    assert candidate_record["secondary_artifact_candidate"]
    assert "primary_vad_dominant" in candidate_record["reasons"]
    assert "secondary_vad_sparse" in candidate_record["reasons"]
    assert secondary_transcript_suppression_reason(candidate_record, "네") is None
    assert secondary_transcript_suppression_reason(candidate_record, "응!") is None
    assert secondary_transcript_suppression_reason(candidate_record, "하") is None
    assert secondary_transcript_suppression_reason(candidate_record, ".") == (
        "secondary_nonlexical_transcript"
    )

    original = np.linspace(-0.6, 0.6, 160, dtype=np.float32)
    full_window = build_full_window_diagnostic(
        original=original,
        speaker_0=original * 0.8,
        speaker_1=np.flip(original) * 0.4,
        vad_results=[vad(True, 0.9), vad(True, 0.2)],
        raw_transcripts=["실제 문장", "네"],
        candidate=True,
        reasons=["primary_vad_dominant", "secondary_vad_sparse"],
        sample_rate=16_000,
    )
    assert full_window["sample_count"] == 160
    assert full_window["speaker_0"]["original_abs_correlation"] > 0.999
    assert full_window["speaker_1"]["original_correlation"] < -0.999
    assert full_window["speaker_1"]["raw_stt"] == "네"

    silence = np.zeros(16, dtype=np.float32)
    non_finite = np.full(16, np.nan, dtype=np.float32)
    safe_full_window = build_full_window_diagnostic(
        original=non_finite,
        speaker_0=silence,
        speaker_1=np.full(16, np.inf, dtype=np.float32),
        vad_results=[],
        raw_transcripts=["", ""],
        candidate=True,
        reasons=[],
        sample_rate=16_000,
    )
    assert safe_full_window["original"]["rms"] is None
    assert safe_full_window["speaker_1"]["peak"] is None
    assert safe_full_window["speaker_pair"]["correlation"] is None

    # Validity evidence is observation-only: only accumulated positive temporal
    # evidence raises a lexical secondary from unknown to likely/strong.
    tracker = SecondaryValidityTracker()
    punctuation = tracker.observe(
        window=1,
        diagnostic=diagnostic(sine, cosine, [vad(True, 0.9), vad(True, 0.2)], ["", "."]),
        raw_stt=".",
    )
    assert punctuation is not None and punctuation["classification"] == "nonlexical"

    short_response = tracker.observe(
        window=2,
        diagnostic=diagnostic(sine, cosine, [vad(True, 0.9), vad(True, 0.1)], ["", "네"]),
        raw_stt="네",
    )
    assert short_response is not None
    assert short_response["classification"] == "unknown_lexical"
    assert secondary_transcript_suppression_reason(candidate_record, "네") is None

    low_vad_lexical = tracker.observe(
        window=3,
        diagnostic=diagnostic(sine, cosine, [vad(True, 0.9), vad(True, 0.05)], ["", "짧은 발화"]),
        raw_stt="짧은 발화",
    )
    assert low_vad_lexical is not None
    assert low_vad_lexical["classification"] == "unknown_lexical"

    continuity_tracker = SecondaryValidityTracker()
    first = continuity_tracker.observe(
        window=10,
        diagnostic=diagnostic(sine, cosine, [vad(True, 0.9), vad(True, 0.9)], ["", "오늘은 게임"]),
        raw_stt="오늘은 게임",
    )
    second = continuity_tracker.observe(
        window=11,
        diagnostic=diagnostic(sine, cosine, [vad(True, 0.9), vad(True, 0.9)], ["", "게임 이야기"]),
        raw_stt="게임 이야기",
    )
    third = continuity_tracker.observe(
        window=12,
        diagnostic=diagnostic(sine, cosine, [vad(True, 0.9), vad(True, 0.9)], ["", "이야기 계속"]),
        raw_stt="이야기 계속",
    )
    assert first is not None and first["classification"] == "unknown_lexical"
    assert second is not None and second["classification"] == "likely_secondary"
    assert second["temporal_text"]["continuity"]["detected"]
    assert third is not None and third["classification"] == "strong_secondary"

    low_correlation_full_window = {
        "speaker_0": {"rms": 1.0, "peak": 1.0, "original_abs_correlation": 0.99},
        "speaker_1": {"rms": 1.0, "peak": 1.0, "original_abs_correlation": 0.01},
        "speaker_pair": {"correlation": 0.01, "abs_correlation": 0.01},
    }
    acoustic_tracker = SecondaryValidityTracker()
    low_correlation_lexical = acoustic_tracker.observe(
        window=20,
        diagnostic=diagnostic(sine, cosine, [vad(True, 0.9), vad(True, 0.2)], ["", "응"]),
        raw_stt="응",
        full_window_diagnostic=low_correlation_full_window,
    )
    assert low_correlation_lexical is not None
    assert low_correlation_lexical["classification"] == "unknown_lexical"
    assert "primary_dominant_acoustic_pattern" in low_correlation_lexical[
        "negative_evidence_reasons"
    ]

    disabled_tracker = SecondaryValidityTracker(enabled=False)
    assert (
        disabled_tracker.observe(
            window=1, diagnostic=candidate_record, raw_stt="네"
        )
        is None
    )
    validity_summary = build_secondary_validity_summary(
        [
            {"window": 10, "validity_evidence": first},
            {"window": 11, "validity_evidence": second},
            {"window": 12, "validity_evidence": third},
        ]
    )
    assert validity_summary["likely_secondary_window_indices"] == [11]
    assert validity_summary["strong_secondary_window_indices"] == [12]
    assert validity_summary["continuity_detected_count"] == 2

    with tempfile.TemporaryDirectory() as temporary_directory:
        output_dir = Path(temporary_directory) / "diagnostic_windows"
        disabled = save_candidate_full_window_wavs(
            enabled=False,
            candidate=True,
            output_dir=output_dir,
            window=3,
            original=original,
            speaker_0=original * 0.8,
            speaker_1=np.flip(original) * 0.4,
            sample_rate=16_000,
        )
        assert not disabled["attempted"] and not output_dir.exists()

        saved = save_candidate_full_window_wavs(
            enabled=True,
            candidate=True,
            output_dir=output_dir,
            window=3,
            original=original,
            speaker_0=original * 0.8,
            speaker_1=np.flip(original) * 0.4,
            sample_rate=16_000,
        )
        assert saved["saved"] and saved["sample_count"] == len(original)
        for path_string in saved["files"].values():
            info = sf.info(path_string)
            assert info.samplerate == 16_000 and info.frames == len(original) and info.channels == 1

        failed = save_candidate_full_window_wavs(
            enabled=True,
            candidate=True,
            output_dir=output_dir,
            window=4,
            original=non_finite,
            speaker_0=silence,
            speaker_1=silence,
            sample_rate=16_000,
        )
        assert failed["attempted"] and not failed["saved"] and "error" in failed
    print("PASS: secondary leakage diagnostic tests")


if __name__ == "__main__":
    main()

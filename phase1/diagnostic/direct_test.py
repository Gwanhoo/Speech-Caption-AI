from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = ROOT / "phase1" / "reference"
OUTPUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "phase0"))

from mix import TARGET_SAMPLE_RATE, mix_wavs  # noqa: E402
from separate import MODEL_NAME, separate_wav  # noqa: E402


def read_mono(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float64")
    if sample_rate != TARGET_SAMPLE_RATE or audio.ndim != 1:
        raise ValueError(f"Expected 16 kHz mono WAV: {path}")
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError(f"Empty or non-finite audio: {path}")
    return audio


def correlation(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    return float(np.dot(reference, estimate) / (np.linalg.norm(reference) * np.linalg.norm(estimate)))


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    projected = np.dot(estimate, reference) / np.dot(reference, reference) * reference
    residual = estimate - projected
    return float(10 * np.log10(np.sum(projected**2) / np.sum(residual**2)))


def main() -> int:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    reference_paths = [REFERENCE_DIR / f"speaker_{name}.wav" for name in ("a", "b")]
    references = [read_mono(path) for path in reference_paths]
    original_hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in reference_paths]
    common_frames = min(map(len, references))

    mixed_path = OUTPUT_DIR / "direct_mixed.wav"
    mix_wavs(*reference_paths, mixed_path)
    mixed = read_mono(mixed_path)
    peak = float(np.max(np.abs(mixed)))
    rms = float(np.sqrt(np.mean(mixed**2)))
    print(f"Mixed: {len(mixed) / TARGET_SAMPLE_RATE:.5f} sec, peak={peak:.6f}, RMS={rms:.6f}", flush=True)
    if peak > 0.98:
        raise ValueError("Mixed audio exceeds Phase 0 peak limit")

    print(f"Separating: {MODEL_NAME} (cpu)", flush=True)
    started = perf_counter()
    temporary_paths = separate_wav(mixed_path, OUTPUT_DIR)
    separation_time = perf_counter() - started
    output_paths = [OUTPUT_DIR / f"direct_speaker_{number}.wav" for number in (1, 2)]
    for source, destination in zip(temporary_paths, output_paths):
        source.rename(destination)
    outputs = [read_mono(path) for path in output_paths]
    if any(len(output) != len(mixed) for output in outputs):
        raise ValueError("Separation output length differs from the mixture")

    if any(hashlib.sha256(path.read_bytes()).hexdigest() != digest for path, digest in zip(reference_paths, original_hashes)):
        raise RuntimeError("Reference WAV changed during the test")

    print(f"Separation time: {separation_time:.2f} sec")
    print(f"Full span: {len(mixed) / TARGET_SAMPLE_RATE:.5f} sec")
    print(f"Common reference span: {common_frames / TARGET_SAMPLE_RATE:.5f} sec")
    full_references = [np.pad(reference, (0, len(mixed) - len(reference))) for reference in references]
    for label, frame_count, reference_set in (
        ("full", len(mixed), full_references),
        ("common", common_frames, references),
    ):
        print(f"[{label}]")
        scores = np.zeros((2, 2))
        for output_index, output in enumerate(outputs):
            for reference_index, reference in enumerate(reference_set):
                measured_reference = reference[:frame_count]
                measured_output = output[:frame_count]
                corr = correlation(measured_reference, measured_output)
                score = si_sdr(measured_reference, measured_output)
                scores[output_index, reference_index] = score
                print(
                    f"output_{output_index + 1} vs reference_{'AB'[reference_index]}: "
                    f"correlation={corr:.6f}, SI-SDR={score:.2f} dB"
                )
        direct_score = scores[0, 0] + scores[1, 1]
        swapped_score = scores[0, 1] + scores[1, 0]
        permutation = "1=A, 2=B" if direct_score >= swapped_score else "1=B, 2=A"
        print(f"Best permutation ({label} SI-SDR sum): {permutation}")

    for number, output in enumerate(outputs, start=1):
        nonzero = np.flatnonzero(output != 0)
        trailing_zero = len(output) - 1 - nonzero[-1]
        print(f"Output {number} trailing exact zeros: {trailing_zero / TARGET_SAMPLE_RATE:.5f} sec")
        print(f"Saved: {output_paths[number - 1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

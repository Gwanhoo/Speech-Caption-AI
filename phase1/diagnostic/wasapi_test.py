from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.signal import correlate

from direct_test import correlation, read_mono, si_sdr
from playback_capture import estimate_offset


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = ROOT / "phase1" / "reference"
sys.path.insert(0, str(ROOT / "phase0"))

from separate import MODEL_NAME, separate_wav  # noqa: E402


def best_offset_near(audio: np.ndarray, reference: np.ndarray, center: int, radius: int) -> tuple[int, float]:
    centered = reference - reference.mean()
    candidates = correlate(audio, centered, mode="valid", method="fft")
    squares = np.concatenate(([0.0], np.cumsum(audio * audio)))
    segment_energy = squares[len(reference):] - squares[:-len(reference)]
    scores = candidates / np.sqrt(np.maximum(segment_energy * np.dot(centered, centered), 1e-20))
    lo = max(0, center - radius)
    hi = min(len(scores), center + radius + 1)
    index = lo + int(np.argmax(scores[lo:hi]))
    return index, float(scores[index])


def main() -> int:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    mixed_path = OUTPUT_DIR / "wasapi_mixed.wav"
    mixed = read_mono(mixed_path)
    references = [read_mono(REFERENCE_DIR / f"speaker_{name}.wav") for name in ("a", "b")]
    full_frames = max(map(len, references))
    padded_references = [np.pad(reference, (0, full_frames - len(reference))) for reference in references]
    if len(mixed) < full_frames:
        raise ValueError("Capture is shorter than the reference evaluation span")

    print(f"Separating {mixed_path.name}: {MODEL_NAME} (cpu)", flush=True)
    started = perf_counter()
    raw_dir = OUTPUT_DIR / "wasapi_raw"
    raw_paths = separate_wav(mixed_path, raw_dir)
    separation_time = perf_counter() - started
    output_paths = [OUTPUT_DIR / f"wasapi_speaker_{number}.wav" for number in (1, 2)]
    for source, destination in zip(raw_paths, output_paths):
        source.rename(destination)
    raw_dir.rmdir()
    outputs = [read_mono(path) for path in output_paths]
    if any(len(output) != len(mixed) for output in outputs):
        raise ValueError("Output length differs from capture")
    print(f"Separation time: {separation_time:.2f} sec")

    for number, (path, output) in enumerate(zip(output_paths, outputs), start=1):
        print(
            f"Output {number}: {len(output) / 16000:.3f} sec, "
            f"peak={np.max(np.abs(output)):.6f}, RMS={np.sqrt(np.mean(output**2)):.6f}, "
            f"non-finite={np.count_nonzero(~np.isfinite(output))}, "
            f"clipped={np.count_nonzero(np.abs(output) >= 0.999)}"
        )
        print(f"Saved: {path}")

    scores = np.zeros((2, 2))
    correlations = np.zeros((2, 2))
    offsets = np.zeros((2, 2))
    capture_offsets = []
    for name, reference in zip("AB", references):
        offset, score = estimate_offset(mixed, reference)
        capture_offsets.append(round(offset * 16000))
        print(f"Capture vs reference {name}: offset={offset:.6f} sec, correlation={score:.6f}")

    for output_index, output in enumerate(outputs):
        for reference_index, reference in enumerate(references):
            offset, _ = best_offset_near(output, reference, capture_offsets[reference_index], radius=1600)
            evaluated = output[offset:offset + full_frames]
            if len(evaluated) != full_frames:
                raise ValueError("Aligned output is shorter than the evaluation span")
            expected = padded_references[reference_index]
            corr = correlation(expected, evaluated)
            score = si_sdr(expected, evaluated)
            correlations[output_index, reference_index] = corr
            scores[output_index, reference_index] = score
            offsets[output_index, reference_index] = offset / 16000
            print(
                f"Output {output_index + 1} vs reference {'AB'[reference_index]}: "
                f"offset={offset / 16000:.6f} sec, correlation={corr:.6f}, SI-SDR={score:.2f} dB"
            )

    direct = scores[0, 0] + scores[1, 1]
    swapped = scores[0, 1] + scores[1, 0]
    if direct >= swapped:
        assignment = ((0, 0), (1, 1))
        print("Best permutation: output 1=A, output 2=B")
    else:
        assignment = ((0, 1), (1, 0))
        print("Best permutation: output 1=B, output 2=A")

    direct_baseline = {0: (0.995802, 20.73), 1: (0.996426, 21.44)}
    for output_index, reference_index in assignment:
        baseline_corr, baseline_si_sdr = direct_baseline[reference_index]
        print(
            f"Matched reference {'AB'[reference_index]}: "
            f"correlation delta={correlations[output_index, reference_index] - baseline_corr:+.6f}, "
            f"SI-SDR delta={scores[output_index, reference_index] - baseline_si_sdr:+.2f} dB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

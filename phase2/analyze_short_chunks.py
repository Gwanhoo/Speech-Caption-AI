from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "phase1" / "diagnostic"))

from playback_capture import estimate_offset  # noqa: E402


def main() -> int:
    references = []
    for name in ("a", "b"):
        audio, sample_rate = sf.read(ROOT / "phase1" / "reference" / f"speaker_{name}.wav", dtype="float32")
        if sample_rate != 16000 or audio.ndim != 1:
            raise ValueError(f"Invalid reference {name}")
        references.append(np.tile(audio, 4))

    for seconds, directory, count in (
        (5, "continuous", 6),
        (2, "short_2s_retry", 10),
        (1, "short_1s_retry", 20),
    ):
        output_dir = ROOT / "phase2" / "output" / directory
        input_dir = ROOT / "phase2" / "input" / directory
        mappings = []
        pair_correlations = []
        matched_scores = []
        print(f"Condition {seconds}s:", flush=True)
        for index in range(count):
            inputs, input_rate = sf.read(input_dir / f"chunk_{index:03d}_mixed_16k.wav", dtype="float32")
            if input_rate != 16000 or inputs.ndim != 1 or len(inputs) != seconds * 16000:
                raise ValueError(f"Invalid input chunk {index} in {directory}")
            outputs = []
            for source in (0, 1):
                path = output_dir / f"chunk_{index:03d}_speaker_{source}.wav"
                audio, sample_rate = sf.read(path, dtype="float32")
                rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
                if sample_rate != 16000 or audio.ndim != 1 or len(audio) != seconds * 16000 or not np.isfinite(audio).all() or rms <= 1e-5:
                    raise ValueError(f"Invalid speaker output: {path}")
                outputs.append(audio)

            scores = np.array([[estimate_offset(reference, output)[1] for reference in references] for output in outputs])
            pair = float(np.corrcoef(outputs)[0, 1])
            pair_correlations.append(pair)
            direct = scores[0, 0] + scores[1, 1]
            swapped = scores[0, 1] + scores[1, 0]
            if abs(direct - swapped) < 0.2 or max(min(scores[0, 0], scores[1, 1]), min(scores[0, 1], scores[1, 0])) < 0.3:
                mapping = "uncertain"
            else:
                mapping = "0=A,1=B" if direct > swapped else "0=B,1=A"
                matched_scores.append(min(scores[0, 0], scores[1, 1]) if direct > swapped else min(scores[0, 1], scores[1, 0]))
            mappings.append(mapping)
            print(
                f"  {index:03d}: scores 0[A,B]=[{scores[0, 0]:.3f},{scores[0, 1]:.3f}] "
                f"1[A,B]=[{scores[1, 0]:.3f},{scores[1, 1]:.3f}], "
                f"pair_corr={pair:.3f}, mapping={mapping}",
                flush=True,
            )
        print(
            f"  Summary: A/B={mappings.count('0=A,1=B')}, B/A={mappings.count('0=B,1=A')}, "
            f"uncertain={mappings.count('uncertain')}, pair_corr_abs_max={max(abs(x) for x in pair_correlations):.3f}, "
            f"collapse_candidates={sum(abs(x) >= 0.98 for x in pair_correlations)}, "
            f"matched_score_min={min(matched_scores) if matched_scores else float('nan'):.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

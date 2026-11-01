"""Build a human-review dataset from the completed A1-A3 results; no inference."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
A13 = ROOT / "evidence" / "accuracy_a1_a3_20261101"
OUT = Path(__file__).resolve().parent
CLIPS = OUT / "clips"
COUNT = 24
SR = 16_000


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_characters(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]+", "", text).lower()


def words(text: str) -> list[str]:
    return re.findall(r"[0-9A-Za-z가-힣]+", text.lower())


def edit_distance(left, right) -> int:
    previous = list(range(len(right) + 1))
    for left_item in left:
        current = [previous[0] + 1]
        for index, right_item in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[index] + 1,
                               previous[index - 1] + (left_item != right_item)))
        previous = current
    return previous[-1]


def disagreement(left: str, right: str) -> dict:
    left_chars, right_chars = normalize_characters(left), normalize_characters(right)
    left_words, right_words = words(left), words(right)
    char_edits = edit_distance(left_chars, right_chars)
    word_edits = edit_distance(left_words, right_words)
    return {
        "exact_output_differs": left != right,
        "normalized_output_differs": left_chars != right_chars,
        "character_edit_distance": char_edits,
        "word_edit_distance": word_edits,
        "normalized_character_disagreement": char_edits / max(len(left_chars), len(right_chars), 1),
        "normalized_word_disagreement": word_edits / max(len(left_words), len(right_words), 1),
    }


def rows_for(report: dict, repeat: int = 0) -> dict[tuple[str, int], dict]:
    run = next(run for run in report["runs"] if run["repeat"] == repeat)
    return {(row["source"], row["end_seconds"]): row for row in run["rows"]}


def a3_outputs(report: dict) -> dict[tuple[str, int], list[str]]:
    result: dict[tuple[str, int], list[str]] = {}
    for run in report["runs"]:
        if run["window_seconds"] != 3:
            continue
        for row in run["rows"]:
            result.setdefault((row["source"], row["end_seconds"]), []).append(row["raw"])
    return result


def option_variability(texts: list[str]) -> dict:
    normalized = [normalize_characters(text) for text in texts]
    unique = sorted(set(normalized))
    maximum = 0.0
    for left, right in combinations(unique, 2):
        maximum = max(maximum, edit_distance(left, right) / max(len(left), len(right), 1))
    return {"unique_normalized_outputs": len(unique), "max_pairwise_character_disagreement": maximum}


def main() -> None:
    if (OUT / "review_dataset.json").exists() or CLIPS.exists():
        raise FileExistsError("Refusing to overwrite an existing A4 dataset")
    CLIPS.mkdir(parents=True)
    manifest = read_json(A13 / "experiment_manifest.json")
    turbo_report = read_json(A13 / "A2_large-v3-turbo.json")
    large_report = read_json(A13 / "A2_large-v3.json")
    turbo_rows, large_rows = rows_for(turbo_report), rows_for(large_report)
    turbo_repeat, large_repeat = rows_for(turbo_report, 1), rows_for(large_report, 1)
    turbo_options = a3_outputs(read_json(A13 / "A3_large-v3-turbo.json"))
    large_options = a3_outputs(read_json(A13 / "A3_large-v3.json"))
    diagnostics = read_json(A13 / "decoder_diagnostics.json")
    diagnostic_keys = {
        (item["source"], item["interval"][1])
        for item in diagnostics["timing_outliers"] + diagnostics["segments_exceeding_context_by_over_100ms"]
        if item["window_seconds"] == 3
    }

    candidates = []
    for key in sorted(turbo_rows):
        turbo, large = turbo_rows[key], large_rows[key]
        comparison = disagreement(turbo["raw"], large["raw"])
        option_texts = turbo_options[key] + large_options[key]
        variability = option_variability(option_texts)
        repeat_unstable = (
            turbo["raw"] != turbo_repeat[key]["raw"]
            or large["raw"] != large_repeat[key]["raw"]
        )
        score = (
            1000 * int(comparison["exact_output_differs"])
            + 100 * comparison["normalized_character_disagreement"]
            + 60 * comparison["normalized_word_disagreement"]
            + 15 * variability["max_pairwise_character_disagreement"]
            + 2 * (variability["unique_normalized_outputs"] - 1)
            + 25 * int(repeat_unstable)
            + 20 * int(key in diagnostic_keys)
        )
        candidates.append({
            "key": key, "turbo": turbo, "large": large, "comparison": comparison,
            "variability": variability, "repeat_unstable": repeat_unstable,
            "decoder_diagnostic": key in diagnostic_keys, "selection_score": score,
        })
    candidates.sort(key=lambda row: (-row["selection_score"], row["key"]))
    selected = candidates[:COUNT]
    assert sum(item["comparison"]["exact_output_differs"] for item in selected) == 16

    sources: dict[str, tuple[Path, np.ndarray]] = {}
    for source in ("A", "B"):
        path = ROOT / "phase3" / "test_audio" / f"{source}.wav"
        audio, rate = sf.read(path, dtype="float32")
        if rate != 24_000 or audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError(f"Unexpected source format: {path}")
        sources[source] = (path, resample_poly(audio, 2, 3).astype(np.float32))

    dataset_rows = []
    for rank, item in enumerate(selected, 1):
        source, end = item["key"]
        turbo, large = item["turbo"], item["large"]
        start = turbo["start_seconds"]
        if (start, end) != (large["start_seconds"], large["end_seconds"]):
            raise ValueError("Model intervals differ")
        source_path, audio = sources[source]
        clip = np.ascontiguousarray(audio[int(start * SR):int(end * SR)], dtype=np.float32)
        float_hash = sha256_bytes(clip.tobytes())
        if float_hash != turbo["audio_sha256_float32"] or float_hash != large["audio_sha256_float32"]:
            raise ValueError(f"Experiment audio hash mismatch: {source} {start}-{end}")
        sample_id = f"A4-{source}-{int(start * 1000):06d}-{int(end * 1000):06d}"
        clip_path = CLIPS / f"{sample_id}.wav"
        sf.write(clip_path, clip, SR, subtype="FLOAT")
        reasons = []
        if item["comparison"]["exact_output_differs"]:
            reasons.append("a2_model_output_difference")
        if item["variability"]["unique_normalized_outputs"] >= 4:
            reasons.append("high_a3_option_variability")
        if item["repeat_unstable"]:
            reasons.append("repeat_output_instability")
        if item["decoder_diagnostic"]:
            reasons.append("decoder_timing_or_timestamp_anomaly")
        if not item["comparison"]["exact_output_differs"]:
            reasons.append("a3_option_variability_priority")
        dataset_rows.append({
            "sample_id": sample_id,
            "selection_rank": rank,
            "selection_score": round(item["selection_score"], 8),
            "selection_reasons": reasons,
            "source_label": source,
            "source_audio_path": str(source_path.relative_to(ROOT)),
            "source_audio_sha256": sha256_file(source_path),
            "start_seconds": start,
            "end_seconds": end,
            "duration_seconds": end - start,
            "review_clip_path": str(clip_path.relative_to(ROOT)),
            "review_clip_sha256": sha256_file(clip_path),
            "experiment_float32_sha256": float_hash,
            "a13_link": {
                "stage": "A2", "window_seconds": 3, "stride_seconds": 2,
                "beam_size": 5, "initial_prompt": None, "repeat": 0,
                "source": source, "end_seconds": end,
            },
            "turbo_transcript": turbo["raw"],
            "large_v3_transcript": large["raw"],
            "turbo_repeat_reproducible": turbo["raw"] == turbo_repeat[item["key"]]["raw"],
            "large_v3_repeat_reproducible": large["raw"] == large_repeat[item["key"]]["raw"],
            "model_disagreement": item["comparison"],
            "a3_option_variability": item["variability"],
            "reference_text": None,
            "review_status": "pending",
            "reviewer": None,
            "reviewed_at": None,
            "review_notes": None,
        })

    dataset = {
        "schema_version": 1,
        "dataset_id": "voxlux-a4-accuracy-review-20261101",
        "source_experiment": "evidence/accuracy_a1_a3_20261101",
        "source_experiment_git_head": manifest["git_head"],
        "selection_count": len(dataset_rows),
        "a2_output_difference_count": sum(r["model_disagreement"]["exact_output_differs"] for r in dataset_rows),
        "reference_status": "unreviewed",
        "cer": None,
        "wer": None,
        "selection_method": (
            "All 16 exact A2 turbo/large-v3 disagreements, then highest A3 3-second "
            "option variability and decoder/repeat anomaly scores; no transcript is treated as truth."
        ),
        "audio_method": "Whole source resampled 24kHz->16kHz with scipy resample_poly(2,3), then exact A2 3-second slice; FLOAT WAV.",
        "source_artifacts_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                A13 / "experiment_manifest.json",
                A13 / "A2_large-v3-turbo.json",
                A13 / "A2_large-v3.json",
                A13 / "A3_large-v3-turbo.json",
                A13 / "A3_large-v3.json",
                A13 / "decoder_diagnostics.json",
            )
        },
        "rows": dataset_rows,
    }
    (OUT / "review_dataset.json").write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    columns = [
        "sample_id", "selection_rank", "selection_reasons", "review_clip_path",
        "source_audio_path", "start_seconds", "end_seconds", "turbo_transcript",
        "large_v3_transcript", "reference_text", "review_status", "reviewer",
        "reviewed_at", "review_notes",
    ]
    with (OUT / "review_template.csv").open("w", newline="", encoding="utf-8-sig") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        for row in dataset_rows:
            csv_row = {
                key: row[key]
                for key in columns
                if key not in {"selection_reasons", "reference_text", "review_status", "reviewer", "reviewed_at", "review_notes"}
            }
            writer.writerow({
                **csv_row,
                "selection_reasons": ";".join(row["selection_reasons"]),
                "reference_text": "", "review_status": "pending", "reviewer": "",
                "reviewed_at": "", "review_notes": "",
            })
    print(json.dumps({"dataset": str(OUT / "review_dataset.json"),
                      "template": str(OUT / "review_template.csv"),
                      "clips": len(dataset_rows), "model_differences": 16}, ensure_ascii=False))


if __name__ == "__main__":
    main()

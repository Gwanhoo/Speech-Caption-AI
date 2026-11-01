"""Score human-verified A4 references. Refuses incomplete review by default."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def normalize_characters(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]+", "", text).lower()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[0-9A-Za-z가-힣]+", text.lower())


def edit_counts(reference, hypothesis) -> dict[str, int]:
    table = [[(0, 0, 0, 0) for _ in range(len(hypothesis) + 1)] for _ in range(len(reference) + 1)]
    for i in range(1, len(reference) + 1):
        table[i][0] = (i, 0, i, 0)
    for j in range(1, len(hypothesis) + 1):
        table[0][j] = (j, 0, 0, j)
    for i, left in enumerate(reference, 1):
        for j, right in enumerate(hypothesis, 1):
            diagonal = table[i - 1][j - 1]
            candidates = [(diagonal[0] + (left != right), diagonal[1] + (left != right), diagonal[2], diagonal[3])]
            deletion = table[i - 1][j]
            candidates.append((deletion[0] + 1, deletion[1], deletion[2] + 1, deletion[3]))
            insertion = table[i][j - 1]
            candidates.append((insertion[0] + 1, insertion[1], insertion[2], insertion[3] + 1))
            table[i][j] = min(candidates, key=lambda value: value[0])
    distance, substitutions, deletions, insertions = table[-1][-1]
    return {"edits": distance, "substitutions": substitutions, "deletions": deletions, "insertions": insertions}


def score(reference_text: str, hypothesis: str) -> dict:
    reference_chars = normalize_characters(reference_text)
    hypothesis_chars = normalize_characters(hypothesis)
    reference_words = tokenize(reference_text)
    hypothesis_words = tokenize(hypothesis)
    characters = edit_counts(reference_chars, hypothesis_chars)
    words = edit_counts(reference_words, hypothesis_words)
    return {
        "character": {**characters, "reference_units": len(reference_chars),
                      "cer": characters["edits"] / len(reference_chars)},
        "word": {**words, "reference_units": len(reference_words),
                 "wer": words["edits"] / len(reference_words)},
    }


def aggregate(rows: list[dict], model_key: str) -> dict:
    character_edits = sum(row[model_key]["character"]["edits"] for row in rows)
    character_units = sum(row[model_key]["character"]["reference_units"] for row in rows)
    word_edits = sum(row[model_key]["word"]["edits"] for row in rows)
    word_units = sum(row[model_key]["word"]["reference_units"] for row in rows)
    return {
        "samples": len(rows),
        "micro_cer": character_edits / character_units,
        "character_edits": character_edits,
        "reference_characters": character_units,
        "micro_wer": word_edits / word_units,
        "word_edits": word_edits,
        "reference_words": word_units,
        "macro_cer": sum(row[model_key]["character"]["cer"] for row in rows) / len(rows),
        "macro_wer": sum(row[model_key]["word"]["wer"] for row in rows) / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(__file__).with_name("review_dataset.json"))
    parser.add_argument("--reviews", type=Path, default=Path(__file__).with_name("review_template.csv"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("evaluation_results.json"))
    parser.add_argument("--allow-partial", action="store_true", help="Score verified rows only and label the result partial")
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    canonical = {row["sample_id"]: row for row in dataset["rows"]}
    with args.reviews.open(newline="", encoding="utf-8-sig") as source:
        reviews = list(csv.DictReader(source))
    if len(reviews) != len(canonical) or len({row["sample_id"] for row in reviews}) != len(reviews):
        raise ValueError("Review CSV must contain every unique dataset sample exactly once")
    if set(row["sample_id"] for row in reviews) != set(canonical):
        raise ValueError("Review CSV sample IDs do not match the canonical dataset")
    immutable = ("review_clip_path", "source_audio_path", "start_seconds", "end_seconds", "turbo_transcript", "large_v3_transcript")
    verified, pending = [], []
    for review in reviews:
        row = canonical[review["sample_id"]]
        for key in immutable:
            if str(review[key]) != str(row[key]):
                raise ValueError(f"Immutable field changed for {row['sample_id']}: {key}")
        status = review["review_status"].strip().lower()
        reference = review["reference_text"].strip()
        if status == "verified":
            if not reference or not normalize_characters(reference) or not tokenize(reference):
                raise ValueError(f"Verified row needs a nonempty scorable reference: {row['sample_id']}")
            if not review["reviewer"].strip() or not review["reviewed_at"].strip():
                raise ValueError(f"Verified row needs reviewer and reviewed_at: {row['sample_id']}")
            verified.append((row, review, reference))
        elif status == "pending" and not reference:
            pending.append(row["sample_id"])
        else:
            raise ValueError(f"Use pending with an empty reference or verified with a reference: {row['sample_id']}")
    if pending and not args.allow_partial:
        raise ValueError(f"Review incomplete: {len(pending)} pending rows; CER/WER not calculated")
    if not verified:
        raise ValueError("No verified references; CER/WER not calculated")
    scored = []
    for row, review, reference in verified:
        scored.append({
            "sample_id": row["sample_id"], "reference_text": reference,
            "reviewer": review["reviewer"], "reviewed_at": review["reviewed_at"],
            "turbo": score(reference, row["turbo_transcript"]),
            "large_v3": score(reference, row["large_v3_transcript"]),
        })
    result = {
        "dataset_id": dataset["dataset_id"],
        "evaluation_status": "partial" if pending else "complete",
        "verified_samples": len(verified), "pending_samples": len(pending),
        "normalization": "CER: keep Hangul/ASCII alphanumeric and lowercase Latin; WER: whitespace/punctuation-separated Hangul/ASCII alphanumeric runs",
        "models": {"large-v3-turbo": aggregate(scored, "turbo"), "large-v3": aggregate(scored, "large_v3")},
        "samples": scored,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["models"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import soundfile as sf


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("evaluate_reviewed", HERE / "evaluate_reviewed.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class MetricTests(unittest.TestCase):
    def test_known_scores(self):
        result = evaluation.score("안녕하세요 세계", "안녕하세여 세계")
        self.assertEqual(result["character"]["edits"], 1)
        self.assertEqual(result["word"]["edits"], 1)
        self.assertEqual(result["character"]["reference_units"], 7)
        self.assertEqual(result["word"]["reference_units"], 2)

    def test_dataset_integrity_and_no_references(self):
        dataset = json.loads((HERE / "review_dataset.json").read_text())
        self.assertEqual(dataset["selection_count"], 24)
        self.assertEqual(dataset["a2_output_difference_count"], 16)
        self.assertIsNone(dataset["cer"])
        self.assertIsNone(dataset["wer"])
        self.assertEqual(len({row["sample_id"] for row in dataset["rows"]}), 24)
        for row in dataset["rows"]:
            self.assertEqual(row["review_status"], "pending")
            self.assertIsNone(row["reference_text"])
            clip_path = HERE.parents[1] / row["review_clip_path"]
            self.assertTrue(clip_path.is_file())
            audio, rate = sf.read(clip_path, dtype="float32")
            self.assertEqual(rate, 16_000)
            self.assertEqual(len(audio), 48_000)
            self.assertEqual(hashlib.sha256(audio.tobytes()).hexdigest(), row["experiment_float32_sha256"])
            self.assertTrue(row["selection_reasons"])
        for relative, expected in dataset["source_artifacts_sha256"].items():
            self.assertEqual(hashlib.sha256((HERE.parents[1] / relative).read_bytes()).hexdigest(), expected)

    def test_unreviewed_template_refuses_to_score(self):
        result = subprocess.run(
            [sys.executable, str(HERE / "evaluate_reviewed.py")],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CER/WER not calculated", result.stderr)

    def test_complete_verified_copy_scores_without_changing_template(self):
        with (HERE / "review_template.csv").open(newline="", encoding="utf-8-sig") as source:
            rows = list(csv.DictReader(source))
            columns = source.readline() if False else list(rows[0])
        for row in rows:
            row["reference_text"] = row["turbo_transcript"]
            row["review_status"] = "verified"
            row["reviewer"] = "unit-test"
            row["reviewed_at"] = "2026-11-01T00:00:00Z"
        with tempfile.TemporaryDirectory() as folder:
            reviewed, output = Path(folder) / "reviewed.csv", Path(folder) / "scores.json"
            with reviewed.open("w", newline="", encoding="utf-8-sig") as destination:
                writer = csv.DictWriter(destination, fieldnames=columns)
                writer.writeheader(); writer.writerows(rows)
            result = subprocess.run(
                [sys.executable, str(HERE / "evaluate_reviewed.py"), "--reviews", str(reviewed), "--output", str(output)],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            scores = json.loads(output.read_text())
            self.assertEqual(scores["evaluation_status"], "complete")
            self.assertEqual(scores["models"]["large-v3-turbo"]["micro_cer"], 0)
            self.assertEqual(scores["models"]["large-v3-turbo"]["micro_wer"], 0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest

from review_store import ReviewStore


HERE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReviewStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.temp = Path(self.temporary.name)
        for name in ("review_template.csv", "review_dataset.json", "evaluate_reviewed.py"):
            shutil.copy2(HERE / name, self.temp / name)
        self.completed = self.temp / "review_completed.csv"
        self.store = ReviewStore(
            self.temp,
            project_root=HERE.parents[1],
            completed_path=self.completed,
            clock=lambda: "2026-11-01T12:00:00Z",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_loads_all_clips_and_audio_paths(self):
        self.assertEqual(self.store.progress(), (0, 24))
        self.assertEqual(len(self.store.validate_audio_paths()), 24)

    def test_verified_requires_audio_reference_and_reviewer(self):
        sample_id = self.store.order[0]
        with self.assertRaisesRegex(ValueError, "오디오"):
            self.store.verify(sample_id, listened=False)
        with self.assertRaisesRegex(ValueError, "정답"):
            self.store.verify(sample_id, listened=True)
        self.store.update_draft(sample_id, "사람이 검수한 정답", "", "")
        with self.assertRaisesRegex(ValueError, "검수자"):
            self.store.verify(sample_id, listened=True)
        self.store.update_draft(sample_id, "사람이 검수한 정답", "reviewer", "")
        self.store.verify(sample_id, listened=True)
        self.assertEqual(self.store.get(sample_id)["review_status"], "verified")
        self.assertEqual(self.store.get(sample_id)["reviewed_at"], "2026-11-01T12:00:00Z")

    def test_atomic_save_preserves_columns_and_reload(self):
        sample_id = self.store.order[0]
        self.store.update_draft(sample_id, "임시 정답", "검수자", "메모")
        self.store.verify(sample_id, listened=True)
        self.store.save()
        self.assertTrue(self.completed.is_file())
        self.assertFalse(list(self.temp.glob(".review_completed.*.tmp")))
        with (HERE / "review_template.csv").open(encoding="utf-8-sig", newline="") as source:
            template_columns = csv.DictReader(source).fieldnames
        with self.completed.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source); completed_rows = list(reader)
        self.assertEqual(reader.fieldnames, template_columns)
        self.assertEqual(len(completed_rows), 24)
        reloaded = ReviewStore(self.temp, project_root=HERE.parents[1], completed_path=self.completed)
        self.assertTrue(reloaded.loaded_from_completed)
        self.assertEqual(reloaded.get(sample_id)["reference_text"], "임시 정답")
        self.assertEqual(reloaded.progress(), (1, 24))

    def test_existing_completed_is_not_initialized_or_overwritten_on_load(self):
        self.store.save()
        before = digest(self.completed)
        ReviewStore(self.temp, project_root=HERE.parents[1], completed_path=self.completed)
        self.assertEqual(digest(self.completed), before)

    def test_incomplete_dataset_refuses_evaluation(self):
        with self.assertRaisesRegex(ValueError, "CER/WER를 계산하지 않습니다"):
            self.store.evaluate(output_path=self.temp / "evaluation.json")
        self.assertFalse((self.temp / "evaluation.json").exists())

    def test_complete_dataset_runs_actual_evaluator(self):
        for sample_id in self.store.order:
            row = self.store.get(sample_id)
            self.store.update_draft(sample_id, row["turbo_transcript"], "unit-test", "")
            self.store.verify(sample_id, listened=True)
        output = self.temp / "evaluation.json"
        result = self.store.evaluate(output_path=output)
        self.assertEqual(result["evaluation_status"], "complete")
        self.assertEqual(result["verified_samples"], 24)
        self.assertEqual(result["models"]["large-v3-turbo"]["micro_cer"], 0)
        self.assertEqual(result["models"]["large-v3-turbo"]["micro_wer"], 0)
        self.assertTrue(output.is_file())

    def test_original_inputs_remain_unchanged(self):
        protected = [HERE / name for name in ("review_template.csv", "review_dataset.json", "evaluate_reviewed.py")]
        before = {path: digest(path) for path in protected}
        self.store.save()
        self.assertEqual({path: digest(path) for path in protected}, before)


if __name__ == "__main__":
    unittest.main()

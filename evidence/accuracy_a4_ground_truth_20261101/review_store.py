"""Persistence and evaluation services for the standalone A4 review GUI."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Callable


IMMUTABLE_COLUMNS = (
    "sample_id",
    "selection_rank",
    "selection_reasons",
    "review_clip_path",
    "source_audio_path",
    "start_seconds",
    "end_seconds",
    "turbo_transcript",
    "large_v3_transcript",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class ReviewStore:
    def __init__(
        self,
        data_dir: Path,
        *,
        project_root: Path | None = None,
        completed_path: Path | None = None,
        clock: Callable[[], str] = utc_now,
    ) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.project_root = Path(project_root).resolve() if project_root else self.data_dir.parents[1]
        self.template_path = self.data_dir / "review_template.csv"
        self.dataset_path = self.data_dir / "review_dataset.json"
        self.completed_path = Path(completed_path).resolve() if completed_path else self.data_dir / "review_completed.csv"
        self.evaluator_path = self.data_dir / "evaluate_reviewed.py"
        self.evaluation_path = self.data_dir / "evaluation_results.json"
        self.clock = clock
        self.columns, template_rows = self._read_csv(self.template_path)
        self.template_rows = {row["sample_id"]: row for row in template_rows}
        self.order = [row["sample_id"] for row in template_rows]
        self.dataset = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        self.dataset_rows = {row["sample_id"]: row for row in self.dataset["rows"]}
        self._validate_canonical_data()
        if self.completed_path.exists():
            completed_columns, rows = self._read_csv(self.completed_path)
            if completed_columns != self.columns:
                raise ValueError("review_completed.csv columns/order differ from review_template.csv")
            self._validate_completed_rows(rows)
            self.rows = {row["sample_id"]: row for row in rows}
            self.loaded_from_completed = True
        else:
            self.rows = {sample_id: dict(self.template_rows[sample_id]) for sample_id in self.order}
            self.loaded_from_completed = False

    @staticmethod
    def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header: {path}")
            return list(reader.fieldnames), list(reader)

    def _validate_canonical_data(self) -> None:
        if len(self.order) != 24 or len(set(self.order)) != 24:
            raise ValueError("A4 template must contain 24 unique sample IDs")
        if set(self.order) != set(self.dataset_rows):
            raise ValueError("Template and review_dataset.json sample IDs differ")
        if self.completed_path.resolve() == self.template_path.resolve():
            raise ValueError("Completed output must never be review_template.csv")
        for sample_id in self.order:
            template = self.template_rows[sample_id]
            canonical = self.dataset_rows[sample_id]
            for column in IMMUTABLE_COLUMNS:
                if column == "selection_reasons":
                    expected = ";".join(canonical[column])
                else:
                    expected = str(canonical[column])
                if template[column] != expected:
                    raise ValueError(f"Template/canonical mismatch for {sample_id}: {column}")

    def _validate_completed_rows(self, rows: list[dict[str, str]]) -> None:
        ids = [row["sample_id"] for row in rows]
        if len(ids) != len(self.order) or len(set(ids)) != len(ids) or set(ids) != set(self.order):
            raise ValueError("review_completed.csv must contain every sample exactly once")
        for row in rows:
            sample_id = row["sample_id"]
            for column in IMMUTABLE_COLUMNS:
                if row[column] != self.template_rows[sample_id][column]:
                    raise ValueError(f"Immutable column changed for {sample_id}: {column}")
            if row["review_status"] not in {"pending", "verified"}:
                raise ValueError(f"Invalid review status for {sample_id}")
            if row["review_status"] == "verified":
                if not row["reference_text"].strip() or not row["reviewer"].strip() or not row["reviewed_at"].strip():
                    raise ValueError(f"Incomplete verified row: {sample_id}")

    def validate_audio_paths(self) -> list[Path]:
        paths = []
        for sample_id in self.order:
            relative = Path(self.rows[sample_id]["review_clip_path"])
            path = relative if relative.is_absolute() else self.project_root / relative
            if not path.is_file():
                raise FileNotFoundError(f"Missing review audio: {path}")
            paths.append(path.resolve())
        return paths

    def audio_path(self, sample_id: str) -> Path:
        relative = Path(self.rows[sample_id]["review_clip_path"])
        return (relative if relative.is_absolute() else self.project_root / relative).resolve()

    def get(self, sample_id: str) -> dict[str, str]:
        return self.rows[sample_id]

    def progress(self) -> tuple[int, int]:
        return sum(self.rows[sample_id]["review_status"] == "verified" for sample_id in self.order), len(self.order)

    def update_draft(self, sample_id: str, reference: str, reviewer: str, notes: str) -> None:
        row = self.rows[sample_id]
        changed_answer = row["reference_text"] != reference or row["reviewer"] != reviewer
        row["reference_text"] = reference
        row["reviewer"] = reviewer
        row["review_notes"] = notes
        if changed_answer and row["review_status"] == "verified":
            row["review_status"] = "pending"
            row["reviewed_at"] = ""

    def mark_pending(self, sample_id: str) -> None:
        row = self.rows[sample_id]
        row["review_status"] = "pending"
        row["reviewed_at"] = ""

    def verify(self, sample_id: str, *, listened: bool) -> None:
        row = self.rows[sample_id]
        if not listened:
            raise ValueError("오디오를 재생한 뒤 verified로 변경할 수 있습니다.")
        if not row["reference_text"].strip():
            raise ValueError("정답 문장을 입력해야 verified로 변경할 수 있습니다.")
        if not row["reviewer"].strip():
            raise ValueError("검수자 이름을 입력해야 verified로 변경할 수 있습니다.")
        row["review_status"] = "verified"
        row["reviewed_at"] = self.clock()

    def save(self) -> None:
        self.completed_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", newline="", encoding="utf-8-sig", delete=False,
                dir=self.completed_path.parent, prefix=".review_completed.", suffix=".tmp",
            ) as destination:
                temporary_name = destination.name
                writer = csv.DictWriter(destination, fieldnames=self.columns, extrasaction="raise")
                writer.writeheader()
                for sample_id in self.order:
                    writer.writerow(self.rows[sample_id])
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary_name, self.completed_path)
            temporary_name = None
            self.loaded_from_completed = True
        finally:
            if temporary_name:
                Path(temporary_name).unlink(missing_ok=True)

    def all_verified(self) -> bool:
        return self.progress()[0] == len(self.order)

    def evaluate(self, *, output_path: Path | None = None, timeout: int = 120) -> dict:
        if not self.all_verified():
            verified, total = self.progress()
            raise ValueError(f"전체 검수가 완료되지 않았습니다: {verified}/{total}. CER/WER를 계산하지 않습니다.")
        self.save()
        output = Path(output_path).resolve() if output_path else self.evaluation_path
        command = [
            sys.executable, str(self.evaluator_path),
            "--dataset", str(self.dataset_path),
            "--reviews", str(self.completed_path),
            "--output", str(output),
        ]
        result = subprocess.run(command, cwd=self.data_dir, capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            message = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
            raise RuntimeError(f"평가 실패: {message}")
        return json.loads(output.read_text(encoding="utf-8"))

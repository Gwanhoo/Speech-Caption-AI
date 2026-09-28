from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmark_stt_model_size import MODELS, read_p2a_pseudo_references


class SttModelSizeBenchmarkTests(unittest.TestCase):
    def test_model_scope_is_limited_to_requested_sizes(self) -> None:
        self.assertEqual(MODELS, ("base", "small", "medium"))

    def test_reads_fixed_p2a_references(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "p2a.json"
            path.write_text(
                json.dumps({"pseudo_references": {"original": {"A_only": {"text": "참조 문장"}}}}),
                encoding="utf-8",
            )
            self.assertEqual(
                read_p2a_pseudo_references(path),
                {"original": {"A_only": "참조 문장"}},
            )

    def test_rejects_missing_reference_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "p2a.json"
            path.write_text(json.dumps({"pseudo_references": {"original": {"A_only": {}}}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_p2a_pseudo_references(path)


if __name__ == "__main__":
    unittest.main()

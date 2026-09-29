from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase4"))

import gpu_processing_server as server


class FasterWhisperModelSourceTests(TestCase):
    def test_uses_base_model_name_when_project_checkpoint_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(server, "FASTER_WHISPER_CHECKPOINT_ROOT", Path(directory)):
                self.assertEqual(server.faster_whisper_model_source(), "base")

    def test_uses_valid_local_snapshot_without_assuming_snapshot_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_root = Path(directory)
            local_model = (
                checkpoint_root
                / "models--Systran--faster-whisper-base"
                / "snapshots"
                / "new-container-cache-id"
            )
            local_model.mkdir(parents=True)
            (local_model / "model.bin").touch()
            with patch.object(server, "FASTER_WHISPER_CHECKPOINT_ROOT", checkpoint_root):
                self.assertEqual(server.faster_whisper_model_source(), str(local_model))


if __name__ == "__main__":
    main()

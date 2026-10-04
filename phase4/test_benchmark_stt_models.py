from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "phase4"))

import benchmark_stt_models as benchmark


class RepetitionMetricTests(TestCase):
    def test_detects_general_adjacent_phrase_repetition_without_editing_text(self):
        text = "네. 네. 네. 그리고 그럼요. 그럼요."
        metrics = benchmark.repetition_metrics(text)
        self.assertEqual(metrics["token_count"], 6)
        self.assertEqual(metrics["maximum_identical_token_run"], 3)
        self.assertEqual(metrics["maximum_consecutive_ngram_repetitions"], 3)
        self.assertEqual(metrics["most_repeated_consecutive_ngram"], ["네"])
        self.assertEqual(text, "네. 네. 네. 그리고 그럼요. 그럼요.")

    def test_empty_transcript_is_supported(self):
        metrics = benchmark.repetition_metrics("")
        self.assertEqual(metrics["token_count"], 0)
        self.assertIsNone(metrics["unique_token_ratio"])


class TranscriptionRecordTests(TestCase):
    def test_records_raw_segments_metadata_and_production_options(self):
        segments = [
            SimpleNamespace(
                id=0,
                start=0.1,
                end=0.7,
                text=" 네.",
                avg_logprob=-0.25,
                no_speech_prob=0.05,
                compression_ratio=1.2,
            ),
            SimpleNamespace(
                id=1,
                start=0.7,
                end=1.2,
                text=" 네.",
                avg_logprob=-0.4,
                no_speech_prob=0.1,
                compression_ratio=1.4,
            ),
        ]
        model = Mock()
        model.transcribe.return_value = (
            iter(segments),
            SimpleNamespace(language="ko", language_probability=0.99),
        )
        result = benchmark.transcribe_audio(
            model, np.zeros(benchmark.SAMPLE_RATE * 2, dtype=np.float32), 2.0
        )
        model.transcribe.assert_called_once()
        self.assertEqual(model.transcribe.call_args.kwargs, benchmark.TRANSCRIBE_OPTIONS)
        self.assertEqual(result["raw_transcript"], "네. 네.")
        self.assertEqual(result["segments"][0]["start"], 0.1)
        self.assertEqual(result["segments"][1]["avg_logprob"], -0.4)
        self.assertEqual(result["segments"][1]["no_speech_prob"], 0.1)
        self.assertEqual(result["segments"][1]["compression_ratio"], 1.4)
        self.assertGreaterEqual(result["inference_latency_seconds"], 0.0)
        self.assertGreaterEqual(result["total_processing_seconds"], 0.0)

    def test_side_by_side_preserves_each_model_transcript_and_segments(self):
        models = [
            {
                "model_name": name,
                "status": "ok",
                "inputs": [
                    {
                        "audio_path": "/fixed.wav",
                        "audio_sha256": "abc",
                        "audio_duration_seconds": 3.0,
                        "raw_transcript": transcript,
                        "repetition": {"token_count": 1},
                        "inference_latency_seconds": 0.1,
                        "real_time_factor": 0.03,
                        "segments": [{"text": transcript}],
                    }
                ],
            }
            for name, transcript in (("small", "네."), ("large-v3-turbo", "예."))
        ]
        comparison = benchmark.side_by_side_comparison(models)
        self.assertEqual(len(comparison), 1)
        self.assertEqual(comparison[0]["models"]["small"]["raw_transcript"], "네.")
        self.assertEqual(
            comparison[0]["models"]["large-v3-turbo"]["segments"],
            [{"text": "예."}],
        )


class FixedAudioTests(TestCase):
    def test_direct_windows_match_complete_production_3s_2s_schedule(self):
        audio = np.zeros(30 * benchmark.SAMPLE_RATE, dtype=np.float32)
        windows = benchmark.window_audio(audio)
        self.assertEqual(len(windows), 14)
        self.assertEqual(
            [(index, start, end) for index, start, end, _ in windows],
            [(index, index * 2.0, index * 2.0 + 3.0) for index in range(14)],
        )

    def test_score_transcript_normalizes_hangul_punctuation_and_space(self):
        score = benchmark.score_transcript("구름이 많습니다.", "구름이   많습니다")
        self.assertEqual(score["cer"], 0.0)
        self.assertEqual(score["wer"], 0.0)

    def test_direct_wav_is_a_supported_benchmark_source(self):
        args = benchmark.parse_args(["--direct-wav", "single.wav"])
        self.assertEqual(args.direct_wav, Path("single.wav"))

    def test_load_fixed_inputs_validates_and_reuses_saved_wav(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "speaker.wav"
            expected = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)
            sf.write(path, expected, benchmark.SAMPLE_RATE, subtype="FLOAT")
            loaded = benchmark.load_fixed_inputs([path])
            self.assertEqual(loaded[0][0], path)
            np.testing.assert_array_equal(loaded[0][1], expected)
            self.assertAlmostEqual(loaded[0][2], 0.1)

    def test_prepare_only_requires_mixed_audio(self):
        with self.assertRaises(SystemExit):
            benchmark.parse_args(
                ["--separated-wav", "speaker.wav", "--prepare-only"]
            )

    def test_prepare_only_writes_manifest_without_loading_whisper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mixed = root / "mixed.wav"
            separated = root / "fixed.wav"
            output = root / "report.json"
            audio = np.zeros(1600, dtype=np.float32)
            sf.write(mixed, audio, benchmark.SAMPLE_RATE, subtype="FLOAT")
            sf.write(separated, audio, benchmark.SAMPLE_RATE, subtype="FLOAT")
            preparation = {"mode": "test"}
            with patch.object(
                benchmark,
                "prepare_separated_audio",
                return_value=([separated], preparation),
            ), patch.object(benchmark, "run_model") as run_model:
                code = benchmark.main(
                    [
                        "--audio",
                        str(mixed),
                        "--prepare-only",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertTrue(output.is_file())
            run_model.assert_not_called()
            self.assertIn("fixed_separated_audio", preparation)


if __name__ == "__main__":
    main()

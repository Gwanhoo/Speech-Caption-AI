"""Recorded numeric seam regression; no audio, model, or network access."""
import json
from pathlib import Path
import unittest

from subtitle_assembler import SpeakerSubtitleState, SubtitleAssembler


RESULTS = Path(__file__).resolve().parents[1] / "experiments/r2_b6/results"


def replay_saved(saved):
    """Replay the recorded admission/overlap boundary through production state."""
    assemblers = [SubtitleAssembler(i, allow_shifted_head=True, enable_sentence_lifecycle=True)
                  for i in (0, 1)]
    states = [SpeakerSubtitleState(i, require_final_support=True) for i in (0, 1)]
    rows, all_events = [], []
    for row in saved["windows"]:
        index, end, primary = row["window"], row["end"], row["primary"]
        assembler, state = assemblers[primary], states[primary]
        vad, overlap = row["vad"][primary], row["overlap_evidence"]
        events = []
        if vad.get("candidate_transition", {}).get("restart"):
            final = state.finalize(index, end, "secondary_candidate_restart")
            events.append(final.to_dict())
            assembler.reset_utterance(final_text=final.text)
        assembly = assembler.process(
            index, row["admitted"][primary], shared_speech=overlap.get("shared_speech"),
            supported_tail_revision=overlap.get("supported_tail_revision", False),
        )
        if assembly.rollover_text and state.partial_text:
            final = state.finalize(index, end, assembly.rollover_reason)
            events.append(final.to_dict())
            assembler.commit_finalized_text(final.text)
        source = (assembly.raw_fragment if assembly.match_type in {
            "fuzzy_replace", "supported_tail_replace", "supported_sentence_revision"
        } or not assembly.new_fragment else assembly.new_fragment)
        for event in state.process(
            index, assembly.utterance_hypothesis, vad["speech_detected"], end,
            confirmed_prefix_length=assembly.confirmed_prefix_length,
            source_supported=vad.get("source_supported", False), source_text=source,
            candidate_transition=vad.get("candidate_transition"),
        ):
            events.append(event.to_dict())
            if event.status == "final":
                assembler.reset_utterance(final_text=event.text)
        rows.append({"window": index, "assembly": assembly.to_dict(), "events": events})
        all_events.extend(events)
    for state in states:
        final = state.flush(index, end)
        if final:
            all_events.append(final.to_dict())
    return rows, all_events


class PercentageOverlapTests(unittest.TestCase):
    def test_all_saved_runs_preserve_other_text_and_final_boundaries(self):
        paths = sorted(RESULTS.glob("*_repeat_*.json"))
        self.assertEqual(len(paths), 12)
        count = 0
        for path in paths:
            with self.subTest(run=path.name):
                saved = json.loads(path.read_text())
                rows, events = replay_saved(saved)
                count += len(rows)
                baseline = path.name.startswith("baseline_")

                def expected(text):
                    return text.replace("75% 75%가", "75%가") if baseline else text

                for old, new in zip(saved["windows"], rows):
                    self.assertEqual(new["assembly"]["utterance_hypothesis"],
                                     expected(old["assembly"]["utterance_hypothesis"]))
                old_events = [event for row in saved["windows"] for event in row["subtitle_events"]]
                old_events.extend(saved["flush_events"])
                self.assertEqual(len(events), len(old_events))
                for old, new in zip(old_events, events):
                    self.assertEqual(new["publication_text"], expected(old["publication_text"]))
                    for key in ("window", "status", "action", "utterance_id", "finalize_reason",
                                "stream_time_seconds", "utterance_start_seconds"):
                        self.assertEqual(new[key], old[key], (path.name, key))
        self.assertEqual(count, 189)

    def test_recorded_baseline_percentage_is_published_once(self):
        saved = json.loads((RESULTS / "baseline_repeat_0.json").read_text())
        self.assertTrue(saved["windows"][17]["overlap_evidence"]["supported_tail_revision"])
        self.assertTrue(saved["windows"][17]["overlap_evidence"]["shared_speech"])
        rows, events = replay_saved(saved)
        self.assertTrue(rows[16]["assembly"]["utterance_hypothesis"].endswith("75%"))
        expected = rows[16]["assembly"]["utterance_hypothesis"] + "가 php를 쓰고"
        self.assertEqual(rows[17]["assembly"]["utterance_hypothesis"], expected)
        self.assertEqual(rows[17]["events"][-1]["publication_text"], expected)
        self.assertEqual(events[-1]["publication_text"], expected)
        self.assertEqual(events[-1]["status"], "final")
        self.assertEqual(events[-1]["finalize_reason"], "pipeline_end")

    def test_short_percentage_requires_adjacent_supported_single_mode(self):
        for shared, supported, window, single in (
            (False, True, 1, True), (None, True, 1, True),
            (True, False, 1, True), (True, True, 2, True), (True, True, 1, False),
        ):
            with self.subTest(shared=shared, supported=supported, window=window, single=single):
                assembler = SubtitleAssembler(0, allow_shifted_head=single,
                                              enable_sentence_lifecycle=single)
                assembler.process(0, "비율은 75%")
                event = assembler.process(window, "75%가 해당합니다", shared_speech=shared,
                                          supported_tail_revision=supported)
                self.assertEqual(event.utterance_hypothesis, "비율은 75% 75%가 해당합니다")

    def test_different_values_and_repeated_current_input_are_preserved(self):
        for previous, current, expected in (
            ("비율은 75%", "76%가 해당합니다", "비율은 75% 76%가 해당합니다"),
            ("비율은 75%", "75% 75%라고 반복했습니다", "비율은 75% 75%라고 반복했습니다"),
            ("비율은 175%", "75%가 해당합니다", "비율은 175% 75%가 해당합니다"),
            ("값은 75", "75가 해당합니다", "값은 75 75가 해당합니다"),
        ):
            with self.subTest(previous=previous, current=current):
                assembler = SubtitleAssembler(0, enable_sentence_lifecycle=True)
                assembler.process(0, previous)
                event = assembler.process(1, current, shared_speech=True,
                                          supported_tail_revision=True)
                self.assertEqual(event.utterance_hypothesis, expected)


if __name__ == "__main__":
    unittest.main()

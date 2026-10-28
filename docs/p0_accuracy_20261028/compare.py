"""CPU-only before/after comparison against the immutable base commit."""
import importlib.util
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'phase3'), str(ROOT / 'phase4'), str(ROOT)]
import subtitle_assembler as after
from caption_diagnostics import start_trace, stop_trace, trace_event

source = subprocess.check_output(['git', 'show', '0a2a33a:phase3/subtitle_assembler.py'], cwd=ROOT, text=True)
import types
before = types.ModuleType('baseline_assembler')
sys.modules[before.__name__] = before
exec(compile(source, '0a2a33a:subtitle_assembler.py', 'exec'), before.__dict__)


def replay(module, texts):
    assembler = module.SubtitleAssembler(0, enable_sentence_lifecycle=True)
    state = module.SpeakerSubtitleState(0, require_final_support=True)
    events = []
    for index, raw in enumerate(texts):
        a = assembler.process(index, raw, shared_speech=True if index else None)
        if a.rollover_text and state.partial_text:
            final = state.finalize(index, index * 2 + 3, a.rollover_reason)
            assembler.commit_finalized_text(final.text)
            events.append(final)
        fragment = a.raw_fragment if not a.new_fragment else a.new_fragment
        events += state.process(index, a.utterance_hypothesis, True, index * 2 + 3,
                                confirmed_prefix_length=a.confirmed_prefix_length,
                                source_supported=True, source_text=fragment)
    final = state.flush(index, index * 2 + 3)
    if final:
        events.append(final)
    return ' '.join(state.final_segments), [(e.status, e.utterance_id, e.publication_text) for e in events]


def distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        previous, row = row, [i]
        for j, y in enumerate(b, 1):
            row.append(min(previous[j]+1, row[-1]+1, previous[j-1]+(x != y)))
    return row[-1]


sentence = '시청해 주셔서 감사합니다.'
following = '다음 소식을 전합니다.'
cases = [(sentence, [sentence]*3), (sentence+' '+following, [sentence, sentence, sentence+' '+following])]
report = {'scope': 'Constructed text fixtures, shared_speech supplied; not live STT or human-labelled audio',
          'base_commit': '0a2a33a', 'live_CER': None, 'live_WER': None, 'live_latency': None, 'cases': []}
for expected, texts in cases:
    item = {'input': texts, 'expected_final_transcript': expected}
    for label, module in [('before', before), ('after', after)]:
        actual, events = replay(module, texts)
        reference = after.normalize_for_matching(expected)
        item[label] = {'final_transcript': actual, 'events': events,
                       'extra_sentence_copies': max(0, actual.count(sentence)-1),
                       'CER': distance(reference, after.normalize_for_matching(actual))/len(reference),
                       'WER': distance(expected.split(), actual.split())/len(expected.split())}
    report['cases'].append(item)
for label, module in [('before', before), ('after', after)]:
    timings = []
    for i in range(220):
        start = time.perf_counter()
        for _, texts in cases:
            replay(module, texts)
        elapsed = (time.perf_counter()-start)*1000
        if i >= 20:
            timings.append(elapsed)
    report[label+'_cpu_ms_per_two_scenarios'] = {'mean': statistics.mean(timings),
        'p95': sorted(timings)[int(.95*(len(timings)-1))], 'samples': len(timings)}
with tempfile.TemporaryDirectory() as folder:
    handler = start_trace(Path(folder)/'trace.jsonl')
    start = time.perf_counter()
    for i in range(1000):
        trace_event('benchmark', session_id='synthetic', window_id=i, text=sentence)
    report['diagnostic_ms_per_small_record_mean'] = (time.perf_counter()-start)
    stop_trace(handler)
Path(__file__).with_name('comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))

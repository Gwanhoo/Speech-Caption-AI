"""CPU-only replay: withdrawn experimental branch versus conservative rollback."""
import json
from pathlib import Path
import sys
import time
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'phase3'))
import subtitle_assembler as after
from test_general_subtitle_lifecycle import GeneralSubtitleHarness

before = types.ModuleType('before_h1_fix')
sys.modules[before.__name__] = before
exec(compile(Path(__file__).with_name('before_subtitle_assembler.py.txt').read_text(),
             'before_h1_fix', 'exec'), before.__dict__)
CASES = {
    'H1': ['오늘 중앙 도서관에서는 휴관했습니다.', '중앙 도서관에서는 다음 주부터 정상 운영을 시작합니다.'],
    'H1_variant': ['어제 지역 박물관에서는 폐관했습니다.', '지역 박물관에서는 다음 달부터 새로운 전시를 시작합니다.'],
    'A': ['오늘 서울 도심에서는 날씨였습니다.', '서울 도심에서는 아침부터 많은 시민들이 출근길에'],
    'M1': ['오늘 서울 도심에서는', '오늘 서울 도심에서는 날씨였습니다.', '서울 도심에서는 아침부터 많은 시민들이 출근길에'],
    'B': ['시민들은 대중교통을 이용하거나', '공교통을 이용하거나 도로 상황을 확인하며 이동하고 있습니다.'],
    'C': ['이어졌지만 오후부터는 일부 지역의 구름', '일부 지역에 구름이 많아질 것으로 예상됩니다.'],
    'D_synthetic': ['네, 현재까지 큰 교통 혼잡은 발생하지', '큰 교통 혼잡은 발생하지 않았습니다.'],
}


def replay(module, texts):
    h = GeneralSubtitleHarness()
    h.assembler = module.SubtitleAssembler(0, enable_sentence_lifecycle=True)
    h.state = module.SpeakerSubtitleState(0, require_final_support=True)
    for i, text in enumerate(texts):
        h.speech(i, text, shared_speech=True, supported_tail_revision=True)
    h.silence(len(texts))
    return h.state.final_segments


if __name__ == '__main__':
    results = {key: {name: replay(module, texts) for name, module in
                    (('before', before), ('after', after))} for key, texts in CASES.items()}
    for _ in range(20):
        for module in (before, after):
            for texts in CASES.values():
                replay(module, texts)
    samples = {'before': [], 'after': []}
    for i in range(300):
        order = [('before', before), ('after', after)]
        for name, module in (order if i % 2 else reversed(order)):
            for texts in CASES.values():
                started = time.perf_counter()
                replay(module, texts)
                samples[name].append((time.perf_counter() - started) * 1000)
    report = {'kind': 'synthetic_text_with_injected_evidence_not_acoustic_accuracy',
              'cases': results, 'cpu_ms_per_scenario': {
                  name: {'n': len(v), 'mean': float(np.mean(v)),
                         'p50': float(np.percentile(v, 50)), 'p95': float(np.percentile(v, 95))}
                  for name, v in samples.items()}}
    Path(__file__).with_name('comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))

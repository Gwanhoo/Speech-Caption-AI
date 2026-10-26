# 저장된 STT 경계 중복 개선

기준 HEAD: `e910015`. 시작 작업 트리 clean. GPU 추론/다운로드/학습, Windows 실행,
commit/push 없이 저장된 RAW와 admission/overlap 근거를 CPU에서 재생했다.
운영 3초/2초, single, 모델 및 디코딩 설정은 변경하지 않았다.

## 조사 및 선정

`PROJECT_STATUS.md`는 2026-10-03 기준, `ACCURACY_AUDIT.md`는 2026-09-30 기준이다.
small 기본값·secondary 상태 설명 등은 현재 single 경로와 구분해 읽었다.
R2-B3/B4 독립 조사 파일은 checkout에서 찾지 못했다. B6 REPORT/DESIGN도 이 부재를
명시한다. 대신 `docs/overlap_accuracy_analysis.md`, production-evidence/lifecycle
회귀, 뉴스 JSON fixtures, `experiments/r2_b6/results/*_repeat_*.json`을 대조했다.

| 결함/후보 | 현재 증거와 원인 | 판단 |
|---|---|---|
| `75% 75%가` | B6 baseline w16→17, shared_speech와 supported_tail_revision 모두 true. `%`를 제거한 정규화 길이 2가 relaxed 최소 3보다 작아 exact 경계조차 미검출 | 선정: 값 변경 없이 정확히 같은 숫자 토큰의 경계만 결합 가능 |
| `취약점은 최악점은`, `인터넷 전문 인터넷 전반에` | baseline w6→7, w12→13에 재현. RAW 대안 가설이 달라 기존 seam 조건 미충족 | 보류: 새 가설의 정답 여부와 변경 범위를 판단할 단어 정렬/청취 근거 부족. 광범위 병합은 정상 반복·다른 문장을 삭제할 위험 |
| `사용된 프로그래밍` → `프레밍` | baseline w15→16 supported_tail_replace가 이전 prefix까지 교체 | 보류: 보존 문제는 보이나 바뀐 어휘를 어디까지 같은 음성으로 대응할지 추가 검수 필요 |
| 조기 FINAL | 과거 뉴스의 구름/건강 문장 수정은 현재 sentence_head_revision 회귀로 이미 보호. B6 현재 baseline FINAL 3개, 과거 기록 6개 | 기존 해결을 다시 수정하지 않음. 3개가 정답 문장 수라는 주장도 하지 않음. B4 전체 해결 여부는 별도 자료 필요 |
| RAW 오인식 | `기운/기온`, `알려줍니다/알려졌습니다` 등 | 후처리로 정답 생성 금지. 검수 전사 없이 CER/WER 및 정확도 개선율 산출하지 않음 |
| source-support 기존 예상 실패 | gap 뒤 발화 숨김, noise, projection artifact 3건 | 이번 단일 경계 수정과 별개로 유지. 실패를 PASS로 간주하지 않음 |

## 실제 처리 경로

- `phase4/gpu_processing_server.py:PipelineService._process_serialized`에서 single 원본 slot,
  Silero VAD 및 `phase3/stt_context.py:transcribe_base`의 RAW를 만든다.
- `phase3/run_overlap_pipeline.py:stt_worker`가 remote RAW/VAD를 받아 primary 선택,
  `resolve_subtitle_fragments`, `admit_subtitle_streams`, `subtitle_overlap_evidence`를 거친다.
- `SubtitleAssembler.process`가 overlap과 가설을 조립한다. rollover 시 기존 state를
  finalize하고 `SpeakerSubtitleState.process`에 hypothesis/source 근거를 전달한다.
- state의 `publication_text`는 source-supported 텍스트이며, `finalize`/`flush`가
  FINAL 또는 discard를 결정한다. `publish_subtitle_state_event`는 비어 있지 않은 공개문이나
  기존 utterance의 retraction만 WebSocket/Qt hook에 전달한다.

## 최소 수정

`SubtitleAssembler.process`에 18줄을 추가했다. 기존 매칭이 실패했고 single용 옵션 두 개가
활성화되었으며, 인접 window의 shared_speech와 supported_tail_revision이 모두 있는 경우만
검사한다. 이전 끝의 완전한 숫자+`%` 토큰과 현재 시작의 **동일한 원문**을 결합하고 기존
`boundary_exact` 조립 경로를 재사용한다. 숫자 값·표기 교정, 전역 중복 제거, 한국어 치환은 없다.
`175%`의 일부를 `75%`로 맞추지 않으며, 현재 RAW 안에 두 번 등장한 값은 두 번 유지한다.

## FAIL-first 및 비교

- 수정 전 새 테스트 3개: 1 PASS, 2 FAIL. 실제 baseline의 중복 공개와 현재 RAW 내부 반복
  앞에 중복 경계를 한 번 더 붙이는 현상을 확인했다.
- 수정 전 재생 helper가 보관된 12회/189 window의 assembly/state와 일치함을 확인했다
  (연산시간 필드 제외). 원본 파형이 없어 admission/overlap은 기록 값을 사용하며 재추정하지 않는다.
- 수정 후 대상 테스트 4개 PASS. 전체 12회 corpus의 공개 텍스트와 이벤트 수, action,
  utterance ID, FINAL 이유 및 stream 시각 보존을 포함한다.
- baseline 3회 모두 w17·18 및 마지막 FINAL에서만 `75% 75%가` → `75%가`.
  나머지 A/B/C 9회는 공개 텍스트 동일. 모든 run의 FINAL 수 3개와 확정 시점 유지.
  반복 3회는 독립 발화 3개가 아니며 한 음원에서의 재현성 확인이다.
- RAW, 다른 중복, 누락 후보, 오인식, 조기 FINAL 정책은 바뀌지 않는다.
- 교차 순서 CPU replay 10회, 버전당 assembler 1,890 호출:
  수정 전 mean/p95 1.2905/1.0352 ms, 수정 후 1.2688/1.0405 ms.
  지연 향상 주장이 아니며 작은 차이는 실행 변동 범위로 본다. 추가 window 대기는 없다.
- B6 산출물·기존 fixture 등 38개 파일 SHA-256 동일. 누락된 원본 WAV/timeline을 생성하지 않았다.

## 회귀 실행

```bash
PYTHONPATH=phase3 .venv/bin/python -m unittest -v phase3.test_percentage_overlap
PYTHONPATH=.:experiments/r2_b6:phase3:phase4 .venv/bin/python experiments/r2_b6/run_checks.py
git diff --check
```

최종 Phase 3: 182개 = 179 PASS + 기존 expected failure 3개.
Phase 4: 67 PASS. 기존 assertion 스크립트 14개 PASS.
실행기에 포함된 B6 단위 검사: 2 PASS + 원본 evidence 부재 1 SKIP; GPU 실험은 미실행.
변경 Python AST syntax/import, diff 공백 검사 PASS.

## 위험과 후속 검증

실제 음성에서 같은 퍼센트를 경계 양쪽에서 독립적으로 반복했는데 기존 acoustic 판정이
revision으로 오판하면 하나를 합칠 가능성은 남는다. 현재 RAW 내부 반복, 음성 overlap 없음,
tail 근거 없음, window gap, overlap 모드, 다른 숫자에 대한 보존을 회귀로 제한했다.
숫자 표기의 전체 범위(부호·한글 숫자·단위)로 규칙을 확대하지 않았다.

원본 오디오 및 단어 시간 정렬을 확보해 경계의 실제 반복/재인식을 구분하는 검수가 다음 단계다.
특히 B3/B4 자료, 누락된 뉴스 원본 WAV/timeline, 사람이 확인한 전사를 확보해야 더 넓은
대안 가설 병합과 FINAL 개선을 평가할 수 있다. Windows 라이브에서는 같은 수치의 실제 반복,
표시 지연, queue drop을 확인한다. 이번 결과는 음향 STT 정확도 또는 Windows GUI 성공 증거가 아니다.

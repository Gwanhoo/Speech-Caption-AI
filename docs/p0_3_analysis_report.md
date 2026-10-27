# VOXLUX P0-3 문장 누락·중복 분석 및 수정 인계

작성: 2026-10-27 UTC. **분석 전용: 운영·테스트 코드 수정, 리팩터링, 커밋, 푸시 없음.**

## 1. 현재 Git 상태와 분석 범위

- 브랜치: `main`.
- HEAD: `8f5220caf4c04e44ce978f286cf73cba0659c5d6` (`docs: record P0-2A commit and push verification`).
- 시작 및 분석 테스트 실행 후 `git status --short`: 출력 없음.
- 이번 변경은 이 보고서 `docs/p0_3_analysis_report.md` 신규 작성뿐이다. 종료 시 이 파일만 미추적 상태로 남긴다.
- 분석 대상: SINGLE, 3초 window/2초 stride, `large-v3-turbo`, callback→Qt→GUI/overlay 경로.
- 사용량은 직접 조회하지 않았다. 최신 출력 폴더·기존 관련 보고서·대표 저장 결과 1회·직접 관련 코드와 기존 테스트로 범위를 제한했다. GPU 추론, 네트워크 요청, 전체 회귀, 모델 다운로드는 실행하지 않았다.

## 2. 자료 가용성과 최신 사례의 공백

**이번 사용자 테스트의 최신 Windows 로그/evidence는 저장소에서 확인되지 않았다.** `phase3/output`, `phase5/output`에는 분석 가능한 실측 파일이 없었다. 최신 파일 대부분은 이전 세션의 `docs/p0_2_evidence/`이며, 이를 이번 누락·중복의 새 테스트로 간주하지 않았다. 다른 저장 위치를 질문했으나 보고서 작성 시점까지 추가 경로는 제공되지 않았다.

확인한 과거 자료:

| 자료 | 용도/한계 |
|---|---|
| `experiments/r2_b6/results/baseline_repeat_0.json` | 3/2 SINGLE의 19개 window, RAW/admitted/VAD/overlap/assembly/state/FINAL. 이번에 현재 코드로 메모리 재생했다. |
| `experiments/r2_b6/results/metadata.json` | 당시 HEAD `1d306bd...`, SINGLE/large-v3-turbo, 실제 loopback HTTP + 모의 캡처 시간·직렬 no-drop 조건. **최신 Windows GUI 실측이 아니다.** |
| `phase3/test_fixtures/news_20261022_160827_windows.json` | 원본 timeline에서 추출한 w0~10 RAW/VAD/조립 기록. |
| `phase3/test_fixtures/news_20261022_160827_replacement_windows.json` | 동일 timeline에서 추출한 w11~13 RAW/state 기록. |
| `docs/r2_b34_followup.md`, 대응 validation JSON | prefix 보존 후보의 실패 및 원복 근거. 같은 실패 접근을 반복하지 않도록 사용. |
| `experiments/r2_b6/REPORT.md` | 과거 FINAL 경계 비교와 원음/독립 B3·B4 조사 자료의 부재 설명. |

두 뉴스 fixture가 명시한 원본 timeline SHA-256은 `a53803fdf0498c145c4c96c7cbfa94550a0e7baa83e448610dbdfd0614131766`이다. baseline의 원본 음원 SHA-256은 `6424daf1e0715411794ccb2e66b258a698f64fbfa7a665a16f477c6431a3964f`이다. 이 세션에서는 원본 `original.wav`와 `timeline.jsonl`을 확보하지 못했다. 저장소의 일반 reference WAV를 해당 뉴스의 원음으로 대체하지 않았다. **실제 음성에 단어가 있었는지, 실제 반복 발화인지, 어떤 RAW가 정답인지 판정하지 않았다.**

R2-B1/B2라는 명칭의 독립 문서와 코드 변경을 일대일로 연결할 근거도 확보되지 않았다. 명칭을 추정하지 않고 아래 실제 커밋·기능·회귀 테스트로 연속성을 확인했다.

## 3. 문제 사례: RAW → Assembler → State/FINAL → GUI

아래 window는 모두 **과거 baseline 자료의 0-based 번호**다. 최신 사용자 세션의 번호가 아니다. stream 시간은 `start=2×window`, `end=start+3`이며 GUI 도착 시각이 아니다.

### M1. w15→16: 이전 RAW 앞부분이 조립 단계에서 사라짐

| 단계 | 확인 결과 |
|---|---|
| w15, 30~33초 RAW/admitted | `사용된 프로그래밍 언어를 식별하며` |
| w15 PARTIAL | 같은 문구, utterance 3. 이 시점에서 앞 utterance 2는 FINAL 처리됨. |
| w16, 32~35초 RAW/admitted | `프레밍 언어를 식별할 수 있는 웹사이트의 75%` |
| w16 저장 overlap | `shared_speech=True`, `supported_tail_revision=True` |
| 현재 Assembler 재생 | `supported_tail_replace`, `confirmed_prefix_length=0`; 가설이 w16 RAW 전체로 바뀌어 `사용된`이 제거됨. |
| 현재 State/PARTIAL | w16 가설과 같은 `publication_text`. 이 사례는 GUI에서 처음 삭제된 것이 아니다. |
| w18, 39초 FINAL | `프레밍 언어를 식별할 수 있는 웹사이트의 75%가 php를 쓰고`, `pipeline_end`. `사용된`이 되살아나지 않음. |
| GUI | 실제 기록 없음. 현재 presentation에 재생한 입력 텍스트는 그대로 유지됨(도착 시간은 합성). 실제 화면 표시 여부는 미확정. |

**확정된 기계적 원인:** `supported_overlap_tail_anchor()`가 w15/w16에 `(2, 10, 0.75, 0)`을 반환한다. 마지막 값인 보호할 이전 prefix 길이가 0이다. 현재 RAW의 anchor가 첫 글자에서 시작하지 않으면(`block.b > 0`) 이전 RAW 앞부분을 보호하지 않는 분기가 있다. `SubtitleAssembler.process()`의 supported tail replacement는 전체 previous가 active hypothesis 끝에 있고 보호 prefix/confirmed prefix가 없으면 새 RAW로 이전 tail을 교체한다.

**미확정:** 이전 `사용된`이 실제 발화였고 새 window 관측 범위 밖이므로 반드시 보존해야 한다는 음향적 근거. `프로그래밍`/`프레밍` 중 어느 것이 원음인지도 미확정이다. “텍스트 삭제 재현”과 “정상 발화 누락 확정”을 구분한다.

### D1. w6→7: 서로 다른 RAW 표현이 연속 연결됨

- w6(12~15초) RAW/admitted: `언어 PHP에서 발견된 취약점은`.
- w7(14~17초) RAW/admitted: `최악점은 해커가 로그인이나 인증 없이`.
- w7: shared speech=true, supported tail revision=false, `match_type=none`.
- 현재 조립/PARTIAL에 `취약점은 최악점은`이 남고, w11/25초 FINAL에도 유지된다.
- **확정:** 서로 다른 RAW가 seam으로 인정되지 않아 append된다. State나 GUI가 동일 이벤트를 두 번 배달해서 생긴 문자열이 아니다.
- **미확정:** 두 표현이 겹친 같은 발화의 서로 다른 STT 가설인지, 원음의 실제 반복/수정 발화인지. `최악점은`을 단순 오인식이라고 확정할 수 없다.

### D2. w12→13: 유사한 어절의 중복 연결

- w12(24~27초) RAW/admitted: `파급력이 큰 건 php가 인터넷 전문`.
- w13(26~29초) RAW/admitted: `인터넷 전반에 광범위하게 쓰이는 핵심`.
- w13: shared speech=true, supported tail revision=false, `match_type=none`.
- 현재 조립/PARTIAL: `이번 사태 파급력이 큰 건 php가 인터넷 전문 인터넷 전반에 광범위하게 쓰이는 핵심`.
- w15/33초 FINAL에도 `인터넷 전문 인터넷 전반에`가 유지된다.
- D1과 동일하게 **append 경로는 확정**, 반복 발화인지 대안 STT 가설인지는 원음 없이 미확정이다. M1의 과도한 교체와는 반대 방향의 문제이므로 하나의 완화 규칙으로 고치면 안 된다.

### D3. w16→17의 `75% 75%`는 현재 재현되지 않음

과거 baseline JSON에는 `75% 75%가 php를 쓰고`가 있다. 현재 코드 재생은 w17에서 `boundary_exact`를 적용하고 `75%가 php를 쓰고`를 PARTIAL 및 w18 FINAL에 한 번만 남긴다. `745688d`의 기존 수정과 `test_percentage_overlap`이 보호한다. **과거 로그에 중복이 있다는 사실만으로 현재 결함이라고 판단하면 안 된다.**

### M2. 미지원 prefix 뒤 정상 suffix가 공개되지 않는 기존 한계

기존 `SourceSupportDiagnosticsTests.gap_state(True)`를 메모리에서 실행:

```text
synthetic w0: old             source_supported=True
synthetic w1: old gap         source_supported=False
synthetic w2: old gap genuine source_supported=True, source_text=genuine
w2 내부 text: old gap genuine
w2 publication_text: old
reason: unsupported_prefix_gap
retained_prefix_length: 3, unsupported_gap_length: 3
```

**확정:** source support를 연속 prefix로만 표현하는 State 정책 때문에 gap 뒤 지원된 suffix도 공개되지 않는다. 관련 테스트는 기존 expected failure다. **최신 Windows 사례와의 연결은 미확정**이며 위 window 번호는 합성 재현용이다. gap을 무조건 공개하거나 source_supported를 전체 가설에 확대하면 유령 자막 방어가 무너진다.

### 표시·캡처 단계에서 구분할 정상 정책

- SINGLE presentation은 최신 FINAL 최대 2개와 PARTIAL을 유지하고 FINAL을 도착 후 4초에 만료한다. 같은 utterance의 FINAL은 PARTIAL을 교체한다. text 내용이 아닌 event/sequence/utterance identity로 재전달을 차단한다.
- Overlay는 화면 높이에 맞춰 오래된 entry를 제외할 수 있다. RAW/FINAL에 존재하지만 화면에서 사라진 문장은 조립 누락과 구분해야 한다.
- baseline은 40초 입력에서 완전 window가 39초까지라 39~40초 tail이 처리되지 않은 과거 실험이다. 이는 해당 부분의 음성 유무를 확인한 STT 누락 사례가 아니다. 마지막 FINAL도 자연 문장 종결이 아닌 `pipeline_end`다.
- 현재 P0-1 stale/STOP 드롭과 Qt worker identity 방어를 임의 변경할 근거는 없다. 최신 `drop_details`와 실제 arrival 로그가 없으므로 이번 사용자 현상의 지연·드롭 기여도는 알 수 없다.

## 4. 관련 코드 위치와 기존 보호 기능

위치는 현재 HEAD 기준이며 수정 시 다시 확인한다.

| 파일:행 / 함수 | 분석 포인트 |
|---|---|
| `phase3/run_overlap_pipeline.py:1181` `capture_worker` | 채널 선택·resampling·이전 1초 tail+새 2초; 새 PCM과 window 좌표 분리 |
| `phase3/audio_diagnostics.py:8` `select_capture_channel` | 실제 다채널 내용 확인 없이 채널 정책 변경 금지 |
| `phase4/gpu_processing_server.py:313` `_process_serialized` | SINGLE 원본 slot 0, VAD 양성일 때 전체 창 STT |
| `phase3/validate_end_to_end_gpu.py:43` `detect_speech_activity` | 실제 서버 VAD 호출부 |
| `phase3/stt_context.py:38` `transcribe_base` | ko/beam 5/condition_on_previous_text=False, RAW·진단 집계 |
| `phase4/remote_gpu_client.py:119` `parse_result` | 요청/window identity·VAD/RAW 계약 검증 |
| `phase3/run_overlap_pipeline.py:1709` `stt_worker` | RAW→primary/routing/admission→조립→State |
| `phase3/secondary_leakage_diagnostics.py:542` `subtitle_overlap_evidence` | 동일 음성 overlap과 revision 근거. 단어별 정답/삭제 범위의 증거와는 다름 |
| `phase3/subtitle_assembler.py:298` `supported_overlap_tail_anchor` | M1에서 anchor와 unobserved prefix 결정 |
| 같은 파일 `:570` `SubtitleAssembler.process`, `:800`~`:865` tail 교체 분기 | M1 교체, D1/D2 무매칭 append, `:660`대 퍼센트 보호 |
| 같은 파일 `:1215` `SpeakerSubtitleState.process`, `:1365` 주변 | M2 `unsupported_prefix_gap`; `:1206` 공개 텍스트 선택, `:1430` finalize |
| `phase3/run_overlap_pipeline.py:731` `publish_subtitle_state_event`, `:2124` 주변 | publication_text 공개, 조립 결과별 source_text 선택. GUI text는 누적 assembled_text와 구분 |
| `phase5/live_caption_controller.py:341,410` | callback→Qt signal, 현재 worker/state 검사 |
| `phase5/gui_app.py:144,452` | `upsert_event`, `apply_subtitle_event` |
| `phase5/subtitle_presentation.py:70,87` | 만료, 순서·중복 차단, PARTIAL/FINAL 상태 |
| `phase5/subtitle_overlay.py:126,240` | entry 표시 및 화면 높이에 따른 선택 |

확인한 기존 수정 이력: `ff000d1` 조기 FINAL 방지, `d4e272b` 한국어 어절 연속 보존, `95e4b91` 미지원 fuzzy 교체 보호, `745688d` 퍼센트 경계 중복 제거. R2-B1~B4 명칭과 완전히 대응한다고 추정하지 않는다. 현재 테스트가 보호하는 뉴스 어미·문장 head revision·파급력 앞부분·confirmed prefix·의도적 반복 기능을 유지해야 한다.

## 5. 이번에 실제 실행한 검증

로그 파일을 추가하지 않고 stdout으로만 실행했다. `-B`와 `PYTHONDONTWRITEBYTECODE=1`로 bytecode 기록도 억제했다.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=phase3:phase4:. python -B -m unittest test_subtitle_assembler_production_evidence test_general_subtitle_lifecycle test_percentage_overlap test_source_support_diagnostics.SourceSupportDiagnosticsTests.test_unresolved_gap_must_not_hide_later_supported_speech test_source_support_diagnostics.SourceSupportDiagnosticsTests.test_gap_diagnostic_distinguishes_blocked_update_from_no_source_support phase5.test_subtitle_presentation
```

실제 결과: **78개 실행 / PASS 77 / 기존 XFAIL 1 / 신규 FAIL·ERROR 0**, 0.272초. XFAIL은 M2이며 해결 또는 PASS로 보고하지 않는다. `test_percentage_overlap`에 포함된 기존 저장 결과 12회/189 window 비교도 통과했다. 이는 텍스트 재생이며 GPU 재추론이 아니다.

별도로 기존 `test_percentage_overlap.replay_saved()`를 baseline 1회에 호출하여 M1/D1/D2/D3의 assertion 4개를 확인했다. 같은 이벤트를 `SubtitlePresentationState`에 합성 sequence와 고정 도착 시간으로 적용하니 마지막 FINAL 2개(utterance 2,3)가 입력 텍스트 그대로 남았다. **Qt 화면·실제 도착 시간·만료 실측이 아닌 순수 presentation state 재생**이다.

다음 명령은 이번 메모리 재생의 핵심 확인을 파일 생성 없이 다시 실행한다:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=phase3:. python -B - <<'PY'
import json
from test_percentage_overlap import replay_saved
saved = json.load(open('experiments/r2_b6/results/baseline_repeat_0.json'))
rows, events = replay_saved(saved)
assert '사용된' not in rows[16]['assembly']['utterance_hypothesis']
assert '취약점은 최악점은' in rows[7]['events'][-1]['publication_text']
assert '인터넷 전문 인터넷 전반에' in rows[13]['events'][-1]['publication_text']
assert '75% 75%' not in events[-1]['publication_text']
for w in (7,13,16,17):
    print(w, rows[w]['assembly']['match_type'], rows[w]['events'][-1]['publication_text'])
for e in events:
    if e['status'] == 'final':
        print(e['window'], e['finalize_reason'], e['publication_text'])
PY
```

이 assertion은 **현 상태 관측**이며 누락을 올바른 동작으로 고정하는 새 회귀 테스트가 아니다. 운영·테스트 파일에 추가하지 않았다.

미실행: 최신 Windows 통합, 실제 GUI/overlay 화면, WASAPI, 원본 WAV 청취·정답 전사, GPU/HTTP 재추론. 최신 대응 자료가 없거나 이번 분석 목적상 필요하지 않아 수행하지 않았다. 기존 P0-2 성공 결과를 이번 테스트 결과로 합산하지 않는다.

## 6. 수정 우선순위와 최소 수정 계획

### 우선 0: 최신 사례를 기존 분류에 연결

새 자료의 session/window/request ID를 기준으로 `raw_transcripts → transcripts/admission → assembly_events → subtitle_state_events.publication_text → subtitle_events.text → voxlux.presentation → overlay`를 한 행으로 맞춘다. RAW부터 없으면 Assembler 수정으로 해결하지 않는다. GUI 입력은 정상인데 화면만 다르면 서버/Assembler를 수정하지 않는다.

### 우선 1: M1 교체 범위의 증거를 보강한 뒤 제한

- 시작 재현: baseline w15→16과 현재 anchor 반환값 `(2,10,0.75,0)`을 사용한다. 원음 확인 전에는 “사용된을 무조건 남긴다”는 기대값을 정답으로 확정하지 않는다.
- 필요한 조건: 기존 RAW 중 수정 대상 음향 구간 밖의 prefix인지, 다음 window가 앞 어절을 잘라 관측한 것인지 구분할 단어/구간 대응 근거.
- 최소 변경 후보: 근거가 확보되면 `supported_overlap_tail_anchor`의 revision 시작 위치와 `SubtitleAssembler.process`의 protected prefix 계산만 제한한다. previous RAW보다 앞의 active text 및 confirmed prefix 보존을 유지한다. 단어별 정렬 도입이 필요한 경우 범위·비용을 별도 판단하고 임의 유사도 기준으로 대체하지 않는다.
- **재사용 금지 접근:** 내부 어절 anchor이면 이전 앞 단어를 항상 남기는 후보는 이미 기각됐다. `docs/r2_b34_followup_validation.json`에서 기존 테스트 2개가 실패했다. `제 손은 그냥과와 높아지는`→`분양가와 높아지는 청약문턱에 지친 세대가`를 교정할 때 잘못된 `제 손은`까지 남았다.

### 우선 2: D1/D2는 동일 음성의 대안 가설인 경우에만 seam 교정

- 원음에서 실제 반복이 아님을 확인하고, 겹친 span의 대응이 확보된 최소 두 window fixture를 만든다.
- 필요하면 기존 boundary/correction 함수에서 해당 관측 span의 이전 suffix와 새 prefix만 조정한다. 현재 `supported_tail_revision=False`인 사례를 True로 위조하지 않는다.
- 전역 fuzzy 임계값 완화, 특정 단어 예외, 과거 FINAL과의 전역 text dedup은 금지한다. 의도적 반복·동일 문구의 새 utterance·서로 다른 수치가 없어질 위험이 크다.
- M1의 교체 범위 변경과 분리하여 검증한다. “중복 감소”만 보고 prefix 삭제 증가를 수용하지 않는다.

### 우선 3: M2는 최신 사례에서 gap이 확인될 때 별도 수정

- 기존 `old → old gap → old gap genuine` 재현부터 시작한다.
- 지원된 suffix를 미지원 prefix와 독립적으로 공개할 수 있는 span 근거나 명시적인 새 utterance 경계를 설계한다. `_source_supported_text`를 가설 전체로 늘리는 수정은 금지한다.
- 필요한 상태 표현 변경이 커지면 이번 seam 최소 수정과 분리한다. 문장 단절·순서 역전·미지원 텍스트 부활·FINAL 중복 위험을 함께 평가한다.

### GUI/캡처/드롭은 증거가 나올 때만

FINAL 만료·화면 행 선택이면 현재 디자인/수명 정책의 문제인지 별도 요구사항으로 판단한다. 새 자막 재생성 오류와 혼동하지 않는다. latest log의 `drop_details`, window 누락, 음원 tail 처리 증거가 없으면 P0-1 queue/timeout/age 정책을 변경하지 않는다. window/stride 및 모델 교체도 이번 최소 수정의 출발점으로 삼지 않는다.

## 7. 다음 작업의 필수 회귀 계획

| 변경 후보 | 실패 먼저 확인할 신규 사례 | 함께 유지할 기존 검증 |
|---|---|---|
| M1 | 실제 단어 구간이 확인된 w15→16 prefix 보존, older active prefix 포함 변형 | `ReplacementSafetyRegressions.test_supported_substitution_of_unconfirmed_tail_still_reaches_final`, `test_supported_tail_revision_cannot_delete_text_before_previous_raw`, confirmed prefix/삭제-only revision 보호 |
| D1/D2 | 원음에서 단일 발화로 확인된 w6→7, w12→13의 seam 중복 | `test_supported_tail_revision_keeps_intentional_repetition`, `test_repetition_and_unrelated_internal_words_are_preserved`, shared_speech=False/비연속 window의 유사 단어 보존 |
| 기수정 기능 | 추가 변경 전후 같은 fixture 결과 비교 | 뉴스 w8→9 `실행하거나`, w11→12 앞부분, sentence_head_revision, `test_percentage_overlap` 전체 |
| M2 | unsupported gap은 비공개, 이후 확인된 suffix는 공개 | ghost/finalization evidence, 미지원 반복의 PARTIAL·FINAL 부활 방지, 정상 짧은 발화 보존 |
| GUI 변경 필요 시 | 동일 event 재전달, 더 큰 sequence의 이전 FINAL, 만료 후 replay, 새 utterance의 같은 문구 | `phase5.test_subtitle_presentation`, 실제 Qt timer/STOP·재시작 테스트. 누락된 녹화 timeline을 가짜로 대체하지 않음 |

저장된 과거 FINAL 전체가 정답은 아니다. 정상 보존용 fixture와 수정 대상 fixture를 구분하되 기존 assertion을 임의 완화하지 않는다. 각 패치마다 RAW 변화 없음/조립 변화/공개 FINAL 변화/GUI 변화 및 영향 window를 보고한다.

## 8. Windows에서 추가 수집할 증거

기존 `--evidence-dir`, `--latency-diagnostics`를 이용해 짧은 재현 세션을 수집한다. 이번 세션에서는 실행하지 않았다.

```powershell
python -u -m phase5.gui_app --window-seconds 3 --stride-seconds 2 --latency-diagnostics --evidence-dir evidence/p0_3_run1
```

1. Windows와 서버 각각의 HEAD, 서버 health의 SINGLE/model/STT options, 시작·STOP 시각. 재시작은 다른 evidence 경로를 사용한다.
2. 같은 세션의 `original.wav`, 문제 window 및 앞뒤 2개 mixed WAV, `timeline.jsonl`, `phase5/output/gui_last_run.json`, 콘솔 출력, `phase5/output/voxlux.log`, request ID가 같은 서버 로그.
3. 실제 화면 녹화 또는 표시 순서/시각 기록. PARTIAL이 수정된 것인지, FINAL과 PARTIAL에 같은 부분이 함께 보인 것인지, 동일 FINAL이 만료 후 돌아온 것인지 구분한다.
4. 최소 검수 전사: 음원 start/end, 정확한 문구, 실제 반복 여부, 불확실 구간. 특히 M1의 앞 단어가 어느 window에 들어가는지와 D1/D2의 동일 음성 여부를 확인한다.
5. 로그의 `session_id`, `window/window_index`, `remote_request_id`, `utterance_id`, `sequence`, `match_type`, overlap flags, `support_update`, `finalize_reason`, drop reason을 보존한다. stream 시간과 Windows/server perf_counter를 서로 다른 시계로 취급한다.

기존 WAV는 Git에 추가하지 않는다. 새 자료가 없으면 **최신 누락·중복의 원인 판정은 보류**한다.

## 9. 다음 Codex 수정 작업용 프롬프트 초안

> VOXLUX P0-3 후속 최소 수정 작업이다. 먼저 `docs/p0_3_analysis_report.md`를 읽고 현재 HEAD/작업 트리와 추가 제공된 Windows 증거 경로를 확인하라. 기존 분석 전체를 반복하지 말고 최신 사례를 M1(교체 범위), D1/D2(대안 RAW seam), M2(source-support gap), 표시 정책/드롭으로 분류하라. 추가 증거 경로: `<실제 경로>`; 문제 session/window: `<실제 식별자>`.
>
> 대응 WAV와 RAW/admitted/assembly/publication/GUI 로그로 최초 손실·중복 단계를 확정한 **사례 하나**부터 실패 재현 테스트를 작성하고 최소 수정하라. WAV가 없으면 저장된 텍스트 변환과 음향 정답을 구분하고 임의 기대 문구를 만들지 마라. M1의 내부 anchor prefix 무조건 보존 후보는 이미 정상 교정 2개를 깨뜨렸으므로 재채택하지 마라. D1/D2에 전역 fuzzy 완화나 text 블랙리스트를 쓰지 마라. M2의 미지원 gap을 공개하여 뒤 suffix를 살리는 방식도 금지한다.
>
> SINGLE, 3초/2초, large-v3-turbo, P0-1 큐·드롭·timeout, GUI 디자인과 기존 분리 비활성 상태를 유지하라. 보고서 7절의 보호 회귀를 함께 실행하고 원음 근거·RAW→FINAL→GUI 차이·정상 발화 손실 위험을 기록하라. 코드 수정 범위가 커지거나 정답 근거가 부족하면 해당 수정은 보류하고 필요한 증거를 명시하라. 별도 승인 전에는 커밋·푸시하지 마라.

## 10. 인계 결론

기존 자료에서는 **w15→16의 앞부분 삭제 경로**, **w6→7 및 w12→13의 무매칭 append**, **합성 source-support gap의 suffix 비공개**를 현재 코드에서 확인했다. 퍼센트 중복은 이미 수정되어 있다. 실제 원음과 최신 Windows 로그가 없으므로 어느 사례가 이번 사용자의 문제인지, 삭제·반복된 텍스트가 음향적으로 잘못됐는지는 확정하지 않았다. 추가 대규모 탐색 대신 이 재현점과 기각된 수정 반례를 남겨 다음 작업이 즉시 증거 대조와 실패 테스트에서 시작하도록 했다.

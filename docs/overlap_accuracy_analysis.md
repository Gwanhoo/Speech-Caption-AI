# VOXLUX 자막 오버랩 정확도 분석 보고서

## 1. 결론

이번 로그에서 확인되는 문제는 두 층으로 나뉩니다.

- RAW STT가 먼저 잘못 인식한 구간:
  - window 3: `아침에서는`
  - window 5: `이전에는`
  - window 6: `구름이 많아졌습니다`
  - window 10: `큰 교육을`
  - window 13: `기운 변화`, `건강을 받았다`
  - window 14: 문두가 잘린 `비해 건강관리에`
- SubtitleAssembler가 오류를 확대하거나 확정한 구간:
  - window 5: 앞 window의 올바른 `오전에는`을 `이전에는`으로 파괴적 교체
  - window 7: `많아졌습니다`를 확정한 뒤 올바른 `많아질 것으로 예상됩니다`를 별도 문장으로 연결
  - window 11: `큰 교육을`을 교정하지 못하고 올바른 다음 RAW를 통째로 추가
  - window 14: `건강을 받았다.`를 FINAL로 확정하고 `비해 건강관리에...`를 새 문장으로 시작

따라서 STT만의 문제도 아니고 Assembler만의 문제도 아닙니다. STT가 오류 후보를 만들었고, Assembler가 후속 overlap 정보를 이용해 정정하지 못하거나 기존의 더 정확한 텍스트를 오히려 덮어썼습니다.

최종 결과는 정답 8문장에 대해 FINAL 10개입니다. 전체 문장 자체가 사라진 경우보다는 정답 2·5번 문장이 각각 두 utterance로 잘못 분리되어 2개가 늘어난 형태입니다. `오전→이전`, `기온→기운` 같은 어휘 누락·대체도 있습니다. 최종 잘못된 조립 결과는 [gui_last_run.json:252](/workspace/Speech-Caption-AI/phase5/output/gui_last_run.json:252), FINAL 목록은 [gui_last_run.json:428](/workspace/Speech-Caption-AI/phase5/output/gui_last_run.json:428)에서 확인됩니다.

로그의 `success: true`는 파이프라인 실행 성공을 뜻할 뿐 정확도 성공이 아닙니다. 또한 `reference_transcript_available=false`이므로 이 실행 자체는 정답 대본 기준 정확도를 계산하지 않았습니다([gui_last_run.json:121](/workspace/Speech-Caption-AI/phase5/output/gui_last_run.json:121)).

---

## 2. 전체 시간순 비교

표의 RAW는 admission 이전 `single_mode.streams[].raw_transcript` 기준입니다. 핵심 primary RAW는 `windows[].transcripts[0]`과 동일하고 모두 억제되지 않았습니다. `P/F`는 실제 공개된 PARTIAL/FINAL입니다. 전체 공개 이벤트 원본은 [gui_last_run.json:530](/workspace/Speech-Caption-AI/phase5/output/gui_last_run.json:530), window 원본은 [gui_last_run.json:5323](/workspace/Speech-Caption-AI/phase5/output/gui_last_run.json:5323)에 있습니다.

| window / 시간 | RAW STT | Assembler 결과 | PARTIAL / FINAL |
|---|---|---|---|
| 0 / 0–3s | 없음 | 없음 | 없음 |
| 1 / 2–5s | 없음 | 없음 | 없음 |
| 2 / 4–7s | `오늘 서울 도심에서는 아침부터` | `none` → 동일 | P u1 시작 |
| 3 / 6–9s | `아침에서는 아침부터 많은 시민들이 출근길에 나섰습니다.` | `anchored_continuation`; 잘못된 앞부분 제거 → `오늘 서울 도심에서는 아침부터 많은 시민들이 출근길에 나섰습니다.` | P u1 연장, 정상 복구 |
| 4 / 8–11s | `출근길에 나섰습니다. 오전에는 비교적` | `token_exact` → 새 가설 `오전에는 비교적` | F u1 정상, P u2 `오전에는 비교적` |
| 5 / 10–13s | `이전에는 비교적 맑은 날씨가 이어졌지만` | `fuzzy_replace`, 0.941 → `이전에는...` | P u2 교체, 최초 공개 오류 |
| 6 / 12–15s | `오후부터는 일부 지역에 구름이 많아졌습니다.` | `none`, 기존 뒤에 추가 | P u2 `...구름이 많아졌습니다.` |
| 7 / 14–17s | `구름이 많아질 것으로 예상됩니다.` | `none`; 이전 문장 rollover | F u2의 잘못된 문장 확정, P u3로 올바른 구절 별도 시작 |
| 8 / 16–19s | `시민들은 대중교통을 이용하거나 도로` | 새 문장 | F u3, P u4 |
| 9 / 18–21s | `이용하거나 도로 상황을 확인하며 이동하고 있습니다.` | `token_exact` → 정상 결합 | P u4 정상 |
| 10 / 20–23s | s0=`이동하고 있습니다. 네, 현재까지 큰 교육을`; s1=`ㄴㄴㄴㄴㄴㄴㄴ` | s0 `token_exact` → `네, 현재까지 큰 교육을`; s1 억제 | F u4, P u5에 STT 오류 공개 |
| 11 / 22–25s | `현재까지 큰 교통 혼잡은 발생하지 않았습니다.` | `none`, 전체 추가 → `네, 현재까지 큰 교육을 현재까지 큰 교통...` | P u5 중복 연결 |
| 12 / 24–27s | `하지 않았습니다. 한편 전문가들은 갑작스러운` | `boundary_exact` | F u5 오류 확정, P u6 시작 |
| 13 / 26–29s | `인가들은 갑작스러운 기운 변화에 대비해 건강을 받았다.` | `fuzzy`, 0.941; `전문가들은 갑작스러운`은 보존, 잘못된 suffix 추가 | P u6 `...기운 변화...건강을 받았다.` |
| 14 / 28–31s | s0=`비해 건강관리에 주의할 것을 당부했습니다.`; s1=`판매` | `none`; u6 rollover 후 잘린 RAW를 새 가설로 시작 | F u6 오류 확정, P u7 `비해 건강관리에...` |
| 15 / 30–33s | s0=`당부했습니다. 그럼요. 외출하기 전`; s1=`반` | `token_exact`; s1 억제 | F u7, P u8 |
| 16 / 32–35s | s0=`외출하기 전에 날씨를 확인하는 것도 좋은 방법입니다.`; s1=`외출하기 전에 날씨를` | `boundary_exact`; s1 억제 | P u8 정상 결합 |
| 17 / 34–37s | `하는 것도 좋은 방법입니다. 오늘 준비한` | `boundary_exact` | F u8, P u9 |
| 18 / 36–39s | s0=`오늘 준비한 소식은 여기까지입니다.`; s1=`이 시각 세계였습니다.` | `boundary_exact`; s1 억제 | P u9 정상 |
| 19 / 38–41s | `시청해 주셔서 감사합니다.` | 새 문장 | F u9, P u10 |
| 20 / 40–43s | `감사합니다.` | `boundary_exact`, duplicate-only | F u10; 재등장 없음 |
| 21 / 42–45s | 없음 | 없음 | 없음 |
| 22 / 44–47s | 없음 | 없음 | 없음 |
| 23 / 46–49s | 없음 | 없음 | 없음 |
| 24 / 48–51s | 없음 | 없음 | 없음 |

secondary stream의 `ㄴㄴㄴ…`, `판매`, `반`, 중복 발화, `이 시각 세계였습니다.`는 모두 `non_primary_stream`으로 억제되어 이번 최종 오류에 유입되지 않았습니다. 기존 재등장 방지와 single-primary 억제는 정상 동작했습니다.

---

## 3. 최초 오류 발생 window_index

| 구분 | 최초 window | 판정 |
|---|---:|---|
| RAW STT 오인식 | 3 | `아침에서는`이 최초. 단, Assembler가 정상 복구했으므로 최종 피해 없음 |
| 사용자에게 보이는 최초 잘못된 PARTIAL | 5 | 올바른 `오전에는`이 `이전에는`으로 교체 |
| 잘못된 문장 내용 | 6 | `구름이 많아졌습니다.`가 PARTIAL에 들어감 |
| 잘못된 문장 경계/연결 | 7 | 잘못된 문장을 FINAL 처리하고 올바른 continuation을 별도 u3로 시작 |
| 실제 중복 연결 | 11 | `큰 교육을 현재까지 큰 교통` 생성 |
| 중복의 FINAL 확정 | 12 | u5가 잘못된 상태로 FINAL |
| 건강 문장 오인식 | 13 | `기운 변화`, `건강을 받았다` |
| 건강 문장 잘못된 분리·연결 | 14 | u6 FINAL + `비해 건강관리에...`를 u7로 시작 |
| 전체 정답 문장 누락 | 없음 | 정답 문장이 통째로 빠진 것이 아니라 2·5번 문장이 손상·과분할됨 |
| 최초 어휘 누락/대체 | 5 | `오전`이 `이전`으로 대체됨 |

---

## 4. 지정된 네 문제 분석

### 4.1 `이전에는`

근거는 [window 5](/workspace/Speech-Caption-AI/phase5/output/gui_last_run.json:8185)입니다.

- window 4 RAW와 PARTIAL에는 정답인 `오전에는 비교적`이 존재했습니다.
- window 5 RAW가 `이전에는 비교적...`으로 오인식했습니다.
- `assembly_events[0]`:
  - `previous="오전에는 비교적"`
  - `match_type="fuzzy_replace"`
  - `similarity=0.941176...`
  - `utterance_hypothesis="이전에는 비교적..."`
  - `audio_overlap_evidence.supported_tail_revision=false`
- 이후 `subtitle_state_events`가 `action="replace"`, `match_type="hypothesis_correction"`으로 이를 공개했습니다.

원인 분류:

- STT: `이전에는`을 생성한 직접 원인.
- Assembler: 더 정확했던 기존 `오전에는`을 파괴적으로 교체한 확대 원인.

`corrected_suffix_prefix()`는 [subtitle_assembler.py:190](/workspace/Speech-Caption-AI/phase3/subtitle_assembler.py:190)에서 후보를 만들고, [subtitle_assembler.py:537](/workspace/Speech-Caption-AI/phase3/subtitle_assembler.py:537)부터 acoustic revision 지원 없이도 correction을 선택할 수 있습니다. 실제 교체는 [subtitle_assembler.py:564](/workspace/Speech-Caption-AI/phase3/subtitle_assembler.py:564)부터 수행됩니다.

특히 현재 회귀 테스트에는 짧은 가설의 새 window 교정을 허용하는 계약이 있습니다([test_subtitle_assembler_production_evidence.py:52](/workspace/Speech-Caption-AI/phase3/test_subtitle_assembler_production_evidence.py:52)). 따라서 단순히 fuzzy replacement를 끄면 기존 정상 교정도 깨집니다. acoustic evidence와 “파괴적 교체/비파괴 연장”을 구분해야 합니다.

### 4.2 `구름이 많아졌습니다`

근거는 [window 6](/workspace/Speech-Caption-AI/phase5/output/gui_last_run.json:8852)과 [window 7](/workspace/Speech-Caption-AI/phase5/output/gui_last_run.json:9519)입니다.

- window 6 RAW 자체가 `오후부터는 일부 지역에 구름이 많아졌습니다.`로 틀렸습니다.
- Assembler는 `match_type="none"`으로 이를 기존 u2 뒤에 추가했습니다.
- window 7에는 정답 구절 `구름이 많아질 것으로 예상됩니다.`가 들어왔습니다.
- 그러나 이전 가설이 마침표로 끝나고, window 7 역시 `match_type="none"`이며 `supported_tail_revision=false`여서:
  - u2를 `sentence_boundary_confirmed`로 FINAL
  - 같은 의미 구간의 올바른 RAW를 u3로 새로 시작했습니다.

문장 수정 경로 `reconcile_supported_sentence_revision()`은 [subtitle_assembler.py:319](/workspace/Speech-Caption-AI/phase3/subtitle_assembler.py:319)에 있지만, 호출은 `supported_tail_revision`이 참일 때로 제한됩니다([subtitle_assembler.py:596](/workspace/Speech-Caption-AI/phase3/subtitle_assembler.py:596)). 조건이 충족되지 않으면 [subtitle_assembler.py:661](/workspace/Speech-Caption-AI/phase3/subtitle_assembler.py:661)의 sentence rollover가 실행됩니다.

원인 분류:

- STT: `많아졌습니다`라는 잘못된 완결형 생성.
- Assembler: 다음 window의 올바른 수정형을 같은 문장 수정으로 보지 못하고 오문장을 먼저 확정.
- 결과: 정답 한 문장이 `...많아졌습니다.`와 `구름이 많아질 것으로...` 두 FINAL로 과분할됨.

### 4.3 `큰 교육을 현재까지 큰 교통`

근거는 [window 10](/workspace/Speech-Caption-AI/phase5/output/gui_last_run.json:11528)과 [window 11](/workspace/Speech-Caption-AI/phase5/output/gui_last_run.json:12309)입니다.

- window 10 RAW: `네, 현재까지 큰 교육을`
- window 11 RAW: `현재까지 큰 교통 혼잡은 발생하지 않았습니다.`
- 로그의 duplicate 분석도 두 window 사이에 `현재까지큰교` 6자를 공통 후보로 검출합니다([gui_last_run.json:382](/workspace/Speech-Caption-AI/phase5/output/gui_last_run.json:382)).
- 실제 Assembler는 `match_type="none"`과 `overlap=""`을 기록하고 window 11 전체를 append했습니다.
- 상태 계층은 이미 잘못 조립된 문자열이 이전 가설로 시작하므로 `hypothesis_extension`으로 분류했습니다.
- `support_update`도 기존 source-supported prefix와 새 suffix 사이에 gap이 없다고 판단하여 전체 조립 문자열을 공개했습니다.

원인 분류:

- STT: window 10의 `교육을` 오인식.
- Assembler: window 11의 정정 가능한 overlap을 찾지 못해 잘못된 꼬리와 올바른 다음 RAW를 모두 유지.
- SpeakerSubtitleState: 원인은 아니지만 잘못된 조립 결과를 정상 extension으로 통과시킨 downstream 증폭 단계.

상태 계층의 prefix/extension 판단은 [subtitle_assembler.py:1035](/workspace/Speech-Caption-AI/phase3/subtitle_assembler.py:1035), source-supported prefix와 새 suffix 결합은 [subtitle_assembler.py:1103](/workspace/Speech-Caption-AI/phase3/subtitle_assembler.py:1103)에 있습니다.

### 4.4 `건강을 받았다. 비해 건강관리에`

근거는 [window 13](/workspace/Speech-Caption-AI/phase5/output/gui_last_run.json:13628)과 [window 14](/workspace/Speech-Caption-AI/phase5/output/gui_last_run.json:14295)입니다.

window 13:

- RAW: `인가들은 갑작스러운 기운 변화에 대비해 건강을 받았다.`
- `인가들은 갑작스러운`은 이전 `한편 전문가들은 갑작스러운`과 fuzzy overlap으로 인식됐습니다.
- Assembler는 기존 `한편 전문가들은`을 보존했지만 새 suffix `기운 변화에 대비해 건강을 받았다.`는 그대로 추가했습니다.

window 14:

- RAW: `비해 건강관리에 주의할 것을 당부했습니다.`
- `shared_speech=true`, waveform correlation도 약 0.99999이나:
  - `supported_tail_revision=false`
  - `previous_tail_in_overlap=true`
  - `tail_covered_by_current_prefix=false`
- 텍스트 match는 `none`.
- window 13이 마침표로 끝났기 때문에 u6를 FINAL 처리하고 window 14를 u7로 시작했습니다.

원인 분류:

- STT:
  - `기온→기운`
  - `건강 관리에→건강을 받았다`
  - window 14 시작이 `대비해`가 아닌 `비해`로 잘림
- Assembler:
  - `대비해 건강...` 주변의 충돌을 revision으로 처리하지 못함
  - 잘못된 `건강을 받았다.`를 확정하고 잘린 후속 RAW를 별도 문장으로 연결

중요하게도 어떤 RAW window에도 정답 `기온 변화`가 없습니다. 따라서 Assembler만 수정해서 정확한 `기온`을 복원할 근거는 없습니다. Assembler 수정으로 잘못된 중복·경계 확정을 줄일 수는 있지만, 정확한 단어 복구에는 STT 개선이 필요합니다.

---

## 5. Acoustic overlap gate 분석

관련 계산은 [secondary_leakage_diagnostics.py:541](/workspace/Speech-Caption-AI/phase3/secondary_leakage_diagnostics.py:541)입니다.

핵심 문제 window들은 모두 `shared_speech=true`이고 waveform correlation도 높지만 `supported_tail_revision=false`입니다. 현재 gate는 다음을 모두 요구합니다.

- 마지막 이전 VAD 구간의 시작점이 overlap 안에 있어야 함
- 그 전체 tail 구간이 현재 window 첫 VAD 구간에 포함되어야 함
- 높은 waveform correlation

특히 [secondary_leakage_diagnostics.py:565](/workspace/Speech-Caption-AI/phase3/secondary_leakage_diagnostics.py:565)의 `previous[-1][0] >= stride` 조건은 발화가 overlap 이전부터 연속되어 overlap을 통과하는 경우에도 false가 됩니다. 실제 window 11처럼 거의 전체 3초가 하나의 연속 VAD 구간이면 shared audio가 있어도 tail revision 권한이 열리지 않습니다.

다만 로그에 단어 timestamp가 없으므로 이 조건을 단순 완화하면 서로 다른 실제 반복 발화를 잘못 지울 위험이 있습니다. gate가 너무 엄격하다는 현상은 확인되지만, 정확한 완화식은 저장된 로그만으로 확정할 수 없습니다.

---

## 6. 데이터 흐름과 관련 코드 위치

1. Whisper 옵션은 한국어, beam 5, `condition_on_previous_text=False`입니다: [stt_context.py:10](/workspace/Speech-Caption-AI/phase3/stt_context.py:10).
2. STT 텍스트는 segment 문자열을 그대로 연결합니다: [stt_context.py:38](/workspace/Speech-Caption-AI/phase3/stt_context.py:38).
3. 원격 서버가 VAD 활성 stream만 transcribe합니다: [gpu_processing_server.py:300](/workspace/Speech-Caption-AI/phase4/gpu_processing_server.py:300).
4. 서버 응답에는 `raw_transcript`만 저장되고 log probability 등 STT confidence는 포함되지 않습니다: [gpu_processing_server.py:317](/workspace/Speech-Caption-AI/phase4/gpu_processing_server.py:317).
5. 클라이언트는 원격 `raw_transcript`를 그대로 사용합니다: [run_overlap_pipeline.py:1540](/workspace/Speech-Caption-AI/phase3/run_overlap_pipeline.py:1540).
6. admission 이전 복사본은 `raw_transcripts`입니다: [run_overlap_pipeline.py:1573](/workspace/Speech-Caption-AI/phase3/run_overlap_pipeline.py:1573).
7. admission 및 single-primary 억제 후 Assembler에 전달됩니다: [run_overlap_pipeline.py:1737](/workspace/Speech-Caption-AI/phase3/run_overlap_pipeline.py:1737), [run_overlap_pipeline.py:1811](/workspace/Speech-Caption-AI/phase3/run_overlap_pipeline.py:1811).
8. sentence rollover가 있으면 기존 PARTIAL을 먼저 FINAL 처리합니다: [run_overlap_pipeline.py:1877](/workspace/Speech-Caption-AI/phase3/run_overlap_pipeline.py:1877).
9. 새 가설은 SpeakerSubtitleState로 전달됩니다: [run_overlap_pipeline.py:1898](/workspace/Speech-Caption-AI/phase3/run_overlap_pipeline.py:1898).
10. 공개 이벤트의 `text`와 누적 `assembled_text` 작성 위치는 [run_overlap_pipeline.py:597](/workspace/Speech-Caption-AI/phase3/run_overlap_pipeline.py:597)입니다.

실행 로그에는 실제 원격 Whisper 모델명이 기록되지 않았습니다. 따라서 특정 모델 크기 때문에 발생했다고 단정할 수 없습니다. STT confidence도 JSON에 없으므로 window 4와 5 중 어느 decoder 결과가 더 강했는지 수치 비교할 수 없습니다.

---

## 7. 최소 수정 계획

### 우선순위 1 — 파괴적 overlap 교체 제한

수정 대상:

- `phase3/subtitle_assembler.py`
- `phase3/test_subtitle_assembler_production_evidence.py`
- `phase3/test_accuracy_regressions.py`

계획:

- `fuzzy_replace`가 기존 active hypothesis의 앞부분까지 바꾸는 경우, 명시적 acoustic revision support가 false라면 새 window를 무조건 정답으로 간주하지 않도록 제한합니다.
- 비파괴적 suffix 추가와 기존 prefix 교체를 분리합니다.
- window 4→5의 `오전에는` 보존 사례를 추가하되, 실제로 새 window가 이전 오인식을 교정하는 기존 테스트는 유지합니다.
- 변경 범위는 active `utterance_hypothesis`로 한정하고 `_finalized_session`은 절대 수정하지 않습니다.

직접 대상 오류: `이전에는`.

### 우선순위 2 — 연속 발화의 acoustic tail revision 판정 보완

수정 대상:

- `phase3/secondary_leakage_diagnostics.py`
- `phase3/test_live_caption_continuity.py`
- `phase3/test_subtitle_assembler_production_evidence.py`

계획:

- 한 VAD 구간이 overlap 이전부터 시작해 overlap 끝까지 이어지는 경우를 별도 테스트합니다.
- 단순히 `shared_speech`만으로 삭제 권한을 주지 않고, overlap 구간의 실제 교집합·파형 상관·bounded lexical anchor를 함께 요구합니다.
- 현재의 “확정 자막 재등장 방지” 및 “실제 반복 발화 보존” 조건은 유지합니다.

직접 대상 오류: `큰 교육을 현재까지 큰 교통`, 일부 sentence revision 실패.

### 우선순위 3 — 잘못된 sentence rollover 보류

수정 대상:

- `phase3/subtitle_assembler.py`
- `phase3/test_general_subtitle_lifecycle.py`

계획:

- 이전 가설이 마침표로 끝났더라도 다음 adjacent window가 동일 overlap 구간의 충돌·수정 후보이면 즉시 FINAL하지 않고 revision 판정을 먼저 수행합니다.
- revision 근거가 없을 때만 기존 `sentence_boundary_confirmed` 동작을 유지합니다.
- 이미 FINAL 처리된 세션을 뒤늦게 다시 쓰는 방식은 사용하지 않습니다.

직접 대상 오류: `많아졌습니다. 구름이 많아질...`, `건강을 받았다. 비해 건강관리에...`.

### 우선순위 4 — 실제 로그 조각 회귀 테스트 추가

수정 대상:

- `phase3/test_subtitle_assembler_production_evidence.py`
- `phase3/test_general_subtitle_lifecycle.py`
- 필요 시 `phase4/test_remote_overlap_pipeline.py`

추가할 연속 fixture:

- window 4–7
- window 10–12
- window 12–15

검증 항목:

- PARTIAL에서 올바른 prefix를 잘못 덮어쓰지 않음
- 같은 overlap 구절을 두 번 연결하지 않음
- 한 reference sentence가 불필요하게 두 FINAL로 분리되지 않음
- duplicate-only window 20 동작 유지
- finalized session 재등장·재수정 금지 유지
- secondary non-primary RAW 억제 유지

### 우선순위 5 — STT 정확도 별도 개선

후보 대상:

- `phase4/benchmark_stt_models.py`
- 결과에 따라 `phase4/gpu_processing_server.py` 또는 `phase3/stt_context.py`

계획:

- 기존 벤치마크에 이미 이번 정답 대본과 `기온 변화`, `현재까지 큰 교통 혼잡` 등의 key expression이 정의되어 있습니다: [benchmark_stt_models.py:39](/workspace/Speech-Caption-AI/phase4/benchmark_stt_models.py:39).
- 원본 또는 동일 조건의 분리 음성이 있어야 모델·옵션별 비교가 가능합니다. 현재 JSON만으로 재전사는 불가능합니다.
- benchmark 결과 없이 모델 또는 decoding 옵션을 바로 변경해서는 안 됩니다.
- 향후 로그에는 실제 모델명과 `avg_logprob`, `no_speech_prob`, `compression_ratio`를 남기는 것이 필요합니다.

Assembler만으로 복원할 수 없는 대상: `기온→기운` 등 어느 RAW에도 정답이 존재하지 않는 오인식.

---

## 작업 종료 상태

1. 수정한 파일: 없음
2. 실행한 테스트: 없음 — 이번 단계는 코드 수정 금지 및 분석 보고서만 요청됨
3. 실행하지 못한 테스트: 해당 없음. GPU/STT 재검증은 원본 음성과 실행 환경이 필요하며 이번 분석 범위에서 실행하지 않음
4. 남은 문제:
   - 파괴적 fuzzy correction의 acoustic evidence 부족
   - 연속 VAD 발화에서 `supported_tail_revision`이 열리지 않는 문제
   - 충돌하는 adjacent window 앞에서 sentence FINAL이 너무 빨리 발생
   - STT 모델명·confidence가 로그에 없어 STT 원인 세분화 불가

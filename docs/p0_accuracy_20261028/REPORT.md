# P0 실시간 한국어 자막 정확도 개선 및 진단 — 2026-10-28

## 범위와 결론

기준 `main`, `0a2a33a`; 시작 작업 트리는 clean. 사용자 관찰은 **중복 출력 최소 1회**, 정확한 횟수와 원인은 미상이다. 최신 Windows 원본 WAV/timeline은 이 checkout에 없다. 이번 변경은 **재현 가능한 확정 자막 재등장 경로와 Evidence 덮어쓰기 결함**을 수정하고, 기존 진단을 연결한다. 실제 한국어 STT 정확도 개선율 또는 사용자 관찰의 원인 확정을 주장하지 않는다.

서버 중지·재시작·운영 endpoint 호출·GPU 추론·모델 다운로드·commit/push는 수행하지 않았다. 기존 Evidence/ZIP/로그는 보존했다. 테스트는 CPU, mock 모델, 별도 임시 localhost 서버, offscreen Qt를 사용했다. 겹친 음성 기능을 개발하거나 실험하지 않았다. 기존 전체 회귀에 포함된 overlap 보호 테스트만 실행했다.

## 1. 실제 확인한 결함과 수정 전후 근거

### 확정 자막 경계 문맥 소실

`SubtitleAssembler.process()`는 완료 문장이 겹친 창에서 그대로 반복되면 duplicate-only 문장 rollover를 만든다. pipeline은 FINAL을 commit하지만 활성 hypothesis와 `previous`는 비어 있다. 다음 인접 창에 공유 음성이 있고 같은 문장 또는 그 문장+새 내용이 오면, 기존에는 이를 새 utterance로 발행했다.

`test_p0_accuracy_diagnostics.py`의 production 순서 harness에서 수정 전 2건 실패. `before.txt`에 실패 원문, `comparison.json`에 기준 커밋과 수정본의 이벤트를 저장했다. **텍스트와 shared_speech를 주입한 합성 재현**이다. 기본 3/2초 창 세 개가 모두 같은 오디오를 공유한다는 뜻이 아니며, 연속 창의 RAW 반복이 실제 발화 반복인지 환각인지는 원음 없이 판정할 수 없다.

수정은 기존 FINAL 시점을 유지한다. duplicate-only rollover에서 실제 commit된 문장에 포함된 RAW tail만 잠시 보관한다. 활성 hypothesis가 없고 바로 다음 창이며 `shared_speech is True`일 때, 기존 최소 길이/어절 경계를 만족하는 정규화 exact suffix-prefix만 제거한다. match_type은 `finalized_tail_exact`. 최종 문장의 fuzzy 교체·전체 이력 검색은 하지 않는다. 공백/문장부호만 정규화하며 바뀐 단어는 삭제하지 않는다. 무음/reset, 창 번호 누락, 공유 음성 False/미확인은 적용 대상이 아니다.

주의: 공유 음성 증거는 단어별 alignment가 아니다. 자연스러운 반복과 동일 STT 재관측이 텍스트·VAD상 구분되지 않는 경우는 여전히 실제 음성 검증이 필요하다. 모든 중복을 없애는 전역 문자열 필터를 추가하지 않았다.

### Evidence 덮어쓰기

기존 `write_evidence_bundle()`는 같은 경로의 `original.wav`, `timeline.jsonl`을 덮어썼으며, capture의 window WAV도 같은 경로를 재사용했다. 신규 실패 테스트에서 실제로 재현했다.

capture 전 `reserve_evidence_directory()`가 기존 audio/timeline/windows를 거부하고 `.capture-reserved`를 exclusive create한다. 중단된 세션도 재사용하지 않는다. bundle 직접 호출도 기존 파일을 검사하고 exclusive create한다. **새 폴더를 지정해야 한다.** 기존 폴더를 자동 삭제·이동하지 않는다. GUI trace는 UUID 파일명을 쓴다.

## 2. 전체 경로 감사

| 경로/위험 | 코드와 확인 결과 | 잔여 불확실성 |
|---|---|---|
| WASAPI → 16kHz mono | `run_overlap_pipeline.capture_worker`, `audio_diagnostics`: 기본 first channel, 3초 창에 이전 tail 1초+새 2초; 의도된 overlap | 실제 채널/드라이버 discontinuity, 빠른 음절 경계 손상 미측정 |
| 큐 → remote worker | bounded queue, live stale/STOP drop, 단일 동기 HTTP worker, 자동 POST retry 없음 | 지연 시 창 drop으로 음성 누락 가능; 이번 운영 큐 수치는 없음 |
| 요청/응답 | `remote_gpu_client.parse_result`: request_id/window_index 일치, finite waveform, VAD-RAW 계약 확인 | 네트워크/프록시 실제 지연 미측정 |
| GPU single | `gpu_processing_server._process_serialized`: 원본 slot0, inactive slot1, separator 미사용 | 실행 중인 서버의 현재 코드 재검증은 하지 않음 |
| VAD → RAW | `validate_end_to_end_gpu.detect_speech_activity`: 최소 200ms, 음성 양성일 때 전체 창 STT; VAD timestamps로 오디오를 자르지 않음 | 짧은 발화 탈락, 잡음 VAD 오탐, 일부 무음 포함 창의 환각 가능 |
| STT | 지속 worker/model 재사용, warm-up, ko/beam5/previous-text false | RAW 자체 반복·한국어 오인식은 정답 corpus 필요 |
| admission → assembler | 기존 source evidence와 경계 조립 재사용, 이번 finalized exact seam만 추가 | D1/D2 유사어 append 및 M1 prefix 손실은 과거 자료로 기계적 재현되지만 원음 정답 미확인 |
| state → PARTIAL/FINAL | published text는 source-supported text. W10 어미·confirmed prefix·fuzzy 안전장치·sentence rollover 유지 | unsupported gap 뒤 supported suffix 비공개 등 기존 XFAIL 유지 |
| callback → Qt → GUI | 같은 utterance PARTIAL 교체, sequence/event/finalized identity 중복 방어; stale worker 차단 | 사용자 관찰의 GUI 이벤트 이력 없음 |
| GUI → overlay | 같은 model snapshot 전달; 최신 FINAL 최대 2개+PARTIAL, FINAL 도착 후 4초 만료, 화면 높이에 따른 row 제한 | 실제 paint/Windows compositor timing은 미측정 |

과거 자료: `docs/p0_3_analysis_report.md`, `experiments/r2_b6/results/*_repeat_*.json`, 뉴스 fixture 두 파일. **과거 수치를 이번 실측으로 재사용하지 않았다.** 저장 RAW 12회/189 windows 회귀는 기존 문구와 FINAL 경계가 유지됨을 검사한다. W13 시제나 `취약점은 최악점은`, `인터넷 전문 인터넷 전반에`, `사용된` 제거는 추정 교정하지 않았다.

## 3. 진단 재사용과 보강

기존 `--evidence-dir`, `--latency-diagnostics`, 원본 FLOAT WAV, 창 WAV, `timeline.jsonl`, 결과 JSON의 assembly/state events, STT confidence, queue/drop/failure 정보, presentation 로그를 재사용했다. 신규 상세 JSONL은 **`--evidence-dir`이 있을 때만 활성화**된다. 기존 기본 `voxlux.log` 동작은 유지한다.

- CLI: 지정 폴더의 `diagnostic.jsonl`.
- GUI: 지정 폴더의 `gui_trace_<UUID>.jsonl`. GUI가 연 writer를 pipeline도 공유하므로, GUI 실행에서는 별도 `diagnostic.jsonl`이 없는 것이 정상이다. 앱 종료까지 FINAL 만료를 기록한다.
- 모든 trace 행: `schema_version`, `wall_time_ns`, `monotonic_ns`, `stage`.
- `session`: session_id와 창/stride/speaker mode. `server_configuration`: 기존 health 응답의 profile/model/ko·beam·previous-text 옵션만 선별 기록.
- `request_sent` → `response_received` → `raw_stt`: session/window/request identity, 오디오 구간, 전송 크기, RAW slot 텍스트와 STT confidence, numeric 서버/클라이언트 timing.
- `request_error`, `window_error`: timeout 등 오류 **유형**. URL, HTTP headers, 예외 메시지는 trace에 전달하지 않는다.
- `window_result`: 원 RAW/admitted text, VAD, assembler 입력/출력/match_type/duplicate_only/review_reasons/audio_overlap_evidence, state support 판단, 큐 깊이와 지연.
- `publication_decision` → `subtitle_dispatch`: 공개/보류/철회 이유, PARTIAL/FINAL, 공개 텍스트, `event_id=session_id:sequence`, WebSocket enqueue 결과.
- `presentation_decision`: accept/reject/expire 사유와 visible entries. `gui_model_delivery`, `overlay_delivery`: 실제 Qt 모델 및 overlay label에 전달한 entries. overlay visible 상태와 source row 수도 기록.
- `window_drop`: stage/reason/age/queue/timing을 `drop` 객체에 보존. `session_end`: 처리·drop 수와 오류 유형.

조인: `(session_id, window/window_id/window_index)`로 요청/RAW/조립을 묶고, `(session_id, sequence)`로 dispatch/GUI/overlay를 묶는다. GUI entries의 speaker_id는 pipeline speaker와 같다. 텍스트만 같다는 이유로 다른 utterance를 같은 이벤트로 계산하지 않는다. 정상 PARTIAL→FINAL의 같은 문구도 중복 오류로 세지 않는다.

서버와 클라이언트의 monotonic 절대값을 서로 빼면 안 된다. 같은 프로세스의 stage 차이 또는 이미 계산된 duration을 사용한다. overlay_delivery는 label 전달 시각이며 실제 화면 paint 시각이 아니다.

JSONL은 행마다 flush하고 logging handler lock으로 thread 간 write를 직렬화한다. 쓰기/직렬화 오류 발생 시 해당 trace를 비활성화하고 경고하여 자막 처리를 유지한다. 큰 오디오 데이터와 인증 설정은 trace에 넣지 않는다. 전사문 자체는 기록되므로 세션 내용은 포함한다. 원본 연속 WAV/timeline은 기존처럼 정상 종료 시 완성되며, 강제 종료 시에는 flush된 trace와 이미 저장된 window WAV까지만 남을 수 있다. 동기 디스크 쓰기이므로 느린 Windows 저장장치의 비용은 후속 측정 대상이다.

## 4. 개선 후보 평가

설치된 faster-whisper **1.2.1** signature를 직접 확인하고 `decoder_options.json`에 저장했다. production ko/beam5/condition_on_previous_text=false는 유지한다. 미지정 temperature는 0 고정이 아니라 `[0,.2,.4,.6,.8,1]` fallback이다. no-speech=.6, logprob=-1, compression=2.4, word timestamps/내부 VAD off. 공식 구현의 hallucination silence 옵션은 word timestamp 처리와 연관되므로 임계값 하나만 켜는 변경은 하지 않았다. [공식 구현](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py), [공식 README](https://github.com/SYSTRAN/faster-whisper/blob/master/README.md).

| 후보 | 정확도 가능성 | 지연·GPU/CPU 비용 | 위험·검증 | 결정 |
|---|---|---|---|---|
| 확정 직후 exact seam 유지 | 재등장 방지, 합성 2건 확인 | GPU 0, 작은 CPU 경계 매칭 | 자연 반복 구분 한계; 보호 조건/회귀 검사 | 적용 |
| temperature=0, best_of/beam 조정 | 반복/오인식 감소 가능성 미확인 | fallback/탐색 비용 변화 | 어려운 발화 복구를 잃을 수 있음; 동일 WAV A/B 필요 | 보류 |
| no-speech/logprob/compression 강화 | 환각 감소 가능 | 기존 decode 결과 필터 비용 작음 | 짧고 조용한 실제 발화 삭제 위험 | 보류 |
| 단어 timestamp+무음 hallucination 억제 | 경계 정렬/환각 진단 가능 | 추가 alignment 연산·지연 | 현재 event/lifecycle 전면 변경 위험, corpus 필요 | 보류 |
| VAD 200→100ms / 내부 VAD / crop | 짧은 어미 복원 가능 | 호출 수 증가 또는 입력 축소 | 잡음 양성/음절 잘림, 기존 실험도 이득 불명확 | 보류 |
| channel mean/normalization/resampling 변경 | 특정 장치의 누락 개선 가능 | CPU 증가, GPU 변화 작음 | 위상 상쇄·잡음 증폭, 장치별 증거 없음 | 보류 |
| window/stride 확대·축소, prompt | 문맥 증가 가능 | 수집 지연 또는 요청/GPU 부하 증가 | 중복/지연·prompt 오염, 과거 실험 재구현 불필요 | 기존 옵션 유지 |
| 유사어 seam/prefix 보호 완화 | D1/D2/M1 개선 가능 | CPU 매칭 증가 | 정상 prefix 훼손 및 다른 말을 병합할 위험 | W10/confirmed/fuzzy 보호 유지 |
| model reload/queue 최적화 | 지연 개선 가능 | 이미 지속 모델·warm-up·bounded queue | 이번 증거로 추가 병목 확정 불가 | 기존 구현 유지 |

## 5. 측정 비교

`compare.py`는 git show로 기준 코드를 메모리에 로드하며 checkout을 바꾸지 않는다. CER은 한글/영문/숫자로 정규화한 Levenshtein, WER은 공백 토큰 기준이다. **구성한 텍스트 정답이며 실제 음성 정확도 지표가 아니다.**

| 합성 입력 | 추가 문장 복사 전→후 | CER 전→후 | WER 전→후 |
|---|---:|---:|---:|
| 같은 완료 문장 3번 관측 | 1→0 | 100%→0% | 100%→0% |
| 완료 문장 반복 후 동일 head+새 문장 | 1→0 | 55%→0% | 50%→0% |

새 문장 내용은 모두 보존된다. 공유 음성 없음/미확인, 창 gap, 바뀐 단어를 보존하는 대조 테스트도 통과한다. 합성 2시나리오 전체 처리의 CPU 평균은 기준 **0.466ms**, 수정 **0.459ms**; p95 **0.748→0.755ms** (warm-up 20회 제외, 각각 200회). 속도 향상의 근거로 해석하지 않는다. 작은 trace 행 1,000개 flush의 평균은 **0.0136ms/행**이며 실제 전체 window JSON이나 Windows 디스크 비용을 대변하지 않는다.

실제 발화 CER/WER, 누락/유령 자막 횟수, live STT/request/화면 지연은 **미측정(null)**. GPU/모델 decode는 변경하지 않았다.

## 6. 테스트와 실행 환경

운영 Python 설치를 변경하지 않고 `/tmp/voxlux-p0-gui-deps`에 PySide6 6.8.3/websockets 16.1.1, 별도 `/tmp/voxlux-p0-lint`에 ruff를 설치했다. soundcard import는 기존 headless runner에서 mock했다. 실제 캡처를 흉내 낸 결과를 WASAPI 성공으로 보고하지 않는다.

| 검증 | 결과 | 로그 |
|---|---|---|
| 신규 실패 재현 | 5 tests 중 기존 코드 3 FAIL, 2 PASS | `before.txt` |
| 관련 lifecycle/production evidence | 63 PASS | `targeted.txt` |
| Phase 3 전체 unittest | 196 실행: **193 PASS, 기존 XFAIL 3** | `phase3_verified.txt` |
| Phase 4 전체 unittest | **88 PASS** | `phase4_complete.txt` |
| Phase 5 offscreen PySide6 | 77 실행: **72 PASS, SKIP 5** | `phase5_verified.txt` |
| 기존 main 방식 회귀 | **9 scripts PASS** | `scripts.txt` |
| 최종 정리 경로·pipeline·진단 재검증 | **42 PASS** | `final_pipeline_checks.txt` |
| syntax/import, ruff E9/F63/F7/F82, diff whitespace | PASS | 실행 기록/`lint_verified.txt` |

초기 Phase 4 실패(의존성 없음)는 `phase4.txt`, 진단 인자 충돌로 발생한 중간 실패는 `phase4_final.txt`에 보존했다. 충돌은 수정 후 `phase4_verified.txt`, `phase4_complete.txt`에서 해결했다. 기존 기대값을 완화하거나 테스트를 삭제하지 않았다. 기존 Phase 4 Evidence 테스트에는 trace 상관관계 assertion만 추가했다.

SKIP 5: 원본 `phase3/output/evidence/news_20261022_160827/timeline.jsonl`이 필요한 recorded GUI tests. 기존 runner가 해당 선행 자료 부재를 명시한다. 그 외 Qt 모델/label/타이머/레이아웃 회귀는 실제 PySide6 offscreen에서 실행했다.

GPU 재추론/운영 서버 부하 테스트는 **SKIP**: 실행 중 서버에 영향을 주지 않기 위해서다. Windows WASAPI/실제 화면/EXE 패키징은 **SKIP**: Linux 환경이며 EXE 변경 작업이 아니다.

## 7. 남아 있는 문제

- 최신 사용자의 중복 원인과 횟수 미확정. 새 수정이 해당 중복을 해결했다고 단정하지 않는다.
- 기존 XFAIL 3: noise above bound, projection artifact, unsupported gap 이후 genuine suffix 비공개. 해결되지 않았으며 PASS로 계산하지 않는다.
- W13 시제, 과거 D1/D2/M1의 실제 발화 정답, RAW 반복/환각은 원음·사람 검수 정답 필요.
- 실제 반복 발화와 인접 RAW 반복의 모호함, 빠른 어미·조사 누락, 200ms 미만 발화, drop 구간 누락이 남을 수 있다.
- trace는 payload/label 전달 증거이며 실제 픽셀 중복 또는 compositor 문제까지 자동 판정하지 않는다.
- 실시간 정확도 목표 전체가 달성되었다고 보고하지 않는다. 이번 코드 변경 검증과 실제 Windows 품질 검증을 구분한다.

## 8. Windows 다음 통합 테스트

서버는 현재 상태를 유지한다. 수정된 **Windows client 코드**로 아래 명령을 실행한다. GPU_SERVER_URL은 기존 환경 설정을 사용하고 터미널에 값을 출력하지 않는다.

```powershell
$run = "phase3/output/evidence/p0_" + (Get-Date -Format "yyyyMMdd_HHmmss")
python -m phase5.gui_app --window-seconds 3 --stride-seconds 2 `
  --evidence-dir $run --latency-diagnostics
```

1. GUI single 모드와 준비 상태를 확인하고 Start. 사용했던 한국어 대본을 같은 음량/출력 장치로 재생한다. 정답 대본과 실제 원음이 일치하는지 사람이 확인한다.
2. 완료 문장 뒤 이어지는 새 문장, 같은 문장을 의도적으로 두 번 말하기, 빠른 조사/어미, 숫자/퍼센트, 5초 이상 무음, 배경 잡음 구간을 포함한다. 각 재생 구간과 실제 관찰 시각을 별도 메모한다.
3. 마지막 발화 뒤 최소 5초 무음을 재생해 자연 FINAL/만료를 관찰한 다음 Stop. 앱 종료까지 기다려 trace를 닫는다. 강제 종료하지 않는다. 재실행은 새 `$run` 폴더를 쓴다.
4. `$run/original.wav`, `windows/`, `timeline.jsonl`, `gui_trace_*.jsonl`을 함께 보관한다. 기존 GUI 요약 `phase5/output/gui_last_run.json`과 append 로그 `phase5/output/voxlux.log`는 기존 동작이므로 비교가 필요하면 새 이름으로 복사한다.
5. 중복 관찰 시 session/window로 RAW부터 확인한다. RAW 반복인지, assembler append인지, publication event 재등장인지, presentation reject 후에도 overlay label이 중복인지 순서대로 좁힌다. 화면 기록은 event_id/sequence와 함께 대조한다.
6. original.wav에 맞춘 사람 검수 정답으로 **FINAL transcript** CER/WER를 산출한다. 모든 PARTIAL 문자열을 이어 붙여 채점하지 않는다. 의도적 반복, 누락, 무음 유령 자막, lifecycle 오류는 각각 별도 집계한다.
7. 요청/서버/조립/dispatch→GUI/overlay 지연 분포와 drop/timeout을 함께 보고한다. 구간/표본 수가 다른 값 또는 서버·클라이언트 절대 monotonic 시각을 섞지 않는다. 원음 없이 의미상 환각 횟수를 자동 추정하지 않는다.

CLI로만 원인을 조사하려면 같은 새 폴더에 `phase3/run_overlap_pipeline.py --live --duration 75 --processing-mode remote --speaker-mode single --assemble --evidence-dir <새 경로> --latency-diagnostics --json-output <새 summary.json>`을 사용한다. GUI/overlay 진단이 필요한 이번 재시험에는 위 GUI 실행을 우선한다.

## 9. 변경 파일과 생성물

운영 코드:
- `phase3/subtitle_assembler.py`: commit된 duplicate-only tail의 bounded exact 중첩 방지.
- `phase3/run_overlap_pipeline.py`: Evidence 예약/보존, 기존 timeline timing 보강, event_id, 단계 trace.
- `phase4/caption_diagnostics.py` (신규): opt-in exclusive JSONL logging과 실패 격리.
- `phase4/remote_gpu_client.py`: 요청/응답/RAW/오류 상관관계. 전송 프로토콜·retry 정책 변경 없음.
- `phase5/live_caption_controller.py`, `phase5/gui_app.py`, `phase5/subtitle_presentation.py`, `phase5/subtitle_overlay.py`: opt-in GUI trace 소유권 및 전달/거부/만료 기록. 디자인 변경 없음.

테스트:
- `phase3/test_p0_accuracy_diagnostics.py` (신규)
- `phase4/test_caption_diagnostics.py` (신규)
- `phase4/test_remote_overlap_pipeline.py` (assertion 추가)
- `phase5/test_caption_trace.py` (신규)

생성 Evidence/보고서: `docs/p0_accuracy_20261028/` 아래 이 보고서, `before.txt`, 회귀 로그, `compare.py`, `comparison.json`, `comparison.txt`, `decoder_options.json`, lint/의존성 로그. 이 경로에는 **새 Windows/GPU 실측 WAV나 live CER 결과가 없다.** 기존 자료와 구분한다.

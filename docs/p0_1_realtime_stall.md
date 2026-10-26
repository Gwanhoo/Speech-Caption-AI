# VOXLUX P0-1 실시간 처리 정체 조사

기준: `main`, `9ce4c61`. 시작 전 `git status --short`는 빈 출력이었다. 기존 미커밋 변경 없음. commit/push 및 유료 자원 시작 없음. 운영 코드 변경은 파이프라인과 서버 2개 파일이며 모델·음성 분리·GUI 디자인·자막 정확도 로직은 변경하지 않았다.

**판정: 재현된 결함을 수정하고 해당 자동 테스트를 통과했다. 과거 Windows 정체의 원인 확정 및 실제 실시간 해결 판정은 보류한다. 전체 GUI 회귀는 원본 fixture 부재로 통과하지 못했다.**

## 1. 조사한 파일 및 처리 경로

- `phase5/gui_app.py`, `live_caption_controller.py`: GUI → `QThread`의 preflight/파이프라인 실행 → Qt signal → 자막 모델/오버레이. GUI thread에서 HTTP/STT를 수행하지 않는다. 중지 후 thread 종료 전 재시작을 거부하며, 이전 worker의 subtitle signal을 GUI thread에서도 검사한다.
- `phase3/run_overlap_pipeline.py`: capture_worker → audio_queue → separation_worker(원격 모드에서는 HTTP/VAD/STT 결과 요청) → separated_queue → stt_worker(원격 모드에서는 자막 후처리) → runtime hook/WebSocket. 이름에 separation이 있어도 서버 `single` 모드에서는 분리 모델을 사용하지 않는다.
- `phase3/run_realtime_pipeline.py`: 기존 `enqueue_or_drop`, `put_stop`, `data_backlog` 확인. 공용 파일 자체는 수정하지 않았다.
- `phase4/remote_gpu_client.py`, `remote_gpu_protocol.py`: WAV encode, HTTP, 응답 decode/parse, 단조 시각 및 connection ID 계측. 클라이언트 운영 코드 변경 없음.
- `phase4/gpu_processing_server.py`: `ThreadingHTTPServer` → WAV decode → 단일 `gpu-inference` executor → VAD/STT → 응답. Faster-Whisper의 지연 평가를 위해 generator 소비 시간을 포함한 기존 STT 계측을 사용한다.
- Phase 3/4/5 기존 테스트, `phase4/output/runpod_a5000_*`, `remote_gpu_integration.json`, `PROJECT_STATUS.md`, `docs`, `experiments` 및 자막 fixture 조사.

정상 생성 주기는 첫 3초, 이후 2초마다 3초 sliding window(1초 겹침). 클라이언트 HTTP는 작업자 하나가 직렬 실행한다. audio_queue=3, 결과 queue=2이며 포화 시 신규 윈도우를 버린다. 캡처의 `put_nowait`는 요청 완료를 기다리지 않는다.

HTTP 기본 connect/read timeout은 5/30초, `requests.Session`, pool size 1, `pool_block=True`, 재시도 0, HTTP/1.1 keep-alive이다. read timeout은 전체 요청의 절대 마감시간이 아니라 읽기 대기 timeout이다. Qt stop은 cooperative event이며 진행 중 HTTP나 WASAPI read를 강제로 취소하지 않는다.

## 2. 확인된 사실과 추정

**이 checkout에서 34개 드롭·74.5초 최대 지연의 원본 로그를 찾지 못했다. 26개 처리·0드롭·평균 약 0.96초의 원본 세션 로그도 식별하지 못했다.** 저장된 A5000 HTTP 검증 및 과거 자막 fixture는 동일 장애 세션임을 입증하지 않는다. 과거 일부 로그는 분리 모델을 쓰던 구성으로 현행 `single` 성능의 기준선이 아니다.

확인된 사실:

- 평균 요청/후처리 속도가 2초 stride보다 느리면 bounded FIFO가 차고 신규 오디오가 드롭된다. 개수 상한만으로 윈도우의 나이를 제한하지 못한다.
- 실제 localhost HTTP 지연 테스트에서 RTT가 증가해도 응답의 STT 시간은 별도로 유지된다. GPU 시간만으로 전체 지연을 설명할 수 없다.
- 기존 GUI는 network/STT와 분리돼 있고 오래된 subtitle signal도 거부한다. GUI event loop 차단이 과거 장애 원인이라는 증거는 없다.
- 연결 재사용은 이미 구현돼 있다. 기존 keep-alive/재연결 테스트도 통과한다. 실제 RunPod proxy의 연결 재사용 실패 여부는 해당 Windows 로그가 필요하다.

미확정: 과거 세션의 GPU 속도, proxy/network 지연, 동시 접속 수, timeout 빈도, stdout/디스크 지연 및 실제 Windows 표시 지연. 이번 결과를 근거로 GPU가 느리다거나 과거 장애가 해결됐다고 결론내리지 않는다.

## 3. 병목·결함과 근거

| 확인 항목 | 근거 및 판단 |
|---|---|
| 큐 계측 경쟁 조건 | enqueue 후 timestamp를 기록했다. 생산자의 반환을 80ms 지연한 재현에서 소비자가 먼저 실행돼 `KeyError: audio_queue_enqueued_perf_counter`, STT 결과 0개로 실패. **확정 결함**, 진단 옵션 사용 시 발생 가능 |
| 중지 후 대기 요청 처리 | stop은 capture만 멈추고 HTTP worker는 FIFO를 계속 소진했다. 재현 시 요청 ID `[0,1,2,3,5]`까지 전송. **확정 결함** |
| 종료 표식 손실 | 공용 `put_stop`은 가득 찬 큐에 10초만 기다린다. HTTP는 30초 읽기 timeout 안에 있을 수 있다. 표식 삽입 실패 후 worker의 `get()`/join이 무기한 대기할 수 있다. full queue 종료 테스트로 새 전달 방식 검증 |
| 오래된 윈도우 처리 | age 검사 없이 FIFO 보존. 캡처 완료를 7초 전으로 주입하면 수정 전 HTTP 실행, 수정 후 요청 전 폐기. 진행 중 응답이 늦어진 경우도 상태 갱신 전에 폐기 |
| 서버 무제한 대기 | 단일 ThreadPoolExecutor에 admission limit 없이 submit. 동시 client 또는 timeout 후 계속 들어오는 요청은 서버에 남을 수 있었다. 수정 후 지연 추론 1개+대기 3개를 유지하고 초과 요청을 즉시 거절하는 테스트 통과 |
| 실패 뒤 큐 정리 | 소비자 실패 시 producer를 중지하고 남은 항목을 드롭 사유와 함께 정리. worker가 모두 종료되는 테스트 통과 |

이는 소스와 합성 재현에서 확인한 결함이다. **34/74.5 장애와의 인과관계는 아직 입증되지 않았다.**

## 4. 수정 파일과 변경 내용

- `phase3/run_overlap_pipeline.py`: 큐 lock 안에서 삽입 시각/깊이를 확정한 뒤 항목을 공개한다. 종료 시 consumer 종료 여부를 확인하며 STOP을 전달한다. runtime stop 이후 대기/뒤늦은 응답을 버린다. 원격 live에서 `audio_queue_maxsize × stride`(기본 **6초**, 캡처 완료 이후)보다 오래된 항목은 요청 전/응답 후/결과 queue에서 폐기한다. offline reference 재생의 age 정책은 바꾸지 않았다. worker 실패 시 남은 큐를 정리한다.
- 같은 파일: 기존 `dropped`와 실패 종료 판정을 유지하고 `drop_details`, `remote_failures`, `remote_live_max_window_age_seconds`, `cancelled`를 추가했다. 순수 사용자 중지는 `CANCELLED`/exit 0, `success=false`로 구분한다. stop 이전 포화·만료·오류는 정상 중지로 숨기지 않는다.
- `phase4/gpu_processing_server.py`: active+pending 합계 **4개**까지만 admission. 초과는 기존 error envelope의 HTTP 503 `server_busy`, `retryable=false`. health에 rejected/max_inflight를 추가한다. GPU 추론은 여전히 단일 스레드. submit/추론 실패에서도 admission slot과 waiting 통계가 복구된다. 응답 socket write 전에 서버 처리 계측을 출력해 timeout으로 client가 사라져도 증거가 남는다.
- `phase4/test_remote_overlap_pipeline.py`, `test_remote_gpu_client.py`, `test_gpu_processing_server.py`: 정상 처리, 진단 경쟁, 지연, full queue, timeout/실패, stop/restart, 늦은 응답, 서버 포화 및 worker 실패 자동 검증 추가. 기존 수동 server fixture에 신규 semaphore 초기화 추가. 기존 검증 삭제/완화 없음.
- 이 문서와 `docs/p0_1_validation/`: 실행 명령/시간/실패 이력, 합성 비교 원자료 및 재현 스크립트.

**손실/지연의 교환:** 6초보다 오래된 음성은 자막이 빠질 수 있다. 6초는 GPU 측정값이나 사용자 SLA가 아니라 기존 3개 queue와 2초 stride에서 정한 freshness 예산이다. 진행 중 HTTP를 취소하거나 응답을 6초 내 보장하지 않는다. 높은 지연에서 자막이 없을 수 있으나 오래된 결과를 최신처럼 표시하지 않고 모든 폐기 ID/단계/사유/나이를 기록한다. 큐 포화 정책은 여전히 drop-newest이므로 stall 도중 최신 오디오 일부도 잃을 수 있다.

계측 위치 (`--latency-diagnostics`):

| 구간 | 기록 |
|---|---|
| window/capture | `windows[].window`, `capture_timestamp`, client_pipeline의 `capture_*_perf_counter` |
| queue 입출 | `audio_queue_*`, `separated_queue_*`, `*_queue_wait_seconds`, queue depth/backlog |
| HTTP/자막 결과 수신 | remote_client의 `http_request_started_perf_counter`, `http_response_received_perf_counter`, `response_parse_ended_perf_counter`; client_pipeline의 `remote_client_returned_perf_counter` |
| 서버 수신/대기/추론 | server의 `server_request_received_perf_counter`, `server_queue_entered_perf_counter`, `server_inference_started_perf_counter`, `server_processing_ended_perf_counter`, 화자별 `stt_*_perf_counter` |
| 구간 지연 | HTTP RTT, server queue wait, processing_seconds, stt_seconds, `capture_to_result_seconds`, `capture_to_first_subtitle_seconds` |
| 손실/오류 | `[WINDOW DROP]`/`drop_details`, `[REMOTE FAILURE]`/`remote_failures`, 기존 `[DROP]`/`dropped`/`errors` |

모든 duration은 해당 호스트의 `perf_counter` 차이이다. **서버와 PC의 절대 단조 시각을 서로 빼지 않는다.** RTT와 서버 처리 duration의 잔차는 upload/download/proxy/serialization 등을 포함하므로 순수 network 시간으로 단정하지 않는다. capture-to-result는 client 결과 생성까지이며 실제 overlay repaint 완료까지 측정한 값은 아니다. 드롭/실패에서는 실행되지 않은 단계의 시각이 없는 것이 정상이다.

## 5. 자동화 테스트 결과

환경: Linux/Python 3.12.3, Qt offscreen, 합성 capture/model 및 localhost HTTP. WASAPI 및 실제 GPU 모델은 실행하지 않았다. [전체 명령과 이력](p0_1_validation/test_results.json), [직접 실행 스크립트 로그](p0_1_validation/script_regressions.txt).

| 검사 | 결과 | 시간 |
|---|---|---:|
| 변경 전 Phase 4 전체 | 67 PASS | 21.558s |
| 추가 재현 2개, 변경 전 | 2 FAIL: 계측 KeyError 및 stop 후 추가 요청 | 1.839s |
| 최종 Phase 4 전체 | **78 PASS** | 25.828s |
| Phase 3 `test_subtitle*.py` unittest | **62 PASS** | 0.601s |
| Phase 3 single/연속 자막/입력 VAD 근거 회귀 unittest | **45 PASS** | 3.921s |
| 직접 실행 Assembler/state/consensus/integration/P1/silence gate | **6 scripts PASS** | 각 0.054–1.891s |
| 실제 localhost HTTP delay/timeout/recovery 단독 | **1 PASS**, 최종 78개에도 포함 | 0.689s |
| Qt offscreen 전체 | **FAIL: 67 PASS, 5 ERROR** | 3.787s |
| 변경한 Python 5개 `py_compile`, `git diff --check` | PASS | 명령 종료 코드 0 |
| Ruff | SKIP: 미설치 | — |

GUI 5 ERROR는 모두 기존 `phase3/output/evidence/news_20261022_160827/timeline.jsonl` 부재이다. 이를 만들어내거나 테스트를 완화하지 않았다. 기록 기반 final expiry/order/geometry 5개는 **원본 확보 전 미검증**이며 PASS로 보고하지 않는다. 이번 요청과 관련된 기존 Qt stop/restart, queued old-worker event rejection 검사는 통과했다.

중간 실패도 보존했다. 새 server semaphore를 수동 생성 fixture에 초기화하지 않아 1 ERROR가 발생했고 준비 코드를 보완한 뒤 통과했다. 중간 76개 실행에서는 기존 `test_new_stream_admission_in_real_workers_with_permutation` residual subcase가 한 번 exit-code assertion으로 실패했다. 단독 재실행 및 이후 전체 78개 실행 2회는 통과했다. 당시 상세 queue 상태는 보존되지 않아 정확한 원인은 확정하지 못했다. 40ms 가속 synthetic capture의 부하 민감성은 가능성이며 확인된 원인으로 취급하지 않는다.

재실행 명령(저장소 루트):

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=phase3:phase4:. python -m unittest discover -s phase4 -p 'test_*.py'
PYTHONPATH=phase3:phase4:. python -m unittest discover -s phase3 -p 'test_subtitle*.py'
PYTHONPATH=phase3:phase4:. python -m unittest test_single_speaker_mode test_general_subtitle_lifecycle test_live_caption_continuity test_live_evidence_regressions test_input_residual_speech_admission
QT_QPA_PLATFORM=offscreen PYTHONPATH=phase3:phase4:. python -m unittest discover -s phase5 -p 'test_*.py'
python phase3/test_subtitle_assembler.py
python phase3/test_subtitle_state.py
python phase3/test_subtitle_consensus.py
python phase3/test_subtitle_assembler_state_integration.py
python phase3/test_subtitle_contract_p1.py
python phase3/test_pre_separation_silence_gate.py
```

## 6. 수정 전후 비교

[원자료](p0_1_validation/synthetic_comparison.json), [비교 스크립트](p0_1_validation/compare_synthetic.py). 기준 코드 `9ce4c61`을 `/tmp`에 읽어 각각 3회 실행했다. 캡처 read당 40ms, 모델/장치는 대역, 실제 GPU/network 성능 비교가 아니다.

| 동일 합성 조건 | 수정 전 | 수정 후 |
|---|---|---|
| 정상 2윈도우 | 매회 요청 2, 드롭 0 | 매회 요청 2, 드롭 0 |
| 진행 중 stop, 응답당 200ms | 요청 5, run 전체 중앙값 1.200s | 요청 1, 0.320s |
| 캡처 나이 7초 주입 | 오래된 window 0도 HTTP 전송 | window 0 요청 전 폐기, 다음 window 처리 |
| 첫 응답 500ms, 8윈도우 포화 | 처리 4, 드롭 4, 중앙값 0.805s | 처리 4, 드롭 4, 0.812s |

stop 시 드롭은 1→6개로 증가했다. 이전에 처리하던 대기/늦은 결과까지 의도적으로 폐기하고 계수한 결과이다. 총 run 시간은 stop 클릭부터의 시간이 아니며 합성 준비·캡처·정리를 포함한다. 정상 경로 시간 차이는 실행 순서/초기화 비용의 영향을 받으므로 처리 속도 개선으로 해석하지 않는다. **실제 Windows 전후 평균/p95/최대 지연은 아직 측정하지 않았다.**

```bash
PYTHONPATH=phase3:phase4:. python docs/p0_1_validation/compare_synthetic.py
# /tmp/p0-comparison.json 생성; stdout에 조건별 3회 집계 출력
```

## 7. Windows 필수 검증과 로그 수집

이미 운영 중인 RunPod를 사용하고 새 유료 자원을 시작하지 않는다. 기존 서버를 갱신하는 절차에서 다음 실행 옵션을 유지한다(같은 포트의 서버를 중복 실행하지 않는다).

```bash
cd /workspace/Speech-Caption-AI
mkdir -p phase4/output
python -u phase4/gpu_processing_server.py --host 0.0.0.0 --port 8787 \
  --processing-profile single --whisper-model large-v3-turbo \
  2>&1 | tee phase4/output/p0_1_server.log
```

Windows PowerShell에서 실제 서버 주소를 지정한다. 아래 `python`은 프로젝트 의존성이 설치된 Python이다.

```powershell
Set-Location C:\Users\kimgwanho\Desktop\Speech-Caption-AI
$env:GPU_SERVER_URL = 'https://<현재-Pod-ID>-8787.proxy.runpod.net'
$runId = Get-Date -Format 'yyyyMMdd_HHmmss'
$runDir = "phase5\output\p0_1_$runId"
New-Item -ItemType Directory -Force $runDir | Out-Null
Invoke-RestMethod "$env:GPU_SERVER_URL/healthz" |
  ConvertTo-Json -Depth 20 | Set-Content "$runDir\health_before.json" -Encoding utf8
python -u -m phase5.gui_app --window-seconds 3 --stride-seconds 2 --latency-diagnostics 2>&1 |
  Tee-Object -FilePath "$runDir\client.log"
# GUI 종료 후, 다음 실행으로 덮어쓰기 전에 보존
Copy-Item phase5\output\gui_last_run.json "$runDir\result.json"
Copy-Item phase5\output\voxlux.log "$runDir\voxlux.log"
Invoke-RestMethod "$env:GPU_SERVER_URL/healthz" |
  ConvertTo-Json -Depth 20 | Set-Content "$runDir\health_after.json" -Encoding utf8
```

health에서 `processing_profile=single`, `stt_model=large-v3-turbo`, `mossformer2_loaded=false`를 확인한다. 첫 정상 측정은 WAV 저장 없이 실행한다. 오디오/자막 근거가 필요하면 별도 회차에서 GUI 명령에 `--evidence-dir "$runDir\evidence"`를 추가한다. WAV 저장은 캡처 thread의 디스크 I/O와 RAM 사용을 추가하므로 두 회차 결과를 섞지 않는다.

각 조건은 동일 출력 장치·동일 음원·동일 window/stride로 반복한다.

1. 정상 5분 이상: 시작 직후/steady state를 나누어 요청당 시간, 큐 대기, post_capture 평균/p95/최대, 드롭 ID/사유를 수집. 전체 창 수·오류 수도 함께 비교한다.
2. 일시적 네트워크 지연/연결 끊김: 승인된 테스트 환경에서 유도하고 적용/해제 시각을 client log와 함께 기록. timeout 오류 후 캡처 지속, bounded queue, 복구 및 stale 폐기 확인. 실제 RunPod proxy 구간은 localhost 결과로 대체하지 않는다.
3. GPU 응답 지연/서버 포화: 테스트 서버에서 동시 요청을 통제하여 검증. active 포함 상한 4, 추론 1개 실행 중 대기 최대 3개, 초과 503 및 rejected 증가를 확인한다. 추론 스레드가 첫 작업을 꺼내기 전에는 waiting이 일시적으로 4일 수 있다. 무한 부하 생성은 필요하지 않다.
4. 요청 진행 중 STOP: UI 응답성, stopping→idle, 뒤늦은 자막 없음, 대기 윈도우 HTTP 미전송 확인. read timeout은 총 deadline이 아니므로 종료 시간이 길어지면 마지막 요청의 header/body 단계도 분석한다.
5. 같은 GUI에서 STOP→IDLE→START를 3회 반복: 이전 자막/worker 이벤트가 새 세션에 나타나지 않는지 확인. **매 STOP 후 다음 START 전에 다른 PowerShell에서 `gui_last_run.json`과 evidence 폴더를 별도 이름으로 복사**한다. 기본 경로는 다음 세션이 덮어쓴다.
6. 실제 overlay 위치·투명도·최상위 표시·무음 final 만료를 화면에서 확인. 가능하면 화면 녹화와 원본 음원 시간을 함께 남긴다.

JSON 핵심 집계 예시:

```powershell
$r = Get-Content "$runDir\result.json" -Raw | ConvertFrom-Json
$r.performance.post_capture_latency
$r.dropped.Count
$r.drop_details | Group-Object reason | Select-Object Name,Count
$r.errors
$r.windows | Select-Object window,remote_request_seconds,post_capture_latency_seconds,audio_queue_backlog
```

성공 window의 `latency_diagnostics`에서 client/server 구간을 분석한다. 느린 결과가 폐기됐다면 `drop_details.remote_timing`, `remote_client_timing`과 서버 `[LATENCY SERVER PROCESS]`를 함께 확인한다. 서버 request_id/window와 PC의 회차별 로그를 대응시키되 두 호스트의 perf_counter를 직접 빼지 않는다. 서버 로그·health 전후·client.log·result.json·해당 회차 evidence를 함께 보존한다.

## 8. 남은 위험 및 미해결 사항

- 34드롭/74.5초 원본 부재로 과거 장애의 발생 조건·원인·재발 방지 여부 미확정.
- Windows WASAPI, 실제 데스크톱/오버레이, 실제 RunPod proxy/GPU 지연 비교는 SKIP/미실행. GPU 모델 추론 시간을 합성 결과로 보고하지 않았다.
- 원본 timeline 없는 GUI 기록 회귀 5개는 미검증이며 전체 GUI suite 실패 상태. 원본을 확보해 같은 명령으로 재실행해야 한다.
- 기존 가속 회귀 1회의 일시적 실패 원인은 미확정. 최종 전체 suite는 통과했지만 동일 부하 조건의 재발 시 상세 큐 통계가 필요하다.
- 이미 실행 중인 HTTP/GPU 추론은 취소하지 못한다. client timeout 뒤에도 서버는 해당 추론을 완료할 수 있다. 서버 대기 개수는 제한되지만 개별 추론 시간/HTTP body read의 절대 상한은 새로 구현하지 않았다.
- 6초 freshness 기준은 일부 자막 손실을 증가시킬 수 있다. 스트림 간 gap이 생겨도 기존 상태/Assembler 로직은 유지한다. 실제 문장 연속성은 Windows 검증 대상이다.
- 오래 실행하면 기존 results/evidence 수집 배열과 로그는 커진다. 요청 큐와 별개인 장시간 진단 보관 문제는 이번 변경 범위에 포함하지 않았다.

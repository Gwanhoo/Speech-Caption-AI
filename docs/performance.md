[← README로 돌아가기](../README.md)

# 성능 측정과 검증 근거

## 결과를 해석하는 기준

현재 SINGLE 구현, 과거 GPU 모델 실험, 과거 separation 실험, 현재 CPU 회귀 실행을 분리합니다. 모델 decode 시간은 실제 자막 표시 지연이 아니며, 테스트 통과는 사람 검수 정확도를 의미하지 않습니다. 문서 작성 중 새로운 GPU 인스턴스나 유료 자원을 시작하지 않았습니다.

## 현재 HEAD의 직접 실행

2026-11-10, HEAD `5da8925`, Linux, Python 3.11.17, CPU PyTorch 2.15.0+cpu, PySide6 6.8.3, Qt offscreen에서 실행했습니다. 모의 capture/model 및 localhost HTTP를 사용하며 WASAPI/CUDA는 사용하지 않았습니다.

| 실행 | 결과 | 검증 범위 |
|---|---|---|
| 아래 unittest 명령 | **131 PASS**, FAIL/ERROR/XFAIL/SKIP 0, 21.426초 | 자막 lifecycle·기록 재생, client/server, 원격 worker, 표시 상태, GUI 복구 |
| `phase3/test_subtitle_assembler.py` 직접 실행 | PASS, exit 0 | 실제 assert 기반 경계 정합·반복·교정 회귀 |
| `phase3/test_subtitle_state.py`, `phase3/test_subtitle_consensus.py` 직접 실행 | 각 PASS, exit 0 | 상태 전환·consensus의 script-style assert 회귀 |
| `phase4/test_websocket_mvp.py` 직접 실행 | PASS, exit 0 | 실제 로컬 WebSocket 연결·2개 이벤트·receipt |

위 시간은 test runner 전체 실행 시간이며 앱 latency가 아닙니다. [검증 요약 JSON](portfolio-validation.json)에 실행 명령·환경·범위를 보관했습니다. script-style 테스트는 unittest discovery만으로 실행되지 않으므로 별도 실행했습니다.

```bash
PYTHONPATH=phase3:phase4:. QT_QPA_PLATFORM=offscreen \
XDG_CACHE_HOME=/workspace/.cache OMP_NUM_THREADS=2 \
.venv/bin/python -m unittest -v \
  phase3.test_general_subtitle_lifecycle \
  phase3.test_case_a_recorded_lifecycle \
  phase3.test_subtitle_state \
  phase4.test_remote_gpu_client \
  phase4.test_gpu_processing_server \
  phase4.test_remote_overlap_pipeline \
  phase5.test_subtitle_presentation \
  phase5.test_gui_app.GuiTests.test_remote_failure_and_recovery_update_gui_without_losing_subtitles \
  phase5.test_gui_app.GuiTests.test_stop_and_restart_clear_remote_error_and_reject_stale_worker_state

PYTHONPATH=phase3:phase4:. .venv/bin/python phase3/test_subtitle_assembler.py
PYTHONPATH=phase3:phase4:. .venv/bin/python phase3/test_subtitle_state.py
PYTHONPATH=phase3:phase4:. .venv/bin/python phase3/test_subtitle_consensus.py
.venv/bin/python phase4/test_websocket_mvp.py
```

전체 GUI suite와 모든 Phase 3/4 테스트를 이번에 재실행한 것은 아닙니다. 이전 현재 환경 검사에서 `phase5.test_gui_app`는 원본 timeline 부재 5 ERROR와 긴 자막 line-budget 1 FAIL이 있었습니다. 이 결과는 해결하지 않은 채 보존하며 이번 선택 회귀 PASS와 구분합니다.

## GPU STT 실험: 현재 운영 기본 옵션과 비교

[2026-10-31 보고서](accuracy_20261031/REPORT.md)와 [원자료](accuracy_20261031/gpu_probe.json)의 기존 RTX A5000 실험입니다. `large-v3-turbo`, CUDA float16, `language=ko`, `condition_on_previous_text=False`, Silero VAD 200 ms gate를 사용했습니다. 운영 기본 beam은 5입니다.

A/B WAV 각각 첫 6개 완전한 3초/2초 창(각 0–13초)을 사용했습니다. 24 kHz 원본을 전체 16 kHz로 변환하고 beam별 warmup 후 두 번 반복했습니다. 총 decode 72회, beam별 24회입니다.

| beam | STT 평균 | p50 | p95 | 관측 process VRAM 최대 |
|---|---:|---:|---:|---:|
| 5 | 0.1926초 | 0.1900초 | 0.2175초 | 2,302 MiB |
| 1 | 0.1521초 | 0.1539초 | 0.1740초 | 2,302 MiB |
| 3 | 0.1695초 | 0.1669초 | 0.1882초 | 2,302 MiB |

VRAM은 창 처리 후 snapshot의 최대이며 순간 peak 보장이 아닙니다. 이 실험은 기존 WAV의 GPU 함수 호출로 Windows WASAPI·외부 proxy·실제 GUI 표시를 거치지 않았습니다. 정답 없이 어느 beam이 더 정확한지 판단하지 않으며 운영 beam을 바꾸지 않았습니다.

같은 보고서의 새 문장 정정 분기는 이후 철회됐습니다. GPU decode 측정 기록과 철회된 Assembler 개선 주장은 별개입니다. H1/M1 후속 CPU 재생의 작은 시간 차이는 스케줄링 잡음을 포함하므로 통계적 속도 개선 성과로 제시하지 않습니다.

## 처리 지연·큐 지표의 정의

| 지표 | 측정 구간과 주의점 |
|---|---|
| `stt_seconds` | 모델 segment generator 소비 완료까지; 캡처·HTTP 제외 |
| `queue_wait_seconds` | 서버 admission 후 추론 시작까지 대기 |
| HTTP roundtrip | 클라이언트 요청/응답 구간; 서버 시간 포함 |
| `audio_queue_wait_seconds` | 클라이언트 큐 대기 |
| `post_capture_latency_seconds` | 수집 완료 이후 결과 생성까지의 파이프라인 지연 |
| `initial_subtitle_seconds` | capture 시작 → 최초 비어 있지 않은 공개 자막 생성; 없는 경우 null |
| queue depth/backlog/drop | 용량 사용량, 적체와 손실; 처리 성공 수와 함께 해석 |
| STT confidence diagnostics | logprob/no_speech/compression 집계; 인식 정확도 그 자체가 아님 |

3초 창·2초 stride는 입력 수집 정책입니다. 평균 요청이 stride를 넘으면 bounded queue가 포화될 수 있고, 평균이 작아도 tail latency로 drop이 발생할 수 있습니다. 서버 4개 admission과 기본 6초 age 폐기는 무한 적체 방지 정책이며 SLA가 아닙니다. 서버·클라이언트 절대 monotonic 시각의 직접 차감도 하지 않습니다.

## 정확도 평가: 정답 검수가 선행되어야 함

[24개 검수 데이터셋](../evidence/accuracy_a4_ground_truth_20261101/README.md)과 [review GUI](../evidence/accuracy_a4_ground_truth_20261101/GUI_README.md)는 기존 결과를 사람 원음 검수에 연결합니다. 현재 템플릿은 `pending`이며 CER/WER는 null입니다. 모델 간 문자열 차이나 더 자연스러운 문장을 정답으로 사용하지 않습니다.

평가기는 reference/status/reviewer/reviewed_at을 검사하고 기본적으로 24개 전체 verified를 요구합니다. 일부만 평가하려면 `--allow-partial`이 필요하며 partial 결과임을 표시합니다. 원본 ID·경로·시간·모델 출력 변경은 기준 JSON과 비교해 차단합니다.

- CER: 한글·ASCII 영숫자만 유지하고 Latin을 소문자로 정규화한 edit distance / reference 문자 수.
- WER: 같은 문자 집합의 연속 구간을 어절 토큰으로 사용한 edit distance / reference 토큰 수.
- 기본 집계는 edit 합/reference unit 합의 micro 지표이며 macro도 별도 기록합니다.
- RAW STT 정확도와 FINAL 자막 품질, 의미적 환각과 문장 경계 오류는 분리 평가해야 합니다.

검수 클립은 모델 출력 차이·불안정성이 큰 사례를 우선 선정한 표본입니다. 검수 후 수치가 생겨도 일반 콘텐츠 전체 정확도로 확장할 수 없습니다. 데이터셋 설명이 참조하는 A1~A3 원본 폴더는 이 checkout에 없어 원래 생성 과정 전체 재현도 추가 자료가 필요합니다.

## 과거 결과와 현재 운영 결과의 구분

| 자료 | 활용 가능 범위 | 현재 성과로 사용하지 않는 내용 |
|---|---|---|
| [기존 정확도 audit](../ACCURACY_AUDIT.md) | 당시 모델·분리·조립 비교의 개발 이력 | separation 포함 latency를 SINGLE 운영 수치로 전용 |
| [P0-1](p0_1_realtime_stall.md) | 합성 지연·STOP·포화 결함 재현 | 원본 없는 과거 실시간 장애의 원인 확정 |
| [H1/M1](accuracy_20261031/h1_m1/REPORT.md) | 206 PASS + 기존 XFAIL 3, Phase4 88 PASS의 당시 기록 | 이번 HEAD 전체 suite를 재실행했다고 주장 |
| [Windows overlay](overlay_validation_20261024/README.md) | 당시 native Qt 위젯·높이 회귀 | 현재 전체 화면 합성·작은 화면 전부 보장 |

## 미측정 항목과 다음 실험

현재 운영 Windows → RunPod → Qt repaint 전체 지연의 p50/p95/p99, 장시간 drop·timeout·메모리, 장치별 채널 품질, 검수된 CER/WER, 짧은 발화 누락률, 다양한 잡음에서 환각률은 확정 수치가 없습니다. Windows 원음·RAW/VAD·queue metric·publication·GUI trace를 같은 세션 ID로 수집하고 네트워크/장치 조건을 기록해야 합니다. A/M1을 포함한 문장 경계 검수와 실제 native GUI 회귀도 필요합니다.

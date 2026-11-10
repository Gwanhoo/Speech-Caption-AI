[← README로 돌아가기](../README.md)

# 문제 해결 기록

현재 HEAD `5da8925`의 구현과 저장된 재현 자료를 기준으로 정리합니다. 테스트의 성공, 실제 장치의 관측, 미해결 가설을 구분합니다. 아래 커밋은 변경 근거이며, 커밋 메시지만으로 동작 성공을 판정하지 않았습니다.

## 1. 실시간 큐 적체와 STOP 이후 오래된 요청

### 문제와 관측

원격 추론이 느릴 때 큐 개수만 제한하면 오래된 창이 뒤늦게 처리될 수 있었습니다. 진단 timestamp를 enqueue 후 기록하는 경로는 소비자가 먼저 접근할 때 `KeyError`를 일으켰습니다. STOP 후 대기 요청을 계속 전송하고, 포화 큐에서 종료 표식 전달이 실패해 join이 끝나지 않을 가능성도 확인됐습니다.

[조사 보고서](p0_1_realtime_stall.md)에는 생산자 반환을 80 ms 지연해 계측 경쟁을 재현한 결과와 capture 완료를 7초 전으로 주입한 stale 검사가 있습니다. 과거 사용자가 관측한 장시간 정체의 원본 세션은 확보되지 않아 동일 원인으로 확정하지 않았습니다.

### 원인과 변경

[322884d](https://github.com/Gwanhoo/Speech-Caption-AI/commit/322884d)는 [run_overlap_pipeline.py](../phase3/run_overlap_pipeline.py)의 큐 내부 lock에서 삽입 계측을 확정하고, STOP/consumer 종료 확인과 stale 검사를 추가했습니다. 기본 remote live age 예산은 6초입니다. 서버 [PipelineService](../phase4/gpu_processing_server.py)는 active+pending 4개 admission으로 무제한 executor 대기를 제한합니다.

### 검증과 한계

[worker 테스트](../phase4/test_remote_overlap_pipeline.py)의 full queue shutdown, 오래된 요청·응답 폐기, worker failure, STOP/재시작과 [서버 테스트](../phase4/test_gpu_processing_server.py)의 포화를 검사합니다. 해당 모듈은 이번 131개 회귀 실행에도 포함됐습니다.

합성 재현의 결함 수정은 확인되지만 실제 Windows 정체 전체의 원인은 미확정입니다. age 폐기는 자막 누락을 수반하며, 진행 중 HTTP를 취소하지 않습니다. drop-newest 정책도 남아 있습니다.

## 2. 무음 유령 자막: 프로토콜 모순과 모델 환각의 분리

### 문제와 관측

VAD=false인데 RAW transcript가 있는 응답을 과거 parser가 허용했습니다. 별도로 저장된 GPU 진단에서는 VAD를 우회해 Whisper에 zero/noise를 직접 주면 문장이 생성됐지만, 정상 SINGLE 서비스 함수의 VAD 경로에서는 같은 입력의 STT가 실행되지 않았습니다. [원자료와 조사](p0_2_ghost_caption_report.md)는 이 두 경우를 구분합니다.

### 원인과 변경

[b3f53cf](https://github.com/Gwanhoo/Speech-Caption-AI/commit/b3f53cf)의 [parse_result](../phase4/remote_gpu_client.py)는 VAD=false/nonempty RAW를 프로토콜 오류로 거부합니다. 서버의 `stt_diagnostics`와 세션 UUID·window/request ID가 RAW, 채택 텍스트, 공개 자막까지 이어져 어느 계층에서 텍스트가 생겼는지 추적할 수 있습니다.

### 검증과 한계

[ghost path 테스트](../phase4/test_ghost_caption_path.py)와 client 계약 검증은 모순 응답 거부 및 정상 짧은 응답 보존을 다룹니다. 이번 client/server 회귀도 통과했습니다. 정상 서버가 문제의 모순 응답을 실제 생성했다는 증거는 없습니다. 무음 환각 전체를 해결했다고 주장하지 않습니다. noise support·unsupported gap 등의 기존 expected failure가 남고, confidence 수치도 의미상 정답 보증이 아닙니다.

## 3. 문장 정정 실험의 철회: 정상 내용 보존을 우선

### 문제와 관측

후속 RAW에서 달라진 서술부를 정정하려던 실험은 합성 사례 A에 유효했지만, 동일 주제의 독립적인 두 정상 문장(H1)을 하나로 합치면서 앞 서술부를 삭제했습니다. M1은 confirmed prefix가 정정 seam보다 뒤에 형성되어 기존 보호 규칙으로 정정되지 않았습니다.

### 원인 분석과 처리

VAD 교집합·tail coverage·높은 waveform correlation은 공유 오디오를 나타내지만 각 RAW 단어가 같은 발화에 대응한다는 증거가 아닙니다. [H1/M1 후속 보고서](accuracy_20261031/h1_m1/REPORT.md)는 whole-predicate 정정 분기를 철회하고 기존 Assembler 범위를 유지한 과정을 보존합니다. 현재 `reconcile_adjacent_sentence_head_revision`은 제한된 기존 교정 조건을 사용합니다.

### 검증과 한계

[일반 수명주기 테스트](../phase3/test_general_subtitle_lifecycle.py)는 H1 변형의 두 문장 보존, confirmed prefix·FINAL history 보호, 근거 없는 source 공개 방지를 검사합니다. 현재 실행에서도 통과했습니다. **A/M1은 부분 해결이 아니라 여전히 미해결**이며, 미해결 동작을 보존하는 테스트 통과를 정확도 개선으로 집계하지 않습니다. 철회 전 보고서의 성공 수치와 현재 구현을 혼용하지 않습니다.

## 4. 최초 자막 지연 지표의 의미 수정

### 문제와 관측

첫 창이 무음이면 첫 처리 완료와 첫 자막 생성은 다릅니다. 기존 `completion_times[0]`를 최초 자막 지연으로 보고하면 자막이 없는 세션도 자막이 나온 것처럼 측정할 수 있었습니다.

### 변경과 검증

[b7fc74b](https://github.com/Gwanhoo/Speech-Caption-AI/commit/b7fc74b)는 [publish_subtitle_state_event](../phase3/run_overlap_pipeline.py)에서 처음 비어 있지 않은 공개 텍스트가 생성된 시각을 capture clock 기준으로 기록합니다. 미공개 후보·빈 결과는 제외하고 전체 무음은 `null`/N/A입니다.

[원격 worker 테스트](../phase4/test_remote_overlap_pipeline.py)는 첫 창 발화, 선행 무음, 여러 창 이후 발화, 전체 무음, 공개 전 STOP 및 진단 옵션 off를 검사합니다. 이번 실행에도 포함됐습니다. 이 지표는 Qt repaint·WebSocket 수신 완료가 아니며 화면 표시 지연 개선율을 의미하지 않습니다.

## 5. remote 처리 실패·복구가 GUI 상태에 반영되지 않음

### 문제와 변경

창 단위 오류를 기록하고 다음 창을 처리해도 GUI가 연결 상태를 구분할 수 있어야 합니다. [f0f2b82](https://github.com/Gwanhoo/Speech-Caption-AI/commit/f0f2b82)는 파이프라인의 `on_remote_state` hook, worker 상태 전환, GUI 오류 표시를 연결했습니다. 정상 응답 뒤 복구를 알리며 STOP/이전 worker의 늦은 상태 신호를 거부합니다.

### 검증과 한계

[GUI 테스트](../phase5/test_gui_app.py)의 `test_remote_failure_and_recovery_update_gui_without_losing_subtitles`, `test_stop_and_restart_clear_remote_error_and_reject_stale_worker_state`를 이번에 직접 실행해 통과했습니다. HTTP/protocol/connection failure 뒤 다음 창 처리도 worker 회귀에 포함됩니다. 모의 backend/서버 검증이며 실제 RunPod 장애·복구 실험은 아닙니다. 복구 전 창의 자막 손실은 복원되지 않습니다.

## 6. native Qt 오버레이 높이와 headless 검증의 차이

### 문제와 원인

기존 offscreen suite는 통과했지만 Windows native Qt에서 2행→1행 전환 후 높이가 118 px에 남고 기대 67 px로 줄지 않는 문제가 재현됐습니다. layout 계산과 top-level 창 크기 반영이 같은 결과를 보장하지 않았습니다.

### 변경·검증·한계

[기존 Windows 보고서](overlay_validation_20261024/README.md)는 `SubtitleOverlay._set_window_height`의 `setFixedHeight` 반영, FAIL-FIRST, Windows/offscreen 각각 66 PASS와 widget 렌더링을 보존합니다. 이 수치는 당시 실행 결과입니다. 현재 코드에는 이후 세 줄 제한 변경 [b53b208](https://github.com/Gwanhoo/Speech-Caption-AI/commit/b53b208)도 포함됩니다.

현재 Linux에서 과거 전체 GUI 검사를 실행했을 때 원본 timeline 부재 오류 5개와 긴 자막 line-budget 실패 1개가 있었습니다. 따라서 과거 66 PASS를 최신 GUI 전체의 PASS로 대체하지 않습니다. 투명도 합성·전체 화면 앱 위 z-order·모든 작은 화면의 잘림 방지는 추가 native 검증이 필요합니다.

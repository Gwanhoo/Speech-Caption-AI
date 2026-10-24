# VOXLUX 오버레이 및 자막 표시 수명 검증

2026-10-24 / Windows 로컬 CPU / `main` / `95e4b9102c534996f3b7cfdcf14c26fd49092126`

## 결과

Windows 네이티브 Qt에서 **2행 → 1행 전환 시 오버레이 높이가 118px에 남는 결함**을 재현하고 수정했다. 필요한 높이는 67px이었다. 기존 offscreen 테스트만으로는 발견되지 않았다. 생산 코드 변경은 `phase5/subtitle_overlay.py`의 `_set_window_height()`에 한정했다. 계산한 높이를 `setFixedHeight()`로 명시하고, 변경 전 높이를 기준으로 기존 하단/드래그 위치 처리 경로를 유지한다. 색상·글꼴·30% 높이 상한·FINAL 정책은 바꾸지 않았다.

Window 13·14에서 기존 FINAL 분절은 그대로 표시된다. 현재 디스플레이의 Qt 렌더링과 모델에서는 텍스트 누락/순서 역전이 없었다. 실제 다른 앱과 합성된 투명도나 전체 화면 위 최상위 표시까지 확인한 것은 아니다.

## 사전 상태와 기존 구현

- 시작 미커밋 파일: `phase4/output/remote_gpu_integration.json`, `r2_b1_diff.txt`, `r2_b1_diff_utf8.txt`, `r2_b1_fixture.json`, `docs/r2_b4_finalization_investigation.md`, `docs/r2_b4_finalization_audit.json`. 보존했다.
- 기존 evidence·fixture·보고서는 읽기만 했다. 조사 직후 확보한 기존 파일 280개 해시와 종료 시 해시를 대조했다.
- L6/L8 문자열을 명시한 저장소 문서/커밋 메시지는 찾지 못했다. 번호를 추정하지 않고 해당 기능의 실제 구현과 이력을 확인했다: `69f60d7`(자막 만료·수명), `eb90470`(presentation 진단), `fe2127e`(오버레이 동적 크기), `a771f40`(색상·Qt lifecycle).
- 기존 상태 관리, 오버레이, preview, controller worker 신호 격리 및 테스트를 재사용했다. `gui_app.py`, `live_caption_controller.py`, `subtitle_presentation.py`, Phase 3/4 운영 코드는 변경하지 않았다.

## 재생 방법과 시간 의미

프로젝트 루트 PowerShell에서:

```powershell
$env:QT_QPA_PLATFORM='windows'
.venv\Scripts\python.exe -m phase5.replay_overlay_timeline
# 원본 Window 14의 마지막 이벤트까지 재생한 뒤 시계를 멈춰 검사
.venv\Scripts\python.exe -m phase5.replay_overlay_timeline --pause-at-sequence 14 --close-after 120
# 전체 시간 및 자막 만료를 함께 8배속으로 재생
.venv\Scripts\python.exe -m phase5.replay_overlay_timeline --speed 8
```

입력은 원본 timeline의 별도 `subtitle_event` 21개다. window 내부 상태 이벤트는 중복 주입하지 않는다. 원본 dictionary의 text, sequence, timestamp, utterance_id와 파일 순서를 유지한다. R2-B4 audit의 현재 코드 재조립 텍스트를 대신 쓰지 않는다. 따라서 화면의 `사급력`, `실행...` 등 원본 인식/조립 결과도 그대로다.

첫 subtitle timestamp를 재생 0초로 삼는다. 최초 자막 전 무음은 생략한다. timestamp 차이를 가상 수신 시각으로 사용하고, presentation clock도 같은 시간으로 움직여 FINAL 수명이 재생 배속과 함께 변한다. 기본 1배속에서는 기록 1초=실제 1초다. Qt 10ms 타이머로 due 이벤트를 순서대로 처리하므로 같은 tick에 들어온 1ms 간격 이벤트는 같은 화면 프레임에 반영될 수 있다. wire timestamp를 바꾸거나 이벤트를 정렬하지 않는다. 이벤트 도착 직전에 만료를 계산한다.

실서비스는 로컬 `time.monotonic()` 수신 시각으로 FINAL 4초를 계산하고 100ms마다 만료를 확인한다. wire epoch나 stream end 시각으로 만료하지 않는다. 재생은 기록 timestamp를 로컬 도착 간격의 대용으로 사용하며 당시 실제 Qt 수신 지연을 복원하지 않는다. `--pause-at-sequence`는 지정 이벤트까지 기록 시간으로 이동한 후 만료까지 정지하는 검사 모드이며 실시간 수명 검증과 구분한다.

오디오 장치 열기, WASAPI, 서버 health, 웹소켓, STT/VAD 추론 없이 production Qt 모델 → production overlay만 실행한다. 1배속 및 8배속 CLI 전체 재생과 정상 종료를 각각 실행했다. 전체 구간은 첫 이벤트부터 27.896초이며 마지막 FINAL 뒤 4.2초까지 유지 후 종료한다.

## Window 13·14 결과

| sequence / 상대 시각 | 이벤트 | 표시 순서 |
| --- | --- | --- |
| 11 / 18.200초 | window 13, FINAL u2 | FINAL u2 |
| 12 / 18.201초 | window 13, PARTIAL u3 | FINAL u2 → PARTIAL u3 |
| 13 / 19.468초 | window 14, FINAL u3 | FINAL u2 → FINAL u3 |
| 14 / 19.469초 | window 14, PARTIAL u4 | FINAL u2 → FINAL u3 → PARTIAL u4 |

sequence 14의 실제 입력/표시 텍스트:

1. `이번 사급력이 큰 건 PHP가 인터넷 정보...` (FINAL)
2. `인터넷 전반의 광범위하게 쓰이는 핵심.` (FINAL)
3. `기술이기 때문입니다.` (PARTIAL)

u2 FINAL은 22.200초, u3 FINAL은 23.468초에 만료한다. u3은 window 15 이벤트가 들어오는 23.373초부터 약 95ms 동안 새 FINAL/PARTIAL과 공존할 수 있다. 마지막 u6 FINAL은 31.896초에 만료한다. 이 짧은 공존 구간은 기록된 도착 간격과 4초 정책의 결과다.

현재 native Qt 작업 영역은 2560×1400 logical pixels, logical DPI 96이었다. 기본 28pt와 최대 52pt 각각 21개 스냅샷에서 label의 실제 높이와 `heightForWidth()`를 비교했다. 이 환경에서는 잘림과 행 생략이 없었다. [재생 측정 JSON](replay_audit.json)에 각 이벤트의 텍스트·timestamp·model/overlay 순서·높이를 보관했다.

아래 이미지는 Qt 위젯 자체의 렌더링이다. Windows 데스크톱 합성 화면 캡처가 아니다. 픽셀을 확인해 Window 14의 3행과 긴 첫 발화의 2줄 줄바꿈을 검사했다.

![Window 14 위젯 렌더링](qt_widget_sequence_14.png)

![긴 첫 발화 위젯 렌더링](qt_widget_sequence_7.png)

## 11개 검사 항목

| 항목 | 자동 검증 결과 / 한계 |
| --- | --- |
| PARTIAL 표시 | 기록 이벤트가 모델·QLabel에 그대로 전달됨. PASS |
| PARTIAL → FINAL | 동일 utterance PARTIAL이 FINAL로 대체되고 중복 행 없음. PASS |
| FINAL 직후 새 PARTIAL | 기존 FINAL 아래 새 utterance가 표시됨. PASS |
| Window 13·14 | 위 표의 순서 및 정확한 원본 텍스트 확인. PASS |
| 다른 utterance_id 순서 | 해당 single-speaker 기록의 u2/u3/u4 순서 확인. 다화자 상태는 기존 회귀 포함. PASS |
| 중복 FINAL | 원본 FINAL 복사 및 더 큰 sequence를 준 합성 중복 모두 거부. 기존 deadline 연장/만료 후 재등장 없음. PASS |
| 무음 수명 | FINAL만 4초 후 제거. 진행 PARTIAL은 자동 만료 대상 아님. upstream FINAL/discard/세션 reset이 필요. 기존 정책 유지 |
| 오래된 자막 제거 | 실제 Qt 100ms 타이머로 새 이벤트 없이 만료 확인. 마지막 만료 후 card 숨김/창 높이 1px. PASS |
| 시작·중지·재시작 | 기존 fake backend 테스트로 clear, 오버레이 재사용, 이전 worker의 queued signal 거부 확인. PASS. 실제 서버/캡처 세션은 미실행 |
| 긴 한국어/영역 초과 | 현재 화면의 28/52pt 기록 재생 PASS. 아래 작은 화면 한계는 미해결 |
| 위치·크기·투명도·최상위 | geometry 화면 내 포함, frameless/topmost/translucent/show-without-activating 속성, 배경 72%(alpha184), 기본 하단 anchor·드래그 후 위치 유지 검증 PASS. OS 합성/실제 drag 입력/전체 화면 앱 위 z-order는 수동 확인 필요 |

**무음에서 배경 바가 사라지는 것은 의도된 동작이다.** FINAL 만료 후 entries가 없으면 `_card.hide()`와 1px 빈 창으로 전환한다. 오버레이 객체 자체를 파괴하는 것이 아니다. 반대로 FINAL을 받지 못한 PARTIAL은 현재 설계상 남는다.

## FAIL-FIRST와 수정 검증

- 수정 전 기존 offscreen suite: 60 PASS.
- 기록 재생 테스트 5개 추가 후 offscreen: 65 PASS.
- 같은 65개를 `QT_QPA_PLATFORM=windows`에서 실행: 64 PASS / 1 FAIL. 기존 `test_overlay_height_shrinks_with_three_two_one_and_zero_rows`가 `118 != 67`로 실패. 해당 테스트 단독 실행에서도 동일하게 실패했다.
- 높이 제약 최소 수정 및 빈 상태 뒤 재확대/anchor 회귀 1개 추가 후: **Windows 66 PASS, offscreen 66 PASS**, 최종 FAIL/ERROR/XFAIL 0.
- 정상 경로와 의도된 mock 실패 경로를 검증하므로 test output의 `mock server unavailable` / `WASAPI capture 실패` traceback은 실제 연결·캡처 시도가 아니다.
- Phase 3 회귀: **137 실행 = 134 PASS + 기존 XFAIL 3**, 예상 밖 FAIL 0. main 기반 assembler/state/consensus/integration/P1/silence-gate/separation/leakage 회귀 8개 PASS.
- 수정/추가 Python 3개 AST syntax, import 및 `git diff --check` PASS. Ruff 미설치로 lint SKIP.

GUI 회귀 명령(프로젝트 루트, 환경변수 windows/offscreen 각각 실행):

```powershell
.venv\Scripts\python.exe -m unittest phase5.test_gui_app phase5.test_subtitle_presentation phase5.test_overlay_preview phase5.test_ui_shell
```

## 미해결 한계와 수동 확인

- 800×800 offscreen 가상 화면에서는 기본 28pt의 긴 첫 발화(sequence 5–8)가 단일 행 블록의 30% 화면 높이 상한을 초과했다. 예: sequence 7/8은 label 222px 대 필요 295px. 52pt에서는 더 많은 이벤트가 잘린다. 이것은 native 2560×1400 검증과 다른 화면 조건이다. 큰 텍스트를 축소하거나 스크롤하거나 상한을 바꾸는 표시 정책은 이번 최소 수정에 도입하지 않았다. 작은 시연 화면/높은 DPI에서는 별도 확인이 필요하다.
- 여러 긴 행이 상한을 넘으면 기존 설계가 오래된 행을 생략하고 최신 행을 우선한다. presentation은 최대 FINAL 2개·총 3개를 유지하므로 4초가 지나기 전에 밀려나는 경우도 정상 bounded-history 정책이다. 모든 과거 텍스트를 4초 동안 반드시 전부 표시하는 계약은 아니다.
- Computer Use의 Windows 창 목록/앱 목록에서 재생 Qt tool 창을 찾지 못했다. 따라서 실제 데스크톱 배경과의 alpha 합성, 다른 앱 위 topmost, 전체 화면 동영상·게임, 모니터 간 이동 및 DPI 변경, 직접 마우스 드래그는 **미검증**이다. native Qt 테스트 통과로 이 항목을 PASS 처리하지 않는다.
- 실제 WASAPI·RunPod/GPU·STT 재추론은 제한에 따라 SKIP. EXE 패키징은 이번 변경 범위에서 미실행. R2-B3와 기존 Phase 3 source-support XFAIL 3건은 그대로 미해결이다.

## 변경 파일

- `phase5/subtitle_overlay.py`: native 창 높이 적용 최소 수정.
- `phase5/test_gui_app.py`: 기록 기반 재생·만료·중복·속성 및 높이 재확대 회귀 6개 추가. 기존 검증 조건 유지.
- `phase5/replay_overlay_timeline.py`: 개발용 오프라인 재생 도구.
- 이 새 보고서 디렉터리: README, replay_audit JSON, Qt 위젯 렌더링 PNG 2개.

기존 보고서/fixture/evidence 및 금지 파일은 변경하지 않았다. staging/commit/push는 수행하지 않았다. 위 변경만 분리해 커밋할 수 있으나, 작은 화면 잘림과 실제 데스크톱 합성 검증은 완료된 것으로 표시하면 안 된다.

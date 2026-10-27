# VOXLUX P0-2A 무음 유령 자막 분석·수정·회귀 보고서

## 1. 기본 정보와 결론

- 작업 날짜: 2026-10-27 (UTC).
- 시작 브랜치: `main`.
- 시작 HEAD: `322884d15a657b2300a31cdef04f293e94bed6f7` (`fix: harden realtime audio queue and GPU request handling`). 이전 감사 기준과 동일.
- 작업 전 `git status --short`: 출력 없음(깨끗함). 기존 미커밋 변경을 침범하지 않았다.
- `AGENTS.md` 확인. `VOXLUX_PROJECT_GUIDELINES_2026-10-27.md`는 저장소에 없어서 사용자 프롬프트와 실제 소스를 기준으로 작업했다.
- 환경: Linux, Python 3.12.3, NVIDIA RTX A4500 20GB, PyTorch 2.8.0+cu128, Silero VAD 6.2.3, Faster-Whisper 1.2.1, PySide6 6.8.3.
- 판정: **부분 완료**. 응답 계약 결함 수정과 진단 계측, 실행 가능한 회귀 검증을 수행했다. 실제 Windows 무음 유령 자막의 원인과 해결 여부는 **미확정**이다.

실제 GPU의 `large-v3-turbo`는 VAD를 우회한 디지털 무음에서 “다음 영상에서 만나요.”, 저진폭 잡음에서 “감사합니다.”를 생성했다. 반면 **같은 입력의 정상 SINGLE 서버 경로에서는 VAD가 STT를 건너뛰고 빈 텍스트를 반환했다.** 이 재현은 모델의 비음성 환각 가능성을 입증하지만 Windows 운영 경로에서 VAD를 통과한 이유까지 입증하지 않는다.

## 2. 운영 경로 확인

`Phase4PipelineBackend.run` → `phase3/run_overlap_pipeline.main` → WASAPI 캡처/48kHz→16kHz 변환 → 3초 window/2초 stride → `RemoteGPUClient.process` → `PipelineService._process_serialized` (SINGLE) → `validate_end_to_end_gpu.detect_speech_activity` → `transcribe_base` → 원격 raw slot/logical mapping → admission/`SubtitleAssembler`/`SpeakerSubtitleState` → runtime callback → `LiveCaptionWorker` → controller의 Qt slot → `SubtitlePresentationState`/GUI overlay.

- GUI 자막 입력은 callback → Qt signal이다. WebSocket은 별도 선택적 출력이다.
- 서버의 실제 VAD 함수는 `phase3/validate_end_to_end_gpu.py`에 있다. 이름만으로 `run_overlap_pipeline.py`의 로컬 VAD 함수와 혼동하지 않았다. 두 구현은 같은 Silero `get_speech_timestamps` API를 호출한다.
- 서버 기본 프로필 SINGLE, 모델 `large-v3-turbo`, GUI 3초/2초를 확인했다. 실행 중인 Windows 배포본이나 별도 원격 프로세스가 이 소스와 동일한지는 이번 환경만으로 확인할 수 없다.
- 분리 기능을 활성화하지 않았다. GPU 검증도 separator 없는 SINGLE을 사용했다.
- P0-1 큐, stale-window 기준, timeout, 요청 동시성 제한 및 P1-3 빈 자막 바를 변경하지 않았다.

## 3. 원인 후보별 판정과 증거

| 계층/후보 | 판정 | 근거와 한계 |
|---|---|---|
| 청감상 무음과 PCM 무음의 차이, 잡음·버퍼 잔류 | 미확정 | 해당 Windows WAV/로그가 없다. 기존 `capture_audio_diagnostics`에 채널별 RMS/peak, 선택 채널, downmix 취소 후보가 있다. 캡처는 새 2초와 이전 tail 1초를 결합하므로 무음으로 전환한 직후 창에는 실제 이전 발화가 남을 수 있다. 이것만으로 캡처 버그라고 볼 수 없다. |
| 다채널/mono 변환 | 미확정 | 기본 첫 채널 선택, 선택적 mean, 유한값/길이 검사와 resampling을 확인했다. 실제 장치의 채널 내용은 미검증이다. 장치 정책을 변경하지 않았다. |
| VAD 상태가 다음 창에 잔류 | 이번 입력에서는 재현 안 됨 | 설치된 `get_speech_timestamps` 구현에서 `model.reset_states()`를 확인했다. 실제 VAD의 발화→무음→발화→무음에서 무음 speech duration은 모두 0ms였다. |
| 비음성 VAD 오탐 + Whisper 환각 | 모델 측 가능성 확정, Windows 연결은 미확정 | 실제 GPU 직접 추론으로 무음/잡음 환각을 관측했다. 정상 서버의 같은 PCM은 VAD가 차단했다. Windows 입력의 VAD false positive 여부는 아직 알 수 없다. |
| VAD=false + nonempty RAW 원격 응답 수용 | **계약 검증 결함 확정·수정** | `parse_result`가 모순된 응답을 허용했다. 수정 전 5개 문자열 모두 `RemoteProtocolError not raised` 실패, 수정 후 통과. 정상 서버가 이 응답을 생성한다는 증거는 없다. 모의 서버를 통한 worker 테스트에서 해당 창을 거부하고 다음 정상 “네”를 보존했다. |
| STT 결과 진단 유실 | **진단 공백 확정·수정** | `SttResult`는 이미 confidence 관련 통계를 계산하지만 서버가 text/time만 응답에 담고 버렸다. 해당 값을 응답→client metric→evidence에 연결했다. |
| Assembler 과거 텍스트·PARTIAL/FINAL 부활 | 검증한 정상 경로에서는 재현 안 됨 | 기존 뉴스 fixture/조립/수명주기 및 신규 SINGLE 반복 “네” 테스트가 통과했다. 무음 FINAL은 기존 발화 확정이지 새 STT가 아니다. 빈 STT의 text 생성은 단위 검증에서 없었다. 아래 기존 미해결 계약 3개는 별도 한계다. |
| 오래된 HTTP 응답, STOP/재시작, Qt 재전달 | 검증한 모의 경로에서는 재현 안 됨 | 기존 P0-1 전송·응답·소비 stale/STOP 검사와 controller의 `sender() is self._worker` 검사를 확인했다. 지연 HTTP·STOP/재시작·Qt queued old worker 회귀가 통과했다. 실제 Windows 시간 측정은 미검증이다. |

### 실제 GPU 증거

재현 명령: `HF_HUB_OFFLINE=1 python docs/p0_2_evidence/check_real_models.py`

이미 할당된 GPU와 프로젝트의 로컬 모델 snapshot만 사용했다. `local_files_only=True`; 다운로드 및 RunPod 자원 시작 없음. HTTP/WASAPI를 거치지 않는 **실제 SINGLE 서비스 함수 호출**이다.

| 입력 | 실제 VAD | 정상 서버 STT/RAW |
|---|---|---|
| 3초 float32 all-zero | false, 0ms | 호출 안 함 / 빈 텍스트 |
| seed=20261027 Gaussian noise, 표준편차 0.0001 | false, 0ms | 호출 안 함 / 빈 텍스트 |
| 기존 `speaker_a.wav`의 0~3초 | true, 2966ms | “데이터베이스는 여러 사용자가 필요한 정보를 저장하고” |
| 발화 직후 3초 zero | false, 0ms | 호출 안 함 / 빈 텍스트 |
| 같은 녹음의 2~5초 | true, 2866ms | “정보를 저장하고 관리하기 위한 시스템입니다.” |
| 다시 3초 zero | false, 0ms | 호출 안 함 / 빈 텍스트 |

VAD를 의도적으로 우회한 직접 STT 진단:

| 입력 | RAW | avg_logprob | max no_speech_prob | max compression_ratio |
|---|---|---:|---:|---:|
| zero | 다음 영상에서 만나요. | -0.405382 | 0.0 | 0.731707 |
| noise | 감사합니다. | -0.388867 | 0.0 | 0.64 |

따라서 `no_speech_prob`가 높을 때만 거르는 필터는 이 두 사례에 효과가 없고, 임의의 logprob/RMS 임계값은 정상 발화 손실 위험이 있다. 필터나 모델·Whisper 옵션을 변경하지 않았다. 현재 옵션은 `language=ko`, `beam_size=5`, `condition_on_previous_text=False`; 나머지는 설치된 라이브러리 기본값(예: no_speech_threshold 0.6, log_prob_threshold -1.0)이다. 진단 값은 반환된 segment들의 집계이며 Whisper 내부에서 이미 폐기된 segment나 실제 발화 정답을 뜻하지 않는다.

원본 수치: [real_models.json](p0_2_evidence/real_models.json). 실제 짧은 “네” 녹음의 인식 정확도는 검증하지 않았으며, 짧은 문자열 보존은 모의 STT를 사용하는 회귀 검증이다.

### 기존 미해결 계약 3개

`phase3/test_source_support_diagnostics.py`의 원래 `@expectedFailure`를 그대로 유지했다. **PASS로 계산하지 않는다.**

1. `test_unresolved_noise_above_existing_bound_must_not_publish`: RMS 약 0.001518의 합성 잡음에 speech 316ms를 주입하면 source 상관관계가 speech identity처럼 채택될 수 있다. SINGLE에서도 관련 가능한 한계다. 실제 Silero 오탐이 재현됐다는 뜻은 아니다.
2. `test_unresolved_projection_artifact_must_not_publish`: 분리 출력의 투영 오류. 현재 SINGLE 운영 범위 밖이며 분리 기능을 손대지 않았다.
3. `test_unresolved_gap_must_not_hide_later_supported_speech`: 미지원 prefix 뒤의 정상 suffix가 보류될 수 있다. 강한 필터 추가 시 정상 발화 보존 위험을 보여준다.

이 한계를 임의 임계값으로 덮지 않았다. 기존 코드·테스트를 변경하지 않았고 새 회귀 실패는 없다. 이 때문에 전체 유령 자막 문제가 해결됐다고 보고하지 않는다.

## 4. 수정 내용과 위험

| 파일/함수 | 변경 전 → 변경 후 |
|---|---|
| `phase4/remote_gpu_client.py:parse_result` | VAD=false/nonempty RAW도 파싱 → 프로토콜 오류로 거부. 기존 오류 처리 경로가 해당 창을 기록·폐기한다. |
| `phase4/gpu_processing_server.py:PipelineService._process_serialized` | STT 통계 유실 → slot에 `stt_diagnostics` 추가: executed/skip_reason, segment_count, 평균 logprob, 최대 no_speech_prob/compression_ratio. 미실행/미지원/비유한 값은 null. |
| `phase3/run_overlap_pipeline.py:main, build_evidence_timeline` | 세션 간 window 번호 구분 부족 → 실행마다 UUID session_id 생성, subtitle/metric/drop/summary와 evidence window에 연결. metric/evidence에 remote_request_id와 logical 순서 STT 진단 연결. |
| `phase5/subtitle_presentation.py:SubtitleFeedEntry, expire, _log_decision` | 만료 시 speaker/utterance 집합만 기록 → session_id/window_index를 accept/reject/개별 FINAL 만료 로그에 유지. 표시 디자인·수명·수용 정책은 동일. |
| `phase4/test_remote_gpu_client.py` | 모순 응답 5개 문자열 거부 + VAD 활성 짧은 4개 문구 보존 테스트 추가. |
| `phase4/test_ghost_caption_path.py` | 서버 VAD gate/진단 binary roundtrip, SINGLE speech/silence 반복·3/2 경계·만료·식별자 전파, 잘못된 응답 다음 정상 발화 복구, 빈 STT/비유한 진단 테스트 추가. |
| `docs/p0_2_evidence/*`, 이 보고서 | 재실행 스크립트와 소규모 로그/JSON 증거. WAV·모델·인증정보 추가 없음. |

정상 발화 위험: **VAD=true인 짧은 텍스트는 길이·단어·확률로 삭제하지 않는다.** 서버가 실제 발화의 text를 반환하면서 VAD=false라고 잘못 표기하면 응답 전체를 거부하므로 그 창의 정상 발화도 잃을 수 있다. 이는 모순된 프로토콜에 한정되며 서버의 정상 생성 규칙과 일치한다. 잘못된 한 slot 때문에 같은 응답의 다른 정상 slot도 거부될 수 있다(현재 SINGLE의 비활성 두 번째 slot은 빈 값). 기존 VAD false negative 위험은 남아 있다.

추가 계측은 기존 결과 객체당 작은 scalar 필드와 UUID 하나다. 오디오 저장이나 별도 누적 버퍼를 추가하지 않았다. 기존 `--evidence-dir`을 명시할 때만 기존 방식대로 WAV를 저장한다. 기존 summary/로그의 장시간 누적·보존 정책은 변경하지 않았다.

## 5. 진단 연결 방법

- `(session_id, window)` → capture 시각/stream start·end/새 오디오 길이/채널 RMS·peak → VAD timestamps·duration → logical `remote_stt_diagnostics` 및 RAW → admission → assembly/state 이벤트.
- metric의 `remote_request_id` → HTTP `X-Request-ID` 및 서버 latency 로그. request ID는 기존 client가 생성하며 정책을 변경하지 않았다.
- subtitle의 `session_id`, `window_index`, speaker, utterance_id, sequence → callback/Qt presentation accept/reject/expire 로그.
- `--latency-diagnostics`에서 기존 client/server 요청·응답/queue 시각 및 드롭 사유를 사용한다. Windows와 서버의 `perf_counter` 절대값은 서로 빼지 않는다.
- `capture_started/ended`는 해당 read의 새 오디오 캡처 시각이고, 겹친 전체 창의 음향 시간은 `stream_start_seconds/stream_end_seconds`로 읽는다.
- 구버전 서버에서 진단 필드가 없으면 null로 남는다. 진단을 위해 서버와 Windows 코드가 모두 갱신되어야 한다. 이번 작업은 실행 중인 배포를 재시작하지 않았다.

## 6. 실행한 테스트와 한계

| 실행 명령 | 결과 | 증거 |
|---|---|---|
| 수정 전 `PYTHONPATH=phase3:phase4:. python -m unittest test_remote_gpu_client.ClientTests.test_inactive_vad_rejects_any_nonempty_raw_transcript` | **FAIL**, 1 test / 5 subtest: 모순 응답을 허용 | `before_fix.txt` |
| 수정 후 위 테스트 + `test_active_vad_preserves_short_text_and_inactive_empty_is_valid` | **PASS 2** | 최종 Phase 4에서도 재검증 |
| `QT_QPA_PLATFORM=offscreen PYTHONPATH=phase3:phase4:. python -m unittest discover -s phase4 -p 'test_*.py'` | **PASS 83** | `phase4_final.txt`; 당시 신규 3개 포함 |
| 마지막 신규 테스트 포함 `PYTHONPATH=phase3:phase4:. python -m unittest test_ghost_caption_path.GhostCaptionPathTests` | **PASS 4** | `new_regressions.txt`; 위 83개와 3개 중복 |
| `PYTHONPATH=phase3:phase4:. python -m unittest discover -s phase3 -p 'test_*.py'` | **ERROR 2**, Linux soundcard/PulseAudio import 불가; 기존 XFAIL 3 | `phase3.txt` |
| `python docs/p0_2_evidence/run_regressions.py phase3` | **PASS 179 / XFAIL 3**, 총 182 | `phase3_headless.txt`; 장치 import만 stub. 뉴스 fixture/자막/Assembler 포함 |
| `QT_QPA_PLATFORM=offscreen PYTHONPATH=phase3:phase4:. python -m unittest discover -s phase5 -p 'test_*.py'` | **PASS 67 / ERROR 5**, 녹화 timeline 없음 | `phase5.txt` |
| `python docs/p0_2_evidence/run_regressions.py phase5` | **PASS 67 / SKIP 5**, 총 72 | `phase5_available.txt`; 원본 assertion 수정 없음 |
| `python docs/p0_2_evidence/run_script_regressions.py` | **PASS 9 스크립트** | `script_regressions.txt`; unittest discovery가 실행하지 않는 main()도 실행 |
| `HF_HUB_OFFLINE=1 python docs/p0_2_evidence/check_real_models.py` | **PASS**, 실제 VAD/GPU 6개 서비스 입력 + 직접 STT 진단 2개 | `real_models.json`, `real_models_run.txt` |
| 아래 `py_compile`, `git diff --check` | **PASS** | syntax와 whitespace 검증 |

```bash
python -m py_compile phase3/run_overlap_pipeline.py phase4/gpu_processing_server.py phase4/remote_gpu_client.py phase4/test_remote_gpu_client.py phase4/test_ghost_caption_path.py phase5/subtitle_presentation.py docs/p0_2_evidence/check_real_models.py docs/p0_2_evidence/run_regressions.py docs/p0_2_evidence/run_script_regressions.py
git diff --check
```

스크립트 회귀 9개: `test_stt_context`, `test_subtitle_state`, `test_subtitle_contract_p1`, `test_subtitle_consensus`, `test_subtitle_assembler`, `test_subtitle_assembler_state_integration`, `test_overlap_diagnostic_schema`, `test_pre_separation_silence_gate`, `test_secondary_leakage_diagnostics`.

- **SKIP**: 실제 Windows WASAPI/장치 채널/물리 GUI·투명 overlay/실제 START→STOP→재시작. Linux 환경이므로 수행 불가. Qt offscreen 통과를 Windows 통합 통과로 표현하지 않는다.
- **SKIP**: 녹화 GUI 타임라인 의존 5개. `phase3/output/evidence/news_20261022_160827/timeline.jsonl` 부재. 기록된 arrival timestamp를 합성해서 대체하지 않았다. 저장된 뉴스 window fixture 회귀는 실행했다.
- **SKIP**: 별도 lint 도구(ruff/flake8 미설치). import는 실행 테스트에서, syntax는 py_compile로 검증했다.
- 유료 자원 시작, 장시간 benchmark, CER/WER 최적화, EXE 패키징은 수행하지 않았다.
- 최초 환경 ERROR 로그도 보존했다. 재실행기는 전제가 없는 테스트만 명시적으로 SKIP하고 원본 테스트 파일/검증식을 삭제·완화하지 않는다.

## 7. 실제 Windows에서 필요한 후속 증거

업데이트된 서버/Windows 코드를 사용하고 운영 프로필 SINGLE을 유지한다. 30~60초의 제한된 재현 세션을 실행해 디지털 무음/실제 시스템 무음/정상 짧은 발화/발화 직후 무음/STOP·재시작을 구분한다.

```powershell
python -u -m phase5.gui_app --window-seconds 3 --stride-seconds 2 --latency-diagnostics --evidence-dir evidence/p0_2_windows_run1
```

수집할 자료: `original.wav`, 문제 window의 mixed WAV, `timeline.jsonl`, `phase5/output/gui_last_run.json`, GUI stdout 및 `phase5/output/voxlux.log`, 같은 request ID의 서버 로그. 재시작은 별도 evidence 디렉터리로 실행해 덮어쓰기를 피한다. 오디오를 Git에 넣지 않는다. GUI에서 들리지 않던 시각과 실제로 표시된 문구·표시/만료 시각도 기록한다.

우선 확인할 질문은 **문제 자막 창의 PCM에 실제 음성 또는 잡음이 있었고 VAD가 왜 speech로 판정했는가**이다. 과거 발화가 포함된 1초 overlap, 3초 창의 완성 대기, 기존 허용 범위 내 응답 지연, 기존 FINAL의 4초 표시를 새 유령 발화와 구분해야 한다.

**다음 세션의 우선 작업 1개:** 실제 Windows 유령 자막 1건의 WAV·세션/윈도우/요청 로그를 확보하여 동일 PCM을 SINGLE 서버에서 재생하고 VAD→RAW→표시 경로를 대조한다.

## 8. Git 결과

- 사전 원격 확인: `git ls-remote origin refs/heads/main` = `322884d15a657b2300a31cdef04f293e94bed6f7`. 예상하지 못한 원격 변경 없음.
- 최종 구현·테스트 커밋: `b3f53cf98af94b256528a2d6120a22ecb299655c` (`fix: reject inconsistent ghost caption responses and trace STT evidence`).
- `git push origin main`: **성공** (`322884d..b3f53cf main -> main`). 이후 `git ls-remote`의 main도 같은 전체 해시인 것을 확인했다. 강제 푸시 없음.
- 구현 커밋 푸시 직후 `git status --short`: **출력 없음(깨끗함)**. P0-2A 관련 21개 파일만 개별 경로로 stage했다. 기존 XFAIL/환경 SKIP은 위에 기록했으며 수정으로 추가된 회귀 실패나 임계값 조정은 없다.
- 이 Git 결과는 위 커밋·푸시를 실제 확인한 뒤 작성했다. 결과 기록 자체는 후속 문서 전용 커밋으로 보존한다. 문서가 자신의 커밋 해시를 포함할 수 없으므로 최종 문서 커밋 해시는 `git log -1 --format=%H -- docs/p0_2_ghost_caption_report.md`로 확인한다. 최종 터미널 응답에도 해당 실제 해시와 푸시·종료 상태를 별도로 출력한다.

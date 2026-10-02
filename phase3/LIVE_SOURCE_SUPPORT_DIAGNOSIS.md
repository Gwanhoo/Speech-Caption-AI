# Live source support 진단 (기준 d2c891d)

이번 변경은 진단과 재현 테스트다. CASE B/C의 실제 acoustic 값과 후속 raw
fragment가 제공되지 않았으므로 정책/threshold 또는 publication gate를 변경하지
않았다. 합성 재현은 실제 Windows/RunPod Live 재현을 대신하지 않는다.

## 실제 경로

`gpu_processing_server.PipelineService._process_serialized()`에서 분리 출력 각각에 VAD/STT를
수행하고 waveform/VAD/raw transcript를 반환한다. Windows의
`run_overlap_pipeline.main()`은 결과를 `PersistentSpeakerTracker`로 logical
speaker에 매핑하고 `remote_result.logical_slots()`로 VAD/STT를 같은 순서로 맞춘다.
그 뒤 기존 secondary suppression, `resolve_subtitle_fragments()`,
`admit_subtitle_streams(mixture=item.source.audio, speakers=item.speakers)` 순서다.
assembler hypothesis와 admission의 `source_supported`, 현재 admitted transcript인
`source_text`를 `SpeakerSubtitleState.process()`에 전달한다.

`publication_text`는 `publish_subtitle_state_event()`의 empty/retraction guard를
지난 뒤 payload의 `text`가 된다. 그때 sequence가 증가하고 WebSocket queue와
runtime hook으로 전달된다. GUI는 `LiveCaptionWorker._publish_subtitle()` →
`LiveCaptionController._forward_subtitle()` → `MainWindow.apply_subtitle_event()` →
`SubtitlePresentationState` → feed/overlay 순서로 payload의 `text`를 표시한다.
remote mode는 `args.vad=True`, state는 `require_final_support=True`다.

## CASE B: 알 수 있는 조건과 없는 수치

`source_supported`는 candidate의 speech interval 중 하나라도 직접 입력 상관
또는 owner 제거 후 잔차 상관이 0.45 이상이면 True다. owner가 zero 등으로
`source_input_evidence()`가 None을 반환하면 직접 입력 상관으로 fallback한다.

`missing_speech_intervals`는 candidate **또는 owner**의 intervals가 없다는 후반부
설명이다. source support 계산은 그 전에 끝난다. candidate intervals가 정말
없었다면 `any([])`는 False이므로, CASE B처럼 source_supported=True와 함께
나온 경우 이 코드에서는 owner intervals가 없는 경우다. 이는 timestamps 부재만으로
실제 짧은 발화를 차단하지 않는 fail-open admission 정책이다.

저레벨 차단은 원본 입력에 다음 조건이 모두 맞아야 한다:

- 0 < speech duration <= 500ms, 0 < VAD ratio <= 0.20
- candidate speech intervals의 원본 RMS <= 0.001, 원본 peak <= 0.005
- speech RMS <= 1.5 × background RMS (수치적 epsilon 적용)

CASE B의 316ms/0.105는 첫 조건에 해당한다. 하지만 Live `[VAD] rms/peak`는
**분리 출력 전체 window** 값이다. 원본 speech/background 수치를 대신할 수 없다.
따라서 어떤 원본 조건이 차단을 통과시켰는지 아직 알 수 없다.

합성 stationary noise를 RMS 0.001518로 생성하고 candidate=mixture, owner=0으로
설정하면 direct fallback correlation=1, source_supported=True이고 기존 joint
noise guard 밖이라 공개된다. 이는 코드의 한계 재현이며 Live 입력 복원이 아니다.

## CASE C: 상관계수의 의미와 반례

평균을 빼고 단위 norm으로 만든 입력 x, owner p, candidate s에 대해:

| 필드 | 계산/사용 |
|---|---|
| pair_correlation | abs(corr(p,s)); positive support 조건은 아님. duplicate suppression에서 >=0.85 사용 |
| owner_input_correlation | abs(corr(x,p)); unsupported_source에서 >=0.95 사용 |
| candidate_input_correlation | abs(corr(x,s)); >=0.45면 직접 support |
| independent_input_correlation | abs(corr(x-(x·p)p, s-(s·p)p)); >=0.45면 잔차 support |
| input_residual_energy_fraction | 1-(x·p)^2; 독립 상관 계산의 numerical degeneracy 검사에만 사용 |
| candidate_residual_energy_fraction | 1-(s·p)^2; 위와 동일 |

두 잔차 에너지가 모두 1e-14 초과일 때 잔차 상관을 계산한다. 이는 소리 크기나
화자 수 threshold가 아니다. `unsupported_source`는 owner>=0.95와 direct<=0.20,
independent<=0.20가 모두 필요하므로 independent>=0.45인 CASE C에는 적용되지 않는다.
`independent_input_support`라는 reason은 두 speaker가 모두 VAD/STT 활성이고,
candidate intervals가 owner 안에 포함되며, owner가 established 또는 tracker의
유일 활성 source인 경우 해당 상관 조건을 통과했음을 뜻한다.

합성 입력 p, primary 출력 p+0.01r, secondary 출력 r (p와 r은 직교)를 주면:

| 필드 | 합성 수치 |
|---|---:|
| pair_correlation | 0.0099995 |
| owner_input_correlation | 0.9999500 |
| candidate_input_correlation | 약 0 |
| independent_input_correlation | 1.0 |
| input_residual_energy_fraction | 0.00009999 |
| candidate_residual_energy_fraction | 0.9999000 |

입력에 r이 없는데도 owner의 분리 오차 때문에 잔차 상관이 높아지는 반례다.
그러나 아주 작은 실제 source도 낮은 입력 잔차 에너지와 높은 잔차 상관을 만들 수
있다. 기존 quiet-secondary 테스트에는 상대 진폭 0.001인 실제 성분도 포함된다.
따라서 residual energy 하한 추가는 현재 증거로 정당화되지 않는다. 실제 MossFormer2
출력이 이 반례였는지, 배경음 성분이었는지, 다른 오차인지 Live 수치/오디오 없이
확정할 수 없다. VAD 양성은 STT 실행의 선행 조건이며 화자 독립성의 증명이 아니다.

## source_supported_text 고정

state는 이전 supported text와 현재 hypothesis의 공통 prefix만 유지한다.
현재 source-supported raw text가 hypothesis의 suffix이고 그 앞의 길이가 유지된
supported prefix 길이 이하일 때만 전체 hypothesis를 supported로 갱신한다.

예: `old`(supported) → `old gap`(unsupported) → `old gap genuine`(현재 raw=genuine,
supported)에서는 `gap`이 미지원이므로 공개는 `old`에서 멈춘다. 처음 `old`가
미지원이었다면 공개는 빈 문자열에서 멈춘다. current source support가 없거나
raw/hypothesis suffix가 불일치해도 갱신되지 않는다. 반대로 `old` → `old genuine`
처럼 빈 구간 없는 supported extension은 정상 갱신된다.

이는 B/C의 acoustic false-positive와 별개의 prefix-only state 제약이다. Live에서
정확히 어느 경로였는지는 후속 source_text와 갱신 조건 로그가 필요하다. 임의로
빈 구간을 지원된 것으로 승격하면 d2c891d의 방어를 깨므로 그대로 두었다.

## 다음 Windows 실행에서 필요한 최소 증거

GUI 프로세스를 완전히 종료하고 업데이트한 client 코드로 다시 실행한다. 시작의
`[PIPELINE CODE]` 경로를 확인하고, CASE B/C 해당 window와 speaker0가 고정되는
첫 후속 window의 다음 기록을 보존한다:

- `[SUBTITLE ADMISSION]`: 모든 활성 window를 JSON 한 줄로 출력. 기존 partial도 포함.
  `speech_intervals`, `owner_speech_intervals`, `speech_local_evidence`의 위 6개 값,
  `source_support_checks`의 direct/independent pass 및 fallback, `source_supported`,
  `weak_speech_evidence`의 원본 speech/background RMS/peak와 각 predicate,
  `weak_speech_check_applied`, `existing_partial`, `suppressed`, `reason`을 확인한다.
- `[SUBTITLE SUPPORT]`: `source_text`, `retained_prefix_length`,
  `observed_suffix_matches`, `unobserved_prefix_length`, `unsupported_gap_length`,
  `reason`을 확인한다. reason은 `no_current_source_support`,
  `source_not_hypothesis_suffix`, `unsupported_prefix_gap`,
  `current_source_covers_unretained_text` 등을 구분한다.
- 기존 `[SUBTITLE PUBLICATION]`: source_supported_text/publication_text/will_publish와
  같은 window의 WebSocket sequence를 대조한다. CASE A는 계속 빈 publication과
  will_publish=False여야 한다.

run JSON의 `windows[].secondary_leakage_diagnostic.subtitle_admission`과
`subtitle_state_events[].support_update`에도 같은 진단이 저장된다. 이 단계에는
RunPod server 코드 변경이 필요 없다. 스칼라만으로 입력 원인을 구분하지 못하면
같은 window의 원본/분리 waveform이 다음으로 필요한 증거다.

## 테스트 해석

`test_source_support_diagnostics.py`는 기존 정상/차단 경로 9개와 미해결 계약 3개를
검증한다. 노이즈 false positive, projection artifact false positive, unsupported
gap 뒤 genuine text 가림은 `expectedFailure`다. 이 3개는 해결되거나 통과한 테스트가
아니다. worker integration은 진단 로그와 저장 JSON 일치, 기존 utterance의 로그,
WebSocket publication 수와 runtime hook 일치를 확인한다.

검증 결과 (Linux, 이번 진단 변경):

| 범위 | passed | failed | skipped | errors | expected failures |
|---|---:|---:|---:|---:|---:|
| phase3 unittest discovery | 56 | 0 | 0 | 2 | 3 |
| phase4 unittest discovery | 36 | 0 | 0 | 0 | 0 |
| phase5 unittest discovery | 13 | 0 | 0 | 2 | 0 |

phase3의 import 오류는 PulseAudio 부재(`test_overlap_diagnostic_schema`,
`test_secondary_validity_live_test`), phase5의 import 오류는 `libEGL.so.1`
부재(`test_gui_app`, `test_ui_shell`)다. 별도 assembler/state/consensus/integration/
contract/WebSocket main() 스크립트 6개도 통과했다. 문법과 `git diff --check` 통과.
기준 d2c891d와 직접 비교한 합성 admission 입력 48개 및 state transition 8개에서
진단 필드/실행 시간 외 기존 판단 결과가 동일했다.

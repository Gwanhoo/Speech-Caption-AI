# Live source support 진단 (기준 d2c891d, 후속 수정 2026-10-03)

최초 작성 시점에는 CASE B/C의 실제 acoustic 값과 후속 raw fragment가 없어 진단만
추가했다. 이후 확인된 live 값(owner correlation 0.995933, candidate correlation
0.518343, residual correlation 0.857164, input residual energy 0.008117)을 근거로
아래의 후속 admission/publication 보호를 구현했다. 합성 회귀는 실제
Windows/RunPod Live 재현을 대신하지 않는다.

## 2026-10-03 후속 수정

- 후속 Live window 003/004에서는 input residual이 각각 3.9136%, 2.3288%라서 기존
  1% residual gate를 통과했다. 두 창 모두 owner correlation이 0.98 이상이고,
  candidate가 owner VAD 안에 포함되며, candidate residual이 input residual보다 컸다.
  window 004의 secondary raw는 primary raw의 정확한 부분 문자열이었다. 따라서 1%
  경계를 올리지 않고, 이 결합 증거를 `owner_candidate_conflict`로 기록한다. 이 조건의
  새 stream 중 primary와 정확한 lexical 구조까지 겹치는 후보는 한 창 동안 내부
  tentative로 보류한다. 다음 창에도 같은 acoustic conflict와
  temporal containment가 있고 두 stream raw 사이에 정확한 포함 또는 짧은 boundary
  overlap이 있으면 current secondary fragment만 leakage로 억제한다. 서로 다른 다음
  transcript나 owner dominance가 해소된 실제 두 번째 화자는 기존 source-support 경로로
  승격된다.
- 확립된 owner가 입력을 지배하고 input residual energy가 1% 미만인 동시에 candidate가
  direct/residual correlation을 모두 통과하며 candidate residual이 input residual보다
  큰 모순 패턴은 새 source로 시작하지 않는다. 이 gate는 기존 weak-speech 경계인
  500ms를 넘는 후보에만 적용한다. 1% 경계는 실제 ghost의 0.8117%를 포함하는 검증
  가능한 경계다. 짧은 응답과 조용하지만 순수한 독립 source는 기존 경로로 통과한다.
- 확립된 owner의 VAD 구간 밖에서 처음 나타난 source는 한 overlap window 동안 내부
  tentative로 보류한다. 다음 창의 새 acoustic support가 있으면 공개하고, 바로 silence가
  오면 publication 없이 discard한다.
- weak speech 검사를 existing partial에도 적용한다. 특히 476ms/ratio 0.159처럼 기존
  weak-VAD 경계에 들고 원본 입력의 speech RMS가 background보다 증가하지 않는 tail은
  절대 RMS가 0.001보다 크더라도 current suffix만 억제한다. 억제된 tail text는
  assembler에 들어가지 않으며 이미 source-supported인 prefix는 정상 FINAL로 유지한다.
- production의 `SpeakerSubtitleState`는 `require_final_support=True`이므로 textual
  consensus만으로는 publication되지 않는다. unsupported tentative의 silence
  finalization은 빈 discard event로 끝나 외부 publisher에 전달되지 않는다.

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
아주 작은 실제 source도 낮은 입력 잔차 에너지와 높은 잔차 상관을 만들 수 있으므로
residual energy만으로 차단하지 않는다. 후속 gate는 실제 ghost에서 함께 관측된 높은
direct correlation과 높은 residual correlation, normalized candidate residual의 크기,
그리고 이미 확립된 owner를 모두 요구한다. 상대 진폭 0.001/0.005인 기존 quiet-source
회귀도 계속 통과한다. VAD 양성은 STT 실행의 선행 조건이며 화자 독립성의 증명이 아니다.

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
  `source_support_checks`의 direct/independent pass, `owner_candidate_conflict` 및 fallback,
  `cross_stream_lexical_overlap`, `speech_contained_in_owner`, `source_support_deferred`,
  `source_supported`,
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

`test_source_support_diagnostics.py`는 기존 정상/차단 경로와 미해결 계약을
검증한다. 노이즈 false positive, projection artifact false positive, unsupported
gap 뒤 genuine text 가림은 `expectedFailure`다. 이 3개는 해결되거나 통과한 테스트가
아니다. `test_ghost_subtitle_regressions.py`는 이번 window 003/004 live 수치,
outside-owner first observation, 476ms existing-partial tail, silence discard/finalize, sustained silence,
정상 overlap deduplication을 별도로 고정한다. worker integration은 진단 로그와 저장
JSON 일치, 기존 utterance의 로그, WebSocket publication 수와 runtime hook 일치를
확인한다.

아래 표는 최초 진단 변경 당시의 역사적 검증 결과다. 최신 결과는 작업 보고를 따른다.

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

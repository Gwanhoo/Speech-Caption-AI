# Secondary source admission investigation — 2026-10-04

코드 수정과 아래 회귀 검증을 수행했다. **Windows w12 해결 확정 보고가 아니다.**
`single_speaker_30s_16k.wav`, `silence_30s_16k.wav`, `astra_ghost_fix_test.json`은
저장소 및 검색한 `/workspace`, `/tmp`에 없었다. 사용자가 제시한 w12의 결정 필드는
확인 가능한 사실로 사용했지만, 제공되지 않은 waveform/상관 수치를 추정해 사실로 쓰지 않았다.
commit/push하지 않았다.

## 1. Root cause

`source_input_evidence()`는 평균 제거·단위 norm 정규화 후 다음 값을 계산한다.

```
x_res = x - dot(x, owner) * owner
c_res = candidate - dot(candidate, owner) * owner
independent_input_correlation = abs(corr(x_res, c_res))
```

이는 owner에 직교하는 두 잔차 방향의 일치도다. 독립 **화자의 speech** 존재 여부가 아니다.
MossFormer의 두 출력이 같은 분리 오차를 공유하면 원본에 두 번째 화자가 없어도 높다.
예를 들어 원본 `p`, owner `p + εe`, candidate `e`에서 원본 잔차는 대략 `-εe`다.
절댓값 상관은 거의 1이다. 실제 설치된 ClearVoice `utils/decode.py`는 각 출력을
입력 RMS로 개별 정규화하므로 artifact도 VAD/STT에 들어갈 만한 크기가 된다.

이 현상을 숫자 threshold 조정만으로 해결할 수 없다. 원본 `p + .001q`, owner
`p + .01q`, candidate `q`에서도 실제 작은 q가 owner의 과대추정분을 상쇄한다.
부호·상쇄 여부만으로 차단하면 실제 작은 화자를 잃는다. 이 반례도 테스트했다.

## 2. 실제 w12 evidence와 코드의 연결

실행 경로를 확인했다.

```
GPU MossFormer raw slots → 각 slot Silero/Whisper → raw waveform/VAD/STT 응답
→ PersistentSpeakerTracker → logical_slots()로 VAD/STT도 같은 mapping 적용
→ resolve_subtitle_fragments → admit_subtitle_streams
→ SubtitleAssembler → SpeakerSubtitleState → WebSocket/runtime hook → UI
```

제공된 `reason=independent_input_support`, `candidate_independent_support=true`는
적어도 한 candidate VAD interval에서 residual correlation ≥ 0.45이고,
`blocked_by_owner_residual=false`, `owner_candidate_conflict=false`였음을 뜻한다.
`existing_utterance`나 TENTATIVE 분기로 먼저 빠지지도 않았다. 뒤쪽 reason 선택은
기존 코드에서 raw residual correlation을 다시 직접 사용했다.

그러나 w12의 실제 owner/direct/residual correlation, 에너지 비율, VAD timestamps,
owner provenance가 없으므로 **owner conflict의 어떤 conjunct가 false였는지는 확정 불가**다.
아래 A.wav window 0은 별도로 재현한 실제 모델 출력이며 Windows w12와 동일시하지 않는다.

## 3. 이전 synthetic PASS와 실제 실패의 차이

기존 `test_artifact_source_admission.py`의 핵심 fixture는 owner 오차와 candidate의
invented component를 서로 다른 random signal로 만들었다. 따라서 residual correlation이
낮아 `contradicted_by_owner`가 작동했다. 실제 분리기는 출력 사이에 오차를 공유할 수 있다.
repository의 `test_source_support_diagnostics.py`에는 이 projection 반례가 이미
expectedFailure로 남아 있었다. 합성 테스트의 통과는 그 반례를 해결했다는 뜻이 아니었다.

## 4. 확인된 false independent support 조건

저장소 A.wav를 scipy `resample_poly(A, 2, 3)`로 16 kHz 변환한 첫 3초에 실제 GPU
MossFormer2, Silero, faster-whisper small을 실행했다. source 파일과 분리 checkpoint,
저장 fixture의 SHA256 및 package 버전은 fixture JSON에 남겼다.

| 항목 | 실제 A.wav window 0 |
|---|---|
| original/primary STT | 저는 주말마다 새로운 음식을 만들어 보는 것을 |
| secondary RAW STT | 구독과 좋아요 부탁드려요! |
| secondary VAD duration | 1362 ms |
| candidate VAD intervals | [22048, 36832], [40992, 48000] |
| owner_input_correlation | 0.9999644 / 0.9999628 |
| candidate_input_correlation | 0.1183729 / 0.1042420 |
| independent_input_correlation | 0.5271399 / 0.4849988 |
| input_residual_energy_fraction | 0.00007119 / 0.00007445 |
| candidate_residual_energy_fraction | 0.9849223 / 0.9882481 |
| 이전 admission | independent_input_support, suppressed=false |

`contradicted_by_owner`는 independent correlation ≤ 0.20이어야 하므로 작동하지 않았다.
기존 residual-energy gate 및 owner conflict는 **direct correlation ≥ 0.45도 동시에
요구**하므로 이 실제 artifact에 모두 false였다. independent route만으로 승인됐다.
원본 residual에 같은 Silero를 실행하면 speech가 없었다. 수정 후에는 이 추가 증거로
차단됐다. 다시 실행한 fresh MossFormer/Whisper에서도 같은 현상을 확인했다.

## 5. 변경 파일

- `secondary_leakage_diagnostics.py`: input residual speech 측정/검증, admission 결합.
- `run_overlap_pipeline.py`: local 모드도 같은 evidence 생성. remote는 서버 evidence 사용.
- `../phase4/gpu_processing_server.py`: raw slot마다 필요한 residual VAD 측정 및 응답 포함.
- `test_input_residual_speech_admission.py`: 실제 모델 fixture replay 및 보존 조건.
- `../phase4/test_gpu_processing_server.py`: evidence 생성, binary transport, slot mapping.
- `../phase4/test_remote_overlap_pipeline.py`: 실제 fixture를 worker/assembler/state/publishers까지 replay.
- `validate_input_residual_speech.py`: 실제 Silero 재검증 및 선택적 fresh GPU 추론.
- `test_audio/source_admission/*`: 실제 waveform NPZ, provenance/evidence JSON, 12-case 결과 요약.
- 이 보고서.

## 6. 변경한 admission/source-evidence rule

추가 증거 `vad.input_residual_speech`는 candidate가 아니라 **원본에서 다른 출력의
최소제곱 projection을 뺀 waveform**을 검사한다. 3초 전체 context에서 검사하고,
결과 timestamps를 각 candidate VAD interval과 교차한다.

잔차를 RMS 0.1로 정규화한다. 이는 입력 speech 판정의 RMS cutoff가 아니라 gain이다.
낮은 원본 잔차 energy만으로 reject하지 않는다. 기존 Silero 모델과 min speech duration,
상관 thresholds, window/stride는 그대로다. 1e-14는 기존 projection의 수치적 퇴화 경계다.

새 `residual_nonspeech`는 다음이 모두 맞는 interval에서만 true다.

- owner가 해당 원본 구간을 지배한다(기존 correlation ≥ 0.95).
- candidate가 기존 residual correlation route를 통과한다.
- 추가 residual VAD가 정상 계산됐고 해당 candidate interval과 겹치는 speech가 없다.
- residual은 수치적으로 유효하고 candidate normalized residual fraction이 input보다 크다.

모든 candidate interval이 이 모순 또는 기존 unsupported/contradicted 증거로 설명될 때
`discard`, `reason=owner_residual_contains_no_candidate_speech`를 적용한다.
direct/independent support를 함께 false로 바꾸고 transcript를 비운다.
existing partial 및 TENTATIVE confirmation보다 먼저 평가하므로 과거 반복으로 우회하지 못한다.
후반 `independent_input_support` reason도 이제 검증을 거친 independent observation을 쓴다.

새 증거가 없거나, 버전/길이/timestamps가 잘못됐거나, 잔차 계산이 퇴화하거나,
검출기가 실패하면 새로운 차단은 적용하지 않는다. 기존 정책은 유지한다.
명확한 supported interval 하나가 있으면 전체 candidate를 이 새 규칙으로 차단하지 않는다.

## 7. genuine two-speaker 보존

실제 독립 source가 잔차에 남아 speech로 검출되면 추가 모순이 성립하지 않는다.
비지배적인 owner에 대해서도 새 reject를 하지 않는다. raw slot이나 logical speaker ID를
상수로 선택하지 않는다. 실제 A/B fresh MossFormer overlap에서 새 suppression이 없었고,
기존 worker의 두 화자 publication 회귀도 통과했다.

12-case 실제 모델 validator의 genuine cases는 기존 outside-owner hold를 그대로 유지한다.
그 PASS는 새 gate가 source를 삭제하지 않고 independent support를 유지한다는 뜻이며,
첫 창에 두 subtitle이 즉시 publication됐다는 주장이 아니다.

## 8. weak secondary 보존

실제 A/B speech와 정확한 source estimates에서 상대 gain 0.005, 0.001, 0.00001을
검사했고, 전체 입력 gain/polarity -0.01에서도 추가 suppression이 없었다.
owner에 q가 실제보다 크게 누출된 경우에도 입력 잔차에 q의 speech가 남으므로 보존됐다.
이는 admission의 quiet-source 보존 검증이다. MossFormer가 그 작은 source를 언제나
분리해 낸다는 보장은 아니다. STT 모델을 바꾸거나 VAD duration을 늘리지 않았다.

## 9. 추가 회귀 테스트

실제 NPZ fixture에서 기존 false admission을 먼저 확인한 후 새 rejection을 검사한다.
임의 한국어/영어 교체, polarity/gain/logical slot 반전, 반복된 window,
NONE/TENTATIVE/VALIDATED 기존 partial, 정상 interval 하나의 보호,
측정 실패/잘못된 metadata fail-open, 실제 작은 source와 owner crosstalk를 검사했다.

server test는 residual waveform에 실제 정규화가 적용되는지와 binary encode/decode 및
raw→logical 매핑을 검사한다. worker test는 실제 fixture의 raw STT를 보존하면서
assembly의 raw/new가 빈 문자열이고 secondary state/publication/hook/UI entry가 없으며,
primary text가 그대로 publication되는지 확인한다. 기존 w13 weak-tail 회귀도 유지된다.

## 10. 전체 테스트 결과

| 검증 | 결과 |
|---|---|
| phase3 unittest, import-only soundcard bootstrap | 104 실행: 101 PASS, 기존 expectedFailure 3 |
| phase4 unittest (remote/server/client/pipeline) | 52 PASS |
| phase5 unittest, Qt offscreen | 34 PASS |
| 기존 standalone main() regressions | 14/14 PASS |
| 실제 Silero 및 fresh GPU validator | 12/12 PASS |
| phase3/4/5 전체 py_compile | PASS |
| git diff --check | PASS |

일반 phase3 discovery는 변경 전부터 soundcard 의존성/import 문제로 두 모듈에서 막혔다.
고정 dependency를 `/tmp/ghost-test-deps`에 설치한 뒤에도 Linux PulseAudio 연결이 없어
그 import는 실패했다. 장치를 쓰지 않는 회귀에 한해 `sys.modules['soundcard']`를 빈
`ModuleType`으로 등록했다. 테스트/assertion을 바꾸거나 skip하지 않았으며,
실제 장치 함수는 제공하지 않아 장치 접근을 성공으로 위장하지 않았다.
phase4 worker 테스트의 capture/HTTP mocking은 원래 테스트 구조 그대로다.
Windows WASAPI/live/실제 Windows loopback은 실행하지 않았다.

실제 모델 validator 재실행:

```
OPENBLAS_NUM_THREADS=1 python phase3/validate_input_residual_speech.py \
  --fresh-separation --json-output /tmp/source-admission-validation.json
```

## 11. 기존 테스트/xfail 상태

기존 assertion 삭제/완화, skip 추가, expectedFailure 제거는 없다. 기존 세 항목
(bound 밖 non-speech, 추가 speech evidence 없는 projection ambiguity,
unsupported gap 이후 genuine text 가림)은 그대로 expectedFailure다.
이들을 해결했다고 주장하지 않는다. 마지막 항목은 이번 범위 밖이다.

## 12. 남은 한계

Windows w12의 full waveform 및 source checks가 없어 정확히 같은 실패 조건을 검증하지 못했다.
새 규칙은 owner-dominant이고 residual VAD에 speech가 없는 경우를 해결한다.
잔차 자체에도 Silero가 false positive를 내거나 owner dominance가 부족하면 여전히
fail-open/기존 admission 경로로 남을 수 있다. 반대로 잔차의 실제 speech를 Silero가
놓치면 false rejection 위험이 있다. gain 정규화와 genuine/weak regression으로
검증했지만 모든 실제 음성에 대한 보장을 주장하지 않는다.

추가 VAD inference 비용이 있다. owner-dominant candidate speech가 있는 출력에만
계산하며 서버 전체 vad_seconds에 포함한다. 서버와 Windows client를 모두 업데이트하고
재시작해야 한다. 이전 서버 응답에 새 evidence가 없으면 호환성상 fail-open이며,
그 실행으로 새 규칙의 성공을 확인할 수 없다.

## 13. Windows 동일 음원 재검증 JSON

`windows[window==12].secondary_leakage_diagnostic.subtitle_admission.streams`에서
실제 secondary logical speaker의 아래 필드를 확인한다.

```
raw_transcript
input_residual_speech.{version,available,reason,speech_detected,timestamps}
source_support_checks[].{raw_independent_pass,input_residual_speech_overlap,residual_nonspeech,independent_pass}
candidate_independent_support
suppressed / reason
transcript_rejected_after_inference / admitted_transcript
```

실패 시 기존 `speech_local_evidence`, 모든 `source_support_checks`, 양쪽 VAD timestamps,
owner/candidate provenance 및 같은 window의 original/두 separated WAV가 필요하다.
`--diagnostic-audio`로 full-window waveform을 보존할 수 있다.

## 14. 성공/실패 판정

RAW STT가 남아도 secondary admitted text가 비어 있고 post-STT rejection=true이며,
assembly raw/new 및 해당 가짜 문장의 state/publication이 비어야 성공이다.
primary가 유지되고 w13 tail 억제도 유지돼야 한다. 실제 두 화자 확인도 필요하다.
새 evidence가 unavailable이면 새 판정 검증 미완료다. 가짜 문장이 admitted로 남거나
genuine secondary가 새 규칙으로 사라지면 실패다. Linux 12-case PASS는 Windows 성공의 대체가 아니다.

## 15–16. git diff / status

최종 응답에 실제 `git diff --stat`, `git status --short`를 제공한다.
일반 diff에는 untracked 새 테스트/fixture/report가 포함되지 않으므로 별도로 명시한다.
staging, commit, push는 수행하지 않았다.

## WINDOWS RETEST CHECKLIST

1. 새 서버/클라이언트를 재시작하고 동일 30초 WAV + silence B, 3초/2초로 실행한다.
2. w12 secondary: `input_residual_speech.available=true` 확인.
3. `transcript_rejected_after_inference=true`, `admitted_transcript=""` 확인.
4. 그 문장의 assembly raw/new 및 secondary publication이 없고 primary가 유지되는지 확인.
5. w13은 기존 tail rejection을 유지하는지 확인. true secondary 입력도 삭제되지 않아야 한다.

실패/미측정이면 w12 admission 전체와 original/owner/candidate WAV를 보존한다.

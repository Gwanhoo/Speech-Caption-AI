# R2-B3/B4 및 RAW STT 후속 검토

기준 HEAD `745688d`, 시작 작업 트리 clean. **새 운영 수정은 채택하지 않았다.**
앞부분 보존 후보를 FAIL-first로 검증했지만 기존 교정 회귀 2개를 깨뜨려 원복했다.
R2-B6/B7·퍼센트 경계 수정, 3초/2초, single, Whisper 모델/디코딩은 그대로다.
GPU 추론, 모델 다운로드, Windows 라이브, commit/push는 수행하지 않았다.

## 증거의 범위

- `PROJECT_STATUS.md`, `ACCURACY_AUDIT.md`, B6 REPORT/DESIGN/RAW_AND_FINAL,
  `docs/overlap_accuracy_analysis.md`, 이전 퍼센트 검토 문서를 대조했다.
  상태·정확도 문서의 small/분리 중심 설명은 과거 기록이며 현재 single 경로와 구분했다.
- B3/B4 독립 조사 파일은 없다. B6 보고서에도 해당 부재가 명시되어 있다.
- 해당 뉴스의 `phase3/output/evidence/news_20261022_160827/original.wav`와
  `timeline.jsonl`은 없다. 미추적 output/evidence 경로도 검색했다.
- B6 12회 결과 JSON, 원본 timeline에서 추출한 뉴스 fixtures, RAW/VAD/overlap 및
  자막 상태 이벤트는 있다. 저장된 waveform 없이 admission/overlap 판정을 재추론하지 않았다.
- 일반 음성은 있다: `phase1/reference/speaker_a.wav` 5.33초, `speaker_b.wav` 6.31625초
  (16 kHz mono), `phase3/test_audio/A.wav` 37.66046초, `B.wav` 36.39초 (24 kHz mono).
  이 파일들을 문제 뉴스의 원음이나 검수된 정답 전사로 취급하지 않았다.

## 실제 경로와 결함 구분

1. `gpu_processing_server.py:PipelineService._process_serialized`의 single 경로는 원본을
   slot 0으로 전달한다. Silero VAD가 speech로 판정한 경우만 `stt_context.transcribe_base`를
   호출한다. VAD timestamp로 잘라서 STT에 주는 것이 아니라 해당 window 전체를 전달한다.
2. `run_overlap_pipeline.py:stt_worker`가 RAW/VAD를 받아 primary 선택, fragment routing,
   admission, audio overlap 근거, `SubtitleAssembler.process` 순서로 처리한다.
3. assembler rollover는 기존 state를 finalize한다. `SpeakerSubtitleState.process`는
   조립 가설과 source_text/support를 받아 공개 가능한 텍스트를 유지한다.
   `publish_subtitle_state_event`는 publication_text 또는 retraction만 전달한다.
4. VAD silence 및 pipeline_end도 FINAL/discard를 만든다. 빈 discard 이벤트를
   공개 FINAL 수에 포함하면 안 된다.

| 대상 | 현재 재현/판정 |
|---|---|
| `취약점은 최악점은`, `인터넷 전문 인터넷 전반에` | baseline에서 서로 다른 RAW가 append되는 현상은 재현된다. 같은 음성의 대안 가설인지 실제 반복인지 원음·단어 정렬 없이 확정할 수 없어 fuzzy 조건을 넓히지 않았다. |
| `사용된 프로그래밍` → `프레밍` | w15→16 `supported_tail_replace`가 앞 단어까지 지우는 출력은 재현된다. 아래 후보는 이를 보존하지만 다른 교정을 악화시켜 기각했다. |
| 구름/건강 문장의 조기 FINAL | 기존 general lifecycle/production-evidence 테스트가 sentence_head_revision 및 후속 FINAL을 검증한다. 이미 해결된 케이스를 다시 수정하지 않았다. |
| `실행하거나` 잘림, 파급력 앞부분, 퍼센트 중복 | 기존 보호 회귀를 유지했다. 재구현하지 않았다. |
| source-support gap 뒤 발화 숨김 | 기존 expected failure로 재현되지만 unsupported 텍스트 공개 방지와 함께 다뤄야 한다. 이번 기록에는 새 수정의 효과를 판단할 실제 대응 원음이 없어 별도 유지했다. |

## 시도한 변경과 기각 근거

`supported_overlap_tail_anchor`는 공통 anchor가 새 RAW 첫 글자에서 시작할 때만 이전의
관측되지 않은 prefix를 보호한다. anchor가 단어 내부에서 시작하면 기존 전체 tail 교체를 허용한다.

실제 기록:

```text
w15: 사용된 프로그래밍 언어를 식별하며
w16: 프레밍 언어를 식별할 수 있는 웹사이트의 75%
기존 공개: 프레밍 언어를 식별할 수 있는 웹사이트의 75%
후보 공개: 사용된 프레밍 언어를 식별할 수 있는 웹사이트의 75%
```

후보는 anchor가 이전/현재 어절 내부에서 시작해 다음 어절까지 이어질 때, 이전 해당 어절
앞의 단어를 보호했다. `프레밍`을 `프로그래밍`으로 임의 치환하지 않았다.

- 새 테스트 3개: 코드 수정 전 1 PASS / 2 FAIL.
- 후보 적용 후 production-evidence 전체 43개: 41 PASS / 2 FAIL. 새 3개는 PASS.
- 실패한 기존 테스트:
  `ReplacementSafetyRegressions.test_supported_substitution_of_unconfirmed_tail_still_reaches_final`,
  `ReplacementSafetyRegressions.test_supported_tail_revision_cannot_delete_text_before_previous_raw`.
- 반례: `제 손은 그냥과와 높아지는` → `분양가와 높아지는 청약문턱에 지친 세대가`에서
  기존 정상 교정은 `분양가와...`인데 후보가 `제 손은 분양가와...`를 공개/FINAL로 남긴다.
- B6 corpus만 보면 baseline w16–18의 앞부분이 보존되고 나머지 설정은 동일하다.
  그러나 기존 별도 정상 문장 회귀가 악화되므로 corpus 일부 개선만으로 채택하지 않았다.

핵심 불확실성: 현재 근거는 **같은 음성이 겹친다**는 사실을 나타내지만, 이전의 `사용된`은
보존해야 하고 `제 손은`은 교정해야 한다는 단어 단위 대응까지 증명하지 못한다.
특정 단어/길이 예외로 반례를 회피하지 않았다. 후보 코드와 후보의 새 기대 테스트만 원복했고,
기존 테스트 파일과 검증 조건은 시작 HEAD와 동일하다. 실패 기록은 validation JSON에 보존했다.

## RAW STT 원인별 분류와 지연 평가

| 분류 | 현재 확인 가능한 것 | 필요한 증거·개선 가능성·지연 |
|---|---|---|
| VAD 음성 손실 | 12회 저장 결과에서 primary VAD-positive/empty RAW 및 primary RAW 전체 admission 거부는 0건. VAD false가 실제 무음인지는 확인 불가. | 원음과 사람 표시 speech 구간 필요. threshold 변경은 보류. 완화 시 수집 대기는 같아도 STT 호출/queue 지연이 늘 수 있음. |
| Window 문맥 손실 | B6 4/3에서 RAW가 달라지고 반복 연결이 줄어드는 관찰. 원인 전부를 경계 잘림으로 단정할 수 없음. 3/2의 39–40초 tail 미처리는 완전 window 정책이며 STT 누락과 구별. | 동일 원음의 경계/연속 문맥 비교 필요. 4/3은 첫 수집과 갱신 주기가 각각 1초 늘지만 실제 end-to-end 추가 지연은 미검증. 기본값 유지. |
| Whisper 오인식 | `취약점/최악점`, `프로그래밍/프레밍`, `알려줍니다/알려졌습니다` 같은 RAW 차이는 존재. 어느 쪽이 실제 발화인지는 검수 전사 없이 확정 불가. | 대응 원음과 검수 전사 필요. 모델/디코딩 변경·추론 실험은 하지 않음. 지연 변화 산출 불가. |
| Assembler 조립 오류 | 중복 append와 prefix 삭제는 저장 RAW로 재현. 후보 prefix 보호는 다른 문장을 악화. | 음성-단어 대응으로 revision 범위를 제한할 근거가 필요. 기각 후보 자체는 추가 window 대기 없으나 정확도 위험이 더 큼. |
| FINAL 문장 손실 | 현재 baseline 공개 FINAL은 25/33/39초, 모두 3개. A/B에는 별도 빈 candidate_restart/discard가 있으나 공개 FINAL은 각 3개. 기존 일부 조기 확정 회귀는 해결 상태. | 발화 종료/수정 단어 시각과 화면 기록 필요. 한 window 확정을 미루면 조건에 따라 최소 한 stride(운영 2초) 더 늦어질 수 있어 일괄 지연하지 않음. |

이 표의 시간은 stream 좌표 또는 설계상 대기이며 실제 Qt 표시 지연이 아니다.
검수된 정답 전사와 해당 뉴스 원음이 없으므로 CER/WER·음향 정확도 개선율을 산출하지 않았다.

## 최종 검증

```bash
PYTHONPATH=.:experiments/r2_b6:phase3:phase4 .venv/bin/python experiments/r2_b6/run_checks.py
git diff --check
git status --short
```

원복 후 Phase 3 182개 = 179 PASS + 기존 XFAIL 3, Phase 4 67 PASS,
기존 assertion 스크립트 14개 PASS. 실행기에 포함된 B6 unit 2 PASS + 원본 evidence 부재 1 SKIP.
12회/189 window의 assembly/state/publication/FINAL은 시작 HEAD와 동일(연산시간 제외).
B6·원본 음성·기존 fixture 등 49개 파일 해시 동일.
Windows WASAPI/실제 GUI, GPU 추론 및 새 CER/WER 측정은 미실행.

## 다음 증거 수집

우선 기존 B3/B4 조사 원문과 당시 `original.wav`, `windows/`, `timeline.jsonl`, `run.json`,
server health/model 정보를 확보한다. 없으면 아래를 **Windows 저장소 루트**에서 실행한다.
이미 준비된 single/large-v3-turbo 서버를 사용하며 서버를 새로 시작하지 않는다.
이 명령은 이번 작업에서 실행하지 않았다. 실제 캡처 시에는 해당 서버의 추론 비용이 발생한다.

```powershell
$server = "https://YOUR-RUNPOD-8787.proxy.runpod.net"
$run = "phase3\output\evidence\r2_b34_" + (Get-Date -Format "yyyyMMdd_HHmmss")
New-Item -ItemType Directory -Path $run | Out-Null
git rev-parse HEAD | Set-Content "$run\head.txt"
Invoke-RestMethod "$server/healthz" | ConvertTo-Json -Depth 20 | Set-Content -Encoding utf8 "$run\health_before.json"
.\.venv\Scripts\python.exe phase3\run_overlap_pipeline.py `
  --live --duration 61 --processing-mode remote --remote-server-url $server `
  --speaker-mode single --window-seconds 3 --stride-seconds 2 --assemble --websocket `
  --evidence-dir $run --latency-diagnostics --json-output "$run\run.json" `
  2>&1 | Tee-Object -FilePath "$run\console.log"
```

health의 프로필/모델이 맞는지 확인한 뒤 캡처한다. 캡처 준비 메시지 후 같은 뉴스 구간을
재생하고 끝에 여유 무음을 둔다. 정상 종료로 WAV/timeline/JSON 저장을 기다린다.
61초는 3초+29×2초로 완전 window에 맞춘 길이다. 실제 분석은 `original.wav`에 담긴 구간 기준이다.

사람 검수 파일에는 최소 `start_seconds`, `end_seconds`, `verbatim_text`, `uncertain`,
`reviewer`를 기록한다. 특히 다음 구간은 단어 시각/실제 반복/문장 종료를 표시한다:
`취약점/최악점`, `인터넷 전문/전반`, `사용된 프로그래밍/프레밍`, `알려줍니다/알려졌습니다`.
조기 FINAL은 원음과 화면을 함께 녹화하고 이벤트 sequence/utterance_id를 대조해야 한다.
명령의 pipeline_end FINAL은 자연 발화 종료와 별도로 평가한다.

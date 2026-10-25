# R2-B6 window/stride 비교 결과

실측일: 2026-10-25 UTC. 기준 HEAD: `1d306bd49401610201db61254c433d25d9be0fb2`.

## 판단

**품질 비교를 이어갈 우선 후보는 B(4초/3초)다. 운영 설정은 3초/2초를 유지했다.**

B는 이 뉴스에서 기준의 반복 연결 세 구간을 피했고, 전체 STT 처리 합계가 약 27.3% 작았다. 그러나 window 시작 기준 지연은 약 1.02초 증가했고, 이 음원의 첫 공개 자막은 약 2.00초 늦었다. `알려줍니다/알려졌습니다` 등 RAW 차이도 남았다. 따라서 이 단일 40초 음원만으로 운영 기본값 변경이나 인식 정확도 향상을 확정할 근거는 부족하다. 지연을 수용하는 뉴스 용도의 후속 비교에는 B를 추천한다. A·C를 기본값으로 채택할 근거는 찾지 못했다.

정답 전사본과 단어별 음향 정렬이 없다. 아래 평가는 **출력 간 일관성·반복 연결·보존 여부**에 관한 관찰이다. 어느 STT 출력도 사람이 검증한 정답으로 취급하지 않았으며 CER/WER 개선을 주장하지 않는다.

## 실험 유효성

- 원본: FLOAT WAV, mono, 16000 Hz, 640000 samples, 40.000초. timeline의 19개 mixed WAV 모두 원본의 해당 sample slice와 완전히 일치했다.
- 원본 `original.wav`, `timeline.jsonl`, `run.json`, `windows/`를 포함한 136개 파일의 실행 전후 SHA-256이 동일했다. 결과는 `experiments/r2_b6/`에만 작성했다.
- GPU: NVIDIA RTX A5000, 24564 MiB, driver 595.91.07. 이미 실행 중이던 포트 8787 서버를 그대로 사용했다. `single` 프로필, `large-v3-turbo`, CUDA/float16 생성 경로, 언어 ko, beam=5, condition_on_previous_text=False, initial_prompt=None.
- faster-whisper 1.2.1 / CTranslate2 4.8.3 / Silero VAD 6.2.3. 기본 디코딩 함수 signature와 서버 health는 [metadata.json](results/metadata.json)에 보존했다. 모든 후보가 같은 서버 인스턴스를 사용했다. 모델 load count는 1로 유지됐다.
- 4개 geometry에 실제 음성 warm-up 각 1회 후 설정별 3회, 총 측정 window 189개. warm-up 포함 성공 요청 증가량은 예상과 같은 193개, 실패 증가량 0, 외부 요청 혼입 징후 없음. 순서: C/B/기준/A → A/기준/B/C → B/A/C/기준.
- 같은 입력·설정의 RAW와 최종 자막은 3회 모두 동일했다. 반복 실행은 시간 변동과 출력 재현성 점검이며, 독립 음원 3개에 대한 정확도 검증은 아니다.
- 운영 파일/설정 변경, 패키지 설치, 모델 다운로드, RunPod 생성·시작, commit/push를 수행하지 않았다.

## 코드 구조와 비교 범위

`run_overlap_pipeline.py`의 연결은 capture/sample 분할 → audio queue → `RemoteGPUClient` → 서버 VAD/STT → logical speaker tracker → single primary 선택 → fragment routing/source admission → audio overlap 근거 → `SubtitleAssembler` → `SpeakerSubtitleState` → publish/UI다.

`gpu_processing_server.py`의 single 경로는 입력을 raw slot 0에 그대로 전달하고 slot 1을 0으로 채운다. MossFormer2는 로드되지 않았다. Silero VAD 통과 시에만 `stt_context.transcribe_base()`를 호출하며, 전 window 텍스트 prompt는 전달하지 않는다. STT 컨텍스트의 증가는 입력 window 길이로만 이루어진다.

실험은 기존 HTTP client·tracker·primary 선택·routing·admission·overlap·assembler/state 함수를 그대로 사용했다. Assembler minimum=6, sentence lifecycle/shifted head 활성화, final support 활성화, silence=2000 ms를 유지했다. 매 설정/반복마다 로컬 상태를 새로 만들었다. **실제 운영 worker에 동일 파형·RAW·VAD를 주입하는 별도 검증에서 네 설정 모두 전체 assembly/state 이벤트와 종료 flush가 일치했다.** 장치 입력과 HTTP 응답만 기존 테스트 harness로 대체했으며 비교 시 실행시간 필드만 제외했다.

Windows WASAPI capture, bounded queue의 실제 시간 동작, RunPod 외부 proxy, WebSocket/GUI 렌더링은 이번 GPU 측정에 포함하지 않았다. 원본 16 kHz 파형이 GPU로 들어간 이후의 계산·자막 처리 비교다. 세부 설계는 [DESIGN.md](DESIGN.md)를 참조한다.

## 실측 성능과 지연

| 설정 | window/stride | windows/STT 호출 | 미처리 tail | STT 합계/회(s) | 서버 합계/회(s) | 처리 평균/p95(s) | window 시작 기준 지연(s) | 최초 공개 시각(s) |
|---|---|---|---|---|---|---|---|---|
| baseline | 3/2 | 19/15 | 1s | 2.718 | 5.560 | 0.338/0.426 | 3.338 | 11.372 |
| A | 4/2 | 19/16 | 0s | 2.915 | 4.742 | 0.359/0.497 | 4.359 | 12.377 |
| B | 4/3 | 13/11 | 0s | 1.975 | 3.847 | 0.358/0.479 | 4.358 | 13.373 |
| C | 5/3 | 12/10 | 2s | 1.802 | 3.346 | 0.365/0.500 | 5.365 | 11.352 |

- STT/서버 합계는 40초 원본에 대한 **반복 1회 합계의 3회 평균**이다. 서버 합계는 VAD·STT 등을 포함하며 순수 GPU kernel 시간이 아니다. `처리`는 로컬 HTTP 호출 전체와 로컬 자막 처리의 합계이며 capture 대기는 제외한다. 평균/p95는 후보별 전체 window를 합쳐 계산했다.
- `window 시작 기준 지연`과 `최초 공개 시각`은 실측 처리시간을 재생 시간축에 적용한 **추정치**다. `available_i=start_i+window`, `finish_i=max(available_i,finish_previous)+measured_service_i`로 계산했다. 실제 사용자 화면 지연이 아니다. 최초 공개 시각은 음원 0초 기준이며 음성 시작 기준 지연이 아니다.
- 네 설정 모두 가상 queue 대기 최대 0초였다. 이는 이 loopback 측정에서 처리 시간이 stride보다 짧았다는 뜻이며, Windows/proxy 부하에서 drop이 없음을 보증하지 않는다.
- 첫 window를 뺀 새 구간의 가장 오래된 샘플 기준 평균 지연은 기준 2.352초, A 2.374초, B 3.380초, C 3.389초였다. stride=3의 새 내용 갱신 주기는 stride=2보다 1초 길다.
- 첫 자막 시각은 window 길이만으로 단조 증가하지 않았다. C는 6–11초 window의 불완전 RAW를 공개했고, A/B는 6–10초의 `웹사이트`를 tentative로 보류한 뒤 다음 window에서 정상 restart했다. 그 때문에 B의 첫 공개가 13.373초로 늦어졌다.

| 설정 | 전송 window 오디오 합계 | VAD 통과 STT 오디오 합계 | STT 합계 3회 범위 | 기준 대비 STT 합계 |
|---|---:|---:|---:|---:|
| 기준 3/2 | 57초 | 45초 | 2.687–2.745초 | 기준 |
| A 4/2 | 76초 | 64초 | 2.901–2.934초 | 약 +7.2% |
| B 4/3 | 52초 | 44초 | 1.950–1.992초 | 약 −27.3% |
| C 5/3 | 60초 | 50초 | 1.780–1.843초 | 약 −33.7% |

긴 window라고 총 비용이 항상 증가하지 않았다. stride와 VAD 통과 호출 수가 함께 달라지므로 **호출당 비용과 음원 전체 비용을 구분**해야 한다. C는 마지막 2초를 처리하지 않아 비용 우위에 그 차이도 포함된다. 서버 시간에는 CPU VAD와 스케줄링 변동이 포함되므로 적은 반복의 작은 차이를 일반화하지 않는다.

## RAW·조립 결과와 중복/누락 의심

전체 window RAW/새 조각/FINAL은 [RAW_AND_FINAL.md](RAW_AND_FINAL.md), 모든 반복의 행별 데이터는 [windows.csv](windows.csv), 원본 구조 데이터는 [results/](results/)에 있다. 아래 시간은 관련 window들이 덮는 범위이며 단어 정렬 시각이 아니다.

| 관련 구간 | 기준 3/2 | A 4/2 | B 4/3 | C 5/3 |
|---|---|---|---|---|
| 시작 약 6–14초 | `웹사이트 개발의 논리성은 사이트 개발에 널리...`로 연결 | 처음 짧은 `웹사이트`는 hold 후 restart, `웹사이트 개발에 널리 쓰이는...` 공개 | A와 같은 hold/restart, 시작 내용이 다음 RAW에 포함됨 | `웹사이트 개발의 논리 웹사이트 개발에 널리...` 반복 연결 |
| 취약점/로그인 약 12–19초 | `취약점은 최악점은 해커가...` | `취약점은 해커가 로그인 최악점은 해커가 로그인이나...`로 반복 범위 확대 | `취약점은 해커가 로그인이나...`로 연결 | `취약점은 해커가 로그인이나...`로 연결 |
| 실행/결함 약 16–24초 | `실행하거나` 보존, 끝 `알려졌습니다` | `알려줍니다` RAW가 후속 `알려졌습니다`와 조정됨 | `실행하거나` 보존, 끝 `알려줍니다`가 FINAL에 남음 | `실행하거나`와 `알려졌습니다` 출력 |
| 인터넷 약 22–32초 | `인터넷 전문 인터넷 전반에...` 반복 연결 | `인터넷 전반에 광범위하게...` | 같은 표현으로 연결 | 같은 표현으로 연결 |
| 사용된/75% 약 30–38초 | `사용된 프로그래밍...` RAW가 후속 수정 뒤 FINAL에서는 `프레밍...`으로 남고 `75% 75%` 반복 | `사용된 프로그램은 사용된 프레밍...` 반복/수정 | `사용된 프로그래밍 언어를 식별할 수 있는 웹사이트의 75%가...` 연결 | B와 유사한 연결, 38초 이후 미처리 |

B에서 사라진 위 반복 연결은 텍스트 수준의 개선 후보다. 특히 B의 `알려줍니다`와 다른 설정의 `알려졌습니다` 중 어느 쪽이 실제 발화인지는 이 실험에서 확정하지 않았다. 다른 후보에만 존재하는 단어를 B의 누락이라고 자동 판정하지도 않았다.

기준의 `사용된 프로그래밍...` → `프레밍...`과 A의 `프로그래밍` → `프레밍`은 RAW 간 수정이 조립 결과에 영향을 준 보존 검토 대상이다. A/B에서 처음 보류된 `웹사이트`는 다음 공개 RAW에 다시 들어 있어 그 이벤트 자체를 누락으로 세지 않았다. 끝부분 `를 쓰고`가 duplicate-only로 소비된 것도 정상 overlap 제거일 수 있다.

자동 검토 플래그는 기준 6행, A 8행, B 5행, C 2행이었다. 이것은 오류 수나 품질 점수가 아니다. 정상 단어 수정·FINAL rollover도 6글자 비교에 걸리며, `75% 75%` 같은 짧은 반복은 fuzzy 최소 길이 조건에서 빠질 수 있다. C의 시작 구간 반복도 수치만으로 평가하면 놓치기 쉽다. VAD-positive/empty RAW와 RAW 전체 admission 거부 플래그는 네 후보 모두 없었다.

완전 window 정책에 따른 입력 tail 미처리는 기준 39–40초, C 38–40초다. A/B는 40초까지 포함했다. 공통 포함 범위는 0–38초이지만 마지막 window의 추가 context는 서로 다르다. 이 tail 차이를 STT 누락이나 WER 차이로 계산하지 않았다. 모든 마지막 FINAL은 문장 완결을 보증하는 이벤트가 아니라 `pipeline_end` flush다.

## 기존 R2-B3 중복 구간 및 R2-B4 FINAL 경계 대조

이 checkout에는 R2-B3/R2-B4라는 이름의 독립 audit 파일이 없었다. 따라서 해당 이슈의 완전한 재현/해결을 선언하지 않는다. 뉴스 원본 timeline과 기존 `news_20261022_160827_windows.json`, `news_20261022_160827_replacement_windows.json`, production-evidence 회귀 테스트를 근거로 관련 구간을 대조했다.

중복 관련해서 원본 window 6→7의 `취약점은/최악점은`, 12→13의 `인터넷...`, 16→17의 `75%`를 위 표로 별도 검토했다. 현재 기준에서도 반복 연결이 남았고 B에서는 해당 연결이 나타나지 않았다. 기존 window 8→9의 `실행.../실행하거나` 연결과 11→12의 `이번 사태 파급력/사급력` 보존 회귀는 기존 테스트로 확인했다. 이는 window 변경만의 효과와 현재 Assembler 수정 효과를 구분하기 위한 점검이다.

| 입력/설정 | 공개 FINAL 수 | FINAL 발생 음원 시간(s) | 이유 |
|---|---:|---|---|
| 과거 evidence | 6 | 25, 29, 31, 33, 35, 39 | 앞 5개 sentence boundary, 마지막 pipeline_end |
| 현재 기준 3/2 | 3 | 25, 33, 39 | 앞 2개 sentence boundary, 마지막 pipeline_end |
| A 4/2 | 3 | 26, 32, 40 | 동일 |
| B 4/3 | 3 | 25, 34, 40 | 동일 |
| C 5/3 | 3 | 26, 35, 38 | 동일 |

과거에는 `인터넷 정보...`, `쓰이는 핵심.`, `기술이기 때문입니다.`, `식별할 수 있습니다.` 등 각 window의 문장 끝 표현에 따라 FINAL이 나뉘었다. 이번에는 모든 설정에서 3개다. **과거 6→현재 3을 window 확대의 성과로 해석할 수 없다. 현재 3/2 기준부터 이미 3개**이고, 과거 모델 식별과 현재 코드/환경의 완전한 일치를 확인할 정보가 부족하다. FINAL 수 3개가 정답 문장 수임을 의미하지도 않는다.

B의 9–13초 RAW는 `PHP.`로 끝나지만 다음 `언어 PHP에서...`와 `supported_sentence_revision`으로 이어져 그 지점에서 즉시 FINAL로 분리되지 않았다. 인터넷 문장 FINAL 발생 시각은 현재 기준 33초, B 34초, C 35초로 stride/window geometry에 따른 대기 증가가 보인다. 이 시각은 음원상의 마지막 단어 시각이나 실제 UI 수신 시각이 아니다.

## 검증 및 제한

| 검증 | 최종 결과 |
|---|---|
| 실험 unit tests | 3 PASS |
| Phase 3 unittest | 178 실행: 175 PASS, 기존 expected failure 3 |
| Phase 4 Client/Server/worker unittest | 67 PASS |
| 기존 assertion-style 스크립트 | 14 PASS; VAD/silence, STT context, Assembler/state/consensus, diagnostic, WebSocket 등 |
| 실제 운영 worker와 adapter 동등성 | 4설정 PASS; 전체 63 window, assembly/state/flush 비교 |
| GPU 최종 측정 | 12 runs, 189 windows PASS; warm-up 제외 |
| 원본 무결성 | 136파일 SHA-256 동일 |
| 새 Python 파일 syntax/import, 결과 구조/요청수 검증 | PASS |
| 실제 WASAPI/Windows GUI/proxy 포함 end-to-end | SKIP: 현재 Linux 서버 환경, 이번 측정 경로 밖 |
| EXE 패키징/실행 | SKIP: 패키징 변경 없음, Windows 환경 없음 |
| 별도 lint 도구 | SKIP: ruff 미설치; 패키지 설치 없이 syntax/import/공백 검증 수행 |

초기 Phase 3 탐색은 PulseAudio 부재로 2개 import 오류, 첫 headless 통합 실행은 저장소 root import 경로 누락으로 Phase 4 2개 import 오류가 있었다. 테스트 실행기에서 기존 원격 worker 테스트와 같은 soundcard import mock 및 root 경로를 적용한 뒤 전체 재실행이 통과했다. 기존 테스트 파일과 검증 조건은 변경하지 않았다. 실제 장치 테스트를 PASS로 표시하지 않았다.

예비 GPU 실행 중 실험 adapter의 candidate restart 분기 차이를 발견했다. A/B에서 정상 공개를 막던 실험 연결 문제를 운영 코드와 동일하게 수정하고, 전체 worker 동등성 검증 후 모든 후보를 다시 측정했다. `pilot_excluded/`와 `pilot_excluded.log`는 중단된 예비 기록으로 **위 결과에서 전부 제외**했다. 최종 측정은 회귀 테스트 프로세스 종료 후 단독 실행했다.

남은 기존 예상 실패 3개는 source-support의 gap 이후 발화 숨김, 기존 bound 위 noise 공개, projection artifact 공개 사례다. 이 실험에서 운영 로직을 수정하지 않았으므로 해결하지 않았다. 정답 전사·음향 검수 부재, 짧은 단일 음원, 다른 발화/소음/실제 네트워크에서의 일반화 한계도 남는다.

## 재현 및 산출물

```bash
PYTHONPATH=.:experiments/r2_b6:phase3:phase4 .venv/bin/python experiments/r2_b6/run_checks.py
.venv/bin/python experiments/r2_b6/compare_windows.py --output /tmp/r2_b6_new_run --repeats 3
PYTHONPATH=.:experiments/r2_b6:phase3:phase4 .venv/bin/python experiments/r2_b6/verify_pipeline_parity.py experiments/r2_b6/results
.venv/bin/python experiments/r2_b6/summarize_results.py
```

`compare_windows.py`는 이미 존재하는 결과 디렉터리 덮어쓰기를 거부한다. 서버가 준비되지 않았거나 모델·프로필·옵션이 다르면 추론 전에 실패한다. 모델을 새로 로드하거나 원격 자원을 시작하지 않는다. `summarize_results.py`는 이 보관된 `results/`의 고정 3회 실험을 집계한다.

- [설계](DESIGN.md), [측정표](MEASUREMENTS.md), [수치 CSV](metrics.csv), [수치 JSON](metrics.json)
- [RAW/FINAL 검토용 원문](RAW_AND_FINAL.md), [전체 window CSV](windows.csv)
- [최종 GPU 로그](gpu_run.log), [회귀 테스트 로그](verification.log), [운영 동등성 결과](pipeline_parity.json)
- [원본 SHA-256 목록](results/source_hashes_before.json), [실행 후 무결성](results/source_integrity.json)
- [실험 코드](compare_windows.py), [검증 실행기](run_checks.py), [동등성 검증](verify_pipeline_parity.py), [집계 코드](summarize_results.py), [실험 테스트](test_compare_windows.py)

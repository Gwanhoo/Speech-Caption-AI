# H1 철회 및 M1 범위 검증 — 2026-10-31

## 현재 결론

H1을 유발한 새 어절 경계 서술부 정정 분기를 철회했다. `phase3/subtitle_assembler.py`는 기반 HEAD `b53b208fa2e5beabad097c675ff231838f808cb5`와 정확히 같다. H1의 도서관/박물관 두 변형 모두 정상 문장 두 개를 FINAL로 보존한다. **A(W003→W004)의 조기 FINAL과 M1은 미해결**이다. 실제 음성 인식 정확도가 개선됐다고 주장하지 않는다.

시작 작업 트리는 Assembler와 두 테스트 수정, accuracy_20261031 문서 미추적 상태였다. 기존 파일은 보존했고 H1 철회 전 코드·테스트를 이 폴더의 `before_*.txt`로 동결했다. 커밋/push, 운영 서버 중단, 모델 추론/다운로드는 수행하지 않았다. L1 모델 캐시 선택 위험은 이번 H1/M1 범위 밖으로 남긴다.

## 공통 원인과 책임 계층

Assembler는 `shared_speech`, `supported_tail_revision`, 현재/이전 RAW 및 confirmed prefix를 받는다. overlap VAD 교집합·연속 tail coverage·높은 파형 상관은 동일 샘플 구간의 존재를 확인하지만, RAW의 각 단어가 그 구간의 어느 발화에 대응하는지는 알려주지 않는다. single의 연속 슬라이딩 창은 정상 문장 전환에서도 공유 샘플이 일치할 수 있다.

H1의 동일 주제 두 문장과 A의 잘못된 초기 서술부는 **고유한 여러 어절 anchor + 한 어절 서술부 변경**이라는 동일 형태를 가진다. 문장부호 역시 STT 출력이므로 구분 근거로 충분하지 않다. 특정 단어/시제/의미에 대한 예외 규칙, anchor 길이 확대, confidence 임계값 변경은 근거 없는 일반화가 된다. 단어·발화 시간 정렬과 원음 검수, 독립된 정정 관측 등 추가 증거 없이는 안전한 whole-predicate 교체를 입증하지 못했다.

M1은 그 위에 정정 seam이 confirmed prefix 시작점보다 앞이라는 제한이 더해진다. `오늘서울도심에서는` 9자가 확인됐어도 교체 seam은 `서울` 앞(정규화 위치 2)이므로 전체 후보가 거부된다. 실제 변경되는 suffix만 비교하도록 완화하면 M1을 통과시킬 수 있으나 H1의 같은 주제 새 문장도 통과한다. 따라서 이 제한은 완화하지 않았다. State/Publication은 Assembler의 rollover를 받아 정상적으로 FINAL 처리하는 것이며, 새로운 State 결함으로 분류하지 않는다.

## 현재 정정 가능 범위

새 전체 서술부 교체는 제거했다. 기존 exact/fuzzy/terminal/tail 정정과 확정 이력 보호는 HEAD 그대로다. 특히 `reconcile_adjacent_sentence_head_revision`은 다음 기존 조건을 유지한다.

- 인접 창, shared_speech True, 기존 후보가 문장부호로 끝나고 앞선 정합 경로가 처리하지 못한 경우.
- current head의 고유 anchor가 previous 내부에서 일치하고 기본 최소 4자, 추가 내용이 충분함.
- 바뀐 old tail 1–4자, anchor 끝이 양쪽 어절 내부여야 함.
- active suffix 일치 및 seam >= confirmed_prefix_length. seam이 확인된 동일 anchor 안에 있더라도 거부하는 보수적 한계는 남음.
- 이미 FINAL인 이력은 수정하지 않음.

이 조건은 모든 자연 발화의 동일성을 증명하는 규칙이 아니다. 이번 작업은 기존 검증 범위를 유지하면서 새로 도입된 H1 회귀를 제거하는 범위다. 전체 fuzzy threshold, GUI, remote protocol, FINAL 대기 시간은 바꾸지 않았다.

## 재현 및 테스트 계약 변경

1. 코드 수정 전에 H1 도서관/박물관 2개 subcase 실패를 `before.txt`에 기록했다. M1은 prefix_length=9 형성과 잘못된 FINAL 유지까지 재현됐다.
2. 분기만 철회하고 이전 실험의 테스트 기대를 그대로 실행했다. `rollback_original_expectations.txt`의 43 tests에서 5 assertion failures가 발생했다. A 정정, 일반화 2개, 그 정정을 전제로 한 source coverage test, 강한 waveform에서의 whole-predicate 교체 기대가 실패했다. 원본 테스트 snapshot도 보존했다.
3. 사용자 지시에 따라 모호한 whole-predicate 입력의 현재 계약을 **두 관측 보존**으로 바꿨다. 기존 실험의 성공 기대를 삭제 사실 없이 PASS로 보고하지 않는다. A 테스트 이름은 미해결을 명시하고, 두 utterance ID·FINAL/PARTIAL 순서·양쪽 전체 문자열 및 후속 창의 연결까지 검사한다. 일반화 사례도 양쪽 FINAL을 완전히 비교한다.
4. source coverage 테스트는 가상의 정정 결과를 State에 직접 주입해 unsupported copied prefix의 공개를 계속 거부하는지 검사한다. 이 검증을 제거하거나 완화하지 않았다.
5. waveform/VAD 검증은 동일 synthetic overlap을 기존 evidence 함수에 통과시킨 후, 지원 여부와 무관하게 H1 두 문장이 보존되는지 검사한다. VAD가 근거 없음/부분 coverage/낮은 correlation인 대조도 유지했다.
6. M1 테스트는 W002→W003에서 실제로 생성된 confirmed_prefix_length=9를 확인한다. 잘못된 FINAL과 내용 보존을 명시적으로 기대하는 **미해결 상태의 계약 검증**이다. M1 정확도 해결을 의미하지 않는다.

기반 HEAD에 있던 기존 테스트의 기대값은 변경하지 않았다. 새 XFAIL 추가도 없다. 기존 B/C, D 합성 및 recorded 부정/시제, 어미 내부 정정, confirmed seam 경계, short reply, silence/reset, finalized history, fuzzy safety, remote STOP/timeout 회귀를 유지했다.

## 최종 실행 결과

| 검증 | 결과 | 증거 |
|---|---|---|
| Phase3 전체 | 209 실행: **206 PASS, 기존 XFAIL 3** | phase3.txt |
| Phase4 전체 | **88 PASS** | phase4.txt |
| 기존 main-style 회귀 | **9 scripts PASS** | scripts.txt |
| lint E9/F63/F7/F82, Python 3.10 grammar, compile/import, diff whitespace | PASS | static.txt |
| 최종 unexpected FAIL/ERROR | **0** | 위 로그 |

기존 XFAIL은 noise support, projection artifact, unsupported gap 이후 정상 suffix 비공개이며 이번 새 정정 분기와 직접 연결되지 않는다. 해결로 집계하지 않았다. headless 장치 import mock 사용은 기존 runner와 같다.

Windows WASAPI/실제 GUI/EXE, 실제 음성 CER/WER, 원격 end-to-end 지연은 SKIP/미측정: Linux 환경이며 해당 최신 WAV/timeline·검수 정답이 없다. 이번 수정은 CPU 조립 경로 철회이므로 추가 GPU 추론은 수행하지 않았다.

## 전후 출력과 처리 지연

비교의 before는 철회 직전 실험 코드 snapshot, after는 현재 코드다. `compare.py`는 같은 RAW와 합성 shared_speech/tail support를 양쪽에 주입하고 기존 production-order harness로 조립→State→silence FINAL까지 실행한다. H1·변형·A·M1·B·C·D 합성 총 7종이며 실제 acoustic 품질 평가가 아니다.

| 시나리오 | 철회 전 | 현재 |
|---|---|---|
| H1 및 박물관 변형 | 앞 서술부 삭제, 합쳐진 FINAL 1개 | 원문 두 FINAL 전부 보존 |
| A (W003부터, prefix 0) | 합성 조건에서 정정 | 이전 오인식 FINAL + 새 PARTIAL, 미해결 |
| M1 (W002부터, prefix 9) | 이전 오인식 FINAL + 새 PARTIAL | 동일, 미해결 |
| B/C/D 합성 대조 | 정상 문구 | 동일 정상 문구 |

다른 테스트 프로세스 종료 후 warmup 20회, 순서를 교대하며 전후 각 2,100개 시나리오를 측정했다. 하나의 시나리오는 2개 또는 3개 speech 창과 silence FINAL을 포함한다.

- 평균: **0.6934 → 0.6812 ms**
- p50: **0.6911 → 0.6859 ms**
- p95: **1.1629 → 1.1443 ms**

이 작은 차이는 분기 제거와 스케줄링 잡음을 포함한다. 속도 개선의 통계적 증명은 아니며, 이 CPU 재생에서 처리 지연 악화는 관측되지 않았다. post-capture/첫 자막/큐 backlog의 실시간 전후 지표로 해석하지 않는다. 상세 결과는 comparison.json에 있다.

## 남은 검증과 커밋 판단

**CONDITIONAL**: H1 유발 실험 철회, 내용 보존 회귀 및 한계 기록으로는 커밋 가능하다. A/M1 정확도 개선 완료로 커밋하거나 배포 승인하는 근거는 없다. 현재 Assembler의 HEAD 대비 diff는 0이며 최종 tracked diff는 두 테스트의 추가 검증이다. 이전 문서·GPU 비교 실험도 아직 미추적이므로 포함 범위를 검토해야 하고 기존 L1은 미해결이다.

Windows에서는 동일 원본으로 정상 문장 전환·동일 주제 반복·의도적 동일 문장 반복, W002부터의 A/M1, B/C/D, 무음/짧은 발화/동시 발화 및 STOP을 재검증한다. 최신 원본 WAV와 VAD/RAW/confirmed prefix/FINAL trace를 함께 확보해야 동일 발화 정정 근거를 추가로 검토할 수 있다.

기존 accuracy_20261031/REPORT.md에는 이번 철회 상태와 이 보고서 링크를 상단에 추가했다. 과거 실험의 기록은 덮어쓰지 않았다. commit/push는 하지 않았다.

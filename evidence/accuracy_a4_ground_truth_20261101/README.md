# VOXLUX A4 사람 검수 데이터셋

이 폴더는 `evidence/accuracy_a1_a3_20261101/`의 기존 CPU/GPU 결과만 재사용한다. 새 STT 추론이나 운영 서버 호출은 수행하지 않았다. 검수 전 상태이므로 CER/WER는 산출하지 않으며 `review_dataset.json`의 값도 `null`이다.

## 구성

- `review_template.csv`: 사람이 편집할 파일. UTF-8 BOM이 있어 Excel에서도 한글을 바로 열 수 있다.
- `review_dataset.json`: ID, 원음/클립 해시, A1~A3 연결 키, 모델 출력과 선정 근거가 담긴 읽기용 기준 데이터.
- `clips/`: A1~A3와 동일하게 원본 전체를 24kHz에서 16kHz로 변환한 뒤 자른 3초 FLOAT WAV 24개.
- `evaluate_reviewed.py`: 검수 완료 후 Turbo와 Large-v3의 CER/WER를 계산한다.
- `prepare_a4_dataset.py`: 기존 A1~A3 산출물로 이 데이터셋을 재현하는 생성기. 기존 폴더 덮어쓰기를 거부한다.
- `test_a4_dataset.py`: 데이터 무결성과 평가 차단/계산 동작 테스트.

## 정답 입력

`review_template.csv`의 아래 네 필드만 편집한다. 모델 출력은 정답 힌트가 아니라 대조 정보이므로 먼저 오디오를 듣고 정답을 작성한다.

1. `reference_text`: 들리는 발화를 그대로 입력한다. 말더듬·반복·숫자·영문 표기 원칙을 모든 행에 일관되게 적용한다.
2. `review_status`: 검수가 끝난 행은 `pending`에서 `verified`로 변경한다.
3. `reviewer`: 검수자 식별자를 입력한다.
4. `reviewed_at`: ISO 8601 시각을 입력한다. 예: `2026-11-01T12:00:00Z`.

불명확한 부분은 임의로 채우지 말고 `pending`으로 남긴다. `review_notes`에는 재청취 필요, 고유명사 불확실성 등을 적을 수 있다. `sample_id`, 경로, 시각, 두 모델 전사는 평가 시 기준 JSON과 대조하므로 수정하면 평가가 중단된다.

원본 템플릿을 보존하려면 복사본을 만들어 검수한다.

```bash
cp evidence/accuracy_a4_ground_truth_20261101/review_template.csv \
  evidence/accuracy_a4_ground_truth_20261101/review_completed.csv
```

## 평가

24개가 모두 `verified`인 경우에만 기본 실행이 CER/WER를 계산한다.

```bash
python evidence/accuracy_a4_ground_truth_20261101/evaluate_reviewed.py \
  --reviews evidence/accuracy_a4_ground_truth_20261101/review_completed.csv \
  --output evidence/accuracy_a4_ground_truth_20261101/evaluation_results.json
```

일부 행만 검수한 중간 결과가 꼭 필요하면 `--allow-partial`을 명시한다. 출력에는 `evaluation_status=partial`, 검수/미검수 개수가 기록되어 전체 데이터셋 결과로 오해하지 않도록 한다.

CER는 한글·ASCII 영숫자만 유지하고 라틴 문자를 소문자로 바꾼 문자열을 사용한다. WER는 같은 문자 집합의 연속 구간을 어절 토큰으로 사용한다. 숫자 읽기나 띄어쓰기 의미 통합은 하지 않는다. 보고할 기본 집계는 전체 edit 합을 전체 reference unit 수로 나눈 micro CER/WER이며 macro 값도 함께 기록한다.

## 선정 범위

검수 대상은 24개다. A2의 Turbo/Large-v3 기본 조건 출력이 정확히 다른 16개를 모두 우선 포함했다. 나머지 8개는 A3의 3초 beam/prompt 출력 변화, 반복 불안정, 디코더 지연·timestamp 이상 신호를 기준으로 선정했다. 모델 출력이나 더 자연스러워 보이는 문장을 정답으로 간주하지 않았다.

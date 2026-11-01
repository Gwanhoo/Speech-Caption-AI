# VOXLUX A4 정답 데이터셋 준비 결과

검수용 데이터셋 24개를 `evidence/accuracy_a4_ground_truth_20261101/`에 생성했다. 기존 A1~A3 결과와 원본 WAV만 읽었으며 새 모델 추론, GPU 사용, 운영 서버 호출은 없었다. 모든 정답은 비어 있고 CER/WER는 `null`이다.

## 선정 결과

- A2 기본 조건에서 Turbo와 Large-v3의 문자열이 다른 16개 구간 전부 포함
- 두 모델이 같아도 A3 beam/prompt 변화가 크거나 반복 출력·decoder 진단 신호가 있었던 8개 추가
- A.wav 12개, B.wav 12개
- 각 항목은 `A4-{source}-{start_ms}-{end_ms}` 형식의 고유 ID 사용
- 각 클립은 3초, mono 16kHz FLOAT WAV이며 A1~A3 입력 float32 SHA-256과 일치

모델 간 차이가 큰 우선 검수 사례는 다음과 같다. 어느 출력이 맞는지는 원음을 사람이 듣기 전에는 판정하지 않는다.

| ID | Turbo | Large-v3 | 정규화 문자 차이 |
|---|---|---|---:|
| A4-A-010000-013000 | 맛있게 완성되었습니다 요리를 할때는 | 맛있게 완성되었습니다. | 0.3750 |
| A4-B-006000-009000 | 평소보다 날씨가 선선해서 천천히 거울 수 있습니다. | 평소보다 날씨가 선선해서 천천히 | 0.3333 |
| A4-A-024000-027000 | 요리가 끝나면 설거지를 | 요리가 끝나면 설거지를 하겠습니다. | 0.3333 |
| A4-A-028000-031000 | 다음 편하게 쉬려고 합니다. | 다음 편하게 쉬려고 합니다 저녁에는 음 | 0.3125 |
| A4-B-008000-011000 | 천천히 걷기 좋았고 공원에는 운동을 하거든요. | 공원에서 천천히 걷기 좋았고 공원에는 운동을 하거였다. | 0.2609 |

## 검수와 평가

검수자는 `review_template.csv`의 `reference_text`, `review_status`, `reviewer`, `reviewed_at`만 채운다. 완료 행의 상태는 `verified`여야 한다. 불명확한 행은 정답을 추측하지 않고 `pending`으로 남긴다.

기본 평가는 24개 전부 검수될 때만 실행된다.

```bash
python evidence/accuracy_a4_ground_truth_20261101/evaluate_reviewed.py \
  --reviews evidence/accuracy_a4_ground_truth_20261101/review_completed.csv \
  --output evidence/accuracy_a4_ground_truth_20261101/evaluation_results.json
```

상세 입력 규칙, partial 평가 방법, CER/WER 정규화 규칙은 `README.md`에 기록했다.

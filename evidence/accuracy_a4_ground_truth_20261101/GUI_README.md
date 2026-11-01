# VOXLUX A4 Windows 검수 GUI

이 도구는 기존 VOXLUX 실시간 자막 GUI와 독립적으로 실행된다. 운영 서버나 모델을 사용하지 않으며 A4의 저장된 CSV, JSON, WAV만 읽는다.

## 실행

Windows 프로젝트 루트에서 실행한다.

```powershell
py -3.10 evidence\accuracy_a4_ground_truth_20261101\a4_review_gui.py
```

또는 `run_review_gui.bat`를 더블클릭한다. `requirements.txt`의 PySide6 6.8.3 설치가 필요하다.

## 사용 순서

1. 왼쪽 목록에서 클립을 선택하고 재생한다. 필요하면 반복과 0.75x/1.0x/1.25x 속도를 사용한다.
2. 색으로 강조된 Turbo와 Large-v3 차이를 참고하되, 실제 오디오를 기준으로 정답을 입력한다.
3. 모델 결과 복사 버튼은 입력을 돕기만 하며 verified 상태로 바꾸지 않는다.
4. 검수자와 정답을 입력한 뒤 `오디오 확인 완료 — verified`를 누른다. 현재 클립에서 재생 버튼을 누르지 않았다면 verified 처리를 거부한다.
5. 불명확한 클립은 `pending으로 저장`하고 검수 메모를 남긴다.

편집 내용은 항목 이동 시 자동으로 `review_completed.csv`에 저장되며 저장 버튼으로도 즉시 저장할 수 있다. 원본 `review_template.csv`는 수정하지 않는다. 기존 `review_completed.csv`가 있으면 그 내용을 먼저 불러온다. 저장은 같은 폴더의 임시 파일을 완전히 기록한 뒤 `os.replace`로 원자 교체한다.

24개가 모두 verified가 되면 `검수 완료 후 CER/WER 평가` 버튼이 활성화된다. 버튼은 아래와 동등한 실제 평가 명령을 실행한다.

```powershell
py -3.10 evidence\accuracy_a4_ground_truth_20261101\evaluate_reviewed.py `
  --dataset evidence\accuracy_a4_ground_truth_20261101\review_dataset.json `
  --reviews evidence\accuracy_a4_ground_truth_20261101\review_completed.csv `
  --output evidence\accuracy_a4_ground_truth_20261101\evaluation_results.json
```

일부만 검수된 상태에서는 평가 버튼이 비활성화되며 정확도 수치를 생성하지 않는다. 저장 또는 평가 실패는 오류 대화상자로 표시된다.

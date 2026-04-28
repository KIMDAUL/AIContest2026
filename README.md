Kaggle API 사용하기: 터미널에서 kaggle competitions download -c [대회이름] 명령어를 치면 데이터를 즉시 내려받을 수 있습니다. 일일이 브라우저에서 다운로드할 필요가 없죠.

파일 분리: EDA.ipynb(데이터 분석), Modeling.ipynb(모델 학습), Submission.ipynb(제출 파일 생성)로 파일을 나누어 관리하면 훨씬 깔끔합니다.

**환경 복제하기: 팀원들은 이 파일을 받아 똑같은 환경을 구축합니다.**
environment.yml에 정의된 라이브러리대로 가상환경 구현하는 코드임.
conda env create -f environment.yml

data/: 데이터 파일 (용량이 크면 .gitignore에 등록)
notebooks/: 각자 실험하는 주피터 노트북
src/: 공통으로 사용하는 전처리 함수나 모델 정의 (.py 파일)

Branch 전략: 각자 이름으로 브랜치를 파서 작업하고, 점수가 잘 나온 코드는 main 브랜치로 합치는 방식을 추천합니다.

여러 명이 각자 실험하다 보면 "어떤 파라미터가 제일 좋았지?"라며 헷갈리게 됩니다. **WandB(Weights & Biases)**라는 툴을 강력 추천합니다.

기능: 실시간 대시보드에서 팀원들이 돌리는 모델의 손실 함수(Loss), 정확도(Accuracy) 그래프를 한눈에 볼 수 있습니다.

장점: VS Code 노트북 셀 상단에 코드 몇 줄만 추가하면 팀원들의 모든 실험 기록이 하나로 모입니다.

터미널에서 wandb login하고 복사한 API 키 붙여넣기

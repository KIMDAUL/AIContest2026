import wandb

# 1. 프로젝트 시작 선언 (로그인 필요)
wandb.init(project="my-first-kaggle", name="baseline-model")

# 2. 하이퍼파라미터 기록
wandb.config.update({
    "learning_rate": 0.01,
    "epochs": 10,
    "batch_size": 32
})

# 3. 학습 루프 내에서 결과 기록
for epoch in range(10):
    loss = (10 - epoch) * 0.1 # 예시용 가짜 로스
    # 웹 대시보드로 전송!
    wandb.log({"loss": loss, "epoch": epoch})

# 4. 종료
wandb.finish()
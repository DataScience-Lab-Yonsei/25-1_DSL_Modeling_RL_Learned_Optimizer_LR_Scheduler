# TODO
## Utilities
- [x] `wandb` Integration
- [x] Config 파일 만들기
- [x] GPU 사용량 체크
- [x] `TrainingTracker`와 loss 사이의 Connector 클래스 구현<br>
    -> `TrainingTracker`에서 Training, Validation Loss를 추적하도록 설정
- [x] `nohup` Std. IO 출력 폴더 만들기
- [x] Imitation Learning, PPO, Benchmark output 경로 따로 만들기

## Implementation
- [x] 다양한 LR Scheduler, Hyperparameter를 시도하는 Shell Script 파일 만들기
- [x] Policy, Advantage 아키텍처 설계 및 구현
- [x] Dataset Concat 및 로드 구현
- [x] Imitation Learning 코드 구현
- [x] Policy Wrapper(Policy를 감싼 `optim._LRScheduler`) 구현
- [x] PPO 학습 코드 구현
- [x] Reward(Episode 단위), Trainee Training Loss(Step 단위), Trainee Eval Loss(Step 단위), lr(Step 단위) 저장해서 wandb 연결

## Training
- [x] Expert Data Collection
- [x] Reward Modeling
- [x] Training Policy and Advantage with Imitation Learning
- [x] Policy RL with PPO

## Paper & Presentation
- [x] Baseline Scheduler들을 사용했을 시와 Loss 값, 안정도 비교
- [x] Baseline Scheduler들을 사용했을 시와 실제 성능 비교(정성적 & 정량적)

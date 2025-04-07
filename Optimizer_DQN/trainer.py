import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from models import DQN_LSTM, TwoLayerNN
from replay_buffer import ReplayBuffer
from learned_optimizer import Learned_Optimizer
from environment import OptimizerEnvironment
from dataset import get_mnist_projected_dataloader
from config import *
import random
import time
import os
import json
import wandb

class Trainer:
    def __init__(self):
        wandb.init(project="DQN_LSTM_Training")  # WandB 프로젝트 초기화
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN_LSTM(100, ACTION_SIZE, STATE_SIZE).to(self.device)
        self.target_net = DQN_LSTM(100, ACTION_SIZE, STATE_SIZE).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(BUFFER_SIZE)
        self.train_loader = get_mnist_projected_dataloader()
        self.track_data_path = "track_data"
        os.makedirs(self.track_data_path, exist_ok=True)
    
    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        # 모든 텐서를 GPU 또는 CPU로 변환
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        dones = dones.float()
        expected_q_values = rewards + (GAMMA * next_q_values * (1 - dones))
        
        q_values = q_values.to(dtype=torch.float32)
        expected_q_values = expected_q_values.to(dtype=torch.float32)
        loss = F.smooth_l1_loss(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def train(self, episodes=500, epochs_per_episode=10):
        total_rewards_data = []
        for episode in range(episodes):
            model = TwoLayerNN().to(self.device)
            params = list(model.parameters())
            if not params:  
                raise ValueError("Model has no trainable parameters!")
            env = OptimizerEnvironment(model)
            learned_optimizer = Learned_Optimizer(params, self.policy_net)
            state = env.get_state()
            correct = 0
            total_samples = 0
            action_counts = defaultdict(int)  # Action 선택 횟수 (에피소드마다 초기화)
            total_episode_reward = 0  # 총 reward 합산
            episode_data = []  # 각 iteration에서의 reward 기록

            print(f"\n=== Episode {episode+1}/{episodes} 시작 ===")
            episode_start_time = time.time()

            for epoch in range(epochs_per_episode):
                epoch_start_time = time.time()

                for batch_idx, (data, target) in enumerate(self.train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    output = model(data)
                    loss = F.cross_entropy(output, target)
                    learned_optimizer.zero_grad()
                    loss.backward()
                    action = env.select_action(self.policy_net, state)
                    action_counts[action] += 1  # Action 선택 횟수 기록
                    learned_optimizer.step(action)  # 수정된 action 적용
                    
                    next_state = env.get_state()
                    reward = env.get_reward(loss)
                    total_episode_reward += float(reward)  # 총 reward 계산
                    done = batch_idx == len(self.train_loader) - 1         
                    episode_data.append({
                        "iteration": batch_idx + 1,
                        "reward": float(reward),
                        "action": action,
                        "loss": loss.item()
                    })  # iteration별 reward 저장       
                    
                    self.memory.push(state, action, reward, next_state, done)
                    state = next_state
                    
                    self.optimize_model()

                    # WandB에 Value Function 로깅 추가
                    value_function = self.policy_net(state.unsqueeze(0)).cpu().detach().numpy()  # Value Function 추출
                    wandb.log({
                        "Episode": episode + 1,
                        "Epoch": epoch + 1,
                        "Iteration": batch_idx + 1,
                        "Loss": loss.item(),
                        "Reward": float(reward),
                        "Value Function": value_function.mean()  # 평균 Value Function 로깅
                    })

                    # Accuracy 계산
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total_samples += target.size(0)

                epoch_time = time.time() - epoch_start_time
                # 로그 출력
                print(f"Episode {episode+1}, Epoch {epoch+1}/{epochs_per_episode}, Loss: {loss.item():.4f}, Accuracy: {correct / total_samples:.4f}, Time: {epoch_time:.2f}s, Action Counts: {dict(action_counts)}")
                if epoch % TARGET_UPDATE == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

            episode_time = time.time() - episode_start_time
            print(f"=== Episode {episode+1} 종료: 총 시간 {episode_time:.2f}s ===")
            print(f"Total Episode Reward: {total_episode_reward:.4f}")
            print(f"Action Counts: {dict(action_counts)}")
            
            #data_track file 저장
            episode_file = os.path.join(self.track_data_path, f"episode_{episode+1}.json")
            with open(episode_file, "w") as f:
                json.dump(episode_data, f, indent=4)
            
            total_rewards_data.append({
                "episode": episode+1,
                "total_reward": total_episode_reward
            })
            total_rewards_file = os.path.join(self.track_data_path, "total_episode.json")
            with open(total_rewards_file, "w") as f:
                json.dump(total_rewards_data, f, indent=4)

    def plot_rewards(self):
        plt.figure(figsize=(10, 5))
        for ep, rewards in enumerate(self.episode_rewards):
            plt.plot(rewards, label=f"Episode {ep+1}", alpha=0.5)
        plt.xlabel("Iteration")
        plt.ylabel("Reward")
        plt.title("Reward Progression per Iteration")
        plt.legend()
        plt.show()
import torch
import random
import numpy as np
from sklearn.random_projection import GaussianRandomProjection
from collections import deque
from config import ACTION_SIZE, EPSILON, PCA_COMPONENTS

class OptimizerEnvironment:
    def __init__(self, model):
        self.model = model
        self.rp = GaussianRandomProjection(n_components=PCA_COMPONENTS)
        self.prev_losses = deque()  # Moving average of past updates
    
    def get_state(self):
        """현재 MLP의 전체 파라미터와 Gradient를 100차원으로 축소 후 반환."""
        weights = []
        gradients = []
      
        for param in self.model.parameters():
            weights.append(param.data.view(-1))
            if param.grad is not None:
                gradients.append(param.grad.data.view(-1))
            else:
                gradients.append(torch.zeros_like(param.data.view(-1)))
        weights = torch.cat(weights).cpu().numpy()
        gradients = torch.cat(gradients).cpu().numpy()
 
        # RP 변환

        reduced_weights = self.rp.fit_transform(weights.reshape(1, -1)) # 2842 -> 100 축소
        reduced_gradients = self.rp.fit_transform(gradients.reshape(1, -1))        
        moving_avg = np.mean(list(self.prev_losses)) if self.prev_losses else 0

        state = reduced_weights[0]
        return torch.tensor(state, device=self.model.fc1.weight.device, dtype=torch.float32)
    
    def get_reward(self, loss):
        """첨부된 수식에 따라 Reward를 계산."""
        
        if len(self.prev_losses) > 0:
            prev_loss = self.prev_losses[-1]
            reward = - (np.log(loss.item()) - np.log(prev_loss)) / (len(self.prev_losses))
        else:
            reward = 0
        self.prev_losses.append(loss.item())
        return reward
    
    def select_action(self, policy_net, state):
        """DQN_LSTM을 사용하여 action 선택 (어떤 optimizer를 사용할지)."""
        if random.random() < EPSILON:
            return random.randint(0, ACTION_SIZE - 1)  # Exploration (랜덤 선택)
        with torch.no_grad():
            return torch.argmax(policy_net(state)).item()  # Exploitation (DQN이 선택)

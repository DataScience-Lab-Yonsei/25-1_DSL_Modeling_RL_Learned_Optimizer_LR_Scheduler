import random
import torch
from collections import deque

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return torch.stack(states), torch.tensor(actions), torch.tensor(rewards), torch.stack(next_states), torch.tensor(dones)
    
    def __len__(self):
        return len(self.buffer)
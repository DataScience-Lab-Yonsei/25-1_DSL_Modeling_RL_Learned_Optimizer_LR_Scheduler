import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoLayerNN(nn.Module):
    def __init__(self):
        super(TwoLayerNN, self).__init__()
        self.fc1 = nn.Linear(48, 48)
        self.fc2 = nn.Linear(48, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Define DQN_LSTM (LSTM-based)
class DQN_LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, state_size=128):
        super(DQN_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, state_size, batch_first=True)
        self.fc = nn.Linear(state_size, output_dim)
    
    def forward(self, x):
        x, _ = self.lstm(x.unsqueeze(0))
        x = self.fc(x.squeeze(0))
        return x
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from tensordict import TensorDict
from icecream import ic

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float=0., max_len: int=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (- torch.tensor(10000.0).log() / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.to(self.pe.device)
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        return x

class Policy(nn.Module):
    def __init__(self, num_model_layers, window_len=512, dist_rate: float=1., ):
        super(Policy, self).__init__()
        self.num_model_layers = num_model_layers
        self.window_len = window_len
        self.d_model = self.num_model_layers + 4
        self.dist_rate = dist_rate
        self.pe = PositionalEncoding(self.d_model, max_len=512)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=2,
            dim_feedforward=2 * self.d_model,
            dropout=0,
            activation=F.gelu,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, 4)
        self.linear1 = nn.Linear(self.d_model, 1)
    
    def forward(
            self, 
            left_steps_frac, 
            train_loss, 
            eval_loss, 
            layerwise_norms, 
            lr
        ):
        batch_size = train_loss.shape[0]
        left_steps_frac = left_steps_frac.view(batch_size, -1, 1)[:, - self.window_len:, :]
        train_loss = train_loss.view(batch_size, -1, 1)[:, - self.window_len:, :]
        eval_loss = eval_loss.view(batch_size, -1, 1)[:, - self.window_len:, :]
        lr = lr.view(batch_size, -1, 1)[:, - self.window_len:]
        layerwise_norms = layerwise_norms.view(batch_size, -1, self.num_model_layers)[:, - self.window_len:, :]
        input_tensor = torch.cat((left_steps_frac, train_loss, eval_loss, layerwise_norms, lr), dim=-1)
        input_tensor = input_tensor.to(torch.float32)

        out = self.pe(input_tensor)
        #ic(out)
        out = self.encoder(out)
        out = out.mean(dim=-2)
        out = self.linear1(out)
        return out

    def act(
            self, 
            left_steps_frac, 
            train_loss, 
            eval_loss, 
            layerwise_norms, 
            lr, 
            dist_rate: float=1., 
            sampling: Optional[bool]=None
        ):
        if sampling is None:
            sampling = self.training
        
        out = self(
            left_steps_frac, 
            train_loss, 
            eval_loss, 
            layerwise_norms, 
            lr
        )
        if sampling:
            sample = dist.Gamma(- out, dist_rate).sample()
            sample = torch.exp(-sample)
            return sample
        else:
            out = out.exp()
            return out

class Critic(nn.Module):
    def __init__(self, num_model_layers, window_len=512):
        super(Critic, self).__init__()
        self.num_model_layers = num_model_layers
        self.window_len = window_len
        self.d_model = self.num_model_layers + 4
        self.pe = PositionalEncoding(self.d_model, max_len=512)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=2,
            dim_feedforward=2 * self.d_model,
            dropout=0,
            activation=F.gelu,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, 4)
        self.linear1 = nn.Linear(self.d_model, 1)

    def forward(
            self, 
            left_steps_frac, 
            train_loss, 
            eval_loss, 
            layerwise_norms, 
            lr
        ):
        batch_size = train_loss.shape[0]
        left_steps_frac = left_steps_frac.view(batch_size, -1, 1)[:, - self.window_len:, :]
        train_loss = train_loss.view(batch_size, -1, 1)[:, - self.window_len:, :]
        eval_loss = eval_loss.view(batch_size, -1, 1)[:, - self.window_len:, :]
        lr = lr.view(batch_size, -1, 1)[:, - self.window_len:]
        layerwise_norms = layerwise_norms.view(batch_size, -1, self.num_model_layers)[:, - self.window_len:, :]
        input_tensor = torch.cat((left_steps_frac, train_loss, eval_loss, layerwise_norms, lr), dim=-1)
        input_tensor = input_tensor.to(torch.float32)

        out = self.pe(input_tensor)
        out = self.encoder(out)
        out = out.mean(dim=-2)
        out = self.linear1(out)
        return out

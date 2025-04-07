import torch
import torch.optim as optim
import torch.nn as nn
from config import LR

class Learned_Optimizer(optim.Optimizer):
    def __init__(self, params, model): #params : models parametrs, model : LSTM
        self.model = model
        self.lr = LR
        self.optimizers = [
            optim.Adam(params, lr=self.lr),  # Adam
            optim.SGD(params, lr=self.lr, momentum=0.9),  # SGD
            optim.RMSprop(params, lr=self.lr),  # RMSprop
            optim.Adagrad(params, lr=self.lr),  # Adagrad
            optim.Adam(params, lr=self.lr, betas=(0.85, 0.999))  # Custom Adam variant
        ]
        defaults = dict(lr=self.lr)
        super(Learned_Optimizer, self).__init__(params, defaults)

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()
    
    def step(self, action):
        """선택된 action(optimizer)에 따라 모델의 파라미터를 업데이트"""
        self.optimizers[action].step()
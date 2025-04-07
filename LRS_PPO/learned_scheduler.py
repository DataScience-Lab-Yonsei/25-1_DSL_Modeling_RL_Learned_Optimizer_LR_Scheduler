from typing import Optional

import numpy as np
import pandas as pd
from icecream import ic

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from training_tracker import get_layerwise_norms
from lr_actor_critic import Policy

# Policy Scheduler Wrapper
class LearnedLRScheduler(_LRScheduler):
    def __init__(
            self, 
            optimizer: Optimizer,
            policy: Policy,
            tgt_model: nn.Module,
            total_steps: int=-1,
            deterministic: bool = True,
            device: torch.device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        ):
        self.current_step = 0
        super(LearnedLRScheduler, self).__init__(optimizer)
        if total_steps <= 0:
            raise ValueError
        self.device = device
        self.policy = policy.to(device)
        self.tgt_model = tgt_model
        self.total_steps = total_steps
        self.last_lr = self.get_last_lr()
        self.train_loss_lt = []
        self.val_loss_lt = []
        self.norms_lt = []
        self.lr_lt = []
        self.deterministic = deterministic

    def get_status(
            self, 
            train_loss: float, 
            val_loss: Optional[float]=None
        ):
        self.train_loss_lt.append(torch.tensor(train_loss))
        self.val_loss_lt.append(
            torch.tensor(val_loss) if val_loss is not None else torch.zeros(1).squeeze()
        )
        self.norms_lt.append(
            torch.tensor(get_layerwise_norms(self.tgt_model))
        )
        self.lr_lt.append(torch.tensor(self.optimizer.param_groups[0]['lr']))
        self.current_step += 1

    def get_lr(self):
        if self.current_step == 0.:
            return [1e-4]
        left_steps_frac = torch.linspace(
            0, self.current_step, self.current_step, dtype=torch.float32,
        ) / self.total_steps
        left_steps_frac = left_steps_frac.view(1, -1, 1)
        train_loss = torch.stack(self.train_loss_lt).view(1, -1, 1)
        eval_loss = torch.stack(self.val_loss_lt).view(1, -1, 1)
        lr = torch.stack(self.lr_lt).view(1, -1, 1)
        layerwise_norms = torch.stack(self.norms_lt).view(1, -1, self.policy.num_model_layers)
        with torch.no_grad():
            if self.deterministic:
                lr_log = self.policy(
                    left_steps_frac.to(self.device),
                    train_loss.to(self.device),
                    eval_loss.to(self.device),
                    layerwise_norms.to(self.device),
                    lr.to(self.device),
                )
                out_lr = (- lr_log).exp()
            else:
                out_lr = self.policy.act(
                    left_steps_frac.to(self.device), 
                    train_loss.to(self.device), 
                    eval_loss.to(self.device), 
                    layerwise_norms.to(self.device), 
                    lr.to(self.device),
                ).squeeze()
        out_lr = float(out_lr)
        return [out_lr]

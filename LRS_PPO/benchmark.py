import argparse
from pathlib import Path

from tqdm.auto import trange, tqdm
from icecream import ic

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from transformers import get_scheduler

from eval_config import EvalConfig
from custom_ds_model import ProjectedMNIST, MLP
from lr_actor_critic import Policy
from learned_scheduler import LearnedLRScheduler
from env import get_reward_type1, get_reward_type2

baseline_keywords = [
    'linear', 
    'cosine_with_restarts',
    'polynomial',
]

def run_benchmark(
        eval_config: EvalConfig,
        device: torch.device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ):
    reward_lt = []
    train_ds = ProjectedMNIST(train=True)
    test_ds = ProjectedMNIST(train=False)
    train_loader = DataLoader(train_ds, batch_size=eval_config.batch_size)
    test_loader = DataLoader(test_ds, batch_size=eval_config.batch_size)
    trainee = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(trainee.parameters(), lr=eval_config.init_learning_rate)
    max_steps = len(train_loader) * eval_config.num_epochs

    is_baseline = eval_config.scheduler in baseline_keywords
    if is_baseline:
        scheduler = get_scheduler(eval_config.scheduler, optimizer, int(max_steps * 0.2), max_steps)
    else:
        policy = Policy(num_model_layers=2, window_len=512)
        with open(eval_config.scheduler, 'rb') as f:
            policy.load_state_dict(torch.load(f, weights_only=False))
        scheduler = LearnedLRScheduler(optimizer, policy, trainee, max_steps)

    eval_loss = 0.
    training_losses = []
    eval_losses = []
    eval_accs = []
    action_lt = []
    for epoch in trange(eval_config.num_epochs):
        trainee.train()
        for batch in train_loader:
            X = batch['img'] if 'img' in batch else batch['image']
            y = batch['label']
            X = X.float().to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = trainee(X)
            loss = criterion(pred, y)
            loss_value = loss.item()
            loss.backward()
            optimizer.step()
            if not is_baseline:
                scheduler.get_status(loss_value, eval_loss)
                if eval_loss != 0.:
                    eval_loss = 0.
            scheduler.step()
            action_lt.append(float(optimizer.param_groups[0]['lr']))
            training_losses.append(loss_value)
            eval_losses.append(0.)
            reward_lt.append(
                get_reward_type1(
                    torch.tensor(training_losses, dtype=torch.float32).unsqueeze(0),
                    torch.tensor(eval_losses, dtype=torch.float32).unsqueeze(0),
                    eval_config,
                ).squeeze().numpy()
            )

        trainee.eval()
        with torch.no_grad():
            total_loss = 0.
            running_acc = 0.
            count = 0
            for batch in test_loader:
                X = batch["img"] if "img" in batch else batch["image"]
                y = batch["label"]
                X = X.float().to(device)
                y = y.to(device)
                outputs = trainee(X)
                loss = criterion(outputs, y)
                running_acc += (outputs.argmax(dim=-1) == y).sum()
                total_loss += loss.item()
                count += X.shape[0]
            eval_losses[-1] = float(total_loss / len(test_loader)) if count > 0 else 0.
            eval_accs.append(float(running_acc / count) if count > 0 else 0.)
            reward_lt[-1] = get_reward_type1(
                torch.tensor(training_losses, dtype=torch.float32).unsqueeze(0),
                torch.tensor(eval_losses, dtype=torch.float32).unsqueeze(0),
                eval_config,
            ).cpu().numpy()
            reward_lt[-1] = float(reward_lt[-1])
    
    return (
        np.stack(reward_lt), 
        training_losses, 
        eval_losses, 
        eval_accs, 
        action_lt
    )

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    reward_data = dict()
    loss_data = dict()
    eval_acc_data = dict()
    action_data = dict()
    model_path_lt = list(Path('./model').glob('*policy.pt'))
    eval_path = model_path_lt + baseline_keywords
    for scheduler_instance in tqdm(eval_path, leave=False):
        eval_config = EvalConfig(scheduler=scheduler_instance)
        if isinstance(scheduler_instance, Path):
            name = scheduler_instance.name
            name = name[:name.rfind('-')]
        else:
            name = scheduler_instance
        reward, training_loss, eval_loss, eval_acc, action = run_benchmark(eval_config, device)
        reward_data[name] = reward
        loss_data[name + '_train_loss'] = training_loss
        loss_data[name + '_eval_loss'] = eval_loss
        eval_acc_data[name] = eval_acc
        action_data[name] = action
    
    pd.DataFrame(reward_data).to_json('./result/mlp-mnist-reward-new.json', double_precision=15)
    pd.DataFrame(loss_data).to_json('./result/mlp-mnist-loss-new.json', double_precision=15)
    pd.DataFrame(eval_acc_data).to_json('./result/mlp-mnist-acc-new.json', double_precision=15)
    ic(action_data)
    pd.DataFrame(action_data).to_json('./result/mlp-mnist-action-new.json', double_precision=15)

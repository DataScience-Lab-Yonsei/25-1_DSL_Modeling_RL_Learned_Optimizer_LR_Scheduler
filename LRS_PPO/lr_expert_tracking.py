import argparse
from typing import Optional
from dataclasses import dataclass, field

from tqdm.auto import trange
import numpy as np
import pandas as pd
from icecream import ic
ic.disable()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import models
from transformers import get_scheduler

from datasets import Dataset, DatasetDict, load_dataset

from train_configs import ExpertTrackingConfig
from custom_ds_model import ProjectedMNIST, MLP
from utils import modify_to_specific_input_shape
from training_tracker import TrainingTracker

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--index', dest='lr_index', action='store', type=int, default=0)
cmd_args = parser.parse_args()

selected_config = ExpertTrackingConfig.product_idx(cmd_args.lr_index)
tracker = TrainingTracker(selected_config)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

if selected_config.model == 'resnet':
    model = models.resnet18().to(device)
    model = modify_to_specific_input_shape(model, selected_config.input_channel)

    ds = load_dataset(selected_config.dataset_name)
    if isinstance(ds, DatasetDict):
        train_ds = ds['train'].shuffle(seed=42)
        test_ds = ds['test'].shuffle(seed=42)
    else:
        ds = ds.shuffle(seed=42).train_test_split(test_size=0.2, seed=42, shuffle=False)
        train_ds = ds['train']
        test_ds = ds['test']
    train_ds = train_ds.with_format('torch')
    test_ds = test_ds.with_format('torch')
else:
    model = MLP().to(device)
    train_ds = ProjectedMNIST(train=True)
    test_ds = ProjectedMNIST(train=False)

train_loader = DataLoader(train_ds, batch_size=selected_config.batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=selected_config.batch_size, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=selected_config.learning_rate)
scheduler = get_scheduler(
    selected_config.lr_scheduler_type, 
    optimizer,
    num_warmup_steps=selected_config.num_warmup_steps,
    num_training_steps=selected_config.num_training_steps,
)

tracker.on_train_begin()
for epoch in trange(selected_config.num_epoch):
    model = model.train()
    for batch in train_loader:
        if 'img' in batch:
            X = batch['img']
        else:
            X = batch['image']
        y = batch['label']
        X = X.float().to(device)
        y = y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        tracker.on_optimizer_step(model, epoch)
        tracker.on_step_end(optimizer, scheduler)
        tracker.on_log(loss.item())
    
    model = model.eval()
    with torch.no_grad():
        acc_cnt = 0
        loss_sum = 0.
        total = 0
        for batch in test_loader:
            if 'img' in batch:
                X = batch['img']
            else:
                X = batch['image']
            y = batch['label']
            X = X.float().to(device)
            y = y.to(device)

            pred = model(X)
            loss = criterion(pred, y)
            loss_sum += loss.item()
            acc_cnt += (y == pred.argmax(dim=-1).view(-1)).sum()
            total += y.shape[0]
        acc = acc_cnt / total
        loss = loss_sum / len(test_loader)
        tracker.on_evaluate(loss, acc)

tracker.on_train_end()

import argparse
from pathlib import Path
import datetime

from tqdm.auto import trange
import numpy as np
import pandas as pd
from icecream import ic
ic.disable()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

from train_configs import ImitationConfig
from lr_actor_critic import Policy
from utils import load_expert_data, DictDataset

config = ImitationConfig()

if torch.cuda.is_available():
    device = torch.device('cuda:0')
elif torch.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser()
parser.add_argument('-ST', '--scheduler_type', dest='scheduler_type', action='store', type=str, default=config.scheduler_type)
cmd_args = parser.parse_args()
scheduler_type = cmd_args.scheduler_type

ds = load_expert_data(
    config.data_path,
    format='torch', 
    cols=['left_steps_frac', 'layerwise_norms', 'eval_loss', 'train_loss', 'lr'],
    scheduler_type=scheduler_type,
    verbose=True,
)
ds = DictDataset(ds)
trainset, testset = random_split(ds, [0.8, 0.2])
train_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=False)
test_loader = DataLoader(testset, batch_size=config.batch_size, shuffle=False)

policy = Policy(num_model_layers=2).to(device)
criterion = nn.MSELoss()
optimizer = AdamW(policy.parameters(), lr=config.learning_rate)

train_stats = {
    'train_loss': [],
    'eval_loss': [],
}
for epoch in trange(config.num_epochs):
    running_losses = 0.
    running_cnt = 0
    for batch in train_loader:
        batch = {key: batch[key].to(device) for key in batch}
        label = batch.pop('label_lr') + 1e-9
        label = label.log().to(torch.float32)
        out = - policy(**batch)
        optimizer.zero_grad()
        ic(out)
        if torch.isnan(label).any() or torch.isnan(out).any():
            raise ValueError('NaN Error')
        loss = criterion(out, label)
        if torch.isnan(loss).any():
            raise ValueError('NaN Error')
        running_losses += loss.item()
        running_cnt += 1
        loss.backward()
        optimizer.step()
    train_stats['train_loss'].append(running_losses / running_cnt)
    
    with torch.no_grad():
        running_losses = 0.
        running_cnt = 0
        for batch in test_loader:
            batch = {key: batch[key].to(device) for key in batch}
            label = batch.pop('label_lr')
            label = label.log()
            out = policy(**batch)
            loss = criterion(out, label)
            running_losses += loss.item()
            running_cnt += 1
        train_stats['eval_loss'].append(running_losses / running_cnt)

save_path = Path(config.save_path)
now = datetime.datetime.now()
pd.DataFrame(train_stats).to_json(save_path / (now.strftime('%m.%d-%H:%M:%S') + f'-{config.scheduler_type}-train-log.json'))
torch.save(policy.state_dict(), save_path / (now.strftime('%m.%d-%H:%M:%S') + f'-{config.scheduler_type}-policy.pt'))

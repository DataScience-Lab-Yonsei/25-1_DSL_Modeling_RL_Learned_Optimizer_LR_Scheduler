from typing import Union
import random
import random
from collections import deque

import numpy as np
import pandas as pd
from icecream import ic

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.models as models
import torchrl
from torchrl.data import Bounded, UnboundedContinuous, Binary, Unbounded, Composite
from torchrl import envs
from tensordict import TensorDict

from datasets import load_dataset, DatasetDict

from train_configs import PPOConfig
from eval_config import EvalConfig
from custom_ds_model import ProjectedMNIST, MLP
from utils import modify_to_specific_input_shape, is_nan
from training_tracker import get_layerwise_norms

NUM_LAYERS = 2

def total_variation(time_series: torch.Tensor):
    return torch.clamp(time_series[:, 1:] - time_series[:, :-1], min=0.).mean().squeeze()

def get_reward_terms(train_loss: torch.Tensor, eval_loss: torch.Tensor, window=128):
    tv = total_variation(train_loss[-window:])
    train_loss_mean = train_loss[-window:].mean().squeeze()
    val_loss_mean = eval_loss[-window:].sum().squeeze()
    return {
        'TV': tv,
        'train_loss': train_loss_mean,
        'val_loss': val_loss_mean, # TODO : Update
    }

def get_reward_type1(train_loss: torch.Tensor, eval_loss: torch.Tensor, config: Union[PPOConfig, EvalConfig]):
    reward_terms = get_reward_terms(train_loss, eval_loss, config.reward_window)
    tv = reward_terms['TV']
    train_loss_mean = reward_terms['train_loss']
    val_loss_mean = reward_terms['val_loss']
    return - 1. - (
        tv
        + train_loss_mean * config.train_loss_weight
        #+ val_loss_mean * config.val_loss_weight
    ) * config.total_reward_weight

def get_reward_type2(train_loss: torch.Tensor, eval_loss: torch.Tensor, config: Union[PPOConfig, EvalConfig]):
    ic(train_loss.shape)
    return - (
        train_loss[-1].log() - train_loss[-2].log()
    ) if train_loss.shape[0] > 1 else 0.

def get_action(action, config: PPOConfig, to_float=True):
    lr_value = (- action['concentration']).exp().cpu()
    lr_value = config.lr_min + (config.lr_max - config.lr_min) * torch.sigmoid(- action['concentration']).cpu()
    if to_float:
        lr_value = float(lr_value)
    return lr_value

class History:
    def __init__(self, cols, maxlen=512):
        self.cols = list(cols)
        self.maxlen = maxlen
        self.obs_history = {col: deque(maxlen=maxlen) for col in self.cols}
    
    def append(self, new_obs: dict):
        for col in self.cols:
            self.obs_history[col].append(new_obs[col])
    
    def reset(self):
        for col in self.cols:
            self.obs_history[col].clear()
    
    def get(self):
        # If the obs_history is not full yet, pad at the beginning with the first value.
        padded = dict()
        for col in self.cols:
            items = list(self.obs_history[col])
            if len(items) < self.maxlen:
                if items:
                    pad_val = items[0]
                else:
                    pad_val = 0.0 if col != 'layerwise_norms' else [0.] * NUM_LAYERS
                pad_items = [pad_val] * (self.maxlen - len(items))
                items = pad_items + items
            padded[col] = np.stack(items, axis=0).reshape(1, self.maxlen, -1).astype(np.float32)  # shape: (maxlen, feature_dim)
        return padded

class LRSchedulingEnv(envs.EnvBase):
    def __init__(
            self, 
            config: PPOConfig, 
            device: torch.device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        ):
        super(LRSchedulingEnv, self).__init__()
        self.device = device
        self.config = config
        if self.config.model == 'resnet':
            self.trainee = models.resnet18()
            self.trainee = modify_to_specific_input_shape(self.trainee, self.config.input_channel)

            ds = load_dataset(self.config.dataset_name)
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
            self.trainee = MLP()
            train_ds = ProjectedMNIST(train=True)
            test_ds = ProjectedMNIST(train=False)
        
        self.trainee.to(self.device)
        
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.train_loader_base = DataLoader(self.train_ds, batch_size=self.config.batch_size, shuffle=True)
        self.train_loader = iter(enumerate(self.train_loader_base))
        self.test_loader = DataLoader(self.test_ds, batch_size=self.config.batch_size, shuffle=False)
        self.optimizer = Adam(self.trainee.parameters(), lr=self.config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        self.action_spec = Unbounded(
            shape=(1,),
            device=self.device,
            dtype=torch.float32,
        )
        self.done_spec = Binary(1, shape=(1, ), dtype=torch.bool)
        self.observation_spec = Composite(
            {
                "left_steps_frac": Bounded(
                    shape=(512, 1),
                    low=0.0,
                    high=1.0,
                    device=self.device,
                    dtype=torch.float32,
                ),
                "train_loss": UnboundedContinuous(
                    shape=(512, 1), device=self.device, dtype=torch.float32
                ),
                "eval_loss": UnboundedContinuous(
                    shape=(512, 1), device=self.device, dtype=torch.float32
                ),
                "layerwise_norms": UnboundedContinuous(
                    shape=(512, NUM_LAYERS), device=self.device, dtype=torch.float32
                ),
                "lr": Bounded(
                    shape=(512, 1),
                    low=self.config.lr_min,
                    high=self.config.lr_max,
                    device=self.device,
                    dtype=torch.float32,
                ),
            }
        )
        self.reward_spec = Unbounded((1,), dtype=torch.float32)
        self.obs_history = History(self.observation_spec.keys())
        self.current_step = 0
        self.num_epoch_step = len(self.train_loader_base)
        self.num_epochs = random.choice([80, 100, 120, 140])
        self.total_steps = self.num_epochs * self.num_epoch_step

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def get_reward(self, train_loss: torch.Tensor, eval_loss: torch.Tensor):
        #ic(train_loss.mean().item())
        return get_reward_type1(train_loss, eval_loss, self.config)

    def __set_lr(self, lr):
        if is_nan(lr):
            raise ValueError('NaN is not supported')
        for param in self.optimizer.param_groups:
            param['lr'] = lr

    def __eval(self) -> float:
        self.trainee.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in self.test_loader:
                X = batch["img"] if "img" in batch else batch["image"]
                y = batch["label"]
                X = X.float().to(self.device)
                y = y.to(self.device)
                outputs = self.trainee(X)
                loss = self.criterion(outputs, y)
                total_loss += loss.item()
                count += 1
        self.trainee.train()
        return total_loss / count if count > 0 else 0.0

    def _step(self, action):
        lr_value = get_action(action, self.config)
        
        self.trainee.train()
        self.__set_lr(lr_value)
        with torch.enable_grad():
            try:
                _, batch = next(self.train_loader)
            except StopIteration:
                self.train_loader_base = DataLoader(self.train_ds, batch_size=self.config.batch_size, shuffle=True)
                self.train_loader = iter(enumerate(self.train_loader_base))
                _, batch = next(self.train_loader)
            X = batch["img"] if "img" in batch else batch["image"]
            y = batch['label']
            X = X.float().to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.trainee(X)
            loss = self.criterion(pred, y)
            loss_value = loss.item()
            loss.backward()
            self.optimizer.step()
        
        self.current_step += 1
        if (self.current_step % self.num_epoch_step) == 0:
            eval_loss = self.__eval()
        else:
            eval_loss = 0.

        obs = {
            'left_steps_frac': np.array([1 - (self.current_step / self.total_steps)], dtype=np.float32),
            'train_loss': np.array([loss_value], dtype=np.float32),
            'eval_loss': np.array([eval_loss], dtype=np.float32),
            'layerwise_norms': np.array(get_layerwise_norms(self.trainee, model_type='mlp'), dtype=np.float32),
            'lr': np.array([lr_value], dtype=np.float32),
        }
        self.obs_history.append(obs)
        # Create a stacked observation from the obs_history
        stacked_obs = self.obs_history.get()
        # Use the full obs_history for reward calculation if needed
        train_loss_hist = torch.tensor(stacked_obs['train_loss'], dtype=torch.float32)
        eval_loss_hist = torch.tensor(stacked_obs['eval_loss'], dtype=torch.float32)
        reward = self.get_reward(train_loss_hist, eval_loss_hist)
        done = (self.current_step >= self.total_steps)
        obs = (
            TensorDict(stacked_obs, batch_size=[])
                .update({'done': done})
                .update({'reward': reward})
        )
        return obs.to(self.device)
    
    def _reset(self, *args, **kwarg):
        if self.config.model == 'resnet':
            self.trainee = models.resnet18()
            self.trainee = modify_to_specific_input_shape(self.trainee, self.config.input_channel)

            ds = load_dataset(self.config.dataset_name)
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
            self.trainee = MLP()
            train_ds = ProjectedMNIST(train=True)
            test_ds = ProjectedMNIST(train=False)
        
        self.trainee.to(self.device)
        
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.train_loader_base = DataLoader(self.train_ds, batch_size=self.config.batch_size, shuffle=True)
        self.train_loader = iter(enumerate(self.train_loader_base))
        self.test_loader = DataLoader(self.test_ds, batch_size=self.config.batch_size, shuffle=False)
        self.optimizer = Adam(self.trainee.parameters(), lr=self.config.learning_rate)

        self.obs_history.reset()
        self.current_step = 0
        self.num_epoch_step = len(self.train_loader_base)
        self.num_epochs = random.choice([60, 90, 120])
        self.total_steps = self.num_epochs * self.num_epoch_step
        init_obs = TensorDict(self.obs_history.get(), batch_size=[])
        #init_obs = TensorDict({
        #    'left_steps_frac': np.ones((1, 512, 1), dtype=np.float32),
        #    'train_loss': np.zeros((1, 512, 1), dtype=np.float32),
        #    'eval_loss': np.zeros((1, 512, 1), dtype=np.float32),
        #    'layerwise_norms': np.zeros((1, 512, NUM_LAYERS), dtype=np.float32),
        #    'lr': np.full((1, 512, 1), self.config.learning_rate, dtype=np.float32)
        #}, batch_size=[])
        return init_obs.to(self.device)

import math
from pathlib import Path
from typing import Union, Tuple, Literal, List, Optional

from tqdm.auto import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data

from datasets import Dataset
from icecream import ic

def modify_to_specific_input_shape(model, in_channel):
    model.conv1 = nn.Conv2d(
        in_channel,
        64,
        kernel_size=(7, 7),
        stride=2,
        padding=(3, 3),
        bias=False
    ).to(model.conv1.weight.device)
    return model

def create_sliding_windows(data, window_len=512):
    return torch.stack([
        data[i : i + window_len]
        for i in range(data.shape[0] - window_len - 1)
    ])

def compose_dataset(
        df: pd.DataFrame,
        cols: Optional[List[str]]=None,
        label_col: Optional[str]='lr',
        window_len: Optional[int]=512,
    ):
    if cols is not None:
        if label_col not in cols:
            cols.append(label_col)
        df = df[cols]
    data_dict = {
        col: torch.tensor(df[col])
        for col in df.columns
    }
    data_dict = {
        col: data_dict[col].view(-1, 1) if len(data_dict[col].shape) == 1 else data_dict[col]
        for col in data_dict
    }
    data = dict()
    data['label_' + label_col] = data_dict[label_col][window_len + 1:]
    for col in data_dict:
        data[col] = create_sliding_windows(data_dict[col], window_len)
    return data

def load_expert_data(
        dir_path: Union[str, Path], 
        format: Literal['hf', 'torch']='torch',
        cols: Optional[List[str]]=None,
        label_col: Optional[str]='lr',
        window_len: Optional[int]=512,
        scheduler_type: Literal['all', 'cosine_with_restarts', 'linear', 'polynomial']='all',
        verbose=False,
    ) -> Union[Dataset, torch.Tensor]:
    dir_path = Path(dir_path)
    
    meta_lt = [pd.read_pickle(path) for path in tqdm(dir_path.glob('meta-*.pkl'), disable=not verbose)]
    total_steps_lt = [df['num_steps'][0] for df in meta_lt]
    data_lt = [pd.read_pickle(path) for path in tqdm(dir_path.glob('main-*.pkl'), disable=not verbose)]
    if scheduler_type != 'all':
        filtered = list(filter(lambda df_zip: df_zip[0]['scheduler'][0] == scheduler_type and np.isclose(df_zip[0]['lr'][0], 1e-4), zip(meta_lt, total_steps_lt, data_lt)))
        meta_lt, total_steps_lt, data_lt = map(list, zip(*filtered))
    for df, total_steps in zip(data_lt, total_steps_lt):
        df['left_steps_frac'] = 1 - df['current_step'] / total_steps
    data_lt = [compose_dataset(data, cols, label_col, window_len) for data in tqdm(data_lt, disable=not verbose)]
    data = {
        col: torch.cat([data[col] for data in data_lt], dim=0)
        for col in data_lt[0]
    }
    if format == 'hf':
        data = Dataset.from_dict(data)

    return data

def is_nan(obj):
    if isinstance(obj, float):
        return math.isnan(obj)
    elif isinstance(obj, np.ndarray):
        return np.isnan(obj).any()
    elif isinstance(obj, torch.Tensor):
        return torch.isnan(obj).any()
    else:
        raise NotImplementedError

class DictDataset(data.Dataset):
    def __init__(self, data: dict):
        self.data = data
        self.keys = list(self.data.keys())
        self.length = self.data[self.keys[0]].shape[0]
        for key in self.keys[1:]:
            if self.length != self.data[key].shape[0]:
                raise IndexError
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        out = dict()
        for key in self.keys:
            out[key] = self.data[key][index]
        return out
        
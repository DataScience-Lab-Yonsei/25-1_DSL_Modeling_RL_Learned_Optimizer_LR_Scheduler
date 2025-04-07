from dataclasses import dataclass
from typing import Tuple, Union, List, Literal
from itertools import product
from math import ceil

@dataclass
class ExpertTrackingConfig:
    trace_save_path: str = './data'
    with_datetime: bool = True

    model: str = 'mlp'
    dataset_name: str = 'ylecun/mnist'
    learning_rate: float = 4e-4
    lr_scheduler_type: str = 'cosine_with_restarts'
    num_warmup_steps: int = 100
    num_training_steps: int = 200

    batch_size: int = 256
    num_epoch: int = 30

    input_dim: Tuple[int, int] = (224, 224)
    input_channel: int = 1

    @classmethod
    def product_idx(cls, idx: int):
        dataset_lt = [
            {
                'ylecun/mnist': {
                    'shape': (1, 28, 28),
                    'num_samples': 60000,
                }
            },
            #{
            #    'uoft-cs/cifar10': {
            #        'shape': (3, 32, 32),
            #        'num_samples': 50000,
            #    }
            #},
            #{
            #    'uoft-cs/cifar100': {
            #        'shape': (3, 32, 32),
            #        'num_samples': 60000,
            #    }
            #},
        ]
        learning_rate_lt = [
            1e-4,
            #1e-2,
        ]
        scheduler_lt = [
            'linear', 
            'cosine_with_restarts',
            'polynomial',
        ]
        num_epoch_lt = [
            80,
            100,
            120,
            140,
        ] * 4
        prod_config = list(product(num_epoch_lt, dataset_lt, learning_rate_lt, scheduler_lt))
        num_epoch, dataset, learning_rate, scheduler = prod_config[idx]

        dataset_name = list(dataset.keys())[0]
        shape = dataset[dataset_name]['shape']
        input_channel = shape[0]
        input_dim = shape[1:]
        num_samples = dataset[dataset_name]['num_samples']
        lr_scheduler_type = scheduler
        num_warmup_steps = ceil(num_samples / (5 * cls.batch_size))
        num_training_steps = ceil(num_samples / (cls.batch_size))

        return cls(
            trace_save_path=cls.trace_save_path + '-' + cls.model,
            dataset_name=dataset_name,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_epoch=num_epoch,
            input_dim=input_dim,
            input_channel=input_channel,
        )

@dataclass
class ImitationConfig:
    data_path: str = './data-mlp'
    save_path: str = './model'
    batch_size: int = 1024 + 512
    num_epochs: int = 150
    learning_rate: float = 1e-4
    scheduler_type: str = 'polynomial'
    
@dataclass
class PPOConfig:
    save_path: str = './model'
    batch_size: int = 256
    num_epochs: Union[int, List[int]] = 110
    model: str = 'mlp'
    dataset_name: str = 'ylecun/mnist'
    input_channel: int = 1

    learning_rate: float = 1e-4
    num_policy_epochs: int = 5
    max_grad_norm: float = 0.4
    frames_per_batch: int = 2500
    num_episodes: int = 50
    clip_epsilon: float = 0.2
    gamma: float = 1. + 1e-4
    #gamma: float = 1.
    lmbda: float = 0.92
    #entropy_eps: float = 1e-4
    entropy_eps: float = 0.
    replay_batch_size: int = 250
    
    reward_window: int = 512
    lr_min: float = 1e-10
    lr_max: float = 1.
    train_loss_weight: float = 1.5
    val_loss_weight: float = 1.8
    total_reward_weight: float = 2.
    
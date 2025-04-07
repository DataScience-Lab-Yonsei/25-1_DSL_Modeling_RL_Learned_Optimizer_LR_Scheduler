from dataclasses import dataclass

@dataclass
class EvalConfig:
    result_save_path: str = './model'
    scheduler: str = 'linear'
    batch_size: int = 256
    num_epochs: int = 120
    dataset_name: str = 'ylecun/mnist'
    input_channel: int = 1
    init_learning_rate: float = 1e-4
    
    reward_window: int = 512
    lr_min: float = 1e-9
    lr_max: float = 1.
    train_loss_weight: float = 0.5
    val_loss_weight: float = 0.7
    total_reward_weight: float = 1
    gamma: float = 1.

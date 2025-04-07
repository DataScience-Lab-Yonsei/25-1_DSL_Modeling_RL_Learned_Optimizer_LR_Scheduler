import argparse
from pathlib import Path
from typing import Optional
import datetime
from collections import defaultdict

from tqdm.auto import trange, tqdm
import numpy as np
import pandas as pd
from icecream import ic
#ic.disable()

import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torch.distributions import Normal, Gamma, TransformedDistribution, ComposeTransform, AffineTransform, ExpTransform
from torchvision import datasets, models
from transformers import get_scheduler

from tensordict.nn import TensorDictModule, TensorDictSequential, InteractionType
from tensordict.nn.distributions import CompositeDistribution

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from lr_actor_critic import Policy, Critic
from train_configs import PPOConfig
from env import LRSchedulingEnv, get_action

config = PPOConfig()
save_path = Path(config.save_path)
now = datetime.datetime.now()

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

base_env = LRSchedulingEnv(config, device)

class GammaExpDistribution(TransformedDistribution):
    def __init__(self, concentration, rate=1., validate_args=None):
        base_dist = Gamma(concentration, rate, validate_args=validate_args)
        transforms = [
            AffineTransform(loc=0., scale=-1.),
            ExpTransform(),
        ]
        super(GammaExpDistribution, self).__init__(base_dist, transforms, validate_args=validate_args)

# Wrap environment with a transformation (e.g. converting doubles to floats)
#env = TransformedEnv(
#    base_env,
#    transform=Compose(
#        DoubleToFloat(),
#    )
#)

env = base_env

class ActionLogger(TensorDictModule):
    def __init__(self, in_keys: list[str] = ["concentration"]):
        super().__init__(lambda x: x, in_keys=in_keys, out_keys=in_keys)
        self.action_lt = []

    def reset(self):
        self.action_lt = []
    
    def forward(self, tensordict):
        action = tensordict.get("concentration")
        self.action_lt.append(action.detach().cpu().numpy())
        return tensordict

cmd_args.policy_path + cmd_args.output_name

policy_base = Policy(num_model_layers=2, window_len=512)
if cmd_args.policy_path is not None:
    with open(cmd_args.policy_path, 'rb') as f:
        policy_base.load_state_dict(torch.load(f, weights_only=False))

for name, param in policy_base.named_parameters():
    if param.isnan().any():
        print(name)

policy = TensorDictModule(
    policy_base,
    in_keys=['left_steps_frac', 'train_loss', 'eval_loss', 'layerwise_norms', 'lr'],
    out_keys=['concentration'],
)

action_logger = ActionLogger()
policy = TensorDictSequential(
    policy,
    action_logger,
)

actor = ProbabilisticActor(
    policy,
    spec=env.action_spec,
    in_keys=['concentration'],
    out_keys=['action'],
    distribution_class=Gamma,
    distribution_kwargs={
        "rate": 0.5,
    },
    return_log_prob=True,
    default_interaction_type=InteractionType.DETERMINISTIC,
).to(device)

critic_base = Critic(num_model_layers=2, window_len=512)
critic = TensorDictModule(
    critic_base,
    in_keys=['left_steps_frac', 'train_loss', 'eval_loss', 'layerwise_norms', 'lr'],
    out_keys=['state_value'],
).to(device)

if config.frames_per_batch > 0:
    frames_per_batch = config.frames_per_batch
else:
    frames_per_batch = config.num_episodes * env.total_steps

# Setup data collector from torchrl
collector = SyncDataCollector(
    env, 
    policy, 
    frames_per_batch=frames_per_batch, 
    total_frames=config.num_episodes * env.total_steps, 
    device=device,
)

ic(config.num_episodes * env.total_steps)

# Replay buffer for storing trajectories
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)

# Advantage estimation using GAE
advantage_module = GAE(
    gamma=config.gamma,
    lmbda=config.lmbda,
    value_network=critic,
    average_gae=True,
).to(device)

# PPO loss module
loss_module = ClipPPOLoss(
    actor,
    critic,
    clip_epsilon=config.clip_epsilon,
    entropy_bonus=bool(config.entropy_eps),
    entropy_coef=config.entropy_eps,
    loss_critic_type='smooth_l1',
    normalize_advantage=True,
).to(device)
loss_module.set_keys(sample_log_prob="action")
loss_module.set_keys(reward='reward')
loss_module.set_keys(done='done')

optimizer = AdamW(loss_module.parameters(), lr=config.learning_rate)
scheduler = CosineAnnealingLR(optimizer, T_max=(config.num_episodes * env.total_steps) // frames_per_batch)

logs = defaultdict(list)
collector.rollout()

pbar = tqdm(total=len(collector))
ic(len(collector))
ic(config.num_episodes * env.total_steps)
ic(frames_per_batch)

for batch_idx, tensordict_data in enumerate(collector):
    # Process each episode in the batch
    for _ in range(config.num_policy_epochs):
        #action_logger.reset()
        tensordict_data.to(device)
        advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        for _ in range(frames_per_batch // config.replay_batch_size):
            subdata = replay_buffer.sample(config.replay_batch_size)
            #for key, value in subdata.items():
            #    if value.isnan().any():
            #        ic(key)
            #    else:
            #        print(f'{key} is not NaN')
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"] 
                + (loss_vals["loss_entropy"] if bool(config.entropy_eps) else 0.)
            )

            #for name, param in loss_module.named_parameters():
            #    if param.isnan().any():
            #        ic(name)
            #    else:
            #        print(f'{name} is not NaN')
            #
            #ic(loss_vals["loss_objective"])
            #ic(loss_vals["loss_critic"])
            #ic(loss_vals["loss_entropy"])
            #print('-' * 100)
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
    scheduler.step()
#        current_action = np.concatenate(action_logger.action_lt, axis=-1)
#        actions.append(current_action)
#    full_action = np.concatenate(actions, axis=-1)
#    ic(full_action.shape)
#    action_logger.reset()
    
    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    pbar.update()
    cum_reward_str = f"average reward={logs['reward'][-1]:.4f} (init={logs['reward'][0]:.4f})"
    mean_action = f"average action={get_action(tensordict_data, env.config, to_float=False).mean():1.3e}"
    if float(logs['reward'][-1]) > 6. and False:
        #np.save(save_path / (now.strftime('%m.%d-%H:%M:%S') + f"-{logs['reward'][-1]:.4f}-action.npy"), action)
        torch.save(policy_base.state_dict(), save_path / (now.strftime('%m.%d-%H:%M:%S') + f"-{logs['reward'][-1]:.4f}-ppo-policy.pt"))
        torch.save(critic_base.state_dict(), save_path / (now.strftime('%m.%d-%H:%M:%S') + f"-{logs['reward'][-1]:.4f}-ppo-critic.pt"))
    logs["left_steps_frac"].append(tensordict_data["left_steps_frac"].max().item())
    #logs["action_lr"].append((- tensordict_data['concentration']).exp().item())
    #lr_str = f"lr action: {logs['action_lr'][-1]: 4.4f}"
    
    if False: # batch_idx % 100 == 0 and batch_idx != 0:
        # Evaluate the policy every 10 batches using a deterministic rollout
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            eval_rollout = env.rollout(env.total_steps, policy)
            eval_reward = eval_rollout["next", "reward"].mean().item()
            eval_reward_sum = eval_rollout["next", "reward"].sum().item()
            eval_step = eval_rollout["left_steps_frac"].max().item()
            logs["eval reward"].append(eval_reward)
            logs["eval reward (sum)"].append(eval_reward_sum)
            eval_str = (
                f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
            )
            del eval_rollout
    else:
        logs["eval reward"].append(0.)
        logs["eval reward (sum)"].append(0.)
        eval_str = ''

    pbar.set_description(", ".join([mean_action, cum_reward_str]))
    
print(logs)
save_path = Path(config.save_path)
now = datetime.datetime.now()
pd.DataFrame(logs).to_json(save_path / (now.strftime('%m.%d-%H:%M:%S') + f'-{cmd_args.output_name}-ppo-train-log.json'))
torch.save(policy_base.state_dict(), save_path / (now.strftime('%m.%d-%H:%M:%S') + f'-{cmd_args.output_name}-final-ppo-policy.pt'))
torch.save(critic_base.state_dict(), save_path / (now.strftime('%m.%d-%H:%M:%S') + f'-{cmd_args.output_name}-final-ppo-critic.pt'))

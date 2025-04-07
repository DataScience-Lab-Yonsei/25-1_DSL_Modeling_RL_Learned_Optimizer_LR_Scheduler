import datetime
from pathlib import Path
from math import sqrt
from typing import Literal

from icecream import ic
#ic.disable()

import numpy as np
import pandas as pd

from train_configs import ExpertTrackingConfig

def get_layerwise_norms(model, model_type: Literal['resnet', 'mlp']='resnet'):
    if model_type == 'resnet':
        module_lt = list(model.children())
        layer0_norm = 0.
        for module in module_lt[:2]:
            for param in module.parameters():
                if param.requires_grad:
                    layer0_norm += (param.grad.detach() ** 2).sum()
        layer0_norm = float(layer0_norm.sqrt().cpu())

        layer1_4_norms = []
        for module in module_lt[4:-2]:
            for block in module:
                norm = 0.
                for param in module.parameters():
                    if param.requires_grad:
                        norm += (param.grad.detach() ** 2).sum()
                layer1_4_norms.append(float(sqrt(norm)))

        head_norm = float(
            (
                (module_lt[-1].weight.grad.detach() ** 2).sum()
                + (module_lt[-1].bias.grad.detach() ** 2).sum()
            )
            .sqrt()
            .cpu()
        )
        layerwise_norms = [layer0_norm] + layer1_4_norms + [head_norm]
    else:
        layerwise_norms = [
            (
                (layer.weight.grad.detach() ** 2).sum()
                + (layer.bias.grad.detach() ** 2).sum()
            )
            .sqrt()
            .cpu()
            for layer in (model.fc1, model.fc2)
        ]
    return layerwise_norms

class TrainingTracker:
    def __init__(self, config: ExpertTrackingConfig):
        self.config = config
        self.save_path = Path(self.config.trace_save_path)
        self.with_datetime = self.config.with_datetime
    
    def on_train_begin(self):
        ic('Train begin')
        self.trace_data = []
        self.meta_data = {
            'lr': [self.config.learning_rate],
            'scheduler': [self.config.lr_scheduler_type],
            'batch_size': [self.config.batch_size],
            'num_epoch': [self.config.num_epoch],
            'num_warmup_steps': [self.config.num_warmup_steps],
            'num_training_steps': [self.config.num_training_steps],
            'dataset': [self.config.dataset_name],
        }
        self.num_steps = 0

    def on_optimizer_step(self, model, epoch):
        ic('Optim step')

        layerwise_norms = get_layerwise_norms(model)

        self.trace_data.append({
            'current_epoch': epoch,
            'layerwise_norms' : layerwise_norms,
            'current_step': self.num_steps,
            'eval_loss': 0.,
            'acc': 0.
        })
    
    def on_step_end(self, optimizer, lr_scheduler):
        self.num_steps += 1
        ic('step end')
        self.trace_data[-1]['lr'] = optimizer.param_groups[0]['lr']
        ic(self.trace_data[-1])

    def on_log(self, loss=None):
        ic('on log')
        if loss is not None:
            self.trace_data[-1]['train_loss'] = loss
        ic(self.trace_data[-1])

    def on_evaluate(self, eval_loss=None, eval_acc=None):
        ic('on eval')
        if eval_loss is not None:
            self.trace_data[-1]['eval_loss'] = eval_loss
        if eval_acc is not None:
            self.trace_data[-1]['eval_acc'] = eval_loss
        ic(self.trace_data[-1])

    def on_train_end(self):
        ic('Train end')
        self.meta_data['num_steps'] = self.num_steps
        self.meta_data = pd.DataFrame(self.meta_data)
        self.trace_data = pd.DataFrame(self.trace_data)

        if self.with_datetime:
            now = datetime.datetime.now()
            self.meta_data.to_pickle(self.save_path / f"meta-{now.strftime('%m.%d-%H:%M:%S')}.pkl")
            self.trace_data.to_pickle(self.save_path / f"main-{now.strftime('%m.%d-%H:%M:%S')}.pkl")
        else:
            self.meta_data.to_pickle(self.save_path / 'meta.pkl')
            self.trace_data.to_pickle(self.save_path / 'main.pkl')

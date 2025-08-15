from typing import Optional
from torch.optim.optimizer import Optimizer
import math
import copy
import numpy as np

class StepwiseLR:

    def __init__(self, optimizer: Optimizer, total_epoch=100, warm_up_iter=0, init_lr: Optional[float] = 0.01,
                 gamma: Optional[float] = 0.001, decay_rate: Optional[float] = 0.75, momentum: Optional[float] = 0.9, weight_decay: Optional[float] = 1e-3, pretrained_flag=False):
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.iter_num = 1
        self.total_epoch = total_epoch
        self.warm_up_iter = warm_up_iter
        self.pretrained_flag = pretrained_flag
        self.backup = copy.deepcopy(optimizer.param_groups)

    def step(self, num_len=1):
        if self.iter_num > self.warm_up_iter * num_len:
            iter1 = max(0, self.iter_num - self.warm_up_iter * num_len)+ 1
            iter2 = max(1, (self.total_epoch - self.warm_up_iter) * num_len)
            progress = (iter1 - 1) / iter2
            lr = max(0.0, 0.5 * (1. + math.cos(math.pi * progress)))
            if self.optimizer:
                for index, param_group in enumerate(self.optimizer.param_groups):
                    pr_grop = self.backup[index]
                    param0 = pr_grop['lr']
                    param_group['weight_decay'] = self.weight_decay
                    param_group['lr'] = param0 * lr
                    param_group['nesterov'] = True
        else:
            ni = self.iter_num
            if ni <= self.warm_up_iter*num_len:
                xi = [0, self.warm_up_iter*num_len]
                if self.optimizer:
                    for index, param_group in enumerate(self.optimizer.param_groups):
                        pr_grop = self.backup[index]
                        lr = pr_grop['lr']
                        rz_lr = np.interp(ni, xi, [0, lr])
                        param_group['lr'] = rz_lr
                        if 'momentum' in param_group:
                            param_group['momentum'] = np.interp(ni, xi, [0, 5e-4])
        self.iter_num += 1
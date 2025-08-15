import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
import math
class MicroCommunity(nn.Module):
    def __init__(self, num_classes, dimension, gar=None):
        super(MicroCommunity, self).__init__()
        if gar is not None:
            self.gar = gar
        else:
            self.gar = 0.2
        mult_class = int(1.0 / self.gar) + 1
        print(f'MultipleCenterMemory mult_class: {mult_class}, gar: {gar:.4f}')
        self.dimension = dimension
        self.mult_class = mult_class
        self.num_classes = num_classes
        self.num_centers_classes = mult_class * num_classes
        # Parameters
        self.micro_community = nn.Parameter(torch.randn((self.num_centers_classes, dimension)))
        self.LSM = nn.Sequential(
            nn.Linear(num_classes, mult_class, bias=True),
            nn.ReLU(),
            nn.Linear(mult_class, mult_class, bias=True),
            nn.Sigmoid(),
        )
        # Label Matrix
        self.center_labels = torch.arange(self.num_centers_classes, dtype=torch.long)
        self.hot_matrix = torch.eye(self.num_centers_classes, dtype=torch.long)
        # K: label from micro-comunnity to class
        self.K = torch.zeros((self.num_centers_classes, self.num_classes), dtype=torch.long)
        for i in range(num_classes):
            self.K[i * mult_class:(i + 1) * mult_class, i] = 1

        self.memory = torch.zeros((self.num_centers_classes, dimension))
        self.memory_weights = torch.zeros((self.num_centers_classes, 1))

    def get_MSML_loss(self, data, source_labels, beta, frozen=False):
        device = data.device
        bs = data.size(0)
        if bs > 0:
            centers = self.micro_community.to(device)
            if frozen:
                centers = centers.detach()
                centers.requires_grad = False
            weights, sum_v = self.get_weights_from_hotmatrix(beta, source_labels)
            center_matrix = []
            for i in range(bs):
                idx_start = source_labels[i] * self.mult_class
                idx_end = idx_start + self.mult_class
                centers_lab = centers[idx_start:idx_end]
                weight = weights[i].view(-1, 1)
                centers_weight_sum = centers_lab * weight
                center_matrix.append(centers_weight_sum.sum(dim=0))
            center_matrix_tensor = torch.stack(center_matrix).to(device)
            loss = F.mse_loss(data, center_matrix_tensor)
            return loss, sum_v
        else:
            return 0.0, 0.0

    def get_weights_from_hotmatrix(self, beta, label):
        device = beta.device
        eps = 1e-4
        hot = F.one_hot(label, self.num_classes).float().to(device)
        out = self.LSM(hot)
        out = out + eps
        column_sum = out.cumsum(dim=1)  
        val = (beta.view(-1, 1) - column_sum) ** 2
        weights = torch.exp(-torch.sqrt(val + 1e-10))
        sum_weights = torch.sum(weights, dim=1, keepdim=True) + eps
        norm_weights = weights / (sum_weights + 1e-10)
        return norm_weights, column_sum

    def init_center(self, is_end=False):
        if not is_end:
            self.memory = torch.zeros((self.num_centers_classes, self.dimension))
            self.memory_weights = torch.zeros((self.num_centers_classes, 1))
            print('init center memory')
        else:
            count = 0
            for i in range(self.num_centers_classes):
                weights = self.memory_weights[i]
                if weights > 0.0:
                    count = count + 1
                    self.memory[i] = self.memory[i] / (weights+1e-10)
            print(f'update center memory, valid: {count}/{self.num_centers_classes}')
            return self.memory

    def update_center_with_memory(self, memory):
        self.micro_community.data = memory.clone()
        print(f'update center with memory <- {self.micro_community.shape}...')

    def update_center(self, data, source_labels, beta):
        device = data.device
        if data.size(0) > 0:
            self.memory = self.memory.to(device)
            self.memory_weights = self.memory_weights.to(device)
            bs = data.size(0)
            weights, sum_v = self.get_weights_from_hotmatrix(beta, source_labels)
            for i in range(bs):
                feat = data[i]
                idx_start = source_labels[i] * self.mult_class
                idx_end = idx_start + self.mult_class
                weight = weights[i].view(-1, 1)
                feat_cent = feat * weight
                self.memory[idx_start:idx_end] = self.memory[idx_start:idx_end] + feat_cent
                self.memory_weights[idx_start:idx_end] = self.memory_weights[idx_start:idx_end] + weight


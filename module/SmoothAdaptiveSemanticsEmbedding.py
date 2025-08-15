import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

class SmoothAdaptiveSemanticsEmbedding(nn.Module):

    def __init__(self, num_class):
        super(SmoothAdaptiveSemanticsEmbedding, self).__init__()
        self.metric = 'euci'
        self.softmax = nn.Softmax(dim=1)
        self.rho = nn.Parameter(torch.tensor(0.5))
        self.rho_list = nn.Parameter(torch.ones(num_class)*0.5)

    def forward(self, source, target, source_pred, target_pred, device, k_in=None):
        if target is None:
            target = source
            target_pred = source_pred
            self_connected = True
        else:
            self_connected = False
        ns = source.size(0)
        nt = target.size(0)
        distances = self.get_dist(source, target).to(device)
        semantic_distances = self.get_dist(source_pred, target_pred).to(device)
        beta = []
        dist1 = []
        dist2 = []
        for i in range(ns):
            row_distances = distances[i]
            row_labels = semantic_distances[i]
            avg_distance_tensor = row_distances.mean()
            avg_label_tensor = row_labels.float().mean()
            sorted_dists_tensor, dist_indices = torch.sort(row_distances)
            sorted_labs_tensor = row_labels[dist_indices]
            distance_B = sorted_dists_tensor - avg_distance_tensor
            distance_A = sorted_labs_tensor - avg_label_tensor
            sorted_ratio = distance_B / (distance_A+1e-4)
            bs = distance_A.size(0)
            # get adaptive k
            large_list = sorted_ratio > 0.0
            idx_list = torch.nonzero(large_list).squeeze()
            k = self.get_index_from_list(idx_list, bs - 1, self_connected=self_connected)
            sort_target = target[dist_indices, :]
            positive_sam = sort_target[:k+1, :].mean(dim=0)
            negative_sam = sort_target[k+1:, :].mean(dim=0)
            beta.append(sorted_ratio[k])
            dist1.append(positive_sam)
            dist2.append(negative_sam)
        beta_tensor = torch.stack(beta).view(-1)
        dist1_tesnor = torch.stack(dist1).view(ns, -1)
        dist2_tesnor = torch.stack(dist2).view(ns, -1)
        return dist1_tesnor, dist2_tesnor, beta_tensor

    def get_dist(self, source, target):
        if self.metric == 'cosine':
            source, target = normalize(source, target)
            target_transport = target.transpose(-2, -1)
            logits = 1.0 - source @ target_transport
            return logits
        else:
            dist = torch.cdist(source, target)
            return dist

    def get_index_from_list(self, flag, bs, self_connected):
        if flag.numel() == 0:
            return bs
        else:
            sorted_indices, _ = torch.sort(flag)
            if self_connected:
                second_min_index = sorted_indices[1] if sorted_indices.numel() > 1 else bs
            else:
                try:
                    second_min_index = sorted_indices[0]
                except:
                    second_min_index = sorted_indices.item()
            return min(second_min_index, bs)



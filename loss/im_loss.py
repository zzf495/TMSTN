import torch
import torch.nn as nn

class InformationMaximization(nn.Module):

    def forward(self, logits):
        softmax_out = nn.Softmax(dim=1)(logits)
        entropy_loss = torch.mean(self.entropy_loss(softmax_out))
        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
        loss = entropy_loss - gentropy_loss
        return loss

    def entropy_loss(self, input_):
        bs = input_.size(0)
        epsilon = 1e-5
        entropy = -input_ * torch.log(input_ + epsilon)
        entropy = torch.sum(entropy, dim=1)
        return entropy
# https://github.com/mbsariyildiz/focal-loss.pytorch/blob/master/focalloss.py

import torch
import torch.nn as nn
        
class FocalLoss(nn.Module):

    def __init__(self, smooth=1e-5, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smooth = smooth
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        pt = input
        logpt = (pt + 1e-5).log()

        # add label smoothing
        num_class = input.shape[1]
        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != input.device:
            one_hot_key = one_hot_key.to(input.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
            logpt = logpt * one_hot_key

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = (-1 * (1 - pt)**self.gamma * logpt).sum(1)
        if self.size_average: return loss.mean()
        else: return loss.sum()
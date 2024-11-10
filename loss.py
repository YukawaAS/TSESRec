import torch
from torch import nn
from torch.nn import functional as F


class BPRLoss(nn.Module):
    def __init__(self, padd_idx):
        super().__init__()
        self.padd_idx = padd_idx

    def forward(self, preds, targets):
        preds = preds[:, :-1, :]                        # 序列前len-1个
        masks = targets.view(-1) != self.padd_idx       # Mask
        sampled_logit = preds.reshape(-1, preds.shape[-1])[masks, :][:, targets.view(-1)[masks]]  # 取每个序列中每个位置，正确index对应的logit

        # differences between the item scores
        diff = sampled_logit.diag().view(-1, 1).expand_as(sampled_logit) - sampled_logit
        # final loss
        loss = -torch.mean(F.logsigmoid(diff))
        return loss


class BPR_max(nn.Module):
    def __init__(self, padd_idx):
        super().__init__()
        self.padd_idx = padd_idx

    def forward(self, preds, targets):
        preds = preds[:, :-1, :]                        # 序列前len-1个
        masks = targets.view(-1) != self.padd_idx       # Mask
        sampled_logit = preds.reshape(-1, preds.shape[-1])[masks, :][:, targets.view(-1)[masks]]  # 取每个序列中每个位置，正确index对应的logit
        logit_softmax = F.softmax(sampled_logit, dim=1)
        diff = sampled_logit.diag().view(-1, 1).expand_as(sampled_logit) - sampled_logit
        loss = -torch.log(torch.mean(logit_softmax * torch.sigmoid(diff)))
        return loss


class TOP1Loss(nn.Module):
    def __init__(self, padd_idx):
        super(TOP1Loss, self).__init__()
        self.padd_idx = padd_idx

    def forward(self, preds, targets):
        preds = preds[:, :-1, :]                        # 序列前len-1个
        masks = targets.view(-1) != self.padd_idx       # Mask
        sampled_logit = preds.reshape(-1, preds.shape[-1])[masks, :][:, targets.view(-1)[masks]]  # 取每个序列中每个位置，正确index对应的logit
        diff = -(sampled_logit.diag().view(-1, 1).expand_as(sampled_logit) - sampled_logit)
        loss = torch.sigmoid(diff).mean() + torch.sigmoid(sampled_logit ** 2).mean()
        return loss


class TOP1_max(nn.Module):
    def __init__(self, padd_idx):
        super(TOP1_max, self).__init__()
        self.padd_idx = padd_idx

    def forward(self, preds, targets):
        preds = preds[:, :-1, :]                        # 序列前len-1个
        masks = targets.view(-1) != self.padd_idx       # Mask
        logit = preds.reshape(-1, preds.shape[-1])[masks, :][:, targets.view(-1)[masks]]
        logit_softmax = F.softmax(logit, dim=1)
        diff = -(logit.diag().view(-1, 1).expand_as(logit) - logit)  # 将当前batch中的其他item作为负样本
        loss = torch.mean(logit_softmax * (torch.sigmoid(diff) + torch.sigmoid(logit ** 2)))  # 原文2式9
        return loss


class MarginLoss(nn.Module):
    def __init__(self, padd_idx, margin=1.0):
        super().__init__()
        self.padd_idx = padd_idx
        self.margin = margin

    def forward(self, preds, targets):
        preds = preds[:, :-1, :]                        # 序列前len-1个
        masks = targets.view(-1) != self.padd_idx       # Mask
        sampled_logit = preds.reshape(-1, preds.shape[-1])[masks, :][:, targets.view(-1)[masks]]  # 取每个序列中每个位置，正确index对应的logit

        diff = self.margin-(sampled_logit.diag().view(-1, 1).expand_as(sampled_logit) - sampled_logit)
        return F.relu(diff).mean()

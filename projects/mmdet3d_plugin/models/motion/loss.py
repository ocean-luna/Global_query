import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class GMMNLLWithModeLoss(nn.Module):
    """
    输入：[B, T, 6]  每帧已是 Top-1 分量 (μx,μy,σx,σy,ρ,π_logit)
    输出：scalar NLL
    """
    def __init__(self, loss_weight=1, min_std=1e-4, max_rho=0.95):
        super().__init__()
        self.loss_weight = loss_weight
        self.min_std = min_std
        self.max_rho = max_rho

    def forward(self, top1_params, target, weight=None, avg_factor=None):
        # 解析
        target = target.cumsum(dim=-2)
        mean = top1_params[..., :2]                 # [B, T, 2]
        mean = mean.cumsum(dim=-2)
        std  = top1_params[..., 2:4].clamp_min(self.min_std)
        rho  = top1_params[..., 4].tanh() * self.max_rho

        # 残差
        dx = target[..., 0] - mean[..., 0]
        dy = target[..., 1] - mean[..., 1]
        sx, sy = std[..., 0], std[..., 1]

        # 单高斯 NLL
        z = (dx/sx)**2 + (dy/sy)**2 - 2*rho*(dx*dy)/(sx*sy)
        denom = 1 - rho**2
        nll_loss = 0.5 * z / denom + torch.tensor(2 * torch.pi) + torch.log(sx) + torch.log(sy) + \
              0.5 * torch.log(denom)
        if weight is not None:
            nll_loss = nll_loss * weight.squeeze(dim=-1)
        nll_loss = nll_loss.sum()
        if avg_factor is not None:
            nll_loss = nll_loss / nll_loss
        else:
            nll_loss = nll_loss / (weight.sum() + 1e-6)
        return nll_loss * self.loss_weight
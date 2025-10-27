import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class LaplaceNLLLoss(nn.Module):
    """
    输入：[B, T, 6]  每帧已是 Top-1 分量 (μx,μy,σx,σy,ρ,π_logit)
    输出：scalar NLL
    """
    def __init__(self, loss_weight = 1, eps = 1e-2, max_scale = 4e1):
        super().__init__()
        self.loss_weight = loss_weight
        self.eps = eps
        self.max_scale = max_scale

    def forward(self, top1_params, target, weight=None, avg_factor=None):
        # 解析
        target = target.cumsum(dim=-2)
        mean = top1_params[..., :2]                 # [B, T, 2]
        mean = mean.cumsum(dim=-2)
        scale  = top1_params[..., 2:4].clamp_min(self.eps).clamp_max(self.max_scale)
        nll_loss = torch.log(2 * scale) + torch.abs(target - mean) / scale
        if weight is not None:
            nll_loss = nll_loss * weight
        nll_loss = nll_loss.sum()
        if avg_factor is not None:
            nll_loss = nll_loss / avg_factor
        else:
            nll_loss = nll_loss / (weight.sum() + 1e-6)
        return nll_loss * self.loss_weight
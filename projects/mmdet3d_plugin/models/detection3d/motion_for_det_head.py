from typing import List, Optional, Tuple, Union
import warnings
import copy

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.utils import build_from_cfg
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import BaseModule, force_fp32
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)
from mmdet.core import reduce_mean
from mmdet.models import HEADS
from mmdet.core.bbox.builder import BBOX_SAMPLERS, BBOX_CODERS
from mmdet.models import build_loss

class QueryDiffusion(nn.Module):
    """
    扩散模型：专门针对 (B, H, n_query, d) 的 det query
    条件ego-action (B, 1, D)
    """
    def __init__(self,
                 H: int = 6,
                 n_query: int = 900,
                 d: int = 256,
                 n_layers: int = 1,
                 n_heads: int = 8,
                 diffusion_steps: int = 100):
        super().__init__()
        self.H, self.n_query, self.d = H, n_query, d
        self.T = diffusion_steps

        # ---------- 1 噪声调度 ----------
        alpha = torch.linspace(0.99, 0.01, diffusion_steps)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_cumprod", torch.cumprod(alpha, 0))

        # ---------- 2 条件编码 ----------
        self.time_embed = nn.Sequential(
            nn.Embedding(diffusion_steps, d),
            nn.SiLU(), nn.Linear(d, d))

        # ---------- 3 DiT 降噪器 ----------
        self.pos_enc = nn.Parameter(torch.randn(H * n_query, d))
        layer = nn.TransformerEncoderLayer(d_model=d,
                                           nhead=n_heads,
                                           dim_feedforward=4*d,
                                           batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Linear(d, d)  # 预测噪声 ε

    def _flat(self, q): return q.reshape(q.size(0), self.H*self.n_query, self.d)
    def _unflat(self, q): return q.reshape(q.size(0), self.H, self.n_query, self.d)

    @torch.no_grad()
    def add_noise(self, q0: torch.Tensor, t: torch.Tensor = None):
        """
        加噪：给定干净 q0，返回任意 t 时刻的 q_t
        q0: [B, H, n_query, d]  真值 query
        t:  [B] int64，None 则随机采样
        return q_t, noise, t
        """
        B = q0.size(0)
        device = q0.device
        if t is None:
            t = torch.randint(0, self.T, (B,), device=device)
        noise = torch.randn_like(q0)
        q0_flat = self._flat(q0)
        noise_flat = self._flat(noise)
        sqrt_alpha = self.alpha_cumprod[t].sqrt().view(B, 1, 1)
        sqrt_1ma = (1 - self.alpha_cumprod[t]).sqrt().view(B, 1, 1)
        q_t_flat = sqrt_alpha * q0_flat + sqrt_1ma * noise_flat
        q_t = self._unflat(q_t_flat)
        return q_t, noise, t

    def denoise(self, q_t: torch.Tensor, t: torch.Tensor, ego_feature: torch.Tensor):
        """
        去噪：给定 q_t, t, action → 预测噪声 ε
        q_t:     [B, H, n_query, d]
        t:       [B] int64
        action:  [B, d]
        return:  ε_pred  与 q_t 同 shape
        """
        B = q_t.size(0)
        q_flat = self._flat(q_t)  # [B, H*n_query, d]
        # 条件向量
        c = self.time_embed(t) + ego_feature  # [B, d]
        h = q_flat + self.pos_enc.unsqueeze(0)              # 位置编码
        # 把条件拼到序列最前面作为全局 token（简单实现）
        h = torch.cat([c.unsqueeze(1), h], dim=1)           # [B, 1+H*n_query, d]
        h = self.transformer(h)[:, 1:]                      # 去掉条件 token
        eps_flat = self.out(h)
        return self._unflat(eps_flat)

    def loss(self, q0_future: torch.Tensor, ego_feature: torch.Tensor):
        """
        训练损失：一次函数完成加噪 + 去噪 + MSE
        """
        q_t, noise, t = self.add_noise(q0_future)      # 随机加噪
        eps_pred = self.denoise(q_t, t, ego_feature)        # 预测噪声
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample(self, ego_feature: torch.Tensor, steps: int = 20):
        """
        采样：DDIM 一键生成未来 H×n_query×d
        ego_feature: [B, d]  未来 ego-action
        return: [B, H, n_query, d]  生成的干净 query
        """
        device = ego_feature.device
        B = ego_feature.size(0)
        q = torch.randn(B, self.H, self.n_query, self.d, device=device)
        step_list = torch.linspace(self.T - 1, 0, steps).long().to(device)
        for i, t in enumerate(step_list):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            eps = self.denoise(q, t_batch, ego_feature)
            alpha_t = self.alpha[t]
            q = (q - eps * (1 - alpha_t).sqrt()) / alpha_t.sqrt()
            if i < len(step_list) - 1:
                beta_t = 1 - alpha_t
                q += beta_t.sqrt() * torch.randn_like(q) * 0.2  # DDIM 随机
        return q

class FusionQueryModel(BaseModule):
    def __init__(self, dropout = 0.1, hidden_dim = 256, H = 6):
        super(FusionQueryModel, self).__init__()
        input_channel = 512
        output_channel = 256
        self.query_mlp = nn.Sequential(
            nn.Linear(input_channel, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, H * output_channel)
        )
        self.H = H
    
    def forward(self, det_query, motion_query):
        # det_query torch.Size([3, 900, 256])
        # motion_query torch.Size([3, 900, 256])
        bs, num_query, channel = det_query.shape
        merge_query = torch.concat([det_query, motion_query], dim = -1)
        output = self.query_mlp(merge_query)
        output = output.reshape(bs, num_query, self.H, channel)
        return output


###### Historical Mot2Det Fusion ######
@HEADS.register_module()
class MotionforDetHead(BaseModule):
    def __init__(
        self,
        norm_layer: dict,
        ffn: dict,
        refine_layer: dict,
        decouple_attn=False,
        embed_dims=256,
        temp_graph_model=None,
        temp_graph_model_no_decouple_attn=None,
        operation_order: Optional[List[str]] = None,
        init_cfg: dict = None,
        **kwargs,
    ):
        super(MotionforDetHead, self).__init__(init_cfg)
        
        self.decouple_attn = decouple_attn
        self.operation_order = operation_order
        self.embed_dims = embed_dims
        self.op_config_map = {
            "temp_gnn": [temp_graph_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "refine": [refine_layer, PLUGIN_LAYERS],
        }
        
        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)
        
        self.layers_interact = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )

        if self.decouple_attn:
            self.fc_before = nn.Linear(
                self.embed_dims, self.embed_dims * 2, bias=False
            )
            self.fc_after = nn.Linear(
                self.embed_dims * 2, self.embed_dims, bias=False
            )
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()
        
        self.fusion_query_model = FusionQueryModel()
        self.query_diffusion = QueryDiffusion()
    
    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers_interact[i] is None:
                continue
            elif op != "refine":
                for p in self.layers_interact[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(
            self.layers_interact[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        )
    
    def forward(
        self,
        det_output,  
        feature_maps,
        metas,
        det_head_anchor_encoder,  # SparseBox3DEncoder
        instance_bank,  # InstanceBank
        det_head_instance_bank_mask,  # None or torch.Size([3])
        det_head_instance_bank_anchor_handler,  # SparseBox3DKeyPointsGenerator
        motion_plan_head_instance_queue=None,  # InstanceQueue
    ):

        classification = det_output['classification']  # len=6 [torch.Size([3, 900, 10])]
        prediction = det_output['prediction']  # len=6 [torch.Size([3, 900, 11])]
        quality = det_output['quality']  # len=6 [torch.Size([3, 900, 2])]
        instance_feature = det_output['instance_feature']  # torch.Size([3, 900, 256])
        anchor_embed = det_output['anchor_embed']  # torch.Size([3, 900, 256])
        time_interval = det_output["time_interval"]  # eg: tensor([0.5000, 0.5000, 0.5000], device='cuda:0')
        anchor = det_output["anchor"]  # torch.Size([3, 900, 11])

        bs, num_anchor, dim = instance_feature.shape  # torch.Size([3, 900, 256])

        instance_queue_get = None
        if motion_plan_head_instance_queue is not None:
            (
                ego_feature,  # torch.Size([3, 1, 256])
                ego_anchor,  # torch.Size([3, 1, 11])
                temp_instance_feature,  # torch.Size([3, 901, 1, 256])
                temp_anchor,  # torch.Size([3, 901, 1, 11])
                temp_mask,  # torch.Size([3, 901, 1])
            ) = motion_plan_head_instance_queue.get(
                det_output,
                feature_maps,
                metas,
                bs,
                det_head_instance_bank_mask,
                det_head_instance_bank_anchor_handler,
            )

            instance_queue_get = (
                ego_feature,  # torch.Size([3, 1, 256])
                ego_anchor,  # torch.Size([3, 1, 11])
                temp_instance_feature,  # torch.Size([3, 901, 1, 256])
                temp_anchor,  # torch.Size([3, 901, 1, 11])
                temp_mask,  # torch.Size([3, 901, 1])
            )

            # ego_query_feature torch.Size([3, 1, 256])
            # motion_query_feature torch.Size([3, 900, 256])
            # motion_query_feature, ego_query_feature = temp_instance_feature[:, :-1, 0], temp_instance_feature[:, -1]
            # ego_query_feature = ego_query_feature.squeeze(1)
            motion_query_feature = temp_instance_feature[:, :-1, -1]
            ego_query_feature = ego_feature.squeeze(1)

        fusion_query = self.fusion_query_model(instance_feature, motion_query_feature) # torch.Size([3, 900, 6, 256])
        fusion_query = fusion_query.transpose(1, 2)
        feature_diffusion = self.query_diffusion.sample(ego_query_feature)
        feature_diffusion = feature_diffusion + fusion_query
        feature_diffusion = feature_diffusion.transpose(1, 2)

        temp_motion_embed_forstate = anchor_embed.unsqueeze(-2).repeat(
            1, 1, 6, 1).detach()  # torch.Size([3, 900, 3, 256])
        
        output = {}

        instance_feature = instance_feature.unsqueeze(-2).reshape(
            -1, 1, dim
        )  # torch.Size([2700, 1, 256])
        motion_query_diffusion = feature_diffusion.reshape(
            -1, feature_diffusion.size(-2), dim
        )  # torch.Size([2700, 6, 256])
        temp_motion_embed_forstate = temp_motion_embed_forstate.reshape(
            -1, temp_motion_embed_forstate.size(-2), dim
        )  # torch.Size([2700, 6, 256])


        for i, op in enumerate(self.operation_order):
            if self.layers_interact[i] is None:
                continue
            elif op == "temp_gnn":  
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    motion_query_diffusion,
                    motion_query_diffusion,
                    query_pos=anchor_embed.flatten(0, 1).unsqueeze(1),
                    key_pos=temp_motion_embed_forstate,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers_interact[i](instance_feature)
            elif op == "refine":
                instance_feature = instance_feature.squeeze(1).reshape(bs, num_anchor, dim)
                anchor, cls, qt = self.layers_interact[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    time_interval=time_interval,
                    return_cls=True,
                )
                prediction.append(anchor)
                classification.append(cls)
                quality.append(qt)

                anchor_embed = det_head_anchor_encoder(anchor)
        
        output.update(
            {
                "classification": classification,
                "prediction": prediction,
                "quality": quality,
                "instance_feature": instance_feature,
                "anchor_embed": anchor_embed,
                "instance_id": det_output["instance_id"],
                "fusion_query": fusion_query,
                "ego_query_feature": ego_query_feature,
            }
        )

        return output, instance_queue_get

    def loss(self, model_outs):
        fusion_query = model_outs["fusion_query"]
        ego_query_feature = model_outs["ego_query_feature"]
        diff_loss = self.query_diffusion.loss(fusion_query, ego_query_feature)
        return {
            "diff_loss": diff_loss,
        }


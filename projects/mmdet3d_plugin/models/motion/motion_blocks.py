import torch
import torch.nn as nn
import numpy as np

from mmcv.cnn import Linear, Scale, bias_init_with_prob
from mmcv.runner.base_module import Sequential, BaseModule
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.registry import (
    PLUGIN_LAYERS,
)

from projects.mmdet3d_plugin.core.box3d import *
from ..blocks import linear_relu_ln
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import ATTENTION

class MotionPlanningTeacherModule(BaseModule):
    def __init__(self, embed_dims=256, ego_fut_ts=6, ego_fut_mode=6):
        super(MotionPlanningTeacherModule, self).__init__()
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.plan_status_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 10),
        )
        multi_config = dict(
            type="MultiheadAttention",
            embed_dims=256,
            num_heads=8,
            batch_first=True,
            dropout=0.1,
        )

        self.motion_input_proj = nn.Linear((256 * 6), 256)
        self.plan_input_proj = nn.Linear((256 * 18), 256)
        self.plan_encoder = build_from_cfg(multi_config, ATTENTION)
        self.plan_decoder = nn.Sequential(
            nn.Linear(268, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 2 + 1),
        )
    
    def forward(self, motion_query, plan_query, plan_reg, ego_feature, ego_anchor_embed):
        bs, num_obs, num_motion_tgt, motion_channel = motion_query.shape
        bs, num_ego, num_plan_tgt, plan_channel = plan_query.shape
        motion_query = motion_query.reshape(bs, num_obs, num_motion_tgt * motion_channel)
        motion_query = self.motion_input_proj(motion_query)
        plan_query = plan_query.reshape(bs, num_ego, num_plan_tgt * plan_channel)
        plan_query = self.plan_input_proj(plan_query)
        plan_attn_output = self.plan_encoder(plan_query, key=motion_query)
        plan_attn_output = plan_attn_output.repeat(1, self.ego_fut_mode * 3, 1)
        plan_reg = plan_reg.squeeze(1)
        plan_reg = plan_reg.reshape(bs, self.ego_fut_mode * 3, -1)
        plan_decode_input = torch.cat([plan_attn_output, plan_reg], dim=-1)
        plan_decode_output = self.plan_decoder(plan_decode_input)
        plan_cls, plan_reg_offset = plan_decode_output[:, :, -1], plan_decode_output[:, :, :-1]
        plan_reg_offset = plan_reg_offset.reshape(bs, 1, self.ego_fut_mode * 3, self.ego_fut_ts, -1)
        plan_cls = plan_cls.unsqueeze(1)
        planning_status = self.plan_status_branch(ego_feature + ego_anchor_embed)
        return plan_cls, plan_reg_offset, planning_status


class MotionPlanningStudentModule(BaseModule):

    def __init__(self, embed_dims=256, ego_fut_ts=6, ego_fut_mode=6):
        super(MotionPlanningStudentModule, self).__init__()
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 2),
        )

    def forward(self, plan_query):
        plan_reg = self.plan_reg_branch(plan_query)
        return plan_reg


@PLUGIN_LAYERS.register_module()
class MotionPlanningRefinementModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=6,
        ego_fut_mode=3,
    ):
        super(MotionPlanningRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

        self.motion_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.motion_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, fut_ts * 2),
        )
        # self.plan_cls_branch = nn.Sequential(
        #     *linear_relu_ln(embed_dims, 1, 2),
        #     Linear(embed_dims, 1),
        # )
        # self.plan_reg_branch = nn.Sequential(
        #     nn.Linear(embed_dims, embed_dims),
        #     nn.ReLU(),
        #     nn.Linear(embed_dims, embed_dims),
        #     nn.ReLU(),
        #     nn.Linear(embed_dims, ego_fut_ts * 2),
        # )
        # self.plan_status_branch = nn.Sequential(
        #     nn.Linear(embed_dims, embed_dims),
        #     nn.ReLU(),
        #     nn.Linear(embed_dims, embed_dims),
        #     nn.ReLU(),
        #     nn.Linear(embed_dims, 10),
        # )
        self.student_model = MotionPlanningStudentModule()
        self.teacher_model = MotionPlanningTeacherModule()

    def init_weight(self):
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.motion_cls_branch[-1].bias, bias_init)
        # nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward(
        self,
        motion_query,
        plan_query,
        ego_feature,
        ego_anchor_embed,
    ):
        bs, num_anchor = motion_query.shape[:2]
        motion_cls = self.motion_cls_branch(motion_query).squeeze(-1)
        motion_reg = self.motion_reg_branch(motion_query).reshape(bs, num_anchor, self.fut_mode, self.fut_ts, 2)
        plan_reg = self.student_model(plan_query).reshape(bs, 1, 3 * self.ego_fut_mode, self.ego_fut_ts, 2)
        plan_cls, plan_reg_offset, planning_status = self.teacher_model(
            motion_query, plan_query, plan_reg,
            ego_feature, ego_anchor_embed
        )
        # plan_cls = self.plan_cls_branch(plan_query).squeeze(-1)
        # plan_reg = self.plan_reg_branch(plan_query).reshape(bs, 1, 3 * self.ego_fut_mode, self.ego_fut_ts, 2)
        # planning_status = self.plan_status_branch(ego_feature + ego_anchor_embed)
        return motion_cls, motion_reg, plan_cls, plan_reg, planning_status, plan_reg_offset
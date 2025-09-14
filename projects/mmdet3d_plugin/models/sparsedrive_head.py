from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn

from mmcv.runner import BaseModule
from mmdet.models import HEADS
from mmdet.models import build_head


class FusionQueryModel(BaseModule):
    def __init__(self, num_class = 6, dropout = 0.1, hidden_dim = 256):
        super(FusionQueryModel, self).__init__()
        self.num_class = num_class
        self.avg_pool = nn.AvgPool1d(kernel_size=num_class)
        # input_channel = 512
        # output_channel = 256
        # self.query_mlp = nn.Sequential(
        #     nn.Linear(input_channel, hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim, output_channel)
        # )
    
    def forward(self, instance_feature, motion_query, classification):
        classification = classification[-1].unsqueeze(-1)  # [B, 900, 6, 1]
        motion_query_feature = motion_query * classification  # [B, 900, 6, 256]
        motion_query_feature = motion_query_feature.transpose(2, 3)
        bs, num_query, channel, _ = motion_query_feature.shape
        motion_query_feature = motion_query_feature.reshape(bs, num_query * channel, self.num_class)
        motion_query_feature = self.avg_pool(motion_query_feature)  # [B, 900, 256, 1]
        motion_query_feature = motion_query_feature.squeeze(-1).reshape(bs, num_query, channel)  # [B, 900, 256]
        # merge_query = torch.concat([instance_feature, motion_query_feature], dim = -1)
        # output = self.query_mlp(merge_query)
        output = motion_query_feature + instance_feature
        return output


@HEADS.register_module()
class SparseDriveHead(BaseModule):
    def __init__(
        self,
        task_config: dict,
        det_head = dict,
        map_head = dict,
        motion_plan_head = dict,
        motion_for_det = dict,
        init_cfg=None,
        **kwargs,
    ):
        super(SparseDriveHead, self).__init__(init_cfg)
        self.task_config = task_config
        if self.task_config['with_det']:
            self.det_head = build_head(det_head)
        if self.task_config['with_map']:
            self.map_head = build_head(map_head)
        
        self.use_motion_for_det = False
        if self.task_config['with_motion_plan']:
            self.use_motion_for_det = True
            self.motion_for_det = build_head(motion_for_det)
            self.motion_plan_head = build_head(motion_plan_head)
        # self.fusion_query_model = FusionQueryModel()

    def init_weights(self):
        if self.task_config['with_det']:
            self.det_head.init_weights()
        if self.task_config['with_map']:
            self.map_head.init_weights()
        if self.task_config['with_motion_plan']:
            self.motion_plan_head.init_weights()

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
    ):
        if self.task_config['with_det']:
            det_output = self.det_head(feature_maps, metas, self.use_motion_for_det)
        else:
            det_output = None

        if self.task_config['with_map']:
            map_output = self.map_head(feature_maps, metas)
        else:
            map_output = None

        instance_queue_get = None
        if self.task_config['with_motion_plan'] and self.use_motion_for_det:
            det_output, instance_queue_get = self.motion_for_det(
                det_output,  
                feature_maps,
                metas,
                self.det_head.anchor_encoder,
                self.det_head.instance_bank,
                self.det_head.instance_bank.mask,
                self.det_head.instance_bank.anchor_handler,
                self.motion_plan_head.instance_queue,
            )
        
        if self.task_config['with_motion_plan']:
            motion_output, planning_output = self.motion_plan_head(
                det_output, 
                map_output, 
                feature_maps,
                metas,
                self.det_head.anchor_encoder,
                self.det_head.instance_bank.mask,
                self.det_head.instance_bank.anchor_handler,
                use_motion_for_det=self.use_motion_for_det,
                instance_queue_get=instance_queue_get,
            )
        else:
            motion_output, planning_output = None, None

        return det_output, map_output, motion_output, planning_output

    def loss(self, model_outs, data):
        det_output, map_output, motion_output, planning_output = model_outs
        losses = dict()
        if self.task_config['with_det']:
            loss_det = self.det_head.loss(det_output, data)
            losses.update(loss_det)
        
        if self.task_config['with_map']:
            loss_map = self.map_head.loss(map_output, data)
            losses.update(loss_map)

        if self.task_config['with_motion_plan']:
            motion_loss_cache = dict(
                indices=self.det_head.sampler.indices, 
            )
            loss_motion = self.motion_plan_head.loss(
                motion_output, 
                planning_output, 
                data, 
                motion_loss_cache
            )
            losses.update(loss_motion)
        
        return losses

    def post_process(self, model_outs, data):
        det_output, map_output, motion_output, planning_output = model_outs
        if self.task_config['with_det']:
            det_result = self.det_head.post_process(det_output)
            batch_size = len(det_result)
        
        if self.task_config['with_map']:
            map_result= self.map_head.post_process(map_output)
            batch_size = len(map_result)

        if self.task_config['with_motion_plan']:
            motion_result, planning_result = self.motion_plan_head.post_process(
                det_output,
                motion_output, 
                planning_output,
                data,
            )

        results = [dict()] * batch_size
        for i in range(batch_size):
            if self.task_config['with_det']:
                results[i].update(det_result[i])
            if self.task_config['with_map']:
                results[i].update(map_result[i])
            if self.task_config['with_motion_plan']:
                results[i].update(motion_result[i])
                results[i].update(planning_result[i])

        return results

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS

__all__ = ["GlobalInstanceBank"]
  

def topk(confidence, k, *inputs):
    bs, N = confidence.shape[:2]
    confidence, indices = torch.topk(confidence, k, dim=1)
    indices = (
        indices + torch.arange(bs, device=indices.device)[:, None] * N
    ).reshape(-1)
    outputs = []
    for input in inputs:
        outputs.append(input.flatten(end_dim=1)[indices].reshape(bs, k, -1))
    return confidence, outputs

@PLUGIN_LAYERS.register_module()
class GlobalInstanceBank(nn.Module):

    def __init__(
        self,
        det_instance_bank: dict,
        map_instance_bank: dict,
        motion_plan_instance_queue: dict,
    ):

        super(GlobalInstanceBank, self).__init__()
        # init instance banks
        self.det_instance_bank = build_from_cfg(det_instance_bank, PLUGIN_LAYERS)
        self.map_instance_bank = build_from_cfg(map_instance_bank, PLUGIN_LAYERS)
        self.motion_plan_instance_queue = build_from_cfg(motion_plan_instance_queue, PLUGIN_LAYERS)
    
    def init_weights(self):
        self.det_instance_bank.init_weight()
        self.map_instance_bank.init_weight()


    def get_det_feature(self, batch_size, metas=None, dn_metas=None):
        return self.det_instance_bank.get(batch_size, metas, dn_metas)

    def update_det_feature(self, instance_feature, anchor, confidence):
       return self.det_instance_bank.update(instance_feature, anchor, confidence)

    def cache_det_feature(
        self,
        instance_feature,
        anchor,
        confidence,
        metas=None,
        feature_maps=None,
    ):
        self.det_instance_bank.cache(
            instance_feature,
            anchor,
            confidence,
            metas,
            feature_maps,
        )

    def get_det_instance_id(self, confidence, anchor=None, threshold=None):
        return self.det_instance_bank.get_instance_id(confidence, anchor, threshold)

    def update_det_instance_id(self, instance_id=None, confidence=None):
        self.det_instance_bank.update_instance_id(instance_id, confidence)
    
    def get_map_feature(self, batch_size, metas=None, dn_metas=None):
        return self.map_instance_bank.get(batch_size, metas, dn_metas)

    def update_map_feature(self, instance_feature, anchor, confidence):
       return self.map_instance_bank.update(instance_feature, anchor, confidence)

    def cache_map_feature(
        self,
        instance_feature,
        anchor,
        confidence,
        metas=None,
        feature_maps=None,
    ):
        self.map_instance_bank.cache(
            instance_feature,
            anchor,
            confidence,
            metas,
            feature_maps,
        )

    def get_map_instance_id(self, confidence, anchor=None, threshold=None):
        return self.map_instance_bank.get_instance_id(confidence, anchor, threshold)

    def update_map_instance_id(self, instance_id=None, confidence=None):
        self.map_instance_bank.update_instance_id(instance_id, confidence)
    
    def get_motion_plan_feature(
        self,
        det_output,
        feature_maps,
        metas,
        batch_size,
        mask,
        anchor_handler,
    ):
        return self.motion_plan_instance_queue.get(
            det_output,
            feature_maps,
            metas,
            batch_size,
            mask,
            anchor_handler,
        )
    
    def prepare_motion(self, det_output, mask):
        self.motion_plan_instance_queue.prepare_motion(det_output, mask)
    
    def prepare_planning(self, feature_maps, mask, batch_size):
        return self.motion_plan_instance_queue.prepare_planning(
            feature_maps,
            mask,
            batch_size,
        )
    
    def cache_motion(self, instance_feature, det_output, metas):
        self.motion_plan_instance_queue.cache_motion(instance_feature, det_output, metas)
    
    def cache_planning(self, ego_feature, ego_status):
        self.motion_plan_instance_queue.cache_planning(ego_feature, ego_status)

    

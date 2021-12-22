# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from siamreppoints.core.config import cfg
from siamreppoints.models.loss import select_cross_entropy_loss, weight_l1_loss
from siamreppoints.models.backbone import get_backbone
from siamreppoints.models.head import get_rpn_head
from siamreppoints.models.neck import get_neck


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build rpn head
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)

    
    def get_feature(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf) 
        return torch.cat([zf[0], zf[1]], dim=1)
        
    def get_template(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf) 
        return zf
    
    def template(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x, zf, instance_size):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, pts_preds_init, pts_preds_refine = self.rpn_head(zf, xf, instance_size)
        
        cls = cls.permute(0, 2, 3, 1)
        cls = cls.reshape(cls.shape[0], -1, 1)
        cls = torch.sigmoid(cls)
        
        return {
                'score': cls,
                'bbox': pts_preds_refine,
                'feature': torch.cat([xf[0], xf[1]], dim=1),
                'search_feat': xf
               }
    
    def tracking(self, xf, zf, instance_size):
        cls, pts_preds_init, pts_preds_refine = self.rpn_head(zf, xf, instance_size)

        cls = cls.permute(0, 2, 3, 1)
        cls = cls.reshape(cls.shape[0], -1, 1)
        cls = torch.sigmoid(cls)

        return {
                'score': cls,
                'bbox': pts_preds_refine,
                'feature': torch.cat([xf[0], xf[1]], dim=1),
                'search_feat': xf
               }

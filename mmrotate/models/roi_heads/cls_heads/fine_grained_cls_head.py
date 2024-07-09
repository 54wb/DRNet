# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from mmcv.runner import BaseModule
from mmcv.utils import to_2tuple

from mmdet.core import multi_apply
from mmdet.models.losses import accuracy
from mmdet.models.utils import build_linear_layer

from ...utils.se_layer import SEModule
from ...builder import ROTATED_HEADS, build_loss


@ROTATED_HEADS.register_module()
class FineClsHead(BaseModule):
    """the fine grainted cls head.
    
    Args:
        in_channels(int): the channels of roi features
        num_classes(int): the num of classes
        roi_feat_size(int): size of Roi size
        with_cls(bool): If True, use classification branch.
        reg_class_agnostic(bool): if ture, regression branch are class agnostic
        cls_predictor_cfg(dict): Config of classifcation predictor
        loss_cls(dict): Config of classification loss
        init_cfg(dict): Config of initialization.
    """

    def __init__(self, 
                 with_avg_pool=False,
                 with_cls=True,
                 roi_feat_size=7, 
                 in_channels=256,
                 fc_out_channels=1024, 
                 num_classes=37, 
                 reg_class_agnostic=False,
                 cls_predictor_cfg=dict(type='Linear'),
                 loss_cls=None,
                 num_shared_fcs=2,
                 init_cfg=None,
                 *args, 
                 **kwargs):
        super(FineClsHead, self).__init__(init_cfg)
        assert with_cls
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.roi_feat_size = to_2tuple(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.fc_out_channels = fc_out_channels
        self.num_classes = num_classes
        self.reg_class_agnostic = reg_class_agnostic
        self.cls_predictor_cfg = cls_predictor_cfg
        self.fp16_enabled = False

        self.loss_cls = build_loss(loss_cls)
        
        self.num_shared_fcs = num_shared_fcs
        
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        # 3x3 conv and don't change the channels 
        self.conv = ConvModule(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='BN')
        ) 
        #mask_conv make a learnable mask for roi feature
        self.mask_conv = ConvModule(
            in_channels=self.in_channels,
            out_channels=1,
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='BN')
        ) 
        #seblock to class channels message
        self.se_block = SEModule(in_channels=self.in_channels, reduction=32)
        
        self.shared_fcs, last_layer_dim = self._add_fc_branch(
            self.num_shared_fcs, self.in_channels,True)
        
        self.shared_out_channels = last_layer_dim
        
        self.relu = nn.ReLU(inplace=True)

        #the last fine-grained cls branch
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.shared_out_channels,
                out_features=cls_channels)
        if init_cfg is None:
            self.init_cfg = [
                dict(
                    type='Kaiming',
                    layer='Linear',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='fc_cls')
                    ])] 


    @property
    def custom_cls_channels(self):
        "The custom cls channels"
        return getattr(self.loss_cls, 'custom_cls_channels', False)
    
    @property    
    def custom_activation(self):
        "The custom activation"
        return getattr(self.loss_cls, 'custom_activation', False)
    
    @property
    def custom_accuracy(self):
        "The custom accuracy"
        return getattr(self.loss_cls, 'custom_accuracy', False)
    
    def _add_fc_branch(self,
                       num_branch_fcs,
                       in_channels,
                       is_shared=False):
        """Add the last fcs for fine-grained cls
        
        fcs -> results
        """
        last_layer_dim = in_channels
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            if (is_shared or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_fcs, last_layer_dim
    
    def _get_target_single(self, pos_bboxes, neg_bboxes,
                           pos_gt_bboxes, pos_gt_labels, cfg):
        
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 5)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 5)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights
    
    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights
    
    @force_fp32(apply_to=('fine_cls_score'))
    def loss(self,
             fine_cls_score,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        """Loss function when use normal loss"""
        losses = dict()
        if fine_cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if fine_cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    fine_cls_score, 
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_fine_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(fine_cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['fine_cls_acc'] = accuracy(fine_cls_score, labels)
        return losses
    
    
    def get_fine_labels(self, fine_cls_score, cfg=None):
        """obtain the final fine_labels"""
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(fine_cls_score)
        else:
            scores = F.softmax(fine_cls_score, dim=1) if fine_cls_score is not None else None
        num_classes = scores.size(1) - 1
        
        #exclude background category
        scores = scores[:,:-1]
        scores_max, inds = torch.max(scores, dim=1)
        return inds

        


    def forward(self, x):
        """Forward function"""
        #mask for align feature
        mask_feat = x
        x = self.conv(x)
        mask = self.mask_conv(mask_feat)
        x = mask*x
        
        # senet block
        x = self.se_block(x)
         
        #two shared fcs for cls
        if self.num_shared_fcs > 0:
            x = x.flatten(1)
            
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        #fine_grained cls
        cls_score = self.fc_cls(x) if self.with_cls else None
        return cls_score









        


            
        


    



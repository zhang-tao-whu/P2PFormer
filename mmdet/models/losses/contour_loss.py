# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
import torch.nn as nn

from ..builder import LOSSES
from .utils import weighted_loss

@LOSSES.register_module()
class DMLoss(nn.Module):
    """Dynamic Matching loss.

    Args:
        key_item_weight (float, 0 <= key_item_weight<= 1): The weight if item key points.
            Defaults to 0.5
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, loss_weight=1.0, offsets_stride=4.,
                 key_item_weight=0.5, crit_type='smoothL1'):
        super(DMLoss, self).__init__()
        self.key_item_weight = key_item_weight
        self.loss_weight = loss_weight
        self.offsets_stride = offsets_stride
        self.crit_type = crit_type
        assert crit_type in ['smoothL1', 'L1']
        if crit_type == 'smoothL1':
            self.crit = smooth_l1_loss
            self.beta = 1.0 / offsets_stride
        else:
            self.crit = l1_loss

    def interpolation(self, poly, time=10):
        ori_points_num = poly.size(1)
        poly_roll = torch.roll(poly, shifts=1, dims=1)
        poly_ = poly.unsqueeze(3).repeat(1, 1, 1, time)
        poly_roll = poly_roll.unsqueeze(3).repeat(1, 1, 1, time)
        step = torch.arange(0, time, dtype=torch.float32).cuda() / time
        poly_interpolation = poly_ * step + poly_roll * (1. - step)
        poly_interpolation = poly_interpolation.permute(0, 1, 3, 2).reshape(poly_interpolation.size(0),
                                                                            ori_points_num * time, 2)
        return poly_interpolation

    def compute_distance(self, pred_poly, gt_poly):
        pred_poly_expand = pred_poly.unsqueeze(1)
        gt_poly_expand = gt_poly.unsqueeze(2)
        gt_poly_expand = gt_poly_expand.expand(gt_poly_expand.size(0), gt_poly_expand.size(1),
                                               pred_poly_expand.size(2), gt_poly_expand.size(3))
        pred_poly_expand = pred_poly_expand.expand(pred_poly_expand.size(0), gt_poly_expand.size(1),
                                                   pred_poly_expand.size(2), pred_poly_expand.size(3))
        distance = torch.sum((pred_poly_expand - gt_poly_expand) ** 2, dim=3)
        return distance

    def get_pred_targets_item1(self, preds, offsets, targets):
        targets = self.interpolation(targets)
        distances = self.compute_distance(preds, targets)
        index_gt = torch.min(distances, dim=1)[1]
        index_0 = torch.arange(index_gt.size(0))
        index_0 = index_0.unsqueeze(1).expand(index_gt.size(0), index_gt.size(1))
        targets = targets[index_0, index_gt, :]
        offsets_target = (targets - preds) / self.offsets_stride
        return offsets, offsets_target.detach()

    def get_pred_targets_item2(self, preds, offsets, key_points, masks):
        masks = masks.to(torch.bool)
        distances = self.compute_distance(key_points, preds)
        #(N, n_key_points, n_pred_points)
        index_pred = torch.min(distances, dim=1)[1]
        index_0 = torch.arange(index_pred.size(0))
        index_0 = index_0.unsqueeze(1).expand(index_pred.size(0), index_pred.size(1))
        preds = preds[index_0, index_pred, :][masks]
        offsets = offsets[index_0, index_pred, :][masks]
        offsets_target = (key_points[masks] - preds) / self.offsets_stride
        return offsets, offsets_target.detach()

    def loss(self, pred, targets, weight, avg_factor=None):
        if self.crit_type == 'smoothL1':
            return self.crit(pred, targets, weight=None, beta=self.beta, reduction='mean', avg_factor=avg_factor) *\
                   weight
        else:
            return self.crit(pred, targets, weight=None, reduction='mean', avg_factor=avg_factor) * weight


    def forward(self,
                pred_contours,
                pred_offsets,
                gt_contours,
                gt_key_points,
                gt_key_points_mask,
                avg_factor=None,
                **kwargs):
        """Forward function.
        Args:
        pred_contours torch.Tensor(N, nums_points, 2)
        gt_contours torch.Tensor(N, nums_points, 2)
        gt_key_points torch.Tensor(N, nums_points, 2)
        gt_key_points_mask torch.Tensor(N, nums_points)
        """
        pred_item1, target_item1 = self.get_pred_targets_item1(pred_contours, pred_offsets, gt_contours)
        pred_item2, target_item2 = self.get_pred_targets_item2(pred_contours, pred_offsets,
                                                               gt_key_points, gt_key_points_mask)
        if avg_factor is None:
            avg_factor1, avg_factor2 = None, None
        else:
            avg_factor1, avg_factor2 = avg_factor
        loss1 = self.loss(pred_item1, target_item1, weight=1 - self.key_item_weight, avg_factor=avg_factor1)
        loss2 = self.loss(pred_item2, target_item2, weight=self.key_item_weight, avg_factor=avg_factor2)
        return (loss1 + loss2) * self.loss_weight

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def smooth_l1_loss(pred, target, beta=1.0):
    """Smooth L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert beta > 0
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    loss = torch.abs(pred - target)
    return loss

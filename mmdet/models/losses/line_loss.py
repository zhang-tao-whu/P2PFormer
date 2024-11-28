# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
import torch.nn as nn
from ..builder import LOSSES
from .smooth_l1_loss import smooth_l1_loss
from .utils import reduce_loss

@LOSSES.register_module()
class LineLoss(nn.Module):
    """Smooth L1 loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0,
                 with_center_loss=False,
                 with_angle_loss=False,
                 center_loss_weight=0,
                 angle_loss_weight=0):
        super(LineLoss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.with_center_loss = with_center_loss
        self.with_angle_loss = with_angle_loss
        self.center_loss_weight = center_loss_weight
        self.angle_loss_weight = angle_loss_weight

    def normalize_vector(self, v):
        distance = (v[..., :1] ** 2 + v[..., 1:] ** 2) ** 0.5 + 1e-4
        return v / distance

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        assert pred.size(-1) == target.size(-1) == 6
        reduction = (
            reduction_override if reduction_override else self.reduction)
        cost = smooth_l1_loss(pred, target, weight=None, beta=self.beta, reduction='none', avg_factor=None).sum(dim=-1)
        cost_ = smooth_l1_loss(pred, torch.cat([target[..., 4:], target[..., 2:4], target[..., :2]], dim=-1),
                               weight=None, beta=self.beta, reduction='none', avg_factor=None).sum(dim=-1)
        cost_line = torch.minimum(cost, cost_)

        if self.with_center_loss:
            center_pred = pred[..., 2:4]
            center_target = target[..., 2:4]
            cost_center = smooth_l1_loss(center_pred, center_target, weight=None,
                                         beta=self.beta, reduction='none', avg_factor=None).sum(dim=-1)
        else:
            cost_center = 0


        cost_line = cost_line + cost_center * self.center_loss_weight
        cost_line = cost_line / (1 + self.center_loss_weight)

        if weight is not None:
            cost_line = cost_line * weight
        if avg_factor is None:
            cost_line = reduce_loss(cost_line, reduction)
        else:
            # if reduction is mean, then average the loss by avg_factor
            if reduction == 'mean':
                # Avoid causing ZeroDivisionError when avg_factor is 0.0,
                # i.e., all labels of an image belong to ignore index.
                eps = torch.finfo(torch.float32).eps
                cost_line = cost_line.sum() / (avg_factor + eps)
            # if reduction is 'none', then do nothing, otherwise raise an error
            elif reduction != 'none':
                raise ValueError('avg_factor can not be used with reduction="sum"')
        loss_line = self.loss_weight * cost_line
        return loss_line

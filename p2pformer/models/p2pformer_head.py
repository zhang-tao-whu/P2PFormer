# Copyright (c) OpenMMLab. All rights reserved.
import copy
from abc import ABCMeta
import random
import torch
from mmdet.core import reduce_mean
from mmcv.runner import BaseModule, force_fp32
from mmcv.cnn import build_plugin_layer, xavier_init
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.core import build_assigner
from mmdet.models.builder import HEADS, build_loss, build_roi_extractor
import torch.nn as nn
from functools import partial
from visualizer import get_local

def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map_results)

@HEADS.register_module(force=True)
class P2PFormerHead(BaseModule, metaclass=ABCMeta):
    def __init__(self,
                 in_channels=256,
                 out_channel=4,
                 num_query=30,
                 roi_wh=(40, 40),
                 expand_scale=1.1,
                 line_query_mode='line',
                 scale_agu=True,
                 roi_extractor=None,
                 positional_encoding=None,
                 line_predictor=None,
                 line_loss=None,
                 score_loss=None,
                 order_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs
                 ):
        super(P2PFormerHead, self).__init__(init_cfg)
        assert line_query_mode in ['line', 'point']
        self.line_query_mode = line_query_mode
        self.in_channels = in_channels
        self.num_query = num_query
        self.roi_w = roi_wh[0]
        self.roi_h = roi_wh[1]
        self.expand_scale = expand_scale
        self.scale_agu = scale_agu
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.roi_extractor = build_roi_extractor(roi_extractor)
        self.positional_encoding = build_positional_encoding(positional_encoding)
        line_predictor_ = copy.deepcopy(line_predictor)
        line_predictor_.update(num_query=num_query)
        line_predictor_.update(out_channel=out_channel)
        line_predictor_.update(roi_wh=roi_wh)
        line_predictor_.update(mode=line_query_mode)
        self.line_predictor = build_plugin_layer(line_predictor_)[1]
        self.hidden_dim = self.line_predictor.line_decoder.hidden_dim
        self.num_layers = self.line_predictor.line_decoder.num_layers
        if train_cfg is not None:
            self.line_assigner = build_assigner(train_cfg.line_assigner)
        self.line_loss = build_loss(line_loss)
        self.score_loss = build_loss(score_loss)
        self.order_loss = build_loss(order_loss)

        if self.line_query_mode == 'line':
            self.query_embeds = nn.Embedding(num_query, self.hidden_dim)
        elif self.line_query_mode == 'point':
            self.query_embeds = nn.Embedding(num_query * 3, self.hidden_dim)
        else:
            raise NotImplementedError
        self.pos_embeds = nn.Embedding(num_query, self.hidden_dim)

        self._init_layers()

    def _init_layers(self):
        self.fusion_1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.fusion_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.fusion_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        self.out_proj_1 = nn.Conv2d(256, self.hidden_dim, kernel_size=1, stride=1, bias=True)
        self.out_proj_2 = nn.Conv2d(256, self.hidden_dim, kernel_size=1, stride=1, bias=True)
        self.out_proj_3 = nn.Conv2d(256, self.hidden_dim, kernel_size=1, stride=1, bias=True)

    def init_weights(self):
        """Initialize weights of the LineFormerHead."""
        self.line_predictor.init_weights()
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    @get_local('bboxes')
    def forward(self, cnn_feature, bboxes, inds, expand_ratio=None):
        """Forward function.

        Args:
            cnn_features (tuple[Tensor]): Multi-scale features from the upstream
                network, each is a 4D-tensor with shape (N, C, H, W).
            contours (Tensor): Shape (total_num_gts, 128, 2).
            bboxes (Tensor/list[Tensor]): Bounding boxes of instances, which are gt_bboxes /
                contour_derived_bboxes when training / testing.
                Shape (total_num_gts, 4).
            inds (Tensor): Indicating which image the indexed bbox belong to.
                Shape (total_num_gts, ).
        """
        ct = (bboxes[..., :2] + bboxes[..., 2:4]) / 2.
        if expand_ratio is None or not self.scale_agu:
            roi_wh = (bboxes[..., 2:4] - bboxes[..., :2]) * self.expand_scale
        else:
            roi_wh = (bboxes[..., 2:4] - bboxes[..., :2]) * expand_ratio

        rois = torch.cat([ct[..., :1] - roi_wh[..., :1] / 2.,
                          ct[..., 1:] - roi_wh[..., 1:] / 2.,
                          ct[..., :1] + roi_wh[..., :1] / 2.,
                          ct[..., 1:] + roi_wh[..., 1:] / 2.], dim=1)  # (n, 4)
        rois = torch.cat([inds[:, None].float(), rois], dim=1)  # (n, 5)

        roi = self.roi_extractor([cnn_feature], rois)  # (ngt, c, roi_h, roi_w)
        roi_1 = self.fusion_1(roi)  # (ngt, c, roi_h, roi_w)
        roi_2 = self.fusion_2(roi_1)  # (ngt, c, roi_h / 2, roi_w / 2)
        roi_3 = self.fusion_3(roi_2)  # (ngt, c, roi_h / 4, roi_w / 4)
        multi_scale_roi = [self.out_proj_3(roi_3),
                           self.out_proj_2(roi_2),
                           self.out_proj_1(roi_1)]
        multi_scale_pos = []
        for x in multi_scale_roi:
            mask = x.new_zeros(size=(x.shape[0], x.shape[2], x.shape[3]))
            multi_scale_pos.append(self.positional_encoding(mask))

        num_gts = bboxes.size(0)
        query_embeds = self.query_embeds.weight.unsqueeze(1).repeat(1, num_gts, 1)  # (nq, ngt, c)
        pos_embeds = self.pos_embeds.weight.unsqueeze(1).repeat(1, num_gts, 1)
        if self.line_query_mode == 'point':
            pos_embeds = pos_embeds.unsqueeze(1).repeat(1, 3, 1, 1).flatten(0, 1)

        multi_scale_memory = [roi.flatten(2).permute(2, 0, 1)
                              for roi in multi_scale_roi]
        multi_scale_memory_pos = [pos_embed.flatten(2).permute(2, 0, 1)
                                  for pos_embed in multi_scale_pos]  # each (h*w, ngt, c)

        normed_lines_pred, lines_score_pred, lines_idxs_pred = \
            self.line_predictor(query=query_embeds,
                                query_pos=pos_embeds,
                                mlvl_feats=multi_scale_memory,
                                mlvl_pos_embeds=multi_scale_memory_pos,
                                )
        return normed_lines_pred, lines_score_pred, lines_idxs_pred, ct, roi_wh

    def get_single_normalized_targets(self, single_gt_lines, single_bbox, expand_ratio=None):
        #  line (n, 4), refer_points (36, 2), bbox(4, )
        single_bbox = single_bbox.unsqueeze(0)
        ct = (single_bbox[:, :2] + single_bbox[:, 2:4]) / 2.
        if expand_ratio is None:
            wh = (single_bbox[:, 2:4] - single_bbox[:, :2]) * self.expand_scale
        else:
            wh = (single_bbox[:, 2:4] - single_bbox[:, :2]) * expand_ratio
        single_gt_lines[:, :2] = (single_gt_lines[:, :2] - ct) / wh + 0.5
        single_gt_lines[:, 2:4] = (single_gt_lines[:, 2:4] - ct) / wh + 0.5
        single_gt_lines[:, 4:] = (single_gt_lines[:, 4:] - ct) / wh + 0.5
        return single_gt_lines, 0

    def get_normalized_targets(self, gt_lines, gt_bboxes, expand_ratio=None):
        ret = multi_apply(self.get_single_normalized_targets, gt_lines, gt_bboxes,
                          expand_ratio=expand_ratio)
        return list(zip(*ret))

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_lines,
                      matched_idxs,
                      **kwargs):
        expand_ratio = random.random() * (self.expand_scale - 1) * 2 + 1
        device = gt_bboxes[0].device
        inds = torch.cat([torch.full([len(gt_bboxes[i])], i) for i in range(len(gt_bboxes))], dim=0).to(device)
        gt_bboxes = torch.cat(gt_bboxes, dim=0)

        normed_lines_pred, lines_score_pred, lines_idxs_pred, ct, roi_wh = \
            self(x, gt_bboxes, inds, expand_ratio)

        gt_lines = [torch.from_numpy(item).to(device) \
                    for single_gt_lines in gt_lines for item in single_gt_lines]

        matched_idxs = [torch.from_numpy(item).to(device) \
                        for single_matched_idxs in matched_idxs for item in single_matched_idxs]

        normed_gt_lines, _ = self.get_normalized_targets(gt_lines, gt_bboxes,
                                                      expand_ratio=expand_ratio)

        lines_preds, lines_target, score_preds, score_target, idxs_pred, idxs_target = \
            self.line_assigner.assign(normed_lines_pred, normed_gt_lines, lines_score_pred,
                                      lines_idxs_pred, matched_idxs)



        losses = self.loss(lines_preds, lines_target,
                           score_preds, score_target,
                           idxs_pred, idxs_target)
        return losses

    def loss(self, lines_preds, lines_target,
             score_preds, score_target,
             idxs_pred, idxs_target):
        losses = dict()
        device = lines_preds[0].device
        num_line = torch.tensor(
            len(lines_preds[0]), dtype=torch.float, device=device)
        num_line = max(reduce_mean(num_line), 1.0)
        num_score = torch.tensor(
            len(score_preds[0]), dtype=torch.float, device=device)
        num_score = max(reduce_mean(num_score), 1.0)

        num_order = torch.tensor(
            len(idxs_pred), dtype=torch.float, device=device)
        num_order = max(reduce_mean(num_order), 1.0)

        for i, (line_pred, score_pred) in enumerate(zip(lines_preds, score_preds)):
            losses.update({'line_loss_{}'.format(i): self.line_loss(line_pred, lines_target, avg_factor=num_line * 6)})
            losses.update({'score_loss_{}'.format(i): self.score_loss(score_pred, score_target, avg_factor=num_score)})
        losses.update({'order_loss': self.order_loss(idxs_pred, idxs_target, avg_factor=num_order)})
        return losses

    def convert_single_imagebboxes2featurebboxes(self, bboxes_, img_meta):
        bboxes = bboxes_.clone()
        img_shape = img_meta['img_shape'][:2]
        ori_shape = img_meta['ori_shape'][:2]
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] / ori_shape[0] * img_shape[0]
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / ori_shape[0] * img_shape[0]
        return bboxes

    def convert_imagebboxes2featurebboxes(self, bboxes, img_metas):
        return multi_apply(self.convert_single_imagebboxes2featurebboxes, bboxes, img_metas)

    def get_line_idxs(self, order_pred=None, mode='order'):
        assert mode == 'order'
        assert order_pred is not None
        order_pred = order_pred.softmax(-1)
        line_idxs = torch.max(order_pred, dim=-1)[1]
        return line_idxs

    def simple_test(self,
                    x,
                    img_metas,
                    bboxes,
                    **kwargs):
        device = bboxes[0].device
        inds = torch.cat([torch.full([len(bboxes[i])], i) for i in range(len(bboxes))], dim=0).to(device)
        bboxes = self.convert_imagebboxes2featurebboxes(bboxes, img_metas)
        bboxes = torch.cat(bboxes, dim=0)
        normed_lines_pred, lines_score_pred, lines_idxs_pred, ct, roi_wh = \
            self(x, bboxes, inds)
        lines_idxs_pred = self.get_line_idxs(lines_idxs_pred, mode='order')
        return self.get_img_coords(normed_lines_pred, ct, roi_wh)[-1], \
               lines_score_pred.softmax(-1)[..., 0][-1], lines_idxs_pred

    def get_img_coords(self, normed_lines, cts, whs):
        cts = cts.unsqueeze(1)
        whs = whs.unsqueeze(1)
        p1 = (normed_lines[..., :2] - 0.5) * whs + cts
        p2 = (normed_lines[..., 2:4] - 0.5) * whs + cts
        p3 = (normed_lines[..., 4:] - 0.5) * whs + cts
        lines = torch.cat([p1, p2, p3], dim=-1)
        return lines

    def get_img_coords_points(self, points, cts, whs):
        cts = cts.unsqueeze(1)
        whs = whs.unsqueeze(1)
        points = (points - 0.5) * whs + cts
        return points
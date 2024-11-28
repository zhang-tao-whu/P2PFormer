from mmdet.core.bbox.assigners.base_assigner import BaseAssigner
import torch
import scipy.optimize as opt
from mmdet.core.bbox.builder import BBOX_ASSIGNERS

@BBOX_ASSIGNERS.register_module()
class LineAssigner(BaseAssigner):

    def __init__(self, line_weight=5., score_weight=1.):
        self.line_weight = line_weight
        self.score_weight = score_weight

    def match(self, lines_pred, lines_gt, scores_pred):
        rets = []
        for i in range(len(lines_gt)):
            ngt = len(lines_gt[i])
            cost_line = self.line_cost(lines_pred[i], lines_gt[i])
            cost_score = -scores_pred[i][..., :1].repeat(1, ngt)
            cost = cost_line * self.line_weight + cost_score * self.score_weight
            cost = cost.detach().cpu().numpy()
            pred_idx, gt_idx = opt.linear_sum_assignment(cost)
            rets.append([pred_idx, gt_idx])
        return rets

    def line_cost(self, line_pred, line_gt):
        npred, ngt = line_pred.size(0), line_gt.size(0)
        line_pred = line_pred.unsqueeze(1).repeat(1, ngt, 1)
        line_gt = line_gt.unsqueeze(0).repeat(npred, 1, 1)
        cost = torch.sum((line_pred - line_gt) ** 2, dim=-1) ** 0.5
        cost_ = torch.sum((line_pred - torch.cat([line_gt[..., -2:], line_gt[..., 2:4], line_gt[..., :2]],
                                                 dim=-1)) ** 2, dim=-1) ** 0.5
        return torch.minimum(cost_, cost)

    def assign(self, normed_lines_preds, normed_gt_lines,
               lines_score_preds,
               lines_idxs_pred, matched_idxs):
        #  normed_lines_pred (N, q, 4), lines_score_pred (N, q, 2), normed_reference_points_pred (N, 36, 2), line_idxs_pred (N, q, 36)
        # normed_gt_lines [[(), (), (), ...], []]
        normed_lines_pred = normed_lines_preds[-1]
        lines_score_pred = lines_score_preds[-1].softmax(-1)

        idxs = self.match(normed_lines_pred, normed_gt_lines, lines_score_pred)
        lines_preds = [[] for i in range(len(normed_lines_preds))]
        lines_target = []
        score_target = []
        idxs_pred, idxs_target = [], []
        for i in range(len(idxs)):
            pred_idx, gt_idx = idxs[i]
            for j in range(len(normed_lines_preds)):
                lines_preds[j].append(normed_lines_preds[j][i][pred_idx])
            lines_target.append(normed_gt_lines[i][gt_idx])

            score_label = torch.ones_like(lines_score_pred[i][:, 0])
            score_label[pred_idx] = 0
            score_target.append(score_label)

            idxs_pred.append(lines_idxs_pred[i][pred_idx])
            idxs_target.append(matched_idxs[i][gt_idx])
        lines_preds = [torch.cat(lines_pred, dim=0) for lines_pred in lines_preds]
        lines_target = torch.cat(lines_target, dim=0)
        score_preds = [lines_score_pred.flatten(0, 1) for lines_score_pred in lines_score_preds]
        score_target = torch.cat(score_target, dim=0)
        idxs_pred = torch.cat(idxs_pred, dim=0)
        idxs_target = torch.cat(idxs_target, dim=0)
        return lines_preds, lines_target, score_preds, score_target.to(torch.int64),\
               idxs_pred, idxs_target.to(torch.int64)

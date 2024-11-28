from mmdet.models.detectors.single_stage import SingleStageDetector
import numpy as np
import torch
import math
from shapely import geometry
from mmdet.core import bbox2result
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.models.builder import DETECTORS, build_head
import pycocotools.mask as maskUtils
import os
import json
from .p2pformer_head import multi_apply

@DETECTORS.register_module()
class P2PFormerSegmentor(SingleStageDetector):
    """Base class for contour based segmentor.

        Single-stage segmentor directly and densely predict instance contours on the
        output features of the backbone+neck.
        """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 line_head=None,
                 line_fpn=False,
                 detector_fpn_start_level=1,
                 line_fpn_start_level=1,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        if neck.type == 'MSDeformAttnFPN':
            self.need_split = True
        else:
            self.need_split = False
        super(P2PFormerSegmentor, self).__init__(
            backbone, neck, bbox_head, train_cfg,
            test_cfg, pretrained, init_cfg)
        line_head.update(train_cfg=train_cfg)
        line_head.update(test_cfg=test_cfg)
        self.line_head = build_head(line_head)
        self.detector_fpn_start_level = detector_fpn_start_level
        self.line_fpn_start_level = line_fpn_start_level
        self.line_fpn = line_fpn

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_lines,
                      gt_bboxes_ignore=None,
                      matched_idxs=None,
                      **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        if self.need_split:
            x_line, x_bbox = x
            if self.line_fpn:
                x_line = [x_line] + list(x_bbox)[::-1]
            else:
                x_line = [x_line]
            x_bbox = list(x_bbox)[::-1]
        else:
            x_line, x_bbox = x, x
        losses = self.bbox_head.forward_train(x_bbox[self.detector_fpn_start_level:], img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        if self.line_fpn:
            losses.update(self.line_head.forward_train(x_line[self.line_fpn_start_level:], img_metas, gt_bboxes,
                                                       gt_lines, matched_idxs))
        else:
            losses.update(self.line_head.forward_train(x_line[self.line_fpn_start_level], img_metas, gt_bboxes,
                                                       gt_lines, matched_idxs))
        return losses

    def simple_test(self, img, img_metas, rescale=False, mode='line', **kwargs):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        if self.need_split:
            x_line, x_bbox = x
            if self.line_fpn:
                x_line = [x_line] + list(x_bbox)[::-1]
            else:
                x_line = [x_line]
            x_bbox = list(x_bbox)[::-1]
        else:
            x_line, x_bbox = x, x

        results_list = self.bbox_head.simple_test(
            x_bbox[self.detector_fpn_start_level:], img_metas, rescale=rescale)
        #results_list [(bboxes, labels), ...]
        # boxes (Tensor): Bboxes with score after nms, has shape (num_bboxes, 5). last dimension 5 arrange as (x1, y1, x2, y2, score)
        # labels (Tensor): has shape (num_bboxes, )
        bboxes_pred = [item[0] for item in results_list]
        labels_pred = [item[1] for item in results_list]
        instance_nums = 0
        for item in bboxes_pred:
            instance_nums += len(item)
        if instance_nums == 0:
            mask_results = [[[] for _ in range(self.bbox_head.num_classes)]] * len(bboxes_pred)
            results_list = list(zip(bboxes_pred, labels_pred))
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in results_list
            ]
            return list(zip(bbox_results, mask_results))
        if self.line_fpn:
            lines, lines_scores, lines_idxs = self.line_head.simple_test(x_line[self.line_fpn_start_level:],
                                                                                       img_metas, bboxes_pred)
        else:
            lines, lines_scores, lines_idxs = self.line_head.simple_test(x_line[self.line_fpn_start_level],
                                                                                       img_metas, bboxes_pred)
        num_polygons_per_img = [len(item) for item in bboxes_pred]
        lines, lines_scores, lines_idxs = lines.split(num_polygons_per_img, dim=0), \
            lines_scores.split(num_polygons_per_img, dim=0), \
            lines_idxs.split(num_polygons_per_img, dim=0)
        polygons = []
        for line, score, idx in zip(lines, lines_scores, lines_idxs):
            polygons.append(self.construct_poly_from_lines(line, score, idx))
        mask_results = self.convert_contour2mask(polygons, labels_pred, bboxes_pred, img_metas)
        bboxes_pred = [item[1] for item in mask_results]
        labels_pred = [item[2] for item in mask_results]
        mask_results = [item[0] for item in mask_results]
        results_list = list(zip(bboxes_pred, labels_pred))
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return list(zip(bbox_results, mask_results))

    def construct_poly_from_lines(self, lines, scores, idxs):
        device = lines.device
        lines, scores, idxs = lines.cpu().numpy(), scores.cpu().numpy(), idxs.cpu().numpy()
        polygons = []
        for line, score, idx in zip(lines, scores, idxs):
            line_polygon = CornerPolygon(line, score, idx)
            polygon = line_polygon.get_polygon()
            polygons.append(polygon)
        return polygons

    def converge_components_single(self, contours_pred, labels_pred, bboxes_pred,
                                   bboxes_from='detection', threthold=0.9):
        assert bboxes_from in ['detection', 'contour']
        scores_pred = bboxes_pred[..., 4:]
        bboxes_pred = bboxes_pred[..., :4]
        if bboxes_from == 'contour':
            min_coords = torch.min(contours_pred, dim=1)[0]
            max_coords = torch.max(contours_pred, dim=1)[0]
            bboxes_pred = torch.cat([min_coords, max_coords], dim=1)
        iof = bbox_overlaps(bboxes_pred, bboxes_pred, is_aligned=False, mode='iof')
        same_label = labels_pred.unsqueeze(1) - labels_pred.unsqueeze(0)
        same_label = (same_label == 0).to(torch.float)
        large_score = scores_pred - scores_pred.transpose(0, 1)
        large_score = (large_score <= 0).to(torch.float)
        iof = iof * same_label * large_score
        npred = iof.size(0)
        # iof (n, n)
        component_rela = torch.arange(npred, device=iof.device)
        iof[component_rela, component_rela] = 0
        max_iof, max_inds = torch.max(iof, dim=1)
        replace = max_iof >= threthold
        component_rela[replace] = max_inds[replace]
        valid_idxs = component_rela[torch.logical_not(replace)]
        return (valid_idxs.detach().cpu().numpy(), component_rela.detach().cpu().numpy())

    def converge_components(self, contours_pred, laels_pred, bboxes_pred, bboxes_from='detection', threthold=0.9):
        return multi_apply(self.converge_components_single, contours_pred, laels_pred, bboxes_pred,
                           bboxes_from=bboxes_from, threthold=threthold)

    def single_convert_contour2mask(self, contours_pred, labels_pred, bboxes_pred,
                                    img_meta, rescore=True, converge_component=True,
                                    ignore_contour2mask=False, iou_threthold=0.0):
        img_shape = img_meta['img_shape'][:2]
        ori_shape = img_meta['ori_shape'][:2]
        mask_pred = [[] for _ in range(self.bbox_head.num_classes)]
        if ignore_contour2mask:
            return (mask_pred, bboxes_pred, labels_pred)
        labels_pred_ret = labels_pred
        labels_pred = labels_pred.detach().cpu().numpy()
        rles = []
        if True:
            saved_vector_contours = []
            for contour in contours_pred:
                contour[..., 0] = contour[..., 0] / img_shape[0] * ori_shape[0]
                contour[..., 1] = contour[..., 1] / img_shape[1] * ori_shape[1]
                contour = contour.flatten().tolist()
                if len(contour) < 6:
                    if len(contour) == 0:
                        contour = [0] * 6
                    else:
                        contour += [contour[-1]] * 6
                saved_vector_contours.append(contour)
                rle = maskUtils.frPyObjects([contour], ori_shape[0], ori_shape[1])
                rles += rle
            masks = maskUtils.decode(rles).transpose(2, 0, 1)

            # # for visualization polygon, save the vector polygon as json and draw it
            # self.save_vector_result(img_meta['ori_filename'], saved_vector_contours,
            #                         list(bboxes_pred[:, -1].cpu().numpy()))
        for mask, label in zip(masks, labels_pred):
            mask_pred[int(label)].append(mask)
        return (mask_pred, bboxes_pred, labels_pred_ret)

    def save_vector_result(self, img_name, polygons, scores, save_dir='./work_dirs/json_pred'):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_dir = os.path.join(save_dir, img_name.split('/')[-1].split('.')[0] + '.json')
        # print(img_name.split('/')[-1].split('.')[0])
        for polygon in polygons:
            for i, item in enumerate(polygon):
                polygon[i] = np.float(item)
        for i, item in enumerate(scores):
            scores[i] = np.float(item)
        ret = {'polygons': polygons, 'scores': scores}
        with open(save_dir, 'w') as f:
            json.dump(ret, f)
        return

    def convert_contour2mask(self, contours_preds, labels_preds, bboxes_pred, img_metas,
                             rescore=True, converge_component=False, ignore_contour2mask=False):
        #masks_pred [single img masks_pred]
        #single img masks_pred [single class instances mask]
        #instance mask (h, w)
        return multi_apply(self.single_convert_contour2mask,
                           contours_preds, labels_preds, bboxes_pred,
                           img_metas, rescore=rescore, converge_component=converge_component,
                           ignore_contour2mask=ignore_contour2mask)

class CornerPolygon:
    def __init__(self, lines, line_scores, line_idxs):
        # lines np.array, shape (num_queries, 4)
        # line_scores np.array, shape (num_queries, )
        # line_idxs np.array, shape (num_queries, )
        self.lines = lines
        self.line_scores = line_scores
        self.line_idxs = line_idxs
        self.polygon_from_lines = self.process()

    def filter_low_score_lines(self, lines, line_scores, line_idxs, threthold=0.8):
        keep = line_scores >= threthold
        keeped_lines = lines[keep]
        keeped_line_scores = line_scores[keep]
        keeped_line_idxs = line_idxs[keep]
        return keeped_lines, keeped_line_scores, keeped_line_idxs

    def process(self):
        lines, scores, line_idxs = self.filter_low_score_lines(self.lines, self.line_scores, self.line_idxs)
        idxs = line_idxs.argsort()
        lines = lines[idxs]
        scores = scores[idxs]
        return lines[:, 2:4]
        #return lines.reshape(-1, 2)

    def get_polygon(self):
        return self.polygon_from_lines

class LinePolygon:
    def __init__(self, lines, line_scores, line_idxs):
        # lines np.array, shape (num_queries, 4)
        # line_scores np.array, shape (num_queries, )
        # line_idxs np.array, shape (num_queries, )
        self.lines = lines
        self.line_scores = line_scores
        self.line_idxs = line_idxs
        self.polygon_from_lines = self.process()

    def filter_low_score_lines(self, lines, line_scores, line_idxs, threthold=0.8):
        keep = line_scores >= threthold
        keeped_lines = lines[keep]
        keeped_line_scores = line_scores[keep]
        keeped_line_idxs = line_idxs[keep]
        return keeped_lines, keeped_line_scores, keeped_line_idxs

    def dis_points2lines(self, points, lines, ignore_value=512.):
        n_points = points.shape[0]
        n_lines = lines.shape[0]
        points = np.expand_dims(points, axis=1).repeat(n_lines, axis=1)
        lines = np.expand_dims(lines, axis=0).repeat(n_points, axis=0)
        v1 = lines[..., :2] - points
        v2 = lines[..., 2:] - points
        area = np.abs(v1[..., 0] * v2[..., 1] - v2[..., 0] * v1[..., 1])
        dis = area / (np.sum((lines[..., :2] - lines[..., 2:]) ** 2, axis=-1) ** 0.5)
        vl1 = lines[..., :2] - lines[..., 2:]
        vl2 = lines[..., 2:] - lines[..., :2]
        keep_1 = (v1[..., 0] * vl1[..., 0] + v1[..., 1] * vl1[..., 1]) >= 0
        keep_2 = (v2[..., 0] * vl2[..., 0] + v2[..., 1] * vl2[..., 1]) >= 0
        ignore = np.logical_not(np.logical_and(keep_1, keep_2))
        dis[ignore] = ignore_value
        return dis

    def lines_intersection(self, line1, line2, expand_ratio=5):
        # p1 = np.array([line1[0], line1[1], 0])
        # s1 = np.array([line1[2] - line1[0], line1[3] - line1[1], 0])
        # p2 = np.array([line2[0], line2[1], 0])
        # s2 = np.array([line2[2] - line2[0], line2[3] - line2[1], 0])
        # point = gm.Coordinate().calCoordinateFrom2Lines(p1, s1, p2, s2)
        x11 = (line1[0] - line1[2]) * expand_ratio + line1[2]
        x12 = (line1[2] - line1[0]) * expand_ratio + line1[0]
        y11 = (line1[1] - line1[3]) * expand_ratio + line1[3]
        y12 = (line1[3] - line1[1]) * expand_ratio + line1[1]
        x21 = (line2[0] - line2[2]) * expand_ratio + line2[2]
        x22 = (line2[2] - line2[0]) * expand_ratio + line2[0]
        y21 = (line2[1] - line2[3]) * expand_ratio + line2[3]
        y22 = (line2[3] - line2[1]) * expand_ratio + line2[1]

        shapely_line1 = geometry.LineString([(x11, y11),
                                             (x12, y12)])
        shapely_line2 = geometry.LineString([(x21, y21),
                                             (x22, y22)])
        intersection = np.array(shapely_line1.intersection(shapely_line2).coords)
        if len(intersection) != 0:
            return intersection[0]
        else:
            return None
        # return intersection

    def relation_neighboor_lines(self, lines):
        lines_ = np.roll(lines, -1, axis=0)
        length = np.sum((lines[..., :2] - lines[..., 2:]) ** 2, axis=-1) ** 0.5
        length_ = np.sum((lines_[..., :2] - lines_[..., 2:]) ** 2, axis=-1) ** 0.5
        vector = lines[..., :2] - lines[..., 2:]
        vector_ = lines_[..., :2] - lines_[..., 2:]
        sin = np.abs(vector[..., 0] * vector_[..., 1] - vector[..., 1] * vector_[..., 0]) / \
              (length * length_ + 1e-4)
        return sin > math.sin(math.pi / 10)

    def get_vertical_line(self, line, point, reverse=False):
        def getFootPoint(point, line_p1, line_p2):
            """
            @point, line_p1, line_p2 : [x, y, z]
            """
            x0 = point[0]
            y0 = point[1]
            z0 = point[2]

            x1 = line_p1[0]
            y1 = line_p1[1]
            z1 = line_p1[2]

            x2 = line_p2[0]
            y2 = line_p2[1]
            z2 = line_p2[2]

            k = -((x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1) + (z1 - z0) * (z2 - z1)) / \
                ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) * 1.0

            xn = k * (x2 - x1) + x1
            yn = k * (y2 - y1) + y1
            zn = k * (z2 - z1) + z1

            return (xn, yn, zn)

        vertical_p = getFootPoint([point[0], point[1], 0],
                                  [line[0], line[1], 0],
                                  [line[2], line[3], 0])
        vertical_p = np.array(vertical_p)[:2]
        if reverse:
            return np.concatenate([point, vertical_p], axis=0)
        else:
            return np.concatenate([vertical_p, point], axis=0)

    def per_corner_compute(self, line1, line2, rela, ret_polys):
        replace_idx = 0
        if rela:
            intersection = self.lines_intersection(line1, line2)
            if intersection is not None:
                ret_polys.append(intersection)
                if np.sum((line2[:2] - intersection) ** 2) > np.sum((line2[2:] - intersection) ** 2):
                    replace_idx = 1
        else:
            if np.sum((line2[:2] - line1[2:]) ** 2) > np.sum((line2[2:] - line1[2:]) ** 2):
                replace_idx = 1
                vertical_line1 = self.get_vertical_line(line1, line2[2:])
                vertical_line2 = self.get_vertical_line(line2, line1[2:], reverse=True)
                vertical_line = (vertical_line1 + vertical_line2) / 2.
                ret_polys.append(vertical_line[:2])
                ret_polys.append(vertical_line[2:])
            else:
                replace_idx = 0
                vertical_line1 = self.get_vertical_line(line1, line2[:2])
                vertical_line2 = self.get_vertical_line(line2, line1[2:], reverse=True)
                vertical_line = (vertical_line1 + vertical_line2) / 2.
                ret_polys.append(vertical_line[:2])
                ret_polys.append(vertical_line[2:])
        return replace_idx

    def construct_poly_from_ordered_lines(self, lines):
        relation = self.relation_neighboor_lines(lines)
        ret = []
        replace_idx = 0
        for i, (rela, line) in enumerate(zip(relation, lines)):
            if i == 0:
                replace_idx = self.per_corner_compute(lines[i], lines[(i + 1) % lines.shape[0]], rela, ret)
            else:
                if replace_idx == 1:
                    refer_line = np.array([line[2], line[3], line[0], line[1]])
                    replace_idx = self.per_corner_compute(refer_line, lines[(i + 1) % lines.shape[0]], rela, ret)
                else:
                    refer_line = line
                    replace_idx = self.per_corner_compute(refer_line, lines[(i + 1) % lines.shape[0]], rela, ret)
        if len(ret) == 0:
            return np.zeros([0, 2])
        else:
            return np.stack(ret, axis=0)

    def unique_lines(self, lines, scores):
        def get_op(line1, score1, line2, score2, angle_sin_threthold=math.sin(math.pi / 6), dis_threthold=3):
            v1 = line1[2:] - line1[:2]
            v2 = line2[2:] - line2[:2]
            angle_sin = abs(v1[0] * v2[1] - v2[0] * v1[1])
            if angle_sin >= angle_sin_threthold:
                return -1
            c1 = (line1[2:] + line1[:2]) / 2.
            c2 = (line2[2:] + line2[:2]) / 2.
            v1_2 = c2 - line1[:2]
            v2_1 = c1 - line2[:2]
            s1_2 = abs(v1_2[0] * v1[1] - v1[0] * v1_2[1])
            s2_1 = abs(v2_1[0] * v2[1] - v2[0] * v2_1[1])
            l1 = (v1[0] ** 2 + v1[1] ** 2) ** 0.5
            l2 = (v2[0] ** 2 + v2[1] ** 2) ** 0.5
            d1 = s1_2 / l1
            d2 = s2_1 / l2
            if d1 >= dis_threthold and d2 >= dis_threthold:
                return -1
            else:
                if score1 > score2:
                    return 1
                else:
                    return 0
        ret = []
        ret_scores = []
        for i in range(len(lines)):
            if len(ret) == 0:
                ret.append(lines[i])
                ret_scores.append(scores[i])
            else:
                op = get_op(ret[-1], ret_scores[-1], lines[i], scores[i])
                if op == -1:
                    ret.append(lines[i])
                    ret_scores.append(scores[i])
                elif op == 0:
                    ret.pop()
                    ret_scores.pop()
                    ret.append(lines[i])
                    ret_scores.append(scores[i])
        if len(ret) > 1:
            op = get_op(ret[0], ret_scores[0], ret[-1], ret_scores[-1])
            if op == 0:
                ret = ret[1:]
                ret_scores = ret_scores[1:]
            elif op == 1:
                ret = ret[:-1]
                ret_scores = ret_scores[:-1]
        if len(ret) == 0:
            return lines, scores
        return np.stack(ret, axis=0), np.stack(ret_scores, axis=0)

    def process(self):
        lines, scores, line_idxs = self.filter_low_score_lines(self.lines, self.line_scores, self.line_idxs)
        idxs = line_idxs.argsort()
        lines = lines[idxs]
        scores = scores[idxs]
        lines, scores = self.unique_lines(lines, scores)
        polygon = self.construct_poly_from_ordered_lines(lines)
        return polygon

    def get_polygon(self):
        return self.polygon_from_lines

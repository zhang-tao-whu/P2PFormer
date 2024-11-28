import numpy as np
from mmdet.datasets.builder import PIPELINES
from shapely.geometry import Polygon

from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines.formatting import to_tensor
try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None

@PIPELINES.register_module(force=True)
class LineSampleWithAlignReference:
    def __init__(self,
                 point_nums=36,
                 with_reference_points=True,
                 reset_bbox=True,
                 ):
        self.point_nums = point_nums
        self.reset_bbox = reset_bbox
        self.with_reference_points = with_reference_points

    def __call__(self, results):
        gt_masks = results['gt_masks']
        gt_labels = results['gt_labels']
        gt_polys = gt_masks.masks
        # height, width = gt_masks.height, gt_masks.width
        lines = []
        if self.with_reference_points:
            reference_points = []
            matched_idxs = []
        else:
            reference_points = None
            matched_idxs = None
        if self.reset_bbox:
            reset_bboxes = []
            reset_labels = []

        for gt_poly, label in zip(gt_polys, gt_labels):
            for comp_poly in gt_poly:
                poly = comp_poly.reshape(-1, 2).astype(np.float32)
                if len(poly) < 3:
                    continue
                bbox = np.concatenate([np.min(poly, axis=0), np.max(poly, axis=0)], axis=0)
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                if h <= 1 or w <= 1:
                    continue
                succeed = self.prepare_line(poly, lines, reference_points=reference_points,
                                            point_nums=self.point_nums, matched_idxs=matched_idxs)
                if succeed and self.reset_bbox:
                    reset_labels.append(label)
                    reset_bboxes.append(bbox)

        results['gt_lines'] = lines
        if self.with_reference_points:
            results['reference_points'] = reference_points
            results['matched_idxs'] = matched_idxs
        if self.reset_bbox:
            if len(lines) != 0:
                results['gt_labels'] = np.stack(reset_labels, axis=0)
                results['gt_bboxes'] = np.stack(reset_bboxes, axis=0)
            else:
                results['gt_labels'] = np.zeros((0, ), dtype=np.int64)
                results['gt_bboxes'] = np.zeros((0, 4), dtype=np.float32)
        return results

    def ignore_poly(self, idxs):
        idxs = np.array(idxs)
        max_idx = np.argmax(idxs)
        idxs = np.roll(idxs, -max_idx - 1)
        ret = idxs[1] > idxs[0] and idxs[2] > idxs[1] and idxs[3] > idxs[2]
        return not ret

    def unique(self, poly):
        poly_ = np.roll(poly, 1)
        dis = np.sum((poly - poly_) ** 2, axis=1) ** 0.5
        valid = dis >= 0.1
        return poly[valid]

    def prepare_line(self, poly, lines, reference_points=None, point_nums=None, matched_idxs=None):
        poly = self.unique(poly)
        if len(poly) < 3:
            return False
        poly = self.get_cw_polys(poly)
        if reference_points is not None:
            assert point_nums is not None and matched_idxs is not None
            ori_nodes = len(poly)
            img_gt_poly = self.uniformsample(poly, ori_nodes * point_nums)
            idx = self.four_idx(img_gt_poly)
            if self.ignore_poly(idx):
                return False
            img_gt_poly = self.get_img_gt(img_gt_poly, idx, t=point_nums)
            reference_points.append(img_gt_poly)
        line = np.concatenate([np.roll(poly, shift=1, axis=0), poly,
                               np.roll(poly, shift=-1, axis=0)], axis=-1)
        lines.append(line)
        if reference_points is not None:
            idxs = self.match_point_line(img_gt_poly, line)
            matched_idxs.append(idxs)
        return True

    def match_point_line(self, points, lines):
        lines_center = lines[:, 2:4]
        n_points, n_lines = points.shape[0], lines.shape[0]
        lines_center = np.expand_dims(lines_center, axis=1).repeat(n_points, axis=1)
        points = np.expand_dims(points, axis=0).repeat(n_lines, axis=0)
        distance = np.sum((lines_center - points) ** 2, axis=-1)
        idxs = np.argmin(distance, axis=1)
        return idxs

    def get_cw_polys(self, poly):
        return poly[::-1] if Polygon(poly).exterior.is_ccw else poly

    @staticmethod
    def uniformsample(pgtnp_px2, newpnum):
        pnum, cnum = pgtnp_px2.shape
        assert cnum == 2

        idxnext_p = (np.arange(pnum, dtype=np.int32) + 1) % pnum
        pgtnext_px2 = pgtnp_px2[idxnext_p]
        edgelen_p = np.sqrt(np.sum((pgtnext_px2 - pgtnp_px2) ** 2, axis=1))
        edgeidxsort_p = np.argsort(edgelen_p)

        # two cases
        # we need to remove gt points
        # we simply remove shortest paths
        if pnum > newpnum:
            edgeidxkeep_k = edgeidxsort_p[pnum - newpnum:]
            edgeidxsort_k = np.sort(edgeidxkeep_k)
            pgtnp_kx2 = pgtnp_px2[edgeidxsort_k]
            assert pgtnp_kx2.shape[0] == newpnum
            return pgtnp_kx2
        # we need to add gt points
        # we simply add it uniformly
        else:
            edgenum = np.round(edgelen_p * newpnum / np.sum(edgelen_p)).astype(np.int32)
            for i in range(pnum):
                if edgenum[i] == 0:
                    edgenum[i] = 1

            # after round, it may has 1 or 2 mismatch
            edgenumsum = np.sum(edgenum)
            if edgenumsum != newpnum:

                if edgenumsum > newpnum:

                    id = -1
                    passnum = edgenumsum - newpnum
                    while passnum > 0:
                        edgeid = edgeidxsort_p[id]
                        if edgenum[edgeid] > passnum:
                            edgenum[edgeid] -= passnum
                            passnum -= passnum
                        else:
                            passnum -= edgenum[edgeid] - 1
                            edgenum[edgeid] -= edgenum[edgeid] - 1
                            id -= 1
                else:
                    id = -1
                    edgeid = edgeidxsort_p[id]
                    edgenum[edgeid] += newpnum - edgenumsum

            assert np.sum(edgenum) == newpnum

            psample = []
            for i in range(pnum):
                pb_1x2 = pgtnp_px2[i:i + 1]
                pe_1x2 = pgtnext_px2[i:i + 1]

                pnewnum = edgenum[i]
                wnp_kx1 = np.arange(edgenum[i], dtype=np.float32).reshape(-1, 1) / edgenum[i]

                pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1
                psample.append(pmids)

            psamplenp = np.concatenate(psample, axis=0)
            return psamplenp

    @staticmethod
    def four_idx(img_gt_poly):
        x_min, y_min = np.min(img_gt_poly, axis=0)
        x_max, y_max = np.max(img_gt_poly, axis=0)
        center = [(x_min + x_max) / 2., (y_min + y_max) / 2.]
        can_gt_polys = img_gt_poly.copy()
        can_gt_polys[:, 0] -= center[0]
        can_gt_polys[:, 1] -= center[1]
        distance = np.sum(can_gt_polys ** 2, axis=1, keepdims=True) ** 0.5 + 1e-6
        can_gt_polys /= np.repeat(distance, axis=1, repeats=2)
        idx_bottom = np.argmax(can_gt_polys[:, 1])
        idx_top = np.argmin(can_gt_polys[:, 1])
        idx_right = np.argmax(can_gt_polys[:, 0])
        idx_left = np.argmin(can_gt_polys[:, 0])
        return [idx_bottom, idx_right, idx_top, idx_left]

    @staticmethod
    def get_img_gt(img_gt_poly, idx, t=128):
        align = len(idx)
        pointsNum = img_gt_poly.shape[0]
        r = []
        k = np.arange(0, t / align, dtype=float) / (t / align)
        for i in range(align):
            begin = idx[i]
            end = idx[(i + 1) % align]
            if begin > end:
                end += pointsNum
            r.append((np.round(((end - begin) * k).astype(int)) + begin) % pointsNum)
        r = np.concatenate(r, axis=0)
        return img_gt_poly[r, :]

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(poly_nums={self.point_nums}, '
        return repr_str

@PIPELINES.register_module(force=True)
class LineDefaultFormatBundle:
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor, \
                       (3)to DataContainer (stack=True)

    Args:
        img_to_float (bool): Whether to force the image to be converted to
            float type. Default: True.
        pad_val (dict): A dict for padding value in batch collating,
            the default value is `dict(img=0, masks=0, seg=255)`.
            Without this argument, the padding value of "gt_semantic_seg"
            will be set to 0 by default, which should be 255.
    """

    def __init__(self,
                 img_to_float=True,
                 pad_val=dict(img=0, masks=0, seg=255)):
        self.img_to_float = img_to_float
        self.pad_val = pad_val

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """

        if 'img' in results:
            img = results['img']
            if self.img_to_float is True and img.dtype == np.uint8:
                # Normally, image is of uint8 type without normalization.
                # At this time, it needs to be forced to be converted to
                # flot32, otherwise the model training and inference
                # will be wrong. Only used for YOLOX currently .
                img = img.astype(np.float32)
            # add default meta keys
            results = self._add_default_meta_keys(results)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(
                to_tensor(img), padding_value=self.pad_val['img'], stack=True)
        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        for key in ['gt_lines', 'reference_points', 'matched_idxs']:
            if key not in results:
                continue
            results[key] = DC(results[key], cpu_only=True, stack=False)
        if 'gt_masks' in results:
            results['gt_masks'] = DC(
                results['gt_masks'],
                padding_value=self.pad_val['masks'],
                cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]),
                padding_value=self.pad_val['seg'],
                stack=True)
        return results

    def _add_default_meta_keys(self, results):
        """Add default meta keys.

        We set default meta keys including `pad_shape`, `scale_factor` and
        `img_norm_cfg` to avoid the case where no `Resize`, `Normalize` and
        `Pad` are implemented during the whole pipeline.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            results (dict): Updated result dict contains the data to convert.
        """
        img = results['img']
        results.setdefault('pad_shape', img.shape)
        results.setdefault('scale_factor', 1.0)
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results.setdefault(
            'img_norm_cfg',
            dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False))
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(img_to_float={self.img_to_float})'

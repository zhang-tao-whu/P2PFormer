# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset


@DATASETS.register_module()
class WhuDataset(CocoDataset):
    CLASSES = ('building', )
    PALETTE = [(220, 20, 60), ]
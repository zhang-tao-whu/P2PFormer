from ..builder import PIPELINES

@PIPELINES.register_module()
class SampleContours:
    """sample the contours as same vertices.
    """

    def __init__(self,
                 poly_nums=128):
        self.poly_nums = poly_nums

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """
        polygons = results['gt_masks'].masks
        polygons = [poly.copy() for polygon in polygons for poly in polygon]
        polygons = self.process(polygons) #handle break polygons and filter tiny polygons
        polygons = self.sample_polygons(polygons, num_points=self.poly_nums) # sample the poly to 128 points
        results['polys'] = polygons


        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(poly_nums={self.poly_nums}, '
        return repr_str
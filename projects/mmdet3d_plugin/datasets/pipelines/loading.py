import mmcv
import numpy as np
from mmdet.datasets.builder import PIPELINES
from mmdet3d.datasets.pipelines.loading import LoadAnnotations3D
from einops import rearrange


@PIPELINES.register_module()
class CustomLoadAnnotations3D(LoadAnnotations3D):
    def __init__(self, with_bbox_2d=False, **kwargs):
        super(CustomLoadAnnotations3D, self).__init__(**kwargs)
        self.with_bbox_2d = with_bbox_2d

    def _load_bboxes_2d(self, results):
        results['gt_bboxes_2d'] = results['ann_info']['gt_bboxes_2d']
        results['gt_labels_2d'] = results['ann_info']['gt_labels_2d']
        results['gt_bboxes_2d_to_3d'] = results['ann_info']['gt_bboxes_2d_to_3d']
        results['gt_bboxes_ignore'] = results['ann_info']['gt_bboxes_ignore']
        return results

    def __call__(self, results):
        results = super().__call__(results)
        if self.with_bbox_2d:
            results = self._load_bboxes_2d(results)
        return results
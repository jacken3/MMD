from mmdet.core import bbox2result
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import TwoStageDetector
import torch
from mmdet.models.builder import build_head

@DETECTORS.register_module()
class Prior2D(TwoStageDetector):
    def __init__(self,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TwoStageDetector, self).__init__(init_cfg)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def forward_train(self,
                      feat,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        x = feat

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            img_metas_origin = img_metas.copy()
            img_metas = []
            for img_meta in img_metas_origin:
                for i in range(len(img_meta["img_shape"])):
                    img_meta_per_img = {}
                    for key in ['filename', 'img_shape', 'lidar2img', 'pad_shape', 'scale_factor', 'img_norm_cfg']:
                        if isinstance(img_meta[key], list):
                            img_meta_per_img[key] = img_meta[key][i]
                        else:
                            img_meta_per_img[key] = img_meta[key]
                    img_metas.append(img_meta_per_img)

            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        for key, value in roi_losses.items():
            losses[key+'_roi'] = value
            
        return losses, proposal_list

    def set_detection_cfg(self, detection_cfg):
        self.roi_head.test_cfg = detection_cfg

    def forward_test(self, feat, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = feat
        img_metas_origin = img_metas.copy()
        img_metas = []
        for img_meta in img_metas_origin:
            for i in range(len(img_meta["img_shape"])):
                img_meta_per_img = {}
                for key in ['filename', 'img_shape', 'lidar2img', 'pad_shape', 'scale_factor', 'img_norm_cfg']:
                    if isinstance(img_meta[key], list):
                        img_meta_per_img[key] = img_meta[key][i]
                    else:
                        img_meta_per_img[key] = img_meta[key]
                img_metas.append(img_meta_per_img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    @staticmethod
    def box_iou(rois_a, rois_b, eps=1e-4):
        rois_a = rois_a[..., None, :]                # [*, n, 1, 4]
        rois_b = rois_b[..., None, :, :]             # [*, 1, m, 4]
        xy_start = torch.maximum(rois_a[..., 0:2], rois_b[..., 0:2])
        xy_end = torch.minimum(rois_a[..., 2:4], rois_b[..., 2:4])
        wh = torch.maximum(xy_end - xy_start, rois_a.new_tensor(0))     # [*, n, m, 2]
        intersect = wh.prod(-1)                                         # [*, n, m]
        wh_a = rois_a[..., 2:4] - rois_a[..., 0:2]      # [*, m, 1, 2]
        wh_b = rois_b[..., 2:4] - rois_b[..., 0:2]      # [*, 1, n, 2]
        area_a = wh_a.prod(-1)
        area_b = wh_b.prod(-1)
        union = area_a + area_b - intersect
        iou = intersect / (union + eps)
        return iou
    
    def merge_aug_bboxes_2d(self, proposals_list, gt_proposals_list, thr=0.5):
        # proposals_list: List[array [n, 6]] of different views 6 : [x1, y1, x2, y2, score, label]
        # gt_proposals_list: List[array [m, 6]] of different views 6: [x1, y1, x2, y2, 1, label]
        result = []
        for view in range(len(proposals_list)):
            proposals = proposals_list[view]
            gt_proposals = gt_proposals_list[view]
            if len(gt_proposals) == 0:
                result.append(proposals)
                continue
            if len(proposals) == 0:
                result.append(gt_proposals)
                continue
            iou = self.box_iou(gt_proposals, proposals)
            max_iou = iou.max(-1)[0]
            complement_ids = max_iou < thr
            min_bbox_size = self.train_cfg['detection_proposal'].get('min_bbox_size', 0)
            wh = gt_proposals[:, 2:4] - gt_proposals[:, 0:2]
            valid_ids = (wh >= min_bbox_size).all(dim=1)
            complement_gts = gt_proposals[complement_ids & valid_ids]
            result.append(torch.cat([proposals, complement_gts], dim=0))
        
        return result
    
    def process_prosals(self, proposals, device):
       
        detections = [torch.cat(
            [torch.cat(
                [torch.tensor(boxes), torch.full((len(boxes), 1), label_id, dtype=torch.float)], dim=1) for
             label_id, boxes in enumerate(res)], dim=0).to(device) for res in proposals]
       
        if self.train_cfg is not None:
            min_bbox_size = self.train_cfg['detection_proposal'].get('min_bbox_size', 0)
        else:
            min_bbox_size = self.test_cfg['detection_proposal'].get('min_bbox_size', 0)
        if min_bbox_size > 0:
            new_detections = []
            for det in detections:
                wh = det[:, 2:4] - det[:, 0:2]
                valid = (wh >= min_bbox_size).all(dim=1)
                new_detections.append(det[valid])
            detections = new_detections

        return detections
    
    def process_gt(self, gt_bboxes, gt_labels, device):
        gt_proposals = [
            torch.cat(
            [
                boxes.to(device), 
                torch.full((len(boxes), 1), 1, dtype=boxes.dtype, device=device),
                label_id.unsqueeze(-1).to(boxes.dtype)
            ], 
            dim=-1
            ).to(device) 
            for label_id, boxes in zip(gt_labels, gt_bboxes)
        ]
        return gt_proposals
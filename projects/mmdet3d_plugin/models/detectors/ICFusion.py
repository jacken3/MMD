# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import mmcv
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.parallel import DataContainer as DC
from os import path as osp

from mmcv.runner import force_fp32, auto_fp16
from mmdet.core import multi_apply
from mmdet.models import DETECTORS
from mmdet.models.builder import build_detector
from mmdet3d.models.builder import build_neck
from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin import SPConvVoxelization
from mmdet.core.visualization.image import imshow_det_bboxes


@DETECTORS.register_module()
class ICFusionDetector(MVXTwoStageDetector):

    def __init__(self,
                 if_2d_prior=False,
                 extra_fpn = False,
                 extra_neck=None,
                 bbox_head_2d=None,
                 if_3d_detection=True,
                 use_grid_mask=False,
                 **kwargs):
        pts_voxel_cfg = kwargs.get('pts_voxel_layer', None)
        kwargs['pts_voxel_layer'] = None
        if if_3d_detection:
            kwargs['pts_bbox_head']['if_2d_prior'] = if_2d_prior
        super(ICFusionDetector, self).__init__(**kwargs)
        
        self.if_2d_prior = if_2d_prior
        self.extra_fpn = extra_fpn
        if extra_fpn and self.if_2d_prior:
            assert extra_neck is not None
            self.extra_neck = build_neck(extra_neck)
        self.if_3d_detection = if_3d_detection
        if if_2d_prior:
            assert bbox_head_2d is not None
            self.detector_2d = build_detector(bbox_head_2d)

        self.use_grid_mask = use_grid_mask
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        if pts_voxel_cfg:
            self.pts_voxel_layer = SPConvVoxelization(**pts_voxel_cfg)

    def init_weights(self):
        """Initialize model weights."""
        super(ICFusionDetector, self).init_weights()

    @auto_fp16(apply_to=('img'), out_fp32=True) 
    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img.float())
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    @force_fp32(apply_to=('pts', 'img_feats'))
    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        if pts is None:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                )
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        if self.with_pts_backbone:
            x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def visualize_2d(self, proposals, gt_bboxes, gt_labels, img_metas, imgs):

        def denormalize(img, img_norm_config):
            img = img.permute(1, 2, 0).numpy().astype(np.float64)
            img = mmcv.imdenormalize(img, img_norm_config['mean'], img_norm_config['std'], img_norm_config['to_rgb'])
            return img
        """Visualize 2D detection results."""
        for i in range(len(proposals)):
            file_name = 'visual/' + '/'.join(img_metas[i//6]['filename'][i%6].split('/')[-2:])
            img = denormalize(imgs[i // 6][i % 6].cpu(), img_metas[0]['img_norm_cfg'])
            img = imshow_det_bboxes(
                    img,
                    gt_bboxes[i].cpu().numpy(),
                    gt_labels[i].cpu().numpy(),
                    class_names=[
                                'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
                                'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
                                ],
                    bbox_color='red',
                    text_color='red',
                    show=False,
                    out_file=None)
            
            img = imshow_det_bboxes(
                        img,
                        proposals[i][:, :5].cpu().numpy(),
                        proposals[i][:, 5].cpu().numpy().astype(np.int32),
                        class_names=[
                                'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
                                'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
                                ],
                        bbox_color='green',
                        text_color='green',
                        show=False,
                        out_file=None)
            mmcv.imwrite(img, file_name)
    
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels_2d=None,
                      gt_bboxes_2d=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels_2d (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes_2d (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """

        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        assert img_feats is not None or pts_feats is not None
        losses = dict()
        if self.if_2d_prior:
            assert img_feats is not None
            # 训练2D检测器
            gt_bboxes_2d_list, gt_labels_2d_list, gt_bboxes_ignore_list = [], [], []
            for gt_bbox_2d, gt_label_2d, gt_bbox_ignore in zip(gt_bboxes_2d, gt_labels_2d, gt_bboxes_ignore):
                gt_bboxes_2d_list.extend(gt_bbox_2d)
                gt_labels_2d_list.extend(gt_label_2d)
                gt_bboxes_ignore_list.extend(gt_bbox_ignore)
            losses, proposals = self.detector_2d.forward_train(img_feats, img_metas, gt_bboxes_2d_list, gt_labels_2d_list, gt_bboxes_ignore_list)
            if self.if_3d_detection:
                with torch.no_grad():
                    proposals = self.detector_2d.forward_test(img_feats, img_metas, proposals)
                    proposals = self.detector_2d.process_prosals(proposals, img.device)
                    if False:
                        self.visualize_2d(proposals, gt_bboxes_2d_list, gt_labels_2d_list, img_metas, img)
                    if self.detector_2d.train_cfg['detection_proposal'].get('complement_2d_gt', -1) > 0:
                        gt_proposals = self.detector_2d.process_gt(gt_bboxes_2d_list, gt_labels_2d_list, img.device)
                        proposals = self.detector_2d.merge_aug_bboxes_2d(proposals, gt_proposals, thr=self.detector_2d.train_cfg['detection_proposal'].get('complement_2d_gt'))
        else:
            proposals = None
        if self.if_3d_detection:
            if self.if_2d_prior:
                if self.extra_fpn:
                    img_feats = self.extra_neck(img_feats)
                else:
                    img_feats = img_feats[-3:]       
            losses_pts = self.forward_pts_train(pts_feats, img_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas, proposals,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        return losses

    @force_fp32(apply_to=('pts_feats', 'img_feats'))
    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          proposals=None,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        if pts_feats is None:
            pts_feats = [None]
        if img_feats is None:
            img_feats = [None]
        outs = self.pts_bbox_head(pts_feats, img_feats, img_metas, proposals)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def visualize_attention_maps(self,attn_weights2visual, img, alpha=0.5, save_dir='./visual'):
        import cv2
        import matplotlib.pyplot as plt
        import os
        from PIL import Image

        def denormalize(img, img_norm_config):
            img = img.permute(1, 2, 0).cpu().numpy().astype(np.float64)
            img = mmcv.imdenormalize(img, img_norm_config['mean'], img_norm_config['std'], img_norm_config['to_rgb'])
            return img
        """
        可视化从 attn_weights2visual 得到的注意力图，并叠加到对应相机图像上。

        Args:
            attn_weights2visual (torch.Tensor): 形状 [Q, N, H, W]
                - Q: Query 数
                - N: 相机数（环视图数）
                - H, W: 特征图尺寸
            img (torch.Tensor): 形状 [N, 3, H_img, W_img]
                - N: 相机数
                - 3: 通道数 (RGB)
                - H_img, W_img: 原图(或网络输入尺寸)的高宽
            alpha (float): 叠加热力图时的透明度系数 (0~1)
        """
        img = img.detach()
        Q, N, H_feat, W_feat = attn_weights2visual.shape
        # img.shape = [N, 3, H_img, W_img]
        _, C, H_img, W_img = img.shape

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for q_idx in range(Q):
            attn_map_q = attn_weights2visual[q_idx]  
            global_min = attn_map_q.min()
            global_max = attn_map_q.max()

            denom = (global_max - global_min).clamp_min(1e-7)

            attn_map_q_norm = (attn_map_q - global_min) / denom
            for cam_idx in range(N):
                attn_map_2d = attn_map_q_norm[cam_idx].numpy()  

                attn_map_2d = cv2.resize(
                    attn_map_2d, (W_img, H_img), interpolation=cv2.INTER_CUBIC
                )

                img_cam = img[cam_idx]
                img_cam_np = denormalize(img_cam, dict(mean=np.array([103.53, 116.28, 123.675]), std=np.array([57.375, 57.12, 58.395]), to_rgb=False))
                img_cam_np = img_cam_np[:,:,[2,1,0]]
        
                # 如果图像是 0~1 浮点，则需要转到 0~255
                if img_cam_np.max() <= 1.0:
                    img_cam_np = (img_cam_np * 255).clip(0, 255).astype(np.uint8)
                else:
                    img_cam_np = img_cam_np.astype(np.uint8)

                heatmap = (attn_map_2d * 255).astype(np.uint8)  # [0,1] -> [0,255]
                mask_transparent = (heatmap < 50)

                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # BGR
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)      # -> RGB
                colored_hm_rgba = np.dstack([heatmap, np.full((H_img,W_img), 255, dtype=np.uint8)])
                
                colored_hm_rgba[mask_transparent, 3] = 0  # alpha=0
                colored_hm_rgba[~mask_transparent, 3] = int(255 * alpha)  # alpha=0.5


                bg_rgba = Image.fromarray(img_cam_np, mode='RGB').convert('RGBA')  # (H×W×4)
                fg_rgba = Image.fromarray(colored_hm_rgba, mode='RGBA')
                out_pil = Image.alpha_composite(bg_rgba, fg_rgba)  # 叠加
                overlay = out_pil.convert('RGB')                  # 转回 RGB (H×W×3)

                # 转回 numpy array (uint8, 0~255)
                # overlay = np.array(overlay, dtype=np.uint8)

                # plt.figure(figsize=(10, 5))
                # plt.suptitle(f"Query {q_idx}, Camera {cam_idx} (Unified Norm)", fontsize=14)

                # plt.subplot(1, 2, 1)
                # plt.imshow(img_cam_np)
                # plt.title("Original Image")
                # plt.axis("off")

                # plt.subplot(1, 2, 2)
                # plt.imshow(overlay)
                # plt.title("Attn Overlay")
                # plt.axis("off")

                save_path = os.path.join(save_dir, f"query_{q_idx}_cam_{cam_idx}.png")
                overlay.save(save_path, format='PNG')
                # plt.savefig(save_path, bbox_inches='tight')
                # plt.close()  # 关闭图窗，防止内存占用过多

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img=None, **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        if points is None:
            points = [None]
        if img is None:
            img = [None]
        for var, name in [(points, 'points'), (img, 'img'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        result, bbox_index, attn_weights = self.simple_test(points[0], img_metas[0], img[0], **kwargs)
        if False:
            bbox_index = [index.cpu().numpy() for index in bbox_index]
            attn_weights = [attn.cpu() for attn in attn_weights]

            attn_weights2visual = attn_weights[3][0][bbox_index[0][torch.where(result[0]['pts_bbox']['scores_3d'] > 0.4)[0]]].reshape(-1,6,20,50)
            self.visualize_attention_maps(attn_weights2visual, img[0], alpha=0.5, save_dir='visual/dep_2dPrior/'+img_metas[0][0]['sample_idx'])
    
        return result
    
    @force_fp32(apply_to=('x', 'x_img'))
    def simple_test_pts(self, x, x_img, img_metas, proposals=None, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, x_img, img_metas, proposals)
        bbox_list, bbox_index = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ] 
        return bbox_results, bbox_index, outs['attn_weights']

    def simple_test(self, points, img_metas, img=None, rescale=False):
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        if pts_feats is None:
            pts_feats = [None]
        if img_feats is None:
            img_feats = [None]
        
        bbox_list = [dict() for i in range(len(img_metas))]
        if self.if_2d_prior:
            assert img_feats is not None
            proposals = self.detector_2d.forward_test(img_feats, img_metas)
            proposals = self.detector_2d.process_prosals(proposals, img.device)
        else:
            proposals = None
        # for i in range(len(bbox_list)):
        #     bbox_list[i]['img_bbox'] = proposals[i]
        if (pts_feats or img_feats) and self.with_pts_bbox:
            if self.if_2d_prior:
                if self.extra_fpn:
                    img_feats = self.extra_neck(img_feats)
                else:
                    img_feats = img_feats[-3:]    
            bbox_pts, bbox_index, attn_weights = self.simple_test_pts(
                pts_feats, img_feats, img_metas, proposals, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list, bbox_index, attn_weights


    def show_results(self, data, result, out_dir, show_img=False):
        """Results visualization.

        Args:
            data (list[dict]): Input points and the information of the sample.
            result (list[dict]): Prediction results.
            out_dir (str): Output directory of visualization result.
            show (bool, optional): Determines whether you are
                going to show result by open3d.
                Defaults to False.
            score_thr (float, optional): Score threshold of bounding boxes.
                Default to None.
        """
        COLOR_MAP = {
            'red': np.array([191, 4, 54]) / 256,
            'light_blue': np.array([4, 157, 217]) / 256,
            'black': np.array([0, 0, 0]) / 256,
            'gray': np.array([140, 140, 136]) / 256,
            'purple': np.array([224, 133, 250]) / 256, 
            'dark_green': np.array([32, 64, 40]) / 256,
            'green': np.array([77, 115, 67]) / 256,
            'brown': np.array([164, 103, 80]) / 256,
            'light_green': np.array([135, 206, 191]) / 256,
            'orange': np.array([229, 116, 57]) / 256,
        }

        COLOR_KEYS = list(COLOR_MAP.keys())

        import matplotlib.pyplot as plt
        for batch_id in range(len(result)):
            if isinstance(data['points'][0], DC):
                points = data['points'][0]._data[0][batch_id].numpy()
            elif mmcv.is_list_of(data['points'][0], torch.Tensor):
                points = data['points'][0][batch_id]
            else:
                ValueError(f"Unsupported data type {type(data['points'][0])} "
                           f'for visualization!')
            if isinstance(data['img_metas'][0], DC):
                pts_filename = data['img_metas'][0]._data[0][batch_id][
                    'pts_filename']
                box_mode_3d = data['img_metas'][0]._data[0][batch_id][
                    'box_mode_3d']
                
            elif mmcv.is_list_of(data['img_metas'][0], dict):
                pts_filename = data['img_metas'][0][batch_id]['pts_filename']
                box_mode_3d = data['img_metas'][0][batch_id]['box_mode_3d']
            else:
                ValueError(
                    f"Unsupported data type {type(data['img_metas'][0])} "
                    f'for visualization!')
            if isinstance(data['img'][0], DC):
                img = data['img'][0]._data[0][batch_id].cpu().numpy()
            file_name = osp.split(pts_filename)[-1].split('.')[0]

            assert out_dir is not None, 'Expect out_dir, got none.'

            pred_bboxes = result[batch_id]['pts_bbox']['boxes_3d']
            pred_labels = result[batch_id]['pts_bbox']['labels_3d']
            scores = result[batch_id]['pts_bbox']['scores_3d']

            # for now we convert points and bbox into depth mode
            if (box_mode_3d == Box3DMode.CAM) or (box_mode_3d
                                                  == Box3DMode.LIDAR):
                points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                                   Coord3DMode.DEPTH)
                pred_bboxes = Box3DMode.convert(pred_bboxes, box_mode_3d,
                                                Box3DMode.DEPTH)
            elif box_mode_3d != Box3DMode.DEPTH:
                ValueError(
                    f'Unsupported box_mode_3d {box_mode_3d} for conversion!')
            pred_bboxes = pred_bboxes.tensor.cpu().numpy()

            self.figure = plt.figure(figsize=(20, 20))
            plt.axis('off')
            mask = (np.max(points, axis=-1) < 60)
            pc = points[mask]
            vis_pc = np.asarray(pc)
            plt.scatter(vis_pc[:, 0], vis_pc[:, 1], marker='o', color=COLOR_MAP['gray'], s=0.01)

            for box, score in pred_bboxes, scores:
                if score < 0.4:
                    continue
                corners = box.corners[0][[0,3,4,7]][:,:2]
                corners = np.concatenate([corners, corners[0:1, :2]])
                plt.plot(corners[:, 0], corners[:, 1], color=COLOR_MAP[0], linestyle='solid')
            
            if show_img:
                plt.imshow(img)    
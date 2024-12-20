# ------------------------------------------------------------------------
# Copyright (c) 2023 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import numpy as np
from numpy import random
import torch
import mmcv
import cv2

from mmcv.utils import build_from_cfg
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import RandomFlip
from mmdet3d.core.bbox import box_np_ops
from mmdet3d.datasets.builder import OBJECTSAMPLERS
from PIL import Image
from mmdet3d.datasets.pipelines.transforms_3d import ObjectRangeFilter, ObjectNameFilter
from mmdet3d.core.bbox import (CameraInstance3DBoxes, DepthInstance3DBoxes,
                               LiDARInstance3DBoxes, box_np_ops)

@PIPELINES.register_module()
class CustomObjectRangeFilter(ObjectRangeFilter):
    def __init__(self, with_bbox_2d=False, **kwargs):
        super(CustomObjectRangeFilter, self).__init__(**kwargs)
        self.with_bbox_2d = with_bbox_2d

    def __call__(self, input_dict):
        if isinstance(input_dict['gt_bboxes_3d'],
                      (LiDARInstance3DBoxes, DepthInstance3DBoxes)):
            bev_range = self.pcd_range[[0, 1, 3, 4]]
        elif isinstance(input_dict['gt_bboxes_3d'], CameraInstance3DBoxes):
            bev_range = self.pcd_range[[0, 2, 3, 5]]

        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        mask = gt_bboxes_3d.in_range_bev(bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        mask_numpy = mask.numpy().astype(np.bool)
        gt_labels_3d = gt_labels_3d[mask_numpy]

        # 2d bboxes to 3d bboxes mapping: -1 for not matched to any 3d bbox
        if self.with_bbox_2d:
            gt_ids = np.zeros(len(mask_numpy), dtype=np.int32)
            gt_ids[mask_numpy] = np.arange(len(gt_labels_3d))
            gt_ids[~mask_numpy] = -1
            gt_bboxes_2d_to_3d = input_dict['gt_bboxes_2d_to_3d']
            # assert all([(mapping > -1).all() for mapping in gt_bboxes_2d_to_3d])
            gt_bboxes_2d_to_3d_filtered = []
            for bboxes_2d_to_3d in gt_bboxes_2d_to_3d:
                bboxes_2d_to_3d[bboxes_2d_to_3d > -1] = gt_ids[bboxes_2d_to_3d[bboxes_2d_to_3d > -1]]
                gt_bboxes_2d_to_3d_filtered.append(bboxes_2d_to_3d)
            input_dict['gt_bboxes_2d_to_3d'] = gt_bboxes_2d_to_3d_filtered

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d

        return input_dict


@PIPELINES.register_module()
class CustomObjectNameFilter(ObjectNameFilter):
    def __init__(self, with_bbox_2d=False, **kwargs):
        super(CustomObjectNameFilter, self).__init__(**kwargs)
        self.with_bbox_2d = with_bbox_2d

    def __call__(self, input_dict):
        gt_labels_3d = input_dict['gt_labels_3d']
        gt_bboxes_mask = np.array([n in self.labels for n in gt_labels_3d],
                                  dtype=np.bool_)
        input_dict['gt_bboxes_3d'] = input_dict['gt_bboxes_3d'][gt_bboxes_mask]
        input_dict['gt_labels_3d'] = input_dict['gt_labels_3d'][gt_bboxes_mask]

        # remove corresponding 2d bboxes
        if self.with_bbox_2d:
            gt_ids = np.zeros(len(gt_bboxes_mask), dtype=np.int32)
            gt_ids[gt_bboxes_mask] = np.arange(len(input_dict['gt_labels_3d']))
            gt_ids[~gt_bboxes_mask] = -1
            gt_bboxes_2d = input_dict['gt_bboxes_2d']
            gt_labels_2d = input_dict['gt_labels_2d']
            gt_bboxes_2d_to_3d = input_dict['gt_bboxes_2d_to_3d']
            gt_bboxes_2d_filtered = []
            gt_labels_2d_filtered = []
            gt_bboxes_2d_to_3d_filtered = []
            for bboxes_2d, labels_2d, bboxes_2d_to_3d in zip(gt_bboxes_2d, gt_labels_2d, gt_bboxes_2d_to_3d):
                # 1. filter out 2d bboxes
                mask_2d = np.array([n in self.labels for n in labels_2d], dtype=np.bool_)
                gt_bboxes_2d_filtered.append(bboxes_2d[mask_2d])
                gt_labels_2d_filtered.append(labels_2d[mask_2d])
                bboxes_2d_to_3d_filtered = bboxes_2d_to_3d[mask_2d]
                # 2. adjust mapping
                bboxes_2d_to_3d_filtered[bboxes_2d_to_3d_filtered > -1] = \
                    gt_ids[bboxes_2d_to_3d_filtered[bboxes_2d_to_3d_filtered > -1]]
                gt_bboxes_2d_to_3d_filtered.append(bboxes_2d_to_3d_filtered)
            input_dict['gt_bboxes_2d'] = gt_bboxes_2d_filtered
            input_dict['gt_labels_2d'] = gt_labels_2d_filtered
            input_dict['gt_bboxes_2d_to_3d'] = gt_bboxes_2d_to_3d_filtered

        return input_dict

@PIPELINES.register_module()
class PadMultiViewImage(object):
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = [mmcv.impad(
                img, shape=self.size, pad_val=self.pad_val) for img in results['img']]
        elif self.size_divisor is not None:
            padded_img = [mmcv.impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val) for img in results['img']]
        results['img'] = padded_img
        results['img_shape'] = [img.shape for img in padded_img]
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results['img'] = [mmcv.imnormalize(
            img, self.mean, self.std, self.to_rgb) for img in results['img']]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class UnifiedObjectSample(object):
    """Sample GT objects to the data.

    Args:
        db_sampler (dict): Config dict of the database sampler.
        sample_2d (bool): Whether to also paste 2D image patch to the images
            This should be true when applying multi-modality cut-and-paste.
            Defaults to False.
    """

    def __init__(self, db_sampler, sample_2d=False, sample_method='depth', modify_points=False, mixup_rate=-1):
        self.sampler_cfg = db_sampler
        self.sample_2d = sample_2d
        self.sample_method = sample_method
        self.modify_points = modify_points
        self.mixup_rate = mixup_rate
        if 'type' not in db_sampler.keys():
            db_sampler['type'] = 'DataBaseSampler'
        self.db_sampler = build_from_cfg(db_sampler, OBJECTSAMPLERS)

    @staticmethod
    def remove_points_in_boxes(points, boxes):
        """Remove the points in the sampled bounding boxes.

        Args:
            points (:obj:`BasePoints`): Input point cloud array.
            boxes (np.ndarray): Sampled ground truth boxes.

        Returns:
            np.ndarray: Points with those in the boxes removed.
        """
        masks = box_np_ops.points_in_rbbox(points.coord.numpy(), boxes)
        points = points[np.logical_not(masks.any(-1))]
        return points

    def __call__(self, input_dict):
        """Call function to sample ground truth objects to the data.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after object sampling augmentation, \
                'points', 'gt_bboxes_3d', 'gt_labels_3d' keys are updated \
                in the result dict.
        """
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']

        # change to float for blending operation
        points = input_dict['points']
        if self.sample_2d:
            # Assume for now 3D & 2D bboxes are the same
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(),
                gt_labels_3d,
                with_img=True)
        else:
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(), gt_labels_3d, with_img=False)

        if sampled_dict is not None:
            sampled_gt_bboxes_3d = sampled_dict['gt_bboxes_3d']
            sampled_points = sampled_dict['points']
            sampled_points_idx = sampled_dict["points_idx"]
            sampled_gt_labels = sampled_dict['gt_labels_3d']

            gt_labels_3d = np.concatenate([gt_labels_3d, sampled_gt_labels],
                                          axis=0)
            gt_bboxes_3d = gt_bboxes_3d.new_box(
                np.concatenate(
                    [gt_bboxes_3d.tensor.numpy(), sampled_gt_bboxes_3d]))

            points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
            points_idx = -1 * np.ones(len(points), dtype=np.int)
            # check the points dimension
            # points = points.cat([sampled_points, points])
            points = points.cat([points, sampled_points])
            points_idx = np.concatenate([points_idx, sampled_points_idx], axis=0)

            if self.sample_2d:
                imgs = input_dict['img']
                lidar2img = input_dict['lidar2img']
                sampled_img = sampled_dict['images']
                sampled_num = len(sampled_gt_bboxes_3d)
                imgs, points_keep = self.unified_sample(imgs, lidar2img, 
                                            points.tensor.numpy(), 
                                            points_idx, gt_bboxes_3d.corners.numpy(), 
                                            sampled_img, sampled_num)
                
                input_dict['img'] = imgs

                if self.modify_points:
                    points = points[points_keep]

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d.astype(np.long)
        input_dict['points'] = points

        return input_dict

    def unified_sample(self, imgs, lidar2img, points, points_idx, bboxes_3d, sampled_img, sampled_num):
        # for boxes
        bboxes_3d = np.concatenate([bboxes_3d, np.ones_like(bboxes_3d[..., :1])], -1)
        is_raw = np.ones(len(bboxes_3d))
        is_raw[-sampled_num:] = 0
        is_raw = is_raw.astype(bool)
        raw_num = len(is_raw)-sampled_num
        # for point cloud
        points_3d = points[:,:4].copy()
        points_3d[:,-1] = 1
        points_keep = np.ones(len(points_3d)).astype(np.bool)
        new_imgs = imgs

        assert len(imgs)==len(lidar2img) and len(sampled_img)==sampled_num
        for _idx, (_img, _lidar2img) in enumerate(zip(imgs, lidar2img)):
            coord_img = bboxes_3d @ _lidar2img.T
            coord_img[...,:2] /= coord_img[...,2,None]
            depth = coord_img[...,2]
            img_mask = (depth > 0).all(axis=-1)
            img_count = img_mask.nonzero()[0]
            if img_mask.sum() == 0:
                continue
            depth = depth.mean(1)[img_mask]
            coord_img = coord_img[...,:2][img_mask]
            minxy = np.min(coord_img, axis=-2)
            maxxy = np.max(coord_img, axis=-2)
            bbox = np.concatenate([minxy, maxxy], axis=-1).astype(int)
            bbox[:,0::2] = np.clip(bbox[:,0::2], a_min=0, a_max=_img.shape[1]-1)
            bbox[:,1::2] = np.clip(bbox[:,1::2], a_min=0, a_max=_img.shape[0]-1)
            img_mask = ((bbox[:,2:]-bbox[:,:2]) > 1).all(axis=-1)
            if img_mask.sum() == 0:
                continue
            depth = depth[img_mask]
            if 'depth' in self.sample_method:
                paste_order = depth.argsort()
                paste_order = paste_order[::-1]
            else:
                paste_order = np.arange(len(depth), dtype=np.int64)
            img_count = img_count[img_mask][paste_order]
            bbox = bbox[img_mask][paste_order]

            paste_mask = -255 * np.ones(_img.shape[:2], dtype=np.int)
            fg_mask = np.zeros(_img.shape[:2], dtype=np.int)
            # first crop image from raw image
            raw_img = []
            for _count, _box in zip(img_count, bbox):
                if is_raw[_count]:
                    raw_img.append(_img[_box[1]:_box[3],_box[0]:_box[2]])

            # then stitch the crops to raw image
            for _count, _box in zip(img_count, bbox):
                if is_raw[_count]:
                    if self.mixup_rate < 0:
                        _img[_box[1]:_box[3],_box[0]:_box[2]] = raw_img.pop(0)
                    else:
                        _img[_box[1]:_box[3],_box[0]:_box[2]] = \
                            _img[_box[1]:_box[3],_box[0]:_box[2]] * (1 - self.mixup_rate) + raw_img.pop(0) * self.mixup_rate
                    fg_mask[_box[1]:_box[3],_box[0]:_box[2]] = 1
                else:
                    img_crop = sampled_img[_count-raw_num]
                    if len(img_crop)==0: continue
                    img_crop = cv2.resize(img_crop, tuple(_box[[2,3]]-_box[[0,1]]))
                    if self.mixup_rate < 0:
                        _img[_box[1]:_box[3],_box[0]:_box[2]] = img_crop
                    else:
                        _img[_box[1]:_box[3],_box[0]:_box[2]] = \
                            _img[_box[1]:_box[3],_box[0]:_box[2]] * (1 - self.mixup_rate) + img_crop * self.mixup_rate

                paste_mask[_box[1]:_box[3],_box[0]:_box[2]] = _count
            
            new_imgs[_idx] = _img

            # calculate modify mask
            if self.modify_points:
                points_img = points_3d @ _lidar2img.T
                points_img[:,:2] /= points_img[:,2,None]
                depth = points_img[:,2]
                img_mask = depth > 0
                if img_mask.sum() == 0:
                    continue
                img_mask = (points_img[:,0] > 0) & (points_img[:,0] < _img.shape[1]) & \
                           (points_img[:,1] > 0) & (points_img[:,1] < _img.shape[0]) & img_mask
                points_img = points_img[img_mask].astype(int)
                new_mask = paste_mask[points_img[:,1], points_img[:,0]]==(points_idx[img_mask]+raw_num)
                raw_fg = (fg_mask == 1) & (paste_mask >= 0) & (paste_mask < raw_num)
                raw_bg = (fg_mask == 0) & (paste_mask < 0)
                raw_mask = raw_fg[points_img[:,1], points_img[:,0]] | raw_bg[points_img[:,1], points_img[:,0]]
                keep_mask = new_mask | raw_mask
                points_keep[img_mask] = points_keep[img_mask] & keep_mask

        return new_imgs, points_keep

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f' sample_2d={self.sample_2d},'
        repr_str += f' data_root={self.sampler_cfg.data_root},'
        repr_str += f' info_path={self.sampler_cfg.info_path},'
        repr_str += f' rate={self.sampler_cfg.rate},'
        repr_str += f' prepare={self.sampler_cfg.prepare},'
        repr_str += f' classes={self.sampler_cfg.classes},'
        repr_str += f' sample_groups={self.sampler_cfg.sample_groups}'
        return repr_str


@PIPELINES.register_module()
class ResizeCropFlipImage(object):
    """Random resize, Crop and flip the image
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(self, data_aug_conf=None, training=True, pic_wise=False):
        self.data_aug_conf = data_aug_conf
        self.training = training
        self.pic_wise = pic_wise
        

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """

        imgs = results["img"]
        N = len(imgs)
        new_imgs = []
        new_depths = []
        resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))

            # augmentation (resize, crop, horizontal flip, rotate)
            if self.pic_wise:
                resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            if "depths" in results.keys():
                depth = results['depths'][i]
                depth = self._depth_transform(
                    depth,
                    resize=resize,
                    resize_dims=self.data_aug_conf["final_dim"],
                    crop=crop,
                    flip=flip,
                    rotate=rotate,
                )
                new_depths.append(depth.astype(np.float32))

            new_imgs.append(np.array(img).astype(np.float32))
            results['cam_intrinsic'][i][:3, :3] = ida_mat @ results['cam_intrinsic'][i][:3, :3]

        results["img"] = new_imgs
        results["depths"] = new_depths
        results['lidar2img'] = [results['cam_intrinsic'][i] @ results['lidar2cam'][i] for i in range(len(results['lidar2cam']))]
        
        if 'gt_bboxes_2d' in results.keys():
            gt_bboxes_2d = results['gt_bboxes_2d']
            gt_labels_2d = results['gt_labels_2d']
            gt_bboxes_2d_to_3d = results['gt_bboxes_2d_to_3d']
            gt_bboxes_ignore = results['gt_bboxes_ignore']
            processed_gt_bboxes_2d = []
            processed_gt_labels_2d = []
            processed_gt_bboxes_2d_to_3d = []
            processed_gt_bboxes_ignore = []
            for i in range(N):
                bboxes_2d = gt_bboxes_2d[i]
                labels_2d = gt_labels_2d[i]
                bboxes_2d_to_3d = gt_bboxes_2d_to_3d[i]
                bboxes_ignore = gt_bboxes_ignore[i]
                # 1. resize
                bboxes_2d = bboxes_2d * resize
                bboxes_ignore = bboxes_ignore * resize
                # 2. crop and filter out-of-image bboxes
                bboxes_2d[:, 0::2] = np.clip(bboxes_2d[:, 0::2], crop[0], crop[2])
                bboxes_2d[:, 1::2] = np.clip(bboxes_2d[:, 1::2], crop[1], crop[3])
                bboxes_2d[:, 0::2] = bboxes_2d[:, 0::2] - crop[0]
                bboxes_2d[:, 1::2] = bboxes_2d[:, 1::2] - crop[1]
                bboxes_area = (bboxes_2d[:, 2:] - bboxes_2d[:, :2]).prod(1)
                valid_mask = bboxes_area > 64
                bboxes_2d = bboxes_2d[valid_mask]
                labels_2d = labels_2d[valid_mask]
                bboxes_2d_to_3d = bboxes_2d_to_3d[valid_mask]

                bboxes_ignore[:, 0::2] = np.clip(bboxes_ignore[:, 0::2], crop[0], crop[2])
                bboxes_ignore[:, 1::2] = np.clip(bboxes_ignore[:, 1::2], crop[1], crop[3])
                bboxes_ignore[:, 0::2] = bboxes_ignore[:, 0::2] - crop[0]
                bboxes_ignore[:, 1::2] = bboxes_ignore[:, 1::2] - crop[1]
                bboxes_area = (bboxes_ignore[:, 2:] - bboxes_ignore[:, :2]).prod(1)
                valid_mask = bboxes_area > 64
                bboxes_ignore = bboxes_ignore[valid_mask]
                # 3. flip
                if flip:
                    flipped_bboxes = bboxes_2d.copy()
                    w = crop[2] - crop[0]
                    flipped_bboxes[..., 0::4] = w - bboxes_2d[..., 2::4]
                    flipped_bboxes[..., 2::4] = w - bboxes_2d[..., 0::4]
                    bboxes_2d = flipped_bboxes

                    flipped_bboxes = bboxes_ignore.copy()
                    w = crop[2] - crop[0]
                    flipped_bboxes[..., 0::4] = w - bboxes_ignore[..., 2::4]
                    flipped_bboxes[..., 2::4] = w - bboxes_ignore[..., 0::4]
                    bboxes_ignore = flipped_bboxes
                # 4. rotate and filter out-of-image bboxes
                A = self._get_rot(rotate / 180 * np.pi)
                b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
                b = A.matmul(-b) + b
                bbox_corners = np.stack([bboxes_2d[:, 0], bboxes_2d[:, 1], bboxes_2d[:, 0], bboxes_2d[:, 3],
                                         bboxes_2d[:, 2], bboxes_2d[:, 3], bboxes_2d[:, 2], bboxes_2d[:, 1]], axis=1).reshape(-1, 4, 2)
                bbox_corners = bbox_corners @ A.numpy().T + b.numpy()[None, None]
                bboxes_2d = np.concatenate([bbox_corners.min(1), bbox_corners.max(1)], axis=1)
                bboxes_2d[:, 0::2] = np.clip(bboxes_2d[:, 0::2], 0, crop[2] - crop[0])
                bboxes_2d[:, 1::2] = np.clip(bboxes_2d[:, 1::2], 0, crop[3] - crop[1])
                bboxes_area = (bboxes_2d[:, 2:] - bboxes_2d[:, :2]).prod(1)
                valid_mask = bboxes_area > 64
                bboxes_2d = bboxes_2d[valid_mask]
                labels_2d = labels_2d[valid_mask]
                bboxes_2d_to_3d = bboxes_2d_to_3d[valid_mask]

                bbox_corners = np.stack([bboxes_ignore[:, 0], bboxes_ignore[:, 1], bboxes_ignore[:, 0], bboxes_ignore[:, 3],
                                         bboxes_ignore[:, 2], bboxes_ignore[:, 3], bboxes_ignore[:, 2], bboxes_ignore[:, 1]], axis=1).reshape(-1, 4, 2)
                bbox_corners = bbox_corners @ A.numpy().T + b.numpy()[None, None]
                bboxes_ignore = np.concatenate([bbox_corners.min(1), bbox_corners.max(1)], axis=1)

                processed_gt_bboxes_2d.append(bboxes_2d)
                processed_gt_labels_2d.append(labels_2d)
                processed_gt_bboxes_2d_to_3d.append(bboxes_2d_to_3d)
                processed_gt_bboxes_ignore.append(bboxes_ignore)

            results['gt_bboxes_2d'] = processed_gt_bboxes_2d
            results['gt_labels_2d'] = processed_gt_labels_2d
            results['gt_bboxes_2d_to_3d'] = processed_gt_bboxes_2d_to_3d
            results['gt_bboxes_ignore'] = processed_gt_bboxes_ignore
        return results

    def _get_rot(self, h):

        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat

    def _sample_augmentation(self):
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if self.training:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def _depth_transform(self, cam_depth, resize, resize_dims, crop, flip, rotate):
        """
        Input:
            cam_depth: Nx3, 3: x,y,d
            resize: a float value
            resize_dims: self.ida_aug_conf["final_dim"] -> [H, W]
            crop: x1, y1, x2, y2
            flip: bool value
            rotate: an angle
        Output:
            cam_depth: [h/down_ratio, w/down_ratio, d]
        """

        H, W = resize_dims
        cam_depth[:, :2] = cam_depth[:, :2] * resize
        cam_depth[:, 0] -= crop[0]
        cam_depth[:, 1] -= crop[1]
        if flip:
            cam_depth[:, 0] = resize_dims[1] - cam_depth[:, 0]

        cam_depth[:, 0] -= W / 2.0
        cam_depth[:, 1] -= H / 2.0

        h = rotate / 180 * np.pi
        rot_matrix = [
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ]
        cam_depth[:, :2] = np.matmul(rot_matrix, cam_depth[:, :2].T).T

        cam_depth[:, 0] += W / 2.0
        cam_depth[:, 1] += H / 2.0

        depth_coords = cam_depth[:, :2].astype(np.int16)

        depth_map = np.zeros((H, W, 3))
        valid_mask = (
            (depth_coords[:, 1] < resize_dims[0])
            & (depth_coords[:, 0] < resize_dims[1])
            & (depth_coords[:, 1] >= 0)
            & (depth_coords[:, 0] >= 0)
        )
        depth_map[depth_coords[valid_mask, 1], depth_coords[valid_mask, 0], :] = cam_depth[valid_mask, :]

        return depth_map


@PIPELINES.register_module()
class GlobalRotScaleTransAll(object):
    """Apply global rotation, scaling and translation to a 3D scene.

    Args:
        rot_range (list[float]): Range of rotation angle.
            Defaults to [-0.78539816, 0.78539816] (close to [-pi/4, pi/4]).
        scale_ratio_range (list[float]): Range of scale ratio.
            Defaults to [0.95, 1.05].
        translation_std (list[float]): The standard deviation of translation
            noise. This applies random translation to a scene by a noise, which
            is sampled from a gaussian distribution whose standard deviation
            is set by ``translation_std``. Defaults to [0, 0, 0]
        shift_height (bool): Whether to shift height.
            (the fourth dimension of indoor points) when scaling.
            Defaults to False.
    """

    def __init__(self,
                 rot_range=[-0.78539816, 0.78539816],
                 scale_ratio_range=[0.95, 1.05],
                 translation_std=[0, 0, 0],
                 shift_height=False):
        seq_types = (list, tuple, np.ndarray)
        if not isinstance(rot_range, seq_types):
            assert isinstance(rot_range, (int, float)), \
                f'unsupported rot_range type {type(rot_range)}'
            rot_range = [-rot_range, rot_range]
        self.rot_range = rot_range

        assert isinstance(scale_ratio_range, seq_types), \
            f'unsupported scale_ratio_range type {type(scale_ratio_range)}'
        self.scale_ratio_range = scale_ratio_range

        if not isinstance(translation_std, seq_types):
            assert isinstance(translation_std, (int, float)), \
                f'unsupported translation_std type {type(translation_std)}'
            translation_std = [
                translation_std, translation_std, translation_std
            ]
        assert all([std >= 0 for std in translation_std]), \
            'translation_std should be positive'
        self.translation_std = translation_std
        self.shift_height = shift_height

    def _trans_bbox_points(self, input_dict):
        """Private function to translate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after translation, 'points', 'pcd_trans' \
                and keys in input_dict['bbox3d_fields'] are updated \
                in the result dict.
        """
        translation_std = np.array(self.translation_std, dtype=np.float32)
        trans_factor = np.random.normal(scale=translation_std, size=3).T
        
        input_dict['points'].translate(trans_factor)
        if 'radar' in input_dict:
            input_dict['radar'].translate(trans_factor)
        input_dict['pcd_trans'] = trans_factor
        for key in input_dict['bbox3d_fields']:
            input_dict[key].translate(trans_factor)

        trans_mat = np.eye(4)
        trans_mat[:3, -1] = trans_factor
        trans_mat_inv = np.linalg.inv(trans_mat)
        for view in range(len(input_dict["lidar2img"])):
            input_dict["lidar2img"][view] = input_dict["lidar2img"][view] @ trans_mat_inv
            input_dict["lidar2cam"][view] = input_dict["lidar2cam"][view] @ trans_mat_inv

    def _rot_bbox_points(self, input_dict):
        """Private function to rotate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after rotation, 'points', 'pcd_rotation' \
                and keys in input_dict['bbox3d_fields'] are updated \
                in the result dict.
        """
        if 'rot_degree' not in input_dict:
            rotation = self.rot_range
            noise_rotation = np.random.uniform(rotation[0], rotation[1])
        else:
            noise_rotation = input_dict['rot_degree']

        # if no bbox in input_dict, only rotate points
        if len(input_dict['bbox3d_fields']) == 0:
            if 'rot_degree' not in input_dict:
                rot_mat_T = input_dict['points'].rotate(noise_rotation)
                if 'radar' in input_dict:
                    input_dict['radar'].rotate(noise_rotation)
            else:
                rot_mat_T = input_dict['points'].rotate(-noise_rotation)
                if 'radar' in input_dict:
                    input_dict['radar'].rotate(-noise_rotation)
            input_dict['pcd_rotation'] = rot_mat_T

            rot_mat = torch.eye(4)
            rot_mat[:3, :3].copy_(rot_mat_T)
            rot_mat[0, 1], rot_mat[1, 0] = -rot_mat[0, 1], -rot_mat[1, 0]
            rot_mat_inv = torch.inverse(rot_mat)
            for view in range(len(input_dict["lidar2img"])):
                input_dict["lidar2img"][view] = (torch.tensor(input_dict["lidar2img"][view]).float() @ rot_mat_inv).numpy()
                input_dict["lidar2cam"][view] = (torch.tensor(input_dict["lidar2cam"][view]).float() @ rot_mat_inv).numpy()
            return

        # rotate points with bboxes
        for key in input_dict['bbox3d_fields']:
            if len(input_dict[key].tensor) != 0:
                points, rot_mat_T = input_dict[key].rotate(
                    noise_rotation, input_dict['points'])
                input_dict['points'] = points
                input_dict['pcd_rotation'] = rot_mat_T
                if 'radar' in input_dict:
                    input_dict['radar'].rotate(-noise_rotation)

                rot_mat = torch.eye(4)
                rot_mat[:3, :3].copy_(rot_mat_T)
                rot_mat[0, 1], rot_mat[1, 0] = -rot_mat[0, 1], -rot_mat[1, 0]
                rot_mat_inv = torch.inverse(rot_mat)
                for view in range(len(input_dict["lidar2img"])):
                    input_dict["lidar2img"][view] = (torch.tensor(input_dict["lidar2img"][view]).float() @ rot_mat_inv).numpy()
                    input_dict["lidar2cam"][view] = (torch.tensor(input_dict["lidar2cam"][view]).float() @ rot_mat_inv).numpy()


    def _scale_bbox_points(self, input_dict):
        """Private function to scale bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points'and keys in \
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        scale = input_dict['pcd_scale_factor']
        points = input_dict['points']
        points.scale(scale)
        if self.shift_height:
            assert 'height' in points.attribute_dims.keys(), \
                'setting shift_height=True but points have no height attribute'
            points.tensor[:, points.attribute_dims['height']] *= scale
        input_dict['points'] = points
        
        if 'radar' in input_dict:
            input_dict['radar'].scale(scale)
            
        for key in input_dict['bbox3d_fields']:
            input_dict[key].scale(scale)

        scale_mat = torch.tensor(
            [
                [scale, 0, 0, 0],
                [0, scale, 0, 0],
                [0, 0, scale, 0],
                [0, 0, 0, 1],
            ]
        )
        scale_mat_inv = torch.inverse(scale_mat)
        for view in range(len(input_dict["lidar2img"])):
            input_dict["lidar2img"][view] = (torch.tensor(input_dict["lidar2img"][view]).float() @ scale_mat_inv).numpy()
            input_dict["lidar2cam"][view] = (torch.tensor(input_dict["lidar2cam"][view]).float() @ scale_mat_inv).numpy()

    def _random_scale(self, input_dict):
        """Private function to randomly set the scale factor.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'pcd_scale_factor' are updated \
                in the result dict.
        """
        scale_factor = np.random.uniform(self.scale_ratio_range[0],
                                         self.scale_ratio_range[1])
        input_dict['pcd_scale_factor'] = scale_factor

    def __call__(self, input_dict):
        """Private function to rotate, scale and translate bounding boxes and \
        points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
                'pcd_scale_factor', 'pcd_trans' and keys in \
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        self._rot_bbox_points(input_dict)

        if 'pcd_scale_factor' not in input_dict:
            self._random_scale(input_dict)
        self._scale_bbox_points(input_dict)

        self._trans_bbox_points(input_dict)

        input_dict['transformation_3d_flow'].extend(['R', 'S', 'T'])
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(rot_range={self.rot_range},'
        repr_str += f' scale_ratio_range={self.scale_ratio_range},'
        repr_str += f' translation_std={self.translation_std},'
        repr_str += f' shift_height={self.shift_height})'
        return repr_str


@PIPELINES.register_module()
class CustomRandomFlip3D(object):
    """Flip the points & bbox.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        sync_2d (bool, optional): Whether to apply flip according to the 2D
            images. If True, it will apply the same flip as that to 2D images.
            If False, it will decide whether to flip randomly and independently
            to that of 2D images. Defaults to True.
        flip_ratio_bev_horizontal (float, optional): The flipping probability
            in horizontal direction. Defaults to 0.0.
        flip_ratio_bev_vertical (float, optional): The flipping probability
            in vertical direction. Defaults to 0.0.
    """

    def __init__(self,
                 flip_ratio_bev_horizontal=0.0,
                 flip_ratio_bev_vertical=0.0,
                 **kwargs):
        super(CustomRandomFlip3D, self).__init__()
        self.flip_ratio_bev_horizontal = flip_ratio_bev_horizontal
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical
        if flip_ratio_bev_horizontal is not None:
            assert isinstance(
                flip_ratio_bev_horizontal,
                (int, float)) and 0 <= flip_ratio_bev_horizontal <= 1
        if flip_ratio_bev_vertical is not None:
            assert isinstance(
                flip_ratio_bev_vertical,
                (int, float)) and 0 <= flip_ratio_bev_vertical <= 1

    def random_flip_data_3d(self, input_dict, direction='horizontal'):
        """Flip 3D data randomly.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            direction (str): Flip direction. Default: horizontal.

        Returns:
            dict: Flipped results, 'points', 'bbox3d_fields' keys are \
                updated in the result dict.
        """
        assert direction in ['horizontal', 'vertical']
        if len(input_dict['bbox3d_fields']) == 0:  # test mode
            input_dict['bbox3d_fields'].append('empty_box3d')
            input_dict['empty_box3d'] = input_dict['box_type_3d'](
                np.array([], dtype=np.float32))
        assert len(input_dict['bbox3d_fields']) == 1
        for key in input_dict['bbox3d_fields']:
            if 'points' in input_dict:
                input_dict['points'] = input_dict[key].flip(
                    direction, points=input_dict['points'])
            else:
                input_dict[key].flip(direction)
            if 'radar' in input_dict:
                input_dict['radar'].flip(direction)

    def __call__(self, input_dict):
        """Call function to flip points, values in the ``bbox3d_fields`` and \
        also flip 2D image and its annotations.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction', \
                'pcd_horizontal_flip' and 'pcd_vertical_flip' keys are added \
                into result dict.
        """
        if 'pcd_horizontal_flip' not in input_dict:
            flip_horizontal = True if np.random.rand(
            ) < self.flip_ratio_bev_horizontal else False
            input_dict['pcd_horizontal_flip'] = flip_horizontal
        if 'pcd_vertical_flip' not in input_dict:
            flip_vertical = True if np.random.rand(
            ) < self.flip_ratio_bev_vertical else False
            input_dict['pcd_vertical_flip'] = flip_vertical

        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        flip_mat = np.eye(4)
        if input_dict['pcd_horizontal_flip']:
            self.random_flip_data_3d(input_dict, 'horizontal')
            input_dict['transformation_3d_flow'].extend(['HF'])
            flip_mat[1, 1] = -1
        if input_dict['pcd_vertical_flip']:
            self.random_flip_data_3d(input_dict, 'vertical')
            input_dict['transformation_3d_flow'].extend(['VF'])
            flip_mat[0, 0] = -1
        for view in range(len(input_dict["lidar2img"])):
            input_dict["lidar2img"][view] = input_dict["lidar2img"][view] @ flip_mat
            input_dict["lidar2cam"][view] = input_dict["lidar2cam"][view] @ flip_mat
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(sync_2d={self.sync_2d},'
        repr_str += f' flip_ratio_bev_vertical={self.flip_ratio_bev_vertical})'
        return repr_str


@PIPELINES.register_module()
class ModalMask3D(object):

    def __init__(self, mode='test', mask_modal='image', **kwargs):
        super(ModalMask3D, self).__init__()
        self.mode = mode
        self.mask_modal = mask_modal

    def __call__(self, input_dict):
        if self.mode == 'test':
            if self.mask_modal == 'image':
                input_dict['img'] = [0. * item for item in input_dict['img']]
            if self.mask_modal == 'points':
                input_dict['points'].tensor = input_dict['points'].tensor * 0.0
        else:
            seed = np.random.rand()
            if seed > 0.75:
                input_dict['img'] = [0. * item for item in input_dict['img']]
            elif seed > 0.5:
                input_dict['points'].tensor = input_dict['points'].tensor * 0.0

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class GlobalRotScaleTransImage(object):
    """Random resize, Crop and flip the image
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(
        self,
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        reverse_angle=False,
        training=True,
        flip_dx_ratio=0.0,
        flip_dy_ratio=0.0
    ):

        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std

        self.reverse_angle = reverse_angle
        self.training = training
        
        self.flip_dx_ratio = flip_dx_ratio
        self.flip_dy_ratio = flip_dy_ratio

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        # random rotate
        rot_angle = np.random.uniform(*self.rot_range) # 我们采样得到一个角度

        self.rotate_bev_along_z(results, rot_angle) # 实际是旋转了一个-rot_angle角度
        if self.reverse_angle:
            rot_angle *= -1
        results["gt_bboxes_3d"].rotate(
            np.array(rot_angle)
        )  # mmdet LiDARInstance3DBoxes存的角度方向是反的(rotate函数实现的是绕着z轴由y向x转)(你说的对，但是在改动lidar2cam的时候，我们是按照rotate函数的方向来的，所以这里不用改)

        # random scale
        scale_ratio = np.random.uniform(*self.scale_ratio_range)
        self.scale_xyz(results, scale_ratio)
        results["gt_bboxes_3d"].scale(scale_ratio)

        # TODO: support translation

        self.flip_xy(results)

        return results

    def rotate_bev_along_z(self, results, angle):
        rot_cos = torch.cos(torch.tensor(angle))
        rot_sin = torch.sin(torch.tensor(angle))

        rot_mat = torch.tensor([[rot_cos, -rot_sin, 0, 0], [rot_sin, rot_cos, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) # oldLiDAR2newLiDAR 但是旋转-θ角度
        rot_mat_inv = torch.inverse(rot_mat) # newLiDAR2oldLiDAR

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ rot_mat_inv).numpy()
            results["lidar2cam"][view] = (torch.tensor(results["lidar2cam"][view]).float() @ rot_mat_inv).numpy()

        return

    def scale_xyz(self, results, scale_ratio):
        rot_mat = torch.tensor(
            [
                [scale_ratio, 0, 0, 0],
                [0, scale_ratio, 0, 0],
                [0, 0, scale_ratio, 0],
                [0, 0, 0, 1],
            ]
        ) # 所有的坐标扩大scale_ratio倍，原来的单位向量在新坐标系下的模为scale_ratio，所以这是oldLiDAR2newLiDAR

        rot_mat_inv = torch.inverse(rot_mat) # newLiDAR2oldLiDAR

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ rot_mat_inv).numpy()
            results["lidar2cam"][view] = (torch.tensor(results["lidar2cam"][view]).float() @ rot_mat_inv).numpy()
        return

    def flip_xy(self, results):
        mat = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
        )
        if np.random.rand() < self.flip_dx_ratio:
            mat[0][0] = -1
            results["gt_bboxes_3d"].flip(bev_direction='vertical')
        if np.random.rand() < self.flip_dy_ratio:
            mat[1][1] = -1
            results["gt_bboxes_3d"].flip(bev_direction='horizontal')
            
        num_view = len(results['lidar2img'])
        for view in range(num_view):
            results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ mat.float()).numpy()
            results["lidar2cam"][view] = (torch.tensor(results["lidar2cam"][view]).float() @ mat.float()).numpy()
        return
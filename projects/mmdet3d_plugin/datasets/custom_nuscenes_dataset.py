# ------------------------------------------------------------------------
# Copyright (c) 2023 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from os import path as osp
import pandas as pd
from refile import smart_open
from prettytable import PrettyTable
import json
from mmdet.datasets.api_wrappers import COCO
from .pipelines.customCompose import CustomCompose

import mmcv


@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, *args, ann_file_2d=None, return_gt_info=False, ft_begin_epoch=None, return_2d_annos=False, **kwargs):
        super(CustomNuScenesDataset, self).__init__(*args, **kwargs)
        self.return_gt_info = return_gt_info
        self.return_2d_annos = return_2d_annos
        if self.return_2d_annos:
            assert ann_file_2d is not None
            self.load_annotations_2d(ann_file_2d)
        self.pipeline = CustomCompose(kwargs['pipeline'], ft_begin_epoch)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.pipeline.__setattr__('epoch', epoch)

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
            img_sweeps=None if 'img_sweeps' not in info else info['img_sweeps'],
            radar_info=None if 'radars' not in info else info['radars']
        )

        if self.return_gt_info:
            input_dict['info'] = info

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            img_timestamp = []
            for cam_type, cam_info in info['cams'].items():
                img_timestamp.append(cam_info['timestamp'] / 1e6)
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_timestamp=img_timestamp,
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos # 在这里的gt_bbox_3d的所有值都是正确的，只不过LidarInstance3DBoxes的dim在类中被认为是wlh,但实际是lwh

            if self.return_2d_annos:
                gt_bboxes_3d = annos['gt_bboxes_3d']  # 3d bboxes
                gt_labels_3d = annos['gt_labels_3d']
                gt_bboxes_2d = []  # per-view 2d bboxes
                gt_bboxes_ignore = []  # per-view 2d bboxes
                gt_bboxes_2d_to_3d = []  # mapping from per-view 2d bboxes to 3d bboxes
                gt_labels_2d = []  # mapping from per-view 2d bboxes to 3d bboxes

                for cam_i in range(len(image_paths)):
                    ann_2d = self.impath_to_ann2d(image_paths[cam_i])
                    labels_2d = ann_2d['labels']
                    bboxes_2d = ann_2d['bboxes_2d']
                    bboxes_ignore = ann_2d['gt_bboxes_ignore']
                    bboxes_cam = ann_2d['bboxes_cam']
                    lidar2cam = lidar2cam_rts[cam_i]

                    centers_lidar = gt_bboxes_3d.gravity_center.numpy()
                    centers_lidar_hom = np.concatenate([centers_lidar, np.ones((len(centers_lidar), 1))], axis=1)
                    centers_cam = (centers_lidar_hom @ lidar2cam.T)[:, :3]
                    match = self.center_match(bboxes_cam, centers_cam)
                    assert (labels_2d[match > -1] == gt_labels_3d[match[match > -1]]).all()

                    gt_bboxes_2d.append(bboxes_2d)
                    gt_bboxes_2d_to_3d.append(match)
                    gt_labels_2d.append(labels_2d)
                    gt_bboxes_ignore.append(bboxes_ignore)

                annos['gt_bboxes_2d'] = gt_bboxes_2d
                annos['gt_labels_2d'] = gt_labels_2d
                annos['gt_bboxes_2d_to_3d'] = gt_bboxes_2d_to_3d
                annos['gt_bboxes_ignore'] = gt_bboxes_ignore
                
        return input_dict
    
    def center_match(self, bboxes_a, bboxes_b):
        cts_a, cts_b = bboxes_a[:, :3], bboxes_b[:, :3]
        if len(cts_a) == 0:
            return np.zeros(len(cts_a), dtype=np.int32) - 1
        if len(cts_b) == 0:
            return np.zeros(len(cts_a), dtype=np.int32) - 1
        dist = np.abs(cts_a[:, None] - cts_b[None]).sum(-1)
        match = dist.argmin(1)
        match[dist.min(1) > 1e-3] = -1
        return match

    def load_annotations_2d(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.impath_to_imgid = {}
        self.imgid_to_dataid = {}
        data_infos = []
        total_ann_ids = []
        for i in self.coco.get_img_ids():
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            self.impath_to_imgid['./data/nuscenes/' + info['file_name']] = i
            self.imgid_to_dataid[i] = len(data_infos)
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        self.data_infos_2d = data_infos

    def impath_to_ann2d(self, impath):
        img_id = self.impath_to_imgid[impath]
        data_id = self.imgid_to_dataid[img_id]
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self.get_ann_info_2d(self.data_infos_2d[data_id], ann_info)
    
    def get_ann_info_2d(self, img_info_2d, ann_info_2d):
        """Parse bbox annotation.

        Args:
            img_info (list[dict]): Image info.
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, labels,
                gt_bboxes_3d, gt_labels_3d, attr_labels, centers2d,
                depths, bboxes_ignore, masks, seg_map
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_bboxes_cam3d = []
        for i, ann in enumerate(ann_info_2d):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info_2d['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info_2d['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_name'] not in self.CLASSES:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.CLASSES.index(ann['category_name']))
                bbox_cam3d = np.array(ann['bbox_cam3d']).reshape(1, -1)
                bbox_cam3d = np.concatenate([bbox_cam3d], axis=-1)
                gt_bboxes_cam3d.append(bbox_cam3d.squeeze())

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_cam3d:
            gt_bboxes_cam3d = np.array(gt_bboxes_cam3d, dtype=np.float32)
        else:
            gt_bboxes_cam3d = np.zeros((0, 6), dtype=np.float32)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes_cam=gt_bboxes_cam3d,
            bboxes_2d=gt_bboxes,
            gt_bboxes_ignore=gt_bboxes_ignore,
            labels=gt_labels, )
        return ann
    
    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            metric (str, optional): Metric name used for evaluation.
                Default: 'bbox'.
            result_name (str, optional): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import NuScenesEval

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        nusc_eval = NuScenesEval(
            nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False)
        nusc_eval.main(render_curves=False)

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        path = output_dir + "/metrics_summary.json"
        with smart_open(path, "rb") as fr:
            result = json.load(fr)

            res_sumarry = PrettyTable()
            res_sumarry.add_column(
                "category",
                ["car", "truck", "bus", "trailer", "construction_vehicle", "pedestrian", "motorcycle", "bicycle", "traffic_cone", "barrier", "over_all"]
            )

            ap_res = list(np.round(list(result["mean_dist_aps"].values()), 3)) + [round(result["mean_ap"], 4)]
            res_sumarry.add_column("AP", ap_res)

            label_tp_error_df = pd.DataFrame(result["label_tp_errors"])
            label_tp_error_df_trans = label_tp_error_df.T
            metric_name_map = {
                "trans_err": "ATE",
                "scale_err": "ASE",
                "orient_err": "AOE",
                "vel_err": "AVE",
                "attr_err": "AAE",
            }
            tp_errors = result["tp_errors"]
            for key in label_tp_error_df_trans.keys():
                res_sumarry.add_column(metric_name_map[key], list(np.round(label_tp_error_df_trans[key].to_list(), 3)) + [round(tp_errors[key], 4)])

            with open(output_dir + "/eval_result_sumarry.txt", "w") as fw:
                fw.write(str(res_sumarry))
                fw.write("\n")
                fw.write("NDS: " + str(round(result["nd_score"], 4)))
            print("Save Result Sumarry to: ", output_dir + "/eval_result_sumarry.txt")
        
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val

        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail
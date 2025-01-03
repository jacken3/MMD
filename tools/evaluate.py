from nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import NuScenesEval
import json
import os
from os import path as osp
from nuscenes.eval.detection.data_classes import DetectionConfig
import mmcv
from prettytable import PrettyTable
import numpy as np
import pandas as pd
from refile import smart_open


def config_factory(configuration_name: str) -> DetectionConfig:
    """
    Creates a DetectionConfig instance that can be used to initialize a NuScenesEval instance.
    Note that this only works if the config file is located in the nuscenes/eval/detection/configs folder.
    :param configuration_name: Name of desired configuration in eval_detection_configs.
    :return: DetectionConfig instance.
    """

    # Check if config exists.
    cfg_path = "detection_cvpr_2019.json"

    # Load config file and deserialize it.
    with open(cfg_path, 'r') as f:
        data = json.load(f)
    cfg = DetectionConfig.deserialize(data)

    return cfg


output_dir = './output/'
result_path = "test/ICFusion_lidar_voxel0075_cbgs_15+5_dn_separate_head_3dQuery/Sun_Dec_29_16_27_10_2024/pts_bbox/results_nusc.json"
eval_detection_configs = config_factory('detection_cvpr_2019')
nusc = NuScenes(
    version='v1.0-trainval', dataroot='data/nuscenes', verbose=False)
eval_set_map = {
    'v1.0-mini': 'mini_val',
    'v1.0-trainval': 'train',
}
nusc_eval = NuScenesEval(
    nusc,
    config=eval_detection_configs,
    result_path=result_path,
    eval_set=eval_set_map['v1.0-trainval'],
    output_dir='./',
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
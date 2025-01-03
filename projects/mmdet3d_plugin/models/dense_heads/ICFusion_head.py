# ------------------------------------------------------------------------
# Copyright (c) 2023 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

from distutils.command.build import build
import enum
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer, Linear, bias_init_with_prob, Conv2d
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmcv.runner import BaseModule, force_fp32
from mmcv.cnn import xavier_init, constant_init, kaiming_init
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean, build_bbox_coder, bbox2roi)
from mmdet.models.utils import build_transformer
from mmdet.models import HEADS, build_loss
from mmdet.models.utils import NormedLinear
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.models.utils.clip_sigmoid import clip_sigmoid
from mmdet3d.models import builder
from mmdet3d.core import (circle_nms, draw_heatmap_gaussian, gaussian_radius,
                          xywhr2xyxyr)
from einops import rearrange
import collections
from mmdet.models.builder import build_roi_extractor

from functools import reduce
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from projects.mmdet3d_plugin.models.utils.query_generator import QueryGenerator
from projects.mmdet3d_plugin.models.utils.gaussian import gaussian_radius


def pos2embed(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = 2 * (dim_t // 2) / num_pos_feats + 1
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb

def pos2emb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, groups, eps):
        ctx.groups = groups
        ctx.eps = eps
        N, C, L = x.size()
        x = x.view(N, groups, C // groups, L)
        mu = x.mean(2, keepdim=True)
        var = (x - mu).pow(2).mean(2, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1) * y.view(N, C, L) + bias.view(1, C, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        groups = ctx.groups
        eps = ctx.eps

        N, C, L = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1)
        g = g.view(N, groups, C//groups, L)
        mean_g = g.mean(dim=2, keepdim=True)
        mean_gy = (g * y).mean(dim=2, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx.view(N, C, L), (grad_output * y.view(N, C, L)).sum(dim=2).sum(dim=0), grad_output.sum(dim=2).sum(
            dim=0), None, None


class GroupLayerNorm1d(nn.Module):

    def __init__(self, channels, groups=1, eps=1e-6):
        super(GroupLayerNorm1d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.groups = groups
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.groups, self.eps)

@HEADS.register_module()
class SeparateTaskHead(BaseModule):
    """SeparateHead for CenterHead.

    Args:
        in_channels (int): Input channels for conv_layer.
        heads (dict): Conv information.
        head_conv (int): Output channels.
            Default: 64.
        final_kernal (int): Kernal size for the last conv layer.
            Deafult: 1.
        init_bias (float): Initial bias. Default: -2.19.
        conv_cfg (dict): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict): Config of norm layer.
            Default: dict(type='BN2d').
        bias (str): Type of bias. Default: 'auto'.
    """

    def __init__(self,
                 in_channels,
                 heads,
                 groups=1,
                 head_conv=64,
                 final_kernel=1,
                 init_bias=-2.19,
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super(SeparateTaskHead, self).__init__(init_cfg=init_cfg)
        self.heads = heads
        self.groups = groups
        self.init_bias = init_bias
        for head in self.heads:
            reg_output_dim, num_conv = self.heads[head]

            conv_layers = []
            c_in = in_channels
            for i in range(num_conv - 1):
                conv_layers.extend([
                    nn.Conv1d(
                        c_in * groups,
                        head_conv * groups,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        groups=groups,
                        bias=False),
                    GroupLayerNorm1d(head_conv * groups, groups=groups),
                    nn.ReLU(inplace=True)
                ])
                c_in = head_conv

            conv_layers.append(
                nn.Conv1d(
                    head_conv * groups,
                    reg_output_dim * groups,
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    groups=groups,
                    bias=True))
            conv_layers = nn.Sequential(*conv_layers)

            self.__setattr__(head, conv_layers)

            if init_cfg is None:
                self.init_cfg = dict(type='Kaiming', layer='Conv1d')

    def init_weights(self):
        """Initialize weights."""
        super().init_weights()
        for head in self.heads:
            if head == 'cls_logits':
                self.__getattr__(head)[-1].bias.data.fill_(self.init_bias)

    def forward(self, x):
        """Forward function for SepHead.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [N, B, query, C].

        Returns:
            dict[str: torch.Tensor]: contains the following keys:

                -reg （torch.Tensor): 2D regression value with the \
                    shape of [N, B, query, 2].
                -height (torch.Tensor): Height value with the \
                    shape of [N, B, query, 1].
                -dim (torch.Tensor): Size value with the shape \
                    of [N, B, query, 3].
                -rot (torch.Tensor): Rotation value with the \
                    shape of [N, B, query, 2].
                -vel (torch.Tensor): Velocity value with the \
                    shape of [N, B, query, 2].
        """
        N, B, query_num, c1 = x.shape
        x = rearrange(x, "n b q c -> b (n c) q")
        ret_dict = dict()

        for head in self.heads:
             head_output = self.__getattr__(head)(x)
             ret_dict[head] = rearrange(head_output, "b (n c) q -> n b q c", n=N)

        return ret_dict

@HEADS.register_module()
class ICFusionHead(BaseModule):

    def __init__(self,
                 num_classes,
                 pts_in_channels,
                 num_query=900,
                 hidden_dim=128,
                 num_reg_fcs=2,
                 depth_num=64,
                 LID=False,
                 with_multiview=False,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 use_separate_head = False,
                 separate_head = None,
                 bg_cls_weight = 0,
                 if_depth_pe=False,
                 share_pe=False,
                 query_3dpe=False,
                 lidar_3d_pe=False,
                 generate_with_pe=False,
                 depth_start=1,
                 downsample_scale=8,
                 scalar=10,
                 noise_scale=1.0,
                 noise_trans=0.0,
                 dn_weight=1.0,
                 with_dn=True,
                 split=0.75,
                 if_2d_prior=False,
                 query_generator=None,
                 bbox_roi_extractor=None,
                 if_3d_prior=False,
                 heatmap_layer=dict(
                    num_conv = 2,
                    proposal_head_kernel = 3
                ),
                 max_foreground_token=10000,
                 num_3d_proposals=200,
                 position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                 train_cfg=None,
                 test_cfg=None,
                 transformer=None,
                 bbox_coder=None,
                 input_modality=dict(
                    use_lidar=False,
                    use_camera=False,
                 ),
                 loss_cls=dict(
                     type="FocalLoss",
                     use_sigmoid=True,
                     reduction="mean",
                     gamma=2, alpha=0.25, loss_weight=1.0
                 ),
                 loss_bbox=dict(
                    type="L1Loss",
                    reduction="mean",
                    loss_weight=0.25,
                 ),
                 loss_heatmap=dict(
                     type="GaussianFocalLoss",
                     reduction="mean"
                 ),
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None
        assert input_modality['use_lidar'] or input_modality['use_camera']
        super(ICFusionHead, self).__init__(init_cfg=init_cfg)
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10

        self.input_modality = input_modality
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_query = num_query
        self.pts_in_channels = pts_in_channels
        self.depth_num = depth_num
        self.LID = LID
        self.with_multiview = with_multiview
        self.query_3dpe = query_3dpe
        self.lidar_3d_pe = lidar_3d_pe
        self.positional_encoding = positional_encoding #for multiview pe

        self.use_separate_head = use_separate_head
        self.separate_head = separate_head

        self.downsample_scale = downsample_scale #
        # dn config
        self.with_dn = with_dn
        self.scalar = scalar
        self.bbox_noise_scale = noise_scale
        self.bbox_noise_trans = noise_trans
        self.dn_weight = dn_weight
        self.bg_cls_weight = bg_cls_weight
        self.split = split
        self.if_depth_pe = if_depth_pe
        self.share_pe = share_pe
        self.depth_start = depth_start

        self.if_2d_prior = if_2d_prior
        self.generate_with_pe = generate_with_pe
        self.query_generator = query_generator
        self.bbox_roi_extractor = bbox_roi_extractor

        self.if_3d_prior = if_3d_prior
        self.heatmap_layer = heatmap_layer
        self.max_foreground_token = max_foreground_token
        self.num_3d_proposals = num_3d_proposals
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_heatmap = build_loss(loss_heatmap)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.position_range = position_range
        self.fp16_enabled = False
        self.num_reg_fcs = num_reg_fcs

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        if input_modality['use_lidar']:   
            self.shared_conv = ConvModule(
                pts_in_channels,
                hidden_dim,
                kernel_size=3,
                padding=1,
                conv_cfg=dict(type="Conv2d"),
                norm_cfg=dict(type="BN2d")
            )

            if not self.lidar_3d_pe and self.query_3dpe:
                self.bev_2d_embedding = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim)
                ) # 用于对BEV位置信息进行编码(主要对Lidar栅格进行编码)
            
            if self.lidar_3d_pe and not self.query_3dpe:
                self.bev_3d_embedding = nn.Sequential(
                    nn.Linear(hidden_dim * 3 // 2, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim)
                ) 

        # transformer
        self.num_pred = transformer.decoder.num_layers
        self.transformer = build_transformer(transformer)
        self.reference_points = nn.Embedding(num_query, 3)

        if self.query_3dpe:
            self.bev_3d_embedding = nn.Sequential(
                nn.Linear(hidden_dim * 3 // 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim)
            ) 
        else:
            self.bev_2d_embedding = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim)
                ) 

        if input_modality['use_camera']: 
            self.input_proj = Conv2d(
                self.hidden_dim, self.hidden_dim, kernel_size=1)
            
            if self.if_depth_pe and (not share_pe or not self.query_3dpe):
                self.bev_depth_embedding = nn.Sequential(
                    nn.Linear(hidden_dim * 3 // 2, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            if not self.if_depth_pe:
                self.rv_embedding = nn.Sequential(
                nn.Conv2d(self.depth_num * 3, hidden_dim*4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(hidden_dim*4, hidden_dim, kernel_size=1, stride=1, padding=0),
            ) # （用于对图像特征点对应的3D射线进行位置编码）
        # assigner
        if train_cfg:
            self.assigner = build_assigner(train_cfg["assigner"])
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self._init_layers()

    def _init_layers(self):

        if self.with_multiview:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.hidden_dim*3//2, self.hidden_dim*4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.hidden_dim*4, self.hidden_dim, kernel_size=1, stride=1, padding=0),
            )
            self.positional_encoding = build_positional_encoding(
                self.positional_encoding)

        if not self.use_separate_head:
            cls_branch = []
            for _ in range(self.num_reg_fcs):
                cls_branch.append(Linear(self.hidden_dim, self.hidden_dim))
                cls_branch.append(nn.LayerNorm(self.hidden_dim))
                cls_branch.append(nn.ReLU(inplace=True))
            cls_branch.append(Linear(self.hidden_dim, self.cls_out_channels))
            fc_cls = nn.Sequential(*cls_branch)

            reg_branch = []
            for _ in range(self.num_reg_fcs):
                reg_branch.append(Linear(self.hidden_dim, self.hidden_dim))
                reg_branch.append(nn.ReLU())
            reg_branch.append(Linear(self.hidden_dim, self.code_size))
            reg_branch = nn.Sequential(*reg_branch)

            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(self.num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(self.num_pred)])

        else:
            self.separate_head = builder.build_head(self.separate_head)

        if self.if_2d_prior:
            assert self.query_generator is not None and self.bbox_roi_extractor is not None
            self.roi_size = self.bbox_roi_extractor['roi_layer']['output_size']
            if isinstance(self.roi_size, int):
                self.roi_size = [self.roi_size, self.roi_size]
            self.query_generator = QueryGenerator(**self.query_generator)
            self.bbox_roi_extractor = build_roi_extractor(self.bbox_roi_extractor)
            if hasattr(self.bbox_roi_extractor, 'fp16_enabled'):
                del self.bbox_roi_extractor.fp16_enabled

        if self.if_3d_prior:
            self.maxpool_2d = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            self.maxpool_2d_small = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)

            num_conv = self.heatmap_layer['num_conv']
            proposal_head_kernel = self.heatmap_layer['proposal_head_kernel']
            fc_list = []
            fc_list = []
            for _ in range(num_conv - 1):
                fc_list.append(
                    nn.Sequential(
                        nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=proposal_head_kernel, padding=proposal_head_kernel//2, bias=False),
                        nn.BatchNorm2d(self.hidden_dim, eps=1e-3, momentum=0.01),
                        nn.ReLU(inplace=True)
                    )
                )
            fc_list.append(
                nn.Conv2d(self.hidden_dim, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )
            self.heatmap_layer = nn.Sequential(*fc_list)
            # 初始化偏置
            self.heatmap_layer[-1].bias.data.fill_(-2.19)

    def init_weights(self):
        # super(ICFusionHead, self).init_weights()
        self.transformer.init_weights()
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        if not self.use_separate_head:
            if self.loss_cls.use_sigmoid:
                bias_init = bias_init_with_prob(0.01)
                for m in self.cls_branches:
                    nn.init.constant_(m[-1].bias, bias_init)

    @property
    def coords_bev(self):
        cfg = self.train_cfg if self.train_cfg else self.test_cfg
        x_size, y_size = (
            cfg['grid_size'][1] // self.downsample_scale,
            cfg['grid_size'][0] // self.downsample_scale
        )
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = (batch_x + 0.5) / x_size
        batch_y = (batch_y + 0.5) / y_size
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)
        coord_base = coord_base.view(2, -1).transpose(1, 0) # (H*W, 2)
        if self.lidar_3d_pe:
            coord_base = torch.cat([coord_base, torch.ones_like(coord_base[:, 0:1])*0.5], dim=1)
        return coord_base

    def prepare_for_dn(self, batch_size, reference_points, img_metas):
        if self.training and self.with_dn:
            targets = [torch.cat((img_meta['gt_bboxes_3d']._data.gravity_center, img_meta['gt_bboxes_3d']._data.tensor[:, 3:]),dim=1) for img_meta in img_metas ]
            labels = [img_meta['gt_labels_3d']._data for img_meta in img_metas ]
            known = [(torch.ones_like(t)).cuda() for t in labels]
            know_idx = known
            unmask_bbox = unmask_label = torch.cat(known)
            known_num = [t.size(0) for t in targets]
            labels = torch.cat([t for t in labels])
            boxes = torch.cat([t for t in targets])
            batch_idx = torch.cat([torch.full((t.size(0), ), i) for i, t in enumerate(targets)])

            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)
            # add noise
            groups = min(self.scalar, len(reference_points[0]) // max(known_num))
            known_indice = known_indice.repeat(groups, 1).view(-1)
            known_labels = labels.repeat(groups, 1).view(-1).long().to(reference_points.device)
            known_labels_raw = labels.repeat(groups, 1).view(-1).long().to(reference_points.device)
            known_bid = batch_idx.repeat(groups, 1).view(-1)
            known_bboxs = boxes.repeat(groups, 1).to(reference_points.device)
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()

            if self.bbox_noise_scale > 0:
                diff = known_bbox_scale / 2 + self.bbox_noise_trans
                rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0
                known_bbox_center += torch.mul(rand_prob,
                                            diff) * self.bbox_noise_scale
                known_bbox_center[..., 0:1] = (known_bbox_center[..., 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
                known_bbox_center[..., 1:2] = (known_bbox_center[..., 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
                known_bbox_center[..., 2:3] = (known_bbox_center[..., 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
                known_bbox_center = known_bbox_center.clamp(min=0.0, max=1.0)
                mask = torch.norm(rand_prob, 2, 1) > self.split
                known_labels[mask] = self.num_classes

            single_pad = int(max(known_num))
            pad_size = int(single_pad * groups)
            padding_bbox = torch.zeros(batch_size, pad_size, 3).to(reference_points.device)
            padded_reference_points = torch.cat([padding_bbox, reference_points], dim=1)

            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(groups)]).long()
            if len(known_bid):
                padded_reference_points[(known_bid.long(), map_known_indice)] = known_bbox_center.to(reference_points.device)

            tgt_size = pad_size + len(reference_points[0])
            attn_mask = torch.ones(tgt_size, tgt_size).to(reference_points.device) < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(groups):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == groups - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True

            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'known_labels_raw': known_labels_raw,
                'know_idx': know_idx,
                'pad_size': pad_size
            }

        else:
            padded_reference_points = reference_points
            attn_mask = None
            mask_dict = None

        return padded_reference_points, attn_mask, mask_dict

    def _rv_pe(self, img_feats, img_metas):
        BN, C, H, W = img_feats.shape
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        coords_h = torch.arange(H, device=img_feats[0].device).float() * pad_h / H
        coords_w = torch.arange(W, device=img_feats[0].device).float() * pad_w / W

        if self.LID:
            index  = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
            index_1 = index + 1
            bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_1
        else:
            coords_d = self.depth_start + torch.arange(self.depth_num, device=img_feats[0].device).float() * (self.position_range[3] - 1) / self.depth_num
        
        coords_h, coords_w, coords_d = torch.meshgrid([coords_h, coords_w, coords_d])

        coords = torch.stack([coords_w, coords_h, coords_d, coords_h.new_ones(coords_h.shape)], dim=-1)
        coords[..., :2] = coords[..., :2] * coords[..., 2:3]
        
        imgs2lidars = np.concatenate([np.linalg.inv(meta['lidar2img']) for meta in img_metas])
        imgs2lidars = torch.from_numpy(imgs2lidars).float().to(coords.device)
        coords_3d = torch.einsum('hwdo, bco -> bhwdc', coords, imgs2lidars)
        coords_3d = (coords_3d[..., :3] - coords_3d.new_tensor(self.position_range[:3])[None, None, None, :] )\
                        / (coords_3d.new_tensor(self.position_range[3:]) - coords_3d.new_tensor(self.position_range[:3]))[None, None, None, :]
        coords_3d = coords_3d.permute(0, 3, 4, 1, 2).contiguous().view(BN, -1, H, W)
        coords_3d = inverse_sigmoid(coords_3d)
        return self.rv_embedding(coords_3d).permute(0, 2, 3, 1).contiguous()

    def query_embed(self, ref_points, img_metas):
        ref_points = inverse_sigmoid(ref_points.clone())
        if self.query_3dpe:
            query_embeds = self.bev_3d_embedding(pos2emb3d(ref_points))
        else:
            query_embeds = self.bev_2d_embedding(pos2embed(ref_points, num_pos_feats=self.hidden_dim))
        return query_embeds

    def _rv_pe_depth(self, img_feats, img_metas):
        """
        Args:
            img_feats: (B*N_view, C, H, W)
            img_metas:
            depth_map: (B*N_view, H, W)
        Returns:
            coords_position_embeding: (BN_view, H, W, embed_dims)
        """
        eps = 1e-5
        depth_map_list = []
        for meta in img_metas:
            depth_map_list.append(torch.tensor(meta['depth_map'], device=img_feats.device, dtype=torch.float32))
        depth_map = torch.stack(depth_map_list, dim=0)

        BN, C, H, W = img_feats.shape
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        coords_h = torch.arange(H, device=img_feats[0].device).float() * pad_h / H  # (H, )
        coords_w = torch.arange(W, device=img_feats[0].device).float() * pad_w / W  # (W, )
        coords_h, coords_w = torch.meshgrid([coords_h, coords_w]) # (H, W)


        coords = torch.stack([coords_w, coords_h],dim=-1) # (H, W, 2) (u, v)
        coords = coords.unsqueeze(0).repeat(BN, 1, 1, 1)  # (BN_view, H, W, 2)

        depth_map = depth_map.reshape(-1, *depth_map.shape[2:]) # (BN_view, H, W)
        depth_map = depth_map.unsqueeze(dim=-1)     # (B*N_view, H, W, 1)
        coords = coords * \
            torch.maximum(depth_map, torch.ones_like(depth_map) * eps)  # (BN_view, W, H, 2)    (du, dv)
        coords = torch.cat([coords, depth_map], dim=-1)     # (BN_view, W, H, 3)   (du, dv, d)
        coords = torch.cat([coords, torch.ones_like(coords[..., :1])], dim=-1)  # (BN_view, W, H, 4)   (du, dv, d, 1)

        imgs2lidars = np.concatenate([np.linalg.inv(meta['lidar2img']) for meta in img_metas]) # (B*N_view, 4, 4)
        imgs2lidars = torch.from_numpy(imgs2lidars).float().to(coords.device)

       
        coords_3d = torch.einsum('bhwc, bdc -> bhwd', coords, imgs2lidars)

        coords_3d = (coords_3d[..., :3] - coords_3d.new_tensor(self.position_range[:3])[None, None, None, :] )\
                        / (coords_3d.new_tensor(self.position_range[3:]) - coords_3d.new_tensor(self.position_range[:3]))[None, None, None, :]
        
        coords_3d = inverse_sigmoid(coords_3d)
        if self.share_pe and self.query_3dpe:
            return self.bev_3d_embedding(pos2emb3d(coords_3d, num_pos_feats=self.hidden_dim//2))
        else:
            assert self.bev_depth_embedding is not None, "bev_depth_embedding is None"
            return self.bev_depth_embedding(pos2emb3d(coords_3d, num_pos_feats=self.hidden_dim//2))

    def process_intrins_feat(self, rois, intrinsics, min_size=4):
        intrinsics = intrinsics.view(intrinsics.shape[0], 16).clone().float()
        intrinsics = intrinsics * 0.1
        wh_bbox = rois[:, 3:5] - rois[:, 1:3]
        invalid_bbox = (wh_bbox < min_size).any(1)
        intrinsics[invalid_bbox] = 0
        return intrinsics
    
    def generate_reference_points(self, x_img, proposals, reference_points, img_metas):
        if sum([len(p) for p in proposals]) == 0:
            proposal = torch.tensor([[0, 50, 50, 100, 100, 0]], dtype=proposals[0].dtype,
                                    device=proposals[0].device)
            proposals = [proposal] + proposals[1:]

        rois = bbox2roi(proposals)
        
        img_metas_origin = img_metas.copy()
        img_metas = []
        for img_meta in img_metas_origin:
            for i in range(len(img_meta["img_shape"])):
                img_meta_per_img = {}
                for key in ['filename', 'img_shape', 'lidar2img', 'pad_shape', 'lidar2cam', 'cam_intrinsic']:
                    img_meta_per_img[key] = img_meta[key][i]
                img_metas.append(img_meta_per_img)
        intrinsics, extrinsics = self.get_box_params(proposals,
                                                     [img_meta['cam_intrinsic'] for img_meta in img_metas],
                                                     [img_meta['lidar2cam'] for img_meta in img_metas],)
        bbox_feats = self.bbox_roi_extractor([x_img], rois)
        
        # intrinsics as extra input feature
        extra_feats = dict(
            intrinsic=self.process_intrins_feat(rois, intrinsics)
        )

        # query generator
        reference_points_proposal, _ = self.query_generator(bbox_feats, intrinsics, extrinsics, extra_feats)
        reference_points_proposal[..., 0:1] = (reference_points_proposal[..., 0:1] - self.pc_range[0]) / (
                self.pc_range[3] - self.pc_range[0])
        reference_points_proposal[..., 1:2] = (reference_points_proposal[..., 1:2] - self.pc_range[1]) / (
                self.pc_range[4] - self.pc_range[1])
        reference_points_proposal[..., 2:3] = (reference_points_proposal[..., 2:3] - self.pc_range[2]) / (
                self.pc_range[5] - self.pc_range[2])
        
        reference_points = torch.cat([reference_points_proposal, reference_points], dim=0)

        return reference_points

    def generate_3d_reference_points(self, x, x_pos_embeds, reference_points, bs):

        device, dtype = x.device, x.dtype
        x_feature = x.clone()  # [B, C, H, W]
        B, C, H, W = x_feature.shape
        
        heatmap = self.heatmap_layer(x)  # [B, num_classes, H, W]
        heatmap_clone = heatmap.clone().detach()  # [B, num_classes, H, W]

        x_hm_max = self.maxpool_2d(heatmap_clone)  # [B, num_classes, H, W]
        x_hm_max_small = self.maxpool_2d_small(heatmap_clone)  # [B, num_classes, H, W]

        selected = (x_hm_max == heatmap_clone)  # [B, num_classes, H, W]
        selected_small = (x_hm_max_small == heatmap_clone)  # [B, num_classes, H, W]

        selected[:, 8:10, :, :] = selected_small[:, 8:10, :, :]

        score = heatmap_clone * selected  # [B, num_classes, H, W]
        score, _ = score.topk(1, dim=1)  # [B, 1, H, W]

        proposal_list = []
        foreground_feature = torch.zeros(bs, self.max_foreground_token, self.hidden_dim, device=device, dtype=dtype)
        foreground_bevemb = torch.zeros(bs, self.max_foreground_token, self.hidden_dim, device=device, dtype=dtype)

        for i in range(bs):
            # 对每个批次，找到顶级提案
            sample_hm = score[i].squeeze(0).view(-1)  # [H*W]
            sample_x_feature = x_feature[i].view(C, -1)  # [C, H*W]
        
            _, proposal_ind = sample_hm.topk(self.num_3d_proposals) # 推荐区域ID
            _, voxel_ind = sample_hm.topk(self.max_foreground_token) # 前景体素ID

            selected_pos = self.coords_bev[proposal_ind].unsqueeze(0).to(device)  # [1, num_3d_proposals, 2]
            selected_features = sample_x_feature[:, voxel_ind]  # [C, max_foreground_token]
            selected_bevemb = x_pos_embeds[voxel_ind]  # [max_foreground_token, C]

            foreground_feature[i, :, :] = selected_features.t()
            foreground_bevemb[i, :, :] = selected_bevemb

            proposal_list.append(selected_pos)
        
        query_pos = torch.cat(proposal_list, dim=0)# 形状：[batch_size, num_3d_proposals, 2]
        init_reference_points = torch.cat([
            query_pos,
            torch.full((*query_pos.shape[:-1], 1), 0.5, device=query_pos.device)
        ], dim=-1)  # 形状：[batch_size, num_3d_proposals, 3]

        reference_points = reference_points.unsqueeze(0).repeat(bs,1,1)
        reference_points = torch.cat([init_reference_points, reference_points],dim=1)

        return reference_points, foreground_feature, foreground_bevemb, heatmap

    @torch.no_grad()
    def get_box_params(self, bboxes, intrinsics, extrinsics):
        # TODO: check grad flow from boxes to intrinsic
        intrinsic_list = []
        extrinsic_list = []
        for img_id, (bbox, intrinsic, extrinsic) in enumerate(zip(bboxes, intrinsics, extrinsics)):
            # bbox: [n, (x, y, x, y)], rois_i: [n, c, h, w], intrinsic: [4, 4], extrinsic: [4, 4]
            intrinsic = torch.from_numpy(intrinsic).to(bbox.device).double()
            extrinsic = torch.from_numpy(extrinsic).to(bbox.device).double()
            intrinsic = intrinsic.repeat(bbox.shape[0], 1, 1)
            extrinsic = extrinsic.repeat(bbox.shape[0], 1, 1)
            # consider corners
            wh_bbox = bbox[:, 2:4] - bbox[:, :2]
            wh_roi = wh_bbox.new_tensor(self.roi_size)
            scale = wh_roi[None] / wh_bbox
            intrinsic[:, :2, 2] = intrinsic[:, :2, 2] - bbox[:, :2] - 0.5 / scale
            intrinsic[:, :2] = intrinsic[:, :2] * scale[..., None]
            intrinsic_list.append(intrinsic)
            extrinsic_list.append(extrinsic)
        intrinsic_list = torch.cat(intrinsic_list, 0)
        extrinsic_list = torch.cat(extrinsic_list, 0)
        return intrinsic_list, extrinsic_list    

    def visualize_ref_points(self, reference_points, img_metas):
        import matplotlib.pyplot as plt
        import time

        # 提取 x 和 y 坐标
        x_coords = reference_points[:, 0].cpu().detach().numpy() * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        y_coords = reference_points[:, 1].cpu().detach().numpy() * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]

        x_coords_gt = img_metas[0]['gt_bboxes_3d']._data.gravity_center[:, 0].cpu().detach().numpy()
        y_coords_gt = img_metas[0]['gt_bboxes_3d']._data.gravity_center[:, 1].cpu().detach().numpy()

        # 创建一个散点图 (scatter plot) 进行 BEV 可视化
        plt.figure(figsize=(8, 8), dpi=100)
        
        # 绘制参考点（蓝色）
        plt.scatter(x_coords, y_coords, c='blue', s=20, marker='.', alpha=0.7, label='Reference Points')
        
        # 绘制真实目标重心（红色）
        plt.scatter(x_coords_gt, y_coords_gt, c='red', s=30, marker='o', alpha=0.6, label='Ground Truth')

        # 去掉坐标轴和网格
        plt.axis('off')

        # 设置坐标轴比例相等
        plt.gca().set_aspect('equal', adjustable='box')

        plt.gca().set_xlim(self.pc_range[0], self.pc_range[3])
        plt.gca().set_ylim(self.pc_range[1], self.pc_range[4])

        # # 添加图例
        # plt.legend(loc='upper right', fontsize=12)

        # 保存图像，防止覆盖文件名，添加时间戳
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        plt.savefig(f'bev_visualization_{timestamp}.png', format='png', bbox_inches='tight', pad_inches=0)

    def forward_single(self, x, x_img, img_metas, proposals=None):
        """
            x: [bs c h w]
            return List(dict(head_name: [num_dec x bs x num_query * head_dim]) ) x task_num
        """
        reference_points = self.reference_points.weight # N * 3

        if self.input_modality['use_camera']:
            assert x_img is not None
            
            bn = x_img.size(0)
            bs = len(img_metas)
            input_img_h, input_img_w, _ = img_metas[0]['pad_shape'][0]
            masks = x_img.new_ones(
                (len(img_metas), bn//len(img_metas), input_img_h, input_img_w))
            for img_id in range(len(img_metas)):
                for cam_id in range(bn//len(img_metas)):
                    img_h, img_w, _ = img_metas[img_id]['img_shape'][cam_id]
                    masks[img_id, cam_id, :img_h, :img_w] = 0
            
            x_img = self.input_proj(x_img)

            masks = F.interpolate(
                masks, size=x_img.shape[-2:]).to(torch.bool)
            
            if self.if_depth_pe:
                rv_pos_embeds = self._rv_pe_depth(x_img, img_metas)
            else:
                rv_pos_embeds = self._rv_pe(x_img, img_metas)
        
            if self.with_multiview:
                sin_embed = self.positional_encoding(masks)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x_img.size()).permute(0, 2, 3, 1)
                rv_pos_embeds = rv_pos_embeds + sin_embed

            if proposals is not None and self.if_2d_prior:
                if self.generate_with_pe:
                    reference_points = self.generate_reference_points(x_img + rv_pos_embeds.permute(0, 3, 1, 2), proposals, reference_points, img_metas)
                else:
                    reference_points = self.generate_reference_points(x_img, proposals, reference_points, img_metas)
        else:
            rv_pos_embeds = None

        if self.input_modality['use_lidar']:
            assert x is not None
            heatmap = None
            x = self.shared_conv(x)

            # point_features_pe
            if not self.lidar_3d_pe:
                bev_pos_embeds = self.bev_2d_embedding(pos2embed(self.coords_bev.to(x.device), num_pos_feats=self.hidden_dim))
            else:
                bev_pos_embeds = self.bev_3d_embedding(pos2emb3d(self.coords_bev.to(x.device)))
            bs = x.size(0)
            if self.if_3d_prior:
                reference_points, x, bev_pos_embeds, heatmap = self.generate_3d_reference_points(x, bev_pos_embeds, reference_points, bs)
            else:
                x = rearrange(x, "b c h w -> b (h w) c")
                bev_pos_embeds = bev_pos_embeds.unsqueeze(0).repeat(bs, 1, 1)
        else:
            bev_pos_embeds = None

        if False:
            self.visualize_ref_points(reference_points, img_metas)
            
        if len(reference_points.shape) == 2:
            reference_points = reference_points.unsqueeze(0).repeat(bs, 1, 1)

        reference_points, attn_mask, mask_dict = self.prepare_for_dn(bs, reference_points, img_metas)
        query_embeds = self.query_embed(reference_points, img_metas)

        outs_dec, _, attn_weights = self.transformer(
                            x, x_img, query_embeds,
                            bev_pos_embeds, rv_pos_embeds,
                            attn_masks=attn_mask,
                            bs=bs
                        )
        outs_dec = torch.nan_to_num(outs_dec)

        outputs_classes = []
        outputs_coords = []
        reference = inverse_sigmoid(reference_points.clone())

        if not self.use_separate_head:
            for lvl in range(outs_dec.shape[0]):
                assert reference.shape[-1] == 3
                outputs_class = self.cls_branches[lvl](outs_dec[lvl])
                tmp = self.reg_branches[lvl](outs_dec[lvl])

                tmp[..., 0:2] += reference[..., 0:2]
                tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
                tmp[..., 4:5] += reference[..., 2:3]
                tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

                outputs_coord = tmp
                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)

            all_cls_scores = torch.stack(outputs_classes)
            all_bbox_preds = torch.stack(outputs_coords)

            all_bbox_preds[..., 0:1] = (all_bbox_preds[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
            all_bbox_preds[..., 1:2] = (all_bbox_preds[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
            all_bbox_preds[..., 4:5] = (all_bbox_preds[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

        else:
            outs = self.separate_head(outs_dec)
            center = (outs['center'] + reference[None, :, :, :2]).sigmoid()
            height = (outs['height'] + reference[None, :, :, 2:3]).sigmoid()
            _center, _height = center.new_zeros(center.shape), height.new_zeros(height.shape)
            _center[..., 0:1] = center[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            _center[..., 1:2] = center[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            _height[..., 0:1] = height[..., 0:1] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            dim = outs['dim']
            rot = outs['rot']
            vel = outs['vel']

            all_cls_scores = outs['cls_logits']
            all_bbox_preds = torch.cat([_center, dim[..., :2], _height, dim[..., 2:3], rot, vel], dim=-1)

        if mask_dict and mask_dict['pad_size'] > 0:
            output_known_class = all_cls_scores[:, :, :mask_dict['pad_size'], :]
            output_known_coord = all_bbox_preds[:, :, :mask_dict['pad_size'], :]
            outputs_class = all_cls_scores[:, :, mask_dict['pad_size']:, :]
            outputs_coord = all_bbox_preds[:, :, mask_dict['pad_size']:, :]
            mask_dict['output_known_lbs_bboxes']=(output_known_class, output_known_coord)
            outs = {
                'all_cls_scores': outputs_class,
                'all_bbox_preds': outputs_coord,
                'dn_mask_dict':mask_dict,
            }
        else:
            outs = {
                'all_cls_scores': all_cls_scores,
                'all_bbox_preds': all_bbox_preds,
                'dn_mask_dict': None,
            }

        outs['attn_weights'] = attn_weights

        if self.input_modality['use_lidar']:
            outs['heatmap'] = heatmap

        return outs

    def forward(self, pts_feats, img_feats=None, img_metas=None, proposals=None):
        """
            list([bs, c, h, w])
        """
        return self.forward_single(pts_feats[0], img_feats[0], img_metas, proposals)
    
    def _get_targets_single(self, gt_bboxes_3d, gt_labels_3d, pred_bboxes, pred_logits):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            
            gt_bboxes_3d (Tensor):  LiDARInstance3DBoxes(num_gts, 9)
            gt_labels_3d (Tensor): Ground truth class indices (num_gts, )
            pred_bboxes (Tensor): (num_query, 10)
            pred_logits (Tensor):  (num_query, task_classes)
        Returns:
            tuple[Tensor]: a tuple containing the following.
                - labels_tasks (Tensor): num_query, .
                - label_weights_tasks (Tensor): (num_query, ).
                - bbox_targets_tasks (Tensor): (num_query, 9).
                - bbox_weights_tasks (Tensor): (num_query, 10).
                - pos_inds (Tensor): Sampled positive indices.
                - neg_inds (Tensor): Sampled negative indices.
        """
        
        num_bboxes = pred_bboxes.shape[0]
        assign_results = self.assigner.assign(pred_bboxes, pred_logits, gt_bboxes_3d, gt_labels_3d)
        sampling_result = self.sampler.sample(assign_results, pred_bboxes, gt_bboxes_3d)
        pos_inds, neg_inds = sampling_result.pos_inds, sampling_result.neg_inds
        # label targets
        labels = gt_bboxes_3d.new_full((num_bboxes, ),
                                self.num_classes,
                                dtype=torch.long)
        labels[pos_inds] = gt_labels_3d[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes_3d.new_ones(num_bboxes)
        
        # bbox_targets
        code_size = gt_bboxes_3d.shape[1]
        bbox_targets = torch.zeros_like(pred_bboxes)[..., :code_size]
        bbox_weights = torch.zeros_like(pred_bboxes)
        bbox_weights[pos_inds] = 1.0
        if len(sampling_result.pos_gt_bboxes) > 0:
            bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        return labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, preds_bboxes, preds_logits):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            gt_bboxes_3d (list[LiDARInstance3DBoxes]): batch_size * (num_gts, 9)
            gt_labels_3d (list[Tensor]): Ground truth class indices. batch_size * (num_gts, )
            pred_bboxes (list[list[Tensor]]): batch_size x num_task x [num_query, 10].
            pred_logits (list[list[Tensor]]): batch_size x num_task x [num_query, task_classes]
        Returns:
            tuple: a tuple containing the following targets.
                - task_labels_list (list(list[Tensor])): num_tasks x batch_size x (num_query, ).
                - task_labels_weight_list (list[Tensor]): num_tasks x batch_size x (num_query, )
                - task_bbox_targets_list (list[Tensor]): num_tasks x batch_size x (num_query, 9)
                - task_bbox_weights_list (list[Tensor]): num_tasks x batch_size x (num_query, 10)
                - num_total_pos_tasks (list[int]): num_tasks x Number of positive samples
                - num_total_neg_tasks (list[int]): num_tasks x Number of negative samples.
        """
        (labels_list, labels_weight_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_targets_single, gt_bboxes_3d, gt_labels_3d, preds_bboxes, preds_logits
        )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, labels_weight_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    pred_bboxes,
                    pred_logits,
                    gt_bboxes_3d,
                    gt_labels_3d):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            pred_bboxes (list[Tensor]): num_tasks x [bs, num_query, 10].
            pred_logits (list(Tensor]): num_tasks x [bs, num_query, task_classes]
            gt_bboxes_3d (list[LiDARInstance3DBoxes]): batch_size * (num_gts, 9)
            gt_labels_list (list[Tensor]): Ground truth class indices. batch_size * (num_gts, )
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        batch_size = pred_bboxes.shape[0]
        cls_scores_list = [pred_logits[i] for i in range(batch_size)]
        bbox_preds_list = [pred_bboxes[i] for i in range(batch_size)]

        cls_reg_targets = self.get_targets(
            gt_bboxes_3d, gt_labels_3d, bbox_preds_list, cls_scores_list
        )
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        
        labels = torch.cat(labels_list, dim=0)
        labels_weights = torch.cat(label_weights_list, dim=0)
        bbox_targets = torch.cat(bbox_targets_list, dim=0)
        bbox_weights = torch.cat(bbox_weights_list, dim=0)
        
        pred_bboxes_flatten = pred_bboxes.flatten(0, 1)
        pred_logits_flatten = pred_logits.flatten(0, 1)
        
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            pred_logits_flatten, labels, labels_weights, avg_factor=cls_avg_factor
        )

        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * bbox_weights.new_tensor(self.train_cfg.code_weights)[None, :]

        loss_bbox = self.loss_bbox(
            pred_bboxes_flatten[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos
        )

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox) 
        return loss_cls, loss_bbox
    
    def prepare_for_loss(self, mask_dict):
        """
        prepare dn components to calculate loss
        Args:
            mask_dict: a dict that contains dn information
        """
        output_known_class, output_known_coord = mask_dict['output_known_lbs_bboxes']
        known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
        map_known_indice = mask_dict['map_known_indice'].long()
        known_indice = mask_dict['known_indice'].long().cpu()
        batch_idx = mask_dict['batch_idx'].long()
        bid = batch_idx[known_indice]
        if len(output_known_class) > 0:
            output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
            output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
        num_tgt = known_indice.numel()
        return known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt
    
    def dn_loss_single(self,
                    cls_scores,
                    bbox_preds,
                    known_bboxs,
                    known_labels,
                    num_total_pos=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 3.14159 / 6 * self.split * self.split  * self.split ### positive rate
        bbox_weights = torch.ones_like(bbox_preds)
        label_weights = torch.ones_like(known_labels)
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, known_labels.long(), label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(known_bboxs, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)

        bbox_weights = bbox_weights * bbox_weights.new_tensor(self.train_cfg.code_weights)[None, :]
        
        loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        
        return self.dn_weight * loss_cls, self.dn_weight * loss_bbox
     
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """"Loss function.
        Args:
            gt_bboxes_3d (list[LiDARInstance3DBoxes]): batch_size * (num_gts, 9)
            gt_labels_3d (list[Tensor]): Ground truth class indices. batch_size * (num_gts, )
            preds_dicts(tuple[list[dict]]): nb_tasks x num_lvl
                center: (num_dec, batch_size, num_query, 2)
                height: (num_dec, batch_size, num_query, 1)
                dim: (num_dec, batch_size, num_query, 3)
                rot: (num_dec, batch_size, num_query, 2)
                vel: (num_dec, batch_size, num_query, 2)
                cls_logits: (num_dec, batch_size, num_query, task_classes)
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # 拿到预测结果
        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_3d[0].device
        gt_bboxes_3d = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_3d]
        
        all_gt_bboxes_list = [gt_bboxes_3d for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_3d for _ in range(num_dec_layers)]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_bbox_preds, all_cls_scores,
            all_gt_bboxes_list, all_gt_labels_list)
        
        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1

    
        if preds_dicts['dn_mask_dict'] is not None:
            known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt = self.prepare_for_loss(preds_dicts['dn_mask_dict'])
            all_known_bboxs_list = [known_bboxs for _ in range(num_dec_layers)]
            all_known_labels_list = [known_labels for _ in range(num_dec_layers)]
            all_num_tgts_list = [
                num_tgt for _ in range(num_dec_layers)
            ]
            
            dn_losses_cls, dn_losses_bbox = multi_apply(
                self.dn_loss_single, output_known_class, output_known_coord,
                all_known_bboxs_list, all_known_labels_list, 
                all_num_tgts_list)
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(dn_losses_cls[:-1],
                                            dn_losses_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                num_dec_layer += 1
        if preds_dicts.get("heatmap", None) is not None:
            heatmap_pred = preds_dicts['heatmap']
            hp_target = []
            for i in range(len(gt_bboxes_3d)):
                heatmap_target = self.get_heatmap_target_single(
                    gt_bboxes_3d[i],
                    gt_labels_3d[i],
                    heatmap_pred.shape[-2:]
                )
                hp_target.append(heatmap_target)
            hp_target = torch.stack(hp_target, dim=0)
            loss_heatmap = self.loss_heatmap(clip_sigmoid(heatmap_pred), hp_target, avg_factor=max(hp_target.eq(1).float().sum().item(), 1))
            loss_dict['loss_heatmap'] = loss_heatmap

        return loss_dict

    def get_heatmap_target_single(self, gt_bboxes_3d, gt_labels_3d, feature_map_size):
        
        num_max_objs = 500
        gaussian_overlap = 0.1
        min_radius = 2

        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])
        H, W = feature_map_size

        gt_bboxes_3d = gt_bboxes_3d[:, :6]

        heatmap = gt_bboxes_3d.new_zeros((self.num_classes, H, W))

        x, y= gt_bboxes_3d[:, 0], gt_bboxes_3d[:, 1]
        dx, dy = gt_bboxes_3d[:, 3], gt_bboxes_3d[:, 4]

        coord_x = (x - pc_range[0]) / voxel_size[0] / self.downsample_scale
        coord_y = (y - pc_range[1]) / voxel_size[1] / self.downsample_scale
        coord_x = torch.clamp(coord_x, min=0, max=W - 0.5)  
        coord_y = torch.clamp(coord_y, min=0, max=H - 0.5) 

        center = torch.stack([coord_y, coord_x], dim=1)  # [num_gt, 2]
        center_int = center.int()

        dx = dx / voxel_size[0] / self.downsample_scale
        dy = dy / voxel_size[1] / self.downsample_scale

        radius = gaussian_radius((dx, dy), min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_bboxes_3d.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            if not (0 <= center_int[k][0] <= H and 0 <= center_int[k][1] <= W):
                continue

            cur_class_id = (gt_labels_3d[k]).long()

            heatmap[cur_class_id] = self.draw_gaussian_heatmap(
                heatmap[cur_class_id], center[k], radius[k].item()
            )

        return heatmap

    def draw_gaussian_heatmap(self, heatmap, center, radius):

        diameter = 2 * radius + 1
        sigma = diameter / 6
        x, y = map(int, center)
        height, width = heatmap.shape

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        y1, y2 = y - top, y + bottom
        x1, x2 = x - left, x + right

        heatmap_slice = heatmap[y1:y2, x1:x2]

        # 生成高斯核
        x_range = torch.arange(-left, right, dtype=heatmap.dtype, device=heatmap.device)
        y_range = torch.arange(-top, bottom, dtype=heatmap.dtype, device=heatmap.device)
        y_cor, x_cor = torch.meshgrid(y_range, x_range)  # 明确指定索引方式

        gaussian = torch.exp(-(x_cor**2 + y_cor**2) / (2 * sigma ** 2))

        # 将高斯核与热图切片取最大值
        torch.max(heatmap_slice, gaussian, out=heatmap_slice)

        return heatmap
    

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False):
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)
        
        ret_list, bbox_index = [], []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            bbox_index.append(preds['bbox_index'])
            ret_list.append([bboxes, scores, labels])
        return ret_list, bbox_index

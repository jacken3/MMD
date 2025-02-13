import numpy as np
from mmdet.datasets.builder import PIPELINES
import matplotlib.pyplot as plt
import copy


@PIPELINES.register_module()
class LoadDepthByMapplingPoints2Images(object):
    def __init__(self, src_size, input_size, downsample=1, min_dist=1e-5, max_dist=None):
        self.src_size = src_size
        self.input_size = input_size
        self.downsample = downsample
        self.min_dist = min_dist
        self.max_dist = max_dist

    def mask_points_by_range(self, points_2d, depths, img_size):
        """
        Args:
            points2d: (N, 2)
            depths:   (N, )
            img_size: (H, W)
        Returns:
            points2d: (N', 2)
            depths:   (N', )
        """
        H, W = img_size
        mask = np.ones(depths.shape, dtype=np.bool)
        mask = np.logical_and(mask, points_2d[:, 0] >= 0)
        mask = np.logical_and(mask, points_2d[:, 0] < W)
        mask = np.logical_and(mask, points_2d[:, 1] >= 0)
        mask = np.logical_and(mask, points_2d[:, 1] < H)
        points_2d = points_2d[mask]
        depths = depths[mask]
        return points_2d, depths

    def mask_points_by_dist(self, points_2d, depths, min_dist, max_dist):
        mask = np.ones(depths.shape, dtype=np.bool)
        mask = np.logical_and(mask, depths >= min_dist)
        if max_dist is not None:
            mask = np.logical_and(mask, depths <= max_dist)
        points_2d = points_2d[mask]
        depths = depths[mask]
        return points_2d, depths

    def get_rot(self, h):
        return np.array([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def transform_points2d(self, points_2d, depths, resize, crop, flip, rotate):
        points_2d = points_2d * resize
        points_2d = points_2d - crop[:2]  # (N_points, 2)
        points_2d, depths = self.mask_points_by_range(points_2d, depths, (crop[3] - crop[1], crop[2] - crop[0]))

        if flip:
            points_2d[:, 0] = (crop[2] - crop[0]) - 1 - points_2d[:, 0]

        A = self.get_rot(rotate / 180 * np.pi)
        b = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.dot(-b) + b

        points_2d = points_2d.dot(A.T) + b

        points_2d, depths = self.mask_points_by_range(points_2d, depths, self.input_size)

        return points_2d, depths

    def __call__(self, results):
        imgs = results["img"]  # List[(H, W, 3), (H, W, 3), ...]

        N_views = len(imgs)
        H, W = self.input_size
        dH, dW = H // self.downsample, W // self.downsample
        depth_map_list = []
        depth_map_mask_list = []
        points_2d_list = []
        depth_list= []

        points_lidar = results['points'].tensor[:, :3].numpy()     # (N_points, 3)
        points_lidar = np.concatenate([points_lidar, np.ones((points_lidar.shape[0], 1))], axis=-1)   # (N_points, 4)

        for idx in range(N_views):

            # lidar --> img
            points_image = points_lidar.dot(results['lidar2img'][idx].T)
            points_image = points_image[:, :3]
            points_2d = points_image[:, :2] / points_image[:, 2:3]
            depths = points_image[:, 2]
            points_2d, depths = self.mask_points_by_range(points_2d, depths, self.input_size)
            points_2d, depths = self.mask_points_by_dist(points_2d, depths, self.min_dist, self.max_dist)
            points_2d_list.append(points_2d)
            depth_list.append(depths)


            # downsample
            points_2d = np.round(points_2d / self.downsample)

            points_2d, depths = self.mask_points_by_range(points_2d, depths, (dH, dW))

            depth_map = np.zeros(shape=(dH, dW), dtype=np.float32)  # (dH, dW)
            depth_map_mask = np.zeros(shape=(dH, dW), dtype=np.bool)   # (dH, dW)

            ranks = points_2d[:, 0] + points_2d[:, 1] * dW
            sort = (ranks + depths / 1000.).argsort()
            points_2d, depths, ranks = points_2d[sort], depths[sort], ranks[sort]

            kept = np.ones(points_2d.shape[0], dtype=np.bool)
            kept[1:] = (ranks[1:] != ranks[:-1])
            points_2d, depths = points_2d[kept], depths[kept]
            points_2d = points_2d.astype(np.long)
           #  print(points_2d[-1])

            depth_map[points_2d[:, 1], points_2d[:, 0]] = depths
            depth_map_mask[points_2d[:, 1], points_2d[:, 0]] = 1
            depth_map_list.append(depth_map)
            depth_map_mask_list.append(depth_map_mask)


        depth_map = np.stack(depth_map_list, axis=0)      # (N_view, dH, dW)
        depth_map_mask = np.stack(depth_map_mask_list, axis=0)    # (N_view, dH, dW)


        # # # # for vis
        # import cv2
        # from matplotlib.colors import Normalize
        # from matplotlib.gridspec import GridSpec
        # for idx in range(len(imgs)):

        #     fig = plt.figure(figsize=(15, 10))
        #     gs = GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])  # 增加一个宽度为0.1的列用于颜色条

        #     ax0 = fig.add_subplot(gs[0, :])
        #     img_origin_ = imgs[idx].astype(np.uint8)
        #     b,g,r = cv2.split(img_origin_)
        #     img_origin = cv2.merge([r,g,b])

        #     ax0.imshow(img_origin)
        #     ax0.axis('off')

        #     points_2d = points_2d_list[idx].astype(int)
        #     depth_values = depth_list[idx]

        #     norm = Normalize(vmin=np.min(depth_values), vmax=np.max(depth_values))
        #     cmap = plt.get_cmap('rainbow')
        #     colors = cmap(norm(depth_values))

        #     # 绘制点云，颜色依据深度值变化
        #     for i in range(len(points_2d)):
        #         ax0.plot(points_2d[i, 0], points_2d[i, 1], 'o', color=colors[i], markersize=3)


        #     ax1 = fig.add_subplot(gs[1, 0])
        #     curr_img = cv2.resize(src=img_origin,
        #                           dsize=(img_origin.shape[1]//self.downsample, img_origin.shape[0]//self.downsample))
        #     ax1.imshow(curr_img)
        #     ax1.axis('off')

        #     ax2 = fig.add_subplot(gs[1, 1])
        #     cur_depth_map = depth_map[idx]
        #     cur_depth_map_mask = depth_map_mask[idx]
        #     cur_depth_map = (cur_depth_map - -np.min(cur_depth_map)) / (np.max(cur_depth_map)-np.min(cur_depth_map)) * 255
        #     cur_depth_map = cur_depth_map.astype(np.uint8)
        #     cur_depth_map = cv2.applyColorMap(cur_depth_map, cv2.COLORMAP_RAINBOW)

        #     curr_img[cur_depth_map_mask] = cur_depth_map[cur_depth_map_mask]
        #     ax2.imshow(curr_img)
        #     ax2.axis('off')

        #     # cbar_ax = fig.add_subplot(gs[2, :])
        #     # cbar_ax.set_visible(False)  # 隐藏轴线
        #     sm = plt.cm.ScalarMappable(cmap='rainbow', norm=norm)
        #     sm.set_array([])
        #     cbar = fig.colorbar(sm, ax=[ax0, ax1, ax2], location='right', shrink=0.8)
        #     cbar.ax.tick_params(labelsize=12)
        #     cbar.set_label('Depth Value', fontsize=14)

        #     fig.patch.set_alpha(0.0)  # 图形背景透明

        #     # plt.subplots_adjust(hspace=0.05)
        #     plt.savefig(str(idx) + '_combined.png', bbox_inches='tight', transparent=True)
        #     plt.close()

        results['depth_map'] = depth_map
        results['depth_map_mask'] = depth_map_mask
        results['points_2d'] =  points_2d

        return results


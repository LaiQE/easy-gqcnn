import logging
import numpy as np
import scipy.spatial.distance as ssd
import cv2
from .grasp_2d import Grasp2D


class ImageGraspSampler():
    """ 从图像中采样夹爪, 步骤主要有以下几个
    1. 采集边缘点
    2. 生成边缘点对, 拒绝无效点
    3. 计算力闭合, 生成抓取
    Note: 这里有一个严重的问题, 在本项目中我们定义
    所有坐标除了图像都是(x, y)这样的次序
    """

    def __init__(self, depth, roi, config, width=None):
        """ depth: 深度图, array数组
            roi: ((h1,w1), (h2,w2))
            width: 夹爪宽度, 像素表示
            config: 配置文件
        """
        self._depth = np.squeeze(depth)
        self._roi = roi
        self._config = config
        if 'sampler' in config.keys():
            self._config = config['sampler']
        self._width = width
        if self._width is None:
            self._width = self._config['max_grasp_width_px']
        self._friction_coef = self._config['friction_coef']

    def _get_edge(self, thresh):
        """ 提取边缘像素, 这里直接使用OpenCV的canny算子
        return: 所有边缘像素的数组, 这里是世界坐标(x, y)
        """
        # 转换深度图到OpenCV接受的灰度
        depth_g = self._depth
        depth_h = (depth_g - np.min(depth_g)) * 255 / (np.max(depth_g) - np.min(depth_g))
        # 这里由于深度值是线性变换，所以梯度值也是线性变换
        t = thresh * 255 / (np.max(depth_g) - np.min(depth_g))
        # 使用canny算子计算边缘
        cann = cv2.Canny(depth_h.astype('uint8'), t, t)
        # 剔除roi外的点
        roi = self._roi
        if roi is not None:
            cann_roi = np.zeros(cann.shape)
            cann_roi[roi[0][0]:roi[1][0], roi[0][1]:roi[1][1]] \
                = cann[roi[0][0]:roi[1][0], roi[0][1]:roi[1][1]]
        else:
            cann_roi = cann
        # 计算边缘点坐标
        edge_px = np.where(cann_roi > 200)
        # 转换到世界坐标下
        edge_px = np.c_[edge_px[1], edge_px[0]]
        return edge_px

    @staticmethod
    def force_closure(p1, p2, n1, n2, mu):
        """ 计算两个点是否力闭合 """
        # line between the contacts
        v = p2 - p1
        v = v / np.linalg.norm(v)

        # compute cone membership
        alpha = np.arctan(mu)
        v1 = n1.dot(-v)
        if v1 > 1:
            v1 = 1
        elif v1 < -1:
            v1 = -1
        v2 = n2.dot(v)
        if v2 > 1:
            v2 = 1
        elif v2 < -1:
            v2 = -1
        in_cone_1 = (np.arccos(v1) < alpha)
        in_cone_2 = (np.arccos(v2) < alpha)
        return (in_cone_1 and in_cone_2)

    def _surface_normals(self, depth_im, edge_pixels):
        """ 返回边缘像素的表面法线的数组
        这里返回的也是世界坐标(x, y)
        """
        # 计算梯度
        grad = np.gradient(depth_im)

        # 计算表面法线
        normals = np.zeros([edge_pixels.shape[0], 2])
        for i, pix in enumerate(edge_pixels):
            dx = grad[1][pix[0], pix[1]]
            dy = grad[0][pix[0], pix[1]]
            normal_vec = np.array([dx, dy])
            if np.linalg.norm(normal_vec) == 0:
                normal_vec = np.array([1, 0])
            normal_vec = normal_vec / np.linalg.norm(normal_vec)
            normals[i, :] = normal_vec

        return normals

    def _find_pair(self, edge_pixels, edge_normals):
        """ 找出所有的点对
        """
        max_grasp_width_px = self._width
        # 返回一个n*n的矩阵，每个值代表梯度方向的点积，由于梯度模长为1
        # 这里的每个值就代表夹角的cos值
        normal_ip = edge_normals.dot(edge_normals.T)
        # 计算每个点之间的距离
        dists = ssd.squareform(ssd.pdist(edge_pixels))
        # 取上对角阵,以消除重复的元素
        dists = np.triu(dists, 1)
        # 1. 点对梯度夹角大于180-摩擦角
        # 2. 点对距离小于最大距离大于0
        valid_indices = np.where((normal_ip < -np.cos(np.arctan(self._friction_coef)))
                                 & (dists < max_grasp_width_px) & (dists > 0.0))
        # 这里valid的两个数就代表在edge_pixels数组里的两行所代表的点
        valid_indices = np.c_[valid_indices[0], valid_indices[1]]
        num_pairs = valid_indices.shape[0]
        if num_pairs == 0:
            logging.warning('没有找到合适的点对')
            return []
        return valid_indices

    def _find_thin_edge(self, edge_pixels, edge_normals):
        """ TODO: 找到所有薄边缘点
        """
        pass

    def sample(self, num_sample):
        # TODO: 这里需要修改抓取轴宽度由图像产生
        depth = self._depth
        edge_pixels = self._get_edge(self._config['depth_grad_thresh'])
        edge_normals = self._surface_normals(depth, edge_pixels)
        valid_indices = self._find_pair(edge_pixels, edge_normals)

        # 随机打乱所有的候选点对
        index = np.arange(len(valid_indices))
        np.random.shuffle(index)
        grasps = []

        for i in index:
            pair0, pair1 = valid_indices[i]
            p0 = edge_pixels[pair0, :]
            p1 = edge_pixels[pair1, :]
            n0 = edge_normals[pair0, :]
            n1 = edge_normals[pair1, :]
            # print(p0, p1)

            # 如果力闭合
            if self.force_closure(p0, p1, n0, n1, self._friction_coef):
                grasp_center = (p0 + p1) / 2
                grasp_axis = p1 - p0
                grasp_axis = grasp_axis / np.linalg.norm(grasp_axis)
                grasp_theta = 0
                if grasp_axis[0] != 0:
                    grasp_theta = np.arctan(grasp_axis[1] / grasp_axis[0])
                grasp = Grasp2D(grasp_center, grasp_theta, 0.0)
                # 改抓取到所有已存在抓取的距离
                grasp_dists = [Grasp2D.image_dist(
                    grasp, g[0], alpha=self._config['angle_dist_weight']) for g in grasps]
                if len(grasps) == 0 or np.min(grasp_dists) > self._config['min_grasp_dist']:
                    # 寻找中心点附近最高的点, 这里修改了原始的算法增加了在p0p1点附近的深度
                    _h = self._config['depth_sample_win_height']
                    _w = self._config['depth_sample_win_width']
                    depth_win_c = depth[int(grasp_center[1]-_h): int(grasp_center[1]+_h),
                                        int(grasp_center[0]-_w): int(grasp_center[0]+_w)]
                    depth_win_p0 = depth[int(p0[1]-_h): int(p0[1]+_h),
                                         int(p0[0]-_w): int(p0[0]+_w)]
                    depth_win_p1 = depth[int(p1[1]-_h): int(p1[1]+_h),
                                         int(p1[0]-_w): int(p1[0]+_w)]
                    depth_win = np.r_[depth_win_c, depth_win_p0, depth_win_p1]
                    center_depth = np.min(depth_win)
                    if center_depth == 0 or np.isnan(center_depth):
                        continue

                    # sample depth between the min and max
                    min_depth = np.min(depth_win) + self._config['min_depth_offset']
                    max_depth = np.max(depth_win) + self._config['max_depth_offset']
                    for i in range(self._config['depth_samples_per_grasp']):
                        sample_depth = min_depth + (max_depth - min_depth) * np.random.rand()
                        candidate_grasp = Grasp2D(grasp_center,
                                                  grasp_theta,
                                                  sample_depth,
                                                  width=self._width)
                        grasps.append([candidate_grasp, p0, p1])

            if len(grasps) >= num_sample:
                return grasps
        return grasps

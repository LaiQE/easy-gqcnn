import os
import logging
import numpy as np
import scipy.spatial.distance as ssd
import cv2


class ImageGraspSampler():
    """ 从图像中采样夹爪, 步骤主要有以下几个
    1. 采集边缘点
    2. 生成边缘点对, 拒绝无效点
    3. 计算力闭合, 生成抓取
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
        self._width = width
        if self._width is None:
            self._width = self._config['max_grasp_width_px']

    def _get_edge(self, thresh):
        """ 提取边缘像素, 这里直接使用OpenCV的canny算子
        return: 所有边缘像素的数组
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
        edge_px = np.c_[edge_px[0], edge_px[1]]
        return edge_px
    
    def _surface_normals(self, depth_im, edge_pixels):
        """ 返回边缘像素的表面法线的数组 """
        # 计算梯度
        grad = np.gradient(depth_im)

        # 计算表面法线
        normals = np.zeros([edge_pixels.shape[0], 2])
        for i, pix in enumerate(edge_pixels):
            dx = grad[1][pix[0], pix[1]]
            dy = grad[0][pix[0], pix[1]]
            normal_vec = np.array([dy, dx])
            if np.linalg.norm(normal_vec) == 0:
                normal_vec = np.array([1, 0])
            normal_vec = normal_vec / np.linalg.norm(normal_vec)
            normals[i, :] = normal_vec

        return normals

    def sample(self):
        depth = self._depth
        edge_pixels = self._get_edge(self._config['depth_grad_thresh'])
        edge_normals = self._surface_normals(depth, edge_pixels)
        max_grasp_width_px = self._width
        # 返回一个n*n的矩阵，每个值代表梯度方向的点积，由于梯度模长为1
        # 这里的每个值就代表夹角的cos值
        normal_ip = edge_normals.dot(edge_normals.T)
        # 计算每个点之间的距离
        dists = ssd.squareform(ssd.pdist(edge_pixels))
        # 1. 点对梯度夹角大于180-摩擦角
        # 2. 点对距离小于最大距离大于0
        valid_indices = np.where((normal_ip < -np.cos(np.arctan(self._friction_coef)))
                                 & (dists < max_grasp_width_px) & (dists > 0.0))
        valid_indices = np.c_[valid_indices[0], valid_indices[1]]
        return valid_indices

import numpy as np
import pandas as pd
import cv2


class GraspCloseWidth(object):
    """ 计算抓取位置在图像中的闭合宽度
    """

    def __init__(self, image, roi, thresh=0.05):
        self._imgae = image
        self._roi = roi
        self._thresh = thresh
        self._edge_px = self._get_edge(self._thresh)

    def _get_edge(self, thresh):
        """ 提取边缘像素, 这里直接使用OpenCV的canny算子
        return: 所有边缘像素的数组, 这里是世界坐标(x, y)
        """
        # 转换深度图到OpenCV接受的灰度
        depth_g = self._imgae
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

    def _get_min_point(self, p, v, min_angle=5):
        """ 获取在一个方向v上离点p最近的点
        """
        v = v/np.linalg.norm(v)
        # 点到边缘点的向量
        p2edg = self._edge_px - p
        # 点到边缘点的距离
        p2edg_l = np.linalg.norm(p2edg, axis=1)
        # 点到边缘点的方向向量
        p2edg_v = p2edg / p2edg_l[..., np.newaxis]
        # 方向向量和给定方向的夹角要小于最小预设角度
        angle = np.arccos(p2edg_v.dot(v))
        candidate = angle < np.deg2rad(min_angle)
        if candidate.sum() < 1:
            return None
        p2edg_l_s = pd.Series(p2edg_l)[candidate]
        p2edg_l_s = p2edg_l_s.sort_values()
        point = self._edge_px[p2edg_l_s.index[0]]
        return point

    def action(self, grasp, default_width):
        """ 计算一个抓取位姿在该图像上的闭合宽度
        """
        p0 = self._get_min_point(grasp.center_float, grasp.axis)
        p1 = self._get_min_point(grasp.center_float, -grasp.axis)
        if p0 is None or p1 is None:
            print('计算宽度失败')
            return default_width, p0, p1
        return np.linalg.norm(p0 - p1), p0, p1

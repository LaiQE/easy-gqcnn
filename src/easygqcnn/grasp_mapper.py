import logging
import numpy as np
import cv2


class GraspMapper(object):
    """ 抓取映射器类, 把抓取映射到对应的图像上
    即获取可直接用于分类的抓取图像, 这里修改了原始dex-net中的算法
    这里增加了把抓取宽度对齐到目标宽度的步骤
    Note: 这里和项目的定义一直, 除了图像其他都是(x,y)这样的顺序
    """

    def __init__(self, depth, config):
        if 'grasp_mapper' in config.keys():
            self._config = config['grasp_mapper']
        else:
            self._config = config
        self._depth = depth

    def render(self, grasp):
        """ 渲染抓取图像
        1. 采样像素抓取宽度
        2. 旋转图像对齐到抓取轴
        3. 缩放图像到目标抓取宽度
        4. 裁剪图像到最终大小
        """
        if not isinstance(grasp, (list)):
            grasp = [grasp]
        image = self._depth
        out_size = [self._config['final_width'], self._config['final_height']]
        max_width_px = self._config['max_width_px_in_tensor']
        min_width_px = self._config['min_width_px_in_tensor']
        out_image = np.zeros(tuple(np.r_[len(grasp), out_size]))
        out_pose = np.zeros((len(grasp), 1))
        # 均匀采样抓取在最终图像中的宽度
        width_list = np.random.uniform(min_width_px, max_width_px, len(grasp))
        for i in range(len(grasp)):
            g = grasp[i]
            w = width_list[i]
            # 对齐抓取轴到x轴
            image_T = self.transform(image, g.center_float, g.angle)
            # 缩放到目标大小
            image_T = self.resize_image(image_T, w / g.width_px)
            # 裁剪到最终大小
            image_T = self.crop_image(image_T, out_size)
            out_image[i] = image_T
            out_pose[i] = g.depth
        return out_image, out_pose

    @staticmethod
    def transform(image, center, angle):
        """ 先把图片平移到给定点，再旋转给定角度, 这里是顺时针旋转
        注意:图片保存时维度0是行(即y轴)，维度1是列(即x轴)
        """
        # 这里我们需要的是顺时针旋转, 而OpenCV默认的是逆时针旋转
        # 由于图像坐标下y轴是朝下的, 而我们默认y轴朝上, 
        # 所以图像逆时针转在默认坐标下就是顺时针转, 这里不需要取负号
        angle_ = np.rad2deg(angle)
        image_size = np.array(image.shape[:2][::-1]).astype(np.int)
        # 这里要减1, 图像的中点是(0 + Xmax) / 2
        image_center = (image_size-1) / 2
        translation = image_center - center
        trans_map = np.c_[np.eye(2), translation]
        rot_map = cv2.getRotationMatrix2D(tuple(image_center), angle_, 1)
        trans_map_aff = np.r_[trans_map, [[0, 0, 1]]]
        rot_map_aff = np.r_[rot_map, [[0, 0, 1]]]
        full_map = rot_map_aff.dot(trans_map_aff)
        full_map = full_map[:2, :]
        im_data_tf = cv2.warpAffine(image, full_map, tuple(
            image_size), flags=cv2.INTER_NEAREST)
        return im_data_tf

    @staticmethod
    def crop_image(image, crop_size):
        """ 从图像的中间裁剪出目标大小的图像 """
        image_size = np.array(image.shape[:2][::-1]).astype(np.int)
        image_center = (image_size-1) / 2
        diag = np.array(crop_size) / 2
        if any(image_size < crop_size):
            logging.error('裁剪大小超过目标图像大小')
            raise IndexError('裁剪大小超过目标图像大小')
        # 这里要向上取整, 后面的切片操作算第一个不算最后一个
        start = np.ceil(image_center - diag)
        end = np.ceil(image_center + diag)
        image_crop = image[int(start[1]):int(
            end[1]), int(start[0]):int(end[0])].copy()
        return image_crop

    @staticmethod
    def resize_image(image, resize_rate):
        """ 缩放图片大小 """
        image_size = np.array(image.shape[:2][::-1]).astype(np.int)
        out_size = np.ceil(image_size * resize_rate).astype(np.int)
        image_out = cv2.resize(image, tuple(out_size))
        return image_out

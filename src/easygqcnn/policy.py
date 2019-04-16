import logging
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from .grasp_mapper import GraspMapper
from .neural_network import NeuralNetWork
from .grasp_sampler import ImageGraspSampler
from .grasp_2d import Grasp2D


class GraspingPolicy(object):
    def __init__(self, config):
        self._config = config
        self._policy_config = config['policy']
        self._network = NeuralNetWork(self._config)

    def action(self, image, roi):
        """ 寻找最佳的抓取策略
        1. 随机采样抓取
        2. 用gqcnn排序抓取质量
        3. 取前P%的抓取
        4. 用高斯混合模型重新采样抓取分布
        5. 重复以上2-4步骤
        """
        sampler = ImageGraspSampler(image, roi, self._config)
        grasps = sampler.sample(self._policy_config['num_seed_samples'])
        grasps = [g[0] for g in grasps]
        if len(grasps) == 0:
            logging.error('采样抓取失败')
            raise Exception('采样抓取失败')
        grasp_mapper = GraspMapper(image, self._config)
        image_tensor, pose_tensor = grasp_mapper.render(grasps)
        image_tensor = image_tensor[..., np.newaxis]
        for _ in range(self._policy_config['num_iters']):
            q_values = self._network.predict(image_tensor, pose_tensor)
            q_series = pd.Series(q_values)
            sorted_index = np.array(q_series.sort_values(ascending=False).index)
            num_refit = int(np.ceil(self._policy_config['gmm_refit_p'] * len(grasps)))
            # 取出前P%的抓取的特征向量, 组成一个数组
            elite_grasp_arr = np.array([grasps[i].feature_vec for i in sorted_index[:num_refit]])

            # 标准化待处理的抓取特征集合
            elite_grasp_mean = np.mean(elite_grasp_arr, axis=0)
            elite_grasp_std = np.std(elite_grasp_arr, axis=0)
            # TODO: 不知道为什么需要这一步??
            elite_grasp_std[elite_grasp_std == 0] = 1.0
            elite_grasp_arr = (elite_grasp_arr - elite_grasp_mean) / elite_grasp_std

            # 建立高斯混合模型，TODO: 这里可以改用变分的高斯混合模型
            # 用来拟合的高斯分布个数由样本点个数决定
            num_components = int(np.ceil(self._policy_config['gmm_component_frac'] * num_refit))
            # 初始权重为1/个数
            uniform_weights = (1.0 / num_components) * np.ones(num_components)
            # 这里给高斯混合模型加了一个比较大的正则化
            gmm = GaussianMixture(n_components=num_components,
                                  weights_init=uniform_weights,
                                  reg_covar=self._policy_config['gmm_reg_covar'])

            # 使用EM法学习高斯混合模型
            gmm.fit(elite_grasp_arr)

            # 使用高斯混合模型采样新的抓取分布
            grasp_vecs, _ = gmm.sample(n_samples=self._policy_config['num_gmm_samples'])
            grasp_vecs = elite_grasp_std * grasp_vecs + elite_grasp_mean

            # 生成新的抓取
            grasps = []
            for grasp_vec in grasp_vecs:
                # TODO: 这里的夹爪宽度需要由深度数据产生
                grasp_width_px = 30
                grasps.append(Grasp2D.from_feature_vec(grasp_vec, grasp_width_px))
            image_tensor, pose_tensor = grasp_mapper.render(grasps)
            image_tensor = image_tensor[..., np.newaxis]
        # 进行最后的夹爪选择
        q_values = self._network.predict(image_tensor, pose_tensor)
        q_series = pd.Series(q_values)
        sorted_index = np.array(q_series.sort_values(ascending=False).index)
        index = sorted_index[0]
        grasp = grasps[index]
        q_value = q_values[index]
        return grasp, q_value

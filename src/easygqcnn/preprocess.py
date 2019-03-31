import os
import glob
import logging
import random
import numpy as np
import cv2
# import scipy.misc as sm
import scipy.ndimage.filters as sf
import scipy.ndimage.morphology as snm
import scipy.stats as ss
import skimage.draw as sd
import tensorflow as tf


class DataProcesser(object):
    """ 数据预处理器类，对所有的由Dex-Net生成的数据进行以下预处理
    1. 划分训练集和验证集
    2. 添加模拟噪声
    3. 计算均值和标准差进行标准化
    4. 保存到tfrecord
    Note: 这里计算标准差使用公式 平方的期望 - 期望的平方
    """

    def __init__(self, config, raw_path=None, out_path=None):
        if 'data_process' in config.keys():
            self._config = config['data_process']
        self._raw_path = raw_path
        self._out_path = out_path
        if self._raw_path is None:
            self._raw_path = self._config['raw_path']
        if self._out_path is None:
            self._out_path = self._config['out_path']
        if not os.path.exists(os.path.join(self._out_path, 'train')):
            os.mkdir(os.path.join(self._out_path, 'train'))
        if not os.path.exists(os.path.join(self._out_path, 'validation')):
            os.mkdir(os.path.join(self._out_path, 'validation'))
        # 初始化均值和方差
        self._depth_mean = None
        self._pose_mean = None
        self._depth_square_mean = None
        self._pose_square_mean = None
        self._mean_num = 0
        # 初始化tfrecord的缓冲区
        self._train_buffer = []
        self._validation_buffer = []
        self._train_file_num = 0
        self._validation_file_num = 0
        # 数据统计
        self._train_num = 0
        self._validation_num = 0

    def process(self, is_dex=False):
        def load(file, is_dex):
            if is_dex:
                return np.load(file)['arr_0']
            else:
                return np.load(file)
        # TODO : 这里有一个bug文件名不一定对上
        for d_name, p_name, l_name in self.file_list(self._raw_path, is_dex):
            depth = load(d_name, is_dex)
            pose = load(p_name, is_dex)
            label = load(l_name, is_dex)
            if depth.shape[0] != pose.shape[0] or depth.shape[0] != label.shape[0]:
                raise ValueError('shape of data was not consistent')
            for i in range(depth.shape[0]):
                raw_dp = [depth[i], pose[i], label[i]]
                processed_dp = self.process_datapoint(raw_dp)
                self.mean_counter(processed_dp[0], processed_dp[1])
                self.save_datapoint(processed_dp)
        self.save_validation()
        self.save_train()
        self.save_men_std()
        datapoint_info = np.array([self._train_num, self._validation_num])
        np.save(os.path.join(self._out_path, 'datapoint_info.npy'), datapoint_info)
        logging.info('output train datapoint : %d' % (self._train_num))
        logging.info('output validation datapoint : %d' %
                     (self._validation_num))
        logging.info('output train files num : %d' % (self._train_file_num))
        logging.info('output validation files num : %d' %
                     (self._validation_file_num))

    def save_men_std(self):
        depth_mean = self._depth_mean
        depth_std = np.sqrt(self._depth_square_mean - np.square(depth_mean))
        pose_mean = self._pose_mean
        pose_std = np.sqrt(self._pose_square_mean - np.square(pose_mean))
        np.save(os.path.join(self._out_path, 'mean.npy'), depth_mean)
        np.save(os.path.join(self._out_path, 'std.npy'), depth_std)
        np.save(os.path.join(self._out_path, 'pose_mean.npy'), pose_mean)
        np.save(os.path.join(self._out_path, 'pose_std.npy'), pose_std)
        # logging.warning(str(pose_mean)+str(pose_mean.shape))

    def mean_counter(self, depth, pose):
        if self._depth_mean is None:
            self._depth_mean = np.zeros(depth.shape, dtype='float64')
            self._depth_square_mean = np.zeros(depth.shape, dtype='float64')
        if self._pose_mean is None:
            self._pose_mean = np.zeros(pose.shape, dtype='float64')
            self._pose_square_mean = np.zeros(pose.shape, dtype='float64')
        alpha = self._mean_num/(self._mean_num + 1)
        self._mean_num = self._mean_num + 1
        self._depth_mean = alpha*self._depth_mean + (1 - alpha)*depth
        self._pose_mean = alpha*self._pose_mean + (1 - alpha)*pose
        depth_square = np.square(depth)
        pose_square = np.square(pose)
        self._depth_square_mean = alpha * \
            self._depth_square_mean + (1 - alpha)*depth_square
        self._pose_square_mean = alpha * \
            self._pose_square_mean + (1 - alpha)*pose_square

    def save_datapoint(self, datapoint):
        depth = datapoint[0]
        pose = datapoint[1]
        label = datapoint[2]
        feature = {'depth': self._floats_feature(depth.flatten()),
                   'pose': self._floats_feature(pose.flatten()),
                   'label': self._floats_feature(label)}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        if np.random.rand() < self._config['validation_percent']:
            self._validation_buffer.append(example)
            self._validation_num += 1
        else:
            self._train_num += 1
            self._train_buffer.append(example)
        if len(self._validation_buffer) >= self._config['datapoint_pre_file']:
            self.save_validation()
            self._validation_buffer = []
        if len(self._train_buffer) >= self._config['datapoint_pre_file']:
            self.save_train()
            self._train_buffer = []

    def save_validation(self):
        file_name = 'validation_%06d.tfrecord' % (self._validation_file_num)
        logging.debug('save file: ' + file_name)
        self._validation_file_num += 1
        file = os.path.join(self._out_path, 'validation', file_name)
        if os.path.exists(file):
            os.remove(file)
        writer = tf.python_io.TFRecordWriter(file)
        for example in self._validation_buffer:
            writer.write(example.SerializeToString())
        writer.close()

    def save_train(self):
        file_name = 'train_%06d.tfrecord' % (self._train_file_num)
        logging.debug('save file: ' + file_name)
        self._train_file_num += 1
        file = os.path.join(self._out_path, 'train', file_name)
        if os.path.exists(file):
            os.remove(file)
        writer = tf.python_io.TFRecordWriter(file)
        for example in self._train_buffer:
            writer.write(example.SerializeToString())
        writer.close()

    def process_datapoint(self, datapoint):
        """ 对每个数据点进行处理 """
        depth = np.squeeze(datapoint[0])
        depth = self.distort(depth, self._config)
        pose = datapoint[1][2:3]
        # 因为dex-net和easy-dexnet生成的这个数据维度不一样
        # 原始的只是一个数值，而easy-dexnet生成的是一个一维的array
        label = np.array(datapoint[2]).reshape((1,))
        return (depth, pose, label)

    @staticmethod
    def _floats_feature(value):
        # 这里的value=后面没有括号, 参考https://blog.csdn.net/zxyhhjs2017/article/details/82774732
        # 千万不要写成return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def file_list(self, raw_path, is_dex=False):
        """ 读取文件名列表
            is_dex: 是否是以原始的dex-net定义的文件名
                    如果是否则使用easy-dexnet定义的文件名
        """
        if is_dex:
            depth_files = os.path.join(
                raw_path, 'depth_ims_tf_table*.npz')
        else:
            depth_files = os.path.join(raw_path, 'depth', '*.npy')
        depth_list = glob.glob(depth_files)
        random.shuffle(depth_list)
        for depth_file in depth_list:
            depth_name = os.path.split(depth_file)[1]
            if is_dex:
                pose_file_name = depth_name.replace('depth_ims_tf_table', 'hand_poses')
                label_file_name = depth_name.replace('depth_ims_tf_table', 'robust_ferrari_canny')
                pose_file = os.path.join(raw_path, pose_file_name)
                label_file = os.path.join(raw_path, label_file_name)
            else:
                pose_file_name = depth_name.replace('depth', 'hand_pose')
                label_file_name = depth_name.replace('depth', 'quality')
                pose_file = os.path.join(raw_path, 'hand_pose', pose_file_name)
                label_file = os.path.join(raw_path, 'quality', label_file_name)
            yield (depth_file, pose_file, label_file)

    @staticmethod
    def distort(imgae, config):
        """ 向图像中添加噪声
        这个函数修改自gqcnn的源程序中，具体原理参考论文
        """
        imgae_ = imgae.copy()
        # config = self._config
        im_height = imgae_.shape[0]
        im_width = imgae_.shape[1]
        im_center = np.array([float(im_height-1)/2, float(im_width-1)/2])
        # denoising and synthetic data generation
        if config['multiplicative_denoising']:
            gamma_shape = config['gamma_shape']
            gamma_scale = 1.0 / gamma_shape
            mult_samples = ss.gamma.rvs(gamma_shape, scale=gamma_scale)
            imgae_ = imgae_ * mult_samples

        # randomly dropout regions of the image for robustness
        if config['image_dropout']:
            if np.random.rand() < config['image_dropout_rate']:
                nonzero_px = np.where(imgae_ > 0)
                nonzero_px = np.c_[nonzero_px[0], nonzero_px[1]]
                num_nonzero = nonzero_px.shape[0]
                num_dropout_regions = ss.poisson.rvs(
                    config['dropout_poisson_mean'])

                # sample ellipses
                dropout_centers = np.random.choice(
                    num_nonzero, size=num_dropout_regions)
                x_radii = ss.gamma.rvs(
                    config['dropout_radius_shape'], scale=config['dropout_radius_scale'], size=num_dropout_regions)
                y_radii = ss.gamma.rvs(
                    config['dropout_radius_shape'], scale=config['dropout_radius_scale'], size=num_dropout_regions)

                # set interior pixels to zero
                for j in range(num_dropout_regions):
                    ind = dropout_centers[j]
                    dropout_center = nonzero_px[ind, :]
                    x_radius = x_radii[j]
                    y_radius = y_radii[j]
                    dropout_px_y, dropout_px_x = sd.ellipse(
                        dropout_center[0], dropout_center[1], y_radius, x_radius, shape=imgae_.shape)
                    imgae_[dropout_px_y, dropout_px_x] = 0.0

        # dropout a region around the areas of the image with high gradient
        if config['gradient_dropout']:
            if np.random.rand() < config['gradient_dropout_rate']:
                grad_mag = sf.gaussian_gradient_magnitude(
                    imgae_, sigma=config['gradient_dropout_sigma'])
                thresh = ss.gamma.rvs(
                    config['gradient_dropout_shape'], config['gradient_dropout_scale'], size=1)
                high_gradient_px = np.where(grad_mag > thresh)
                imgae_[high_gradient_px[0], high_gradient_px[1]] = 0.0

        # add correlated Gaussian noise
        if config['gaussian_process_denoising']:
            gp_rescale_factor = config['gaussian_process_scaling_factor']
            gp_sample_height = int(im_height / gp_rescale_factor)
            gp_sample_width = int(im_width / gp_rescale_factor)
            gp_num_pix = gp_sample_height * gp_sample_width
            if np.random.rand() < config['gaussian_process_rate']:
                gp_noise = ss.norm.rvs(scale=config['gaussian_process_sigma'], size=gp_num_pix).reshape(
                    gp_sample_height, gp_sample_width)
                # sm.imresize 有警告将被弃用
                # gp_noise = sm.imresize(
                #     gp_noise, gp_rescale_factor, interp='bicubic', mode='F')
                # st.resize 用来替用将被弃用的sm.imresize
                # gp_noise = st.resize(gp_noise, (im_height, im_width))
                gp_noise = cv2.resize(
                    gp_noise, (im_height, im_width), interpolation=cv2.INTER_CUBIC)
                imgae_[imgae_ > 0] += gp_noise[imgae_ > 0]

        # run open and close filters to
        if config['morphological']:
            sample = np.random.rand()
            morph_filter_dim = ss.poisson.rvs(
                config['morph_poisson_mean'])
            if sample < config['morph_open_rate']:
                imgae_ = snm.grey_opening(
                    imgae_, size=morph_filter_dim)
            else:
                closed_imgae_ = snm.grey_closing(
                    imgae_, size=morph_filter_dim)

                # set new closed pixels to the minimum depth, mimicing the table
                new_nonzero_px = np.where(
                    (imgae_ == 0) & (closed_imgae_ > 0))
                closed_imgae_[new_nonzero_px[0], new_nonzero_px[1]] = np.min(
                    imgae_[imgae_ > 0])
                imgae_ = closed_imgae_.copy()

        # randomly dropout borders of the image for robustness
        if config['border_distortion']:
            grad_mag = sf.gaussian_gradient_magnitude(
                imgae_, sigma=config['border_grad_sigma'])
            high_gradient_px = np.where(
                grad_mag > config['border_grad_thresh'])
            high_gradient_px = np.c_[
                high_gradient_px[0], high_gradient_px[1]]
            num_nonzero = high_gradient_px.shape[0]
            num_dropout_regions = ss.poisson.rvs(
                config['border_poisson_mean'])

            # sample ellipses
            dropout_centers = np.random.choice(
                num_nonzero, size=num_dropout_regions)
            x_radii = ss.gamma.rvs(
                config['border_radius_shape'], scale=config['border_radius_scale'], size=num_dropout_regions)
            y_radii = ss.gamma.rvs(
                config['border_radius_shape'], scale=config['border_radius_scale'], size=num_dropout_regions)

            # set interior pixels to zero or one
            for j in range(num_dropout_regions):
                ind = dropout_centers[j]
                dropout_center = high_gradient_px[ind, :]
                x_radius = x_radii[j]
                y_radius = y_radii[j]
                dropout_px_y, dropout_px_x = sd.ellipse(
                    dropout_center[0], dropout_center[1], y_radius, x_radius, shape=imgae_.shape)
                if np.random.rand() < 0.5:
                    imgae_[dropout_px_y, dropout_px_x] = 0.0
                else:
                    imgae_[dropout_px_y, dropout_px_x] = imgae_[
                        dropout_center[0], dropout_center[1]]

        # randomly replace background pixels with constant depth
        if config['background_denoising']:
            if np.random.rand() < config['background_rate']:
                imgae_[imgae_ > 0] = config['background_min_depth'] + (
                    config['background_max_depth'] - config['background_min_depth']) * np.random.rand()

        # symmetrize images
        if config['symmetrize']:
            # rotate with 50% probability
            if np.random.rand() < 0.5:
                theta = 180.0
                rot_map = cv2.getRotationMatrix2D(
                    tuple(im_center), theta, 1)
                imgae_ = cv2.warpAffine(
                    imgae_, rot_map, (im_height, im_width), flags=cv2.INTER_NEAREST)
            # reflect left right with 50% probability
            if np.random.rand() < 0.5:
                imgae_ = np.fliplr(imgae_)
            # reflect up down with 50% probability
            if np.random.rand() < 0.5:
                imgae_ = np.flipud(imgae_)
        return imgae_

import os
import numpy as np
import tensorflow as tf


class NeuralNetWork(object):
    """ 神经网络定义类
    """

    def __init__(self, config):
        if 'gqcnn_config' in config.keys():
            self._config = config['gqcnn_config']
        else:
            self._config = config
        self._graph = tf.Graph()
        self.initialize_network()
    
    @property
    def graph(self):
        return self._graph

    def initialize_network(self, add_softmax=False):
        with self._graph.as_default():
            self.image_input = tf.placeholder(tf.float32, [None, self._config['im_width'],
                                                           self._config['im_height'],
                                                           self._config['im_channels']], name='image_input')
            self.pose_input = tf.placeholder(tf.float32, [None, self._config['pose_dim']], name='pose_input')
            self.output = self._build_network(self.image_input, self.pose_input, self._config)
            if add_softmax:
                self.output = tf.nn.softmax(self.output)
    
    def load_weights(self, sess, file, remap=False):
        """ 从文件中加载神经网络权重
            file: 文件名
            remap: 映射GQCNN的参数到easy-gqcnn的参数
        """
        if os.path.splitext(file)[-1] == '.npz':
            weights_dict = np.load(file)
            get_weights = lambda name : weights_dict[name] 
        elif os.path.splitext(file)[-1] == '.ckpt':
            weights_dict = tf.train.NewCheckpointReader(file)
            get_weights = lambda name: weights_dict.get_tensor(name)
        else:
            raise Exception('file extension must .npz or .ckpt')
        for name in self._config['architecture'].keys():
            with self._graph.as_default():
                with tf.variable_scope(name, reuse=True) as scope:
                    if name != 'fc4':
                        d = zip(['biases', 'weights'],['b', 'W'])
                    else:
                        d = zip(['biases', 'weights_x', 'weights_y'],['b', 'W_im', 'W_pose'])
                    for v,r in d:
                        if remap:
                            w = get_weights(name + r)
                        else:
                            w = get_weights(name + '/' + v)
                        var = tf.get_variable(v, trainable=False)
                        sess.run(var.assign(w))
    
    def save_to_npz(self, sess, file):
        weights = {}
        for name in self._config['architecture'].keys():
            with self._graph.as_default():
                with tf.variable_scope(name, reuse=True) as scope:
                    if name != 'fc4':
                        d = ['biases', 'weights']
                    else:
                        d = ['biases', 'weights_x', 'weights_y']
                    for v in d:
                        var = tf.get_variable(v, trainable=False)
                        weights[name + '/' + v] = sess.run(var)
        np.savez_compressed(file, **weights)

    def _build_network(self, image_node, pose_node, config):
        conv_pool_1_1 = self._conv_pool(image_node, 'conv1_1', config)
        conv_pool_1_2 = self._conv_pool(conv_pool_1_1, 'conv1_2', config)
        conv_pool_2_1 = self._conv_pool(conv_pool_1_2, 'conv2_1', config)
        conv_pool_2_2 = self._conv_pool(conv_pool_2_1, 'conv2_2', config)
        fc_input = conv_pool_2_2
        if 'conv3_1' in config['architecture'].keys():
            conv_pool_3_1 = self._conv_pool(
                conv_pool_2_2, 'conv3_1', config)
            sconv_pool_3_2 = self._conv_pool(
                conv_pool_3_1, 'conv3_2', config)
            fc_input = conv_pool_3_2
        fc_size = int(np.prod(fc_input.get_shape()[1:]))
        fc_input_falt = tf.reshape(fc_input, [-1, fc_size])
        fc3 = self._fc(fc_input_falt, 'fc3', config)
        pc1 = self._fc(pose_node, 'pc1', config)
        pc_out = pc1
        if 'pc2' in config['architecture'].keys() and config['architecture']['pc2']['out_siz'] > 0:
            pc2 = self._fc(pc1, 'pc2', config)
            pc_out = pc2
        fc4 = self._tow_input_fc(fc3, pc_out, 'fc4', config)
        fc5 = self._fc(fc4, 'fc5', config, relu=False)
        return fc5

    @staticmethod
    def _conv_pool(x, name, config):
        """ 通过配置文件生成一个卷积层和一个池化层 """
        # 结构配置
        try:
            arc_config = config['architecture'][name]
        except Exception as e:
            raise e('config is not key architecture or %s' % name)
        # 过滤器尺寸
        filt_dim = arc_config['filt_dim']
        # 过滤器深度
        num_filt = arc_config['num_filt']
        # 输入层深度
        input_channels = int(x.get_shape()[-1])

        weights_shape = [filt_dim, filt_dim, input_channels, num_filt]

        pool_shape = [1, arc_config['pool_size'], arc_config['pool_size'], 1]
        pool_stride = [1, arc_config['pool_stride'],
                       arc_config['pool_stride'], 1]

        std = np.sqrt(2.0 / (filt_dim**2 * input_channels))

        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weights', shape=weights_shape,
                                      initializer=tf.truncated_normal_initializer(stddev=std))
            biases = tf.get_variable('biases', shape=[num_filt],
                                     initializer=tf.truncated_normal_initializer(stddev=std))
            conv = tf.nn.conv2d(x, weights, [1, 1, 1, 1], 'SAME', name='conv')
            bias = tf.nn.bias_add(conv, biases, name='bias')
            relu = tf.nn.relu(bias, name='relu')
            if arc_config['norm'] and arc_config['norm_type'] == "local_response":
                conv_out = tf.nn.local_response_normalization(
                    relu, depth_radius=config['radius'], alpha=config['alpha'], beta=config['beta'], bias=config['bias'], name='lrn')
            else: conv_out = relu
            pool = tf.nn.max_pool(conv_out, pool_shape, pool_stride, 'SAME')
        return pool

    @staticmethod
    def _fc(x, name, config, relu=True):
        """ 通过配置文件生成一个全连接层 """
        try:
            arc_config = config['architecture'][name]
        except Exception as e:
            raise e('config is not key architecture or %s' % name)
        input_size = int(x.get_shape()[-1])
        output_size = arc_config['out_size']
        std = np.sqrt(2.0 / input_size)
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weights', shape=[input_size, output_size],
                                      initializer=tf.truncated_normal_initializer(stddev=std))
            biases = tf.get_variable('biases', shape=[output_size],
                                     initializer=tf.truncated_normal_initializer(stddev=std))
            mat = tf.matmul(x, weights, name='mat')
            bias = tf.nn.bias_add(mat, biases, name='bias')
            out_node = bias
            if relu:
                relu = tf.nn.relu(bias, name='relu')
                out_node = relu
                if 'drop_out' in arc_config.keys() and arc_config['drop_out']:
                    drop = tf.nn.dropout(
                        relu, arc_config['drop_rate'], name='drop')
                    out_node = drop_out
        return out_node
    
    @staticmethod
    def _tow_input_fc(x, y, name, config, relu=True):
        """ 两个输入的全连接层 """
        try:
            arc_config = config['architecture'][name]
        except Exception as e:
            raise e('config is not key architecture or %s' % name)
        input_size_x = int(x.get_shape()[-1])
        input_size_y = int(y.get_shape()[-1])
        output_size = arc_config['out_size']
        std = np.sqrt(2.0 / (input_size_x + input_size_y))
        with tf.variable_scope(name) as scope:
            weights_x = tf.get_variable('weights_x', shape=[input_size_x, output_size],
                                      initializer=tf.truncated_normal_initializer(stddev=std))
            weights_y = tf.get_variable('weights_y', shape=[input_size_y, output_size],
                                      initializer=tf.truncated_normal_initializer(stddev=std))
            biases = tf.get_variable('biases', shape=[output_size],
                                     initializer=tf.truncated_normal_initializer(stddev=std))
            mat = tf.matmul(x, weights_x, name='mat_x') + tf.matmul(y, weights_y, name='mat_y')
            bias = tf.nn.bias_add(mat, biases, name='bias')
            out_node = bias
            if relu:
                relu = tf.nn.relu(bias, name='relu')
                out_node = relu
                if 'drop_out' in arc_config.keys() and arc_config['drop_out']:
                    drop = tf.nn.dropout(
                        relu, arc_config['drop_rate'], name='drop')
                    out_node = drop_out
        return out_node
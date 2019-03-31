import os
import glob
import logging
import numpy as np
import tensorflow as tf


class GQCNNTraing(object):
    """ 训练GQCNN神经网络
    1. 读取预处理数据, 这里使用tf的Dataset
    2. 建立神经网络结构
    3. 定义损失和优化器
    4. 迭代训练
    """

    def __init__(self, config, network, data_path=None, out_path=None):
        if 'training' in config.keys():
            self._config = config['training']
        else:
            self._config = config
        if data_path is None:
            data_path = self._config['data_path']
        self._data_path = data_path
        if out_path is None:
            out_path = self._config['out_path']
        self._out_path = out_path
        self._network = network
        # 先载入预处理生成的一些参数
        self.pre_load()
        # 创建数据集和神经网络配置
        self._train_out, self._val_out, self._train_label, self._val_label = self.creat_network()
        # 创建损失函数
        self._loss = self.creat_loss()
        # 创建优化器
        self._optimizer = self.create_optimizer()
        # 创建准确率计算
        self._val_accuracy = self.val_accuracy()

    def optimize(self, epoch_num):
        """ 进行模型训练
        """
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        with self._network.graph.as_default():
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
        with tf.Session(graph=self._network.graph, config=gpu_config) as sess:
            sess.run(init)
            if self._config['fine_tune']:
                self._network.load_weights(sess, self.config['model_file'])
            # 训练过程
            for epoch in range(epoch_num):
                logging.info("%d epoch training is start!" % (epoch + 1))
                sess.run(self.train_iterator.initializer)
                while True:
                    try:
                        sess.run(self._optimizer)
                    except tf.errors.OutOfRangeError:
                        logging.info("%d epoch training is finish!" %
                                     (epoch + 1))
                        break
                self.validation(sess, epoch)
                saver.save(sess, os.path.join(self._out_path,
                                              'model.ckpt'), global_step=epoch)

    def validation(self, sess, i):
        logging.info("%d epoch validation is start!" % (i + 1))
        acc = 0
        con = 0
        sess.run(self.val_iterator.initializer)
        while True:
            try:
                acc += sess.run(self._val_accuracy)
                con += 1
            except tf.errors.OutOfRangeError:
                break
        logging.info("%d epoch training is finish!" % (i + 1))
        final_acc = acc / con * 100
        logging.info("%d epoch validation is %.3f!" % (i+1, final_acc))

    def pre_load(self):
        """ 预加载预处理好的数据信息
        datapoint个数, mean, std
        """
        info = np.load(os.path.join(self._data_path, 'datapoint_info.npy'))
        self._train_num = info[0]
        self._val_num = info[1]
        self._im_mean = np.load(os.path.join(self._data_path, 'mean.npy'))
        self._im_std = np.load(os.path.join(self._data_path, 'std.npy'))
        self._pose_mean = np.load(os.path.join(
            self._data_path, 'pose_mean.npy'))
        self._pose_std = np.load(os.path.join(self._data_path, 'pose_std.npy'))

    def creat_network(self):
        """ 初始化神经网络的配置
        """
        # 初始化训练数据和验证数据
        with self._network.graph.as_default():
            self.train_iterator = self.dataset(os.path.join(self._data_path, 'train'),
                                               self._config['train_batch_size'])
            train_im, train_pose, train_label = self.train_iterator.get_next()
            self.val_iterator = self.dataset(os.path.join(self._data_path, 'validation'),
                                             self._config['val_batch_size'])
            val_im, val_pose, val_label = self.val_iterator.get_next()
        train_out = self._network.inference(
            train_im, train_pose, add_softmax=False, drop_out=self._config['train_drop_out'])
        val_out = self._network.inference(val_im, val_pose, add_softmax=True)
        return train_out, val_out, train_label, val_label

    def creat_loss(self):
        with self._network.graph.as_default():
            cross_entropy_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self._train_label, logits=self._train_out))
            regularizer = tf.contrib.layers.l2_regularizer(
                self._config['train_l2_regularizer'])
            for variable in self._network.get_variables('all', 'all'):
                tf.add_to_collection('losses', regularizer(variable))
            loss = cross_entropy_loss + tf.add_n(tf.get_collection('losses'))
        return loss

    def create_optimizer(self):
        var_list = self._network.get_variables('all', 'all')
        if self._config['fine_tune'] and self._config['update_fc_only']:
            var_list = self._network.get_variables('fc', 'all')
        with self._network.graph.as_default():
            global_step = tf.Variable(0)
            learning_rate = tf.train.exponential_decay(
                self._config['base_lr'],                    # 基础学习率.
                global_step,                                # 表示当前轮次的变量
                self._train_num / self._config['train_batch_size'],  # 多少轮衰减一次.
                self._config['decay_rate'],                     # 学习率衰减率.
                staircase=True)
            if self._config['optimizer'] == 'momentum':
                optimizer = tf.train.MomentumOptimizer(
                    learning_rate, self._config['momentum_rate'])
            elif self._config['optimizer'] == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            elif self._config['optimizer'] == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(learning_rate)
            return optimizer.minimize(self._loss, global_step=global_step, var_list=var_list)

    def val_accuracy(self):
        logging.warning(str(self._val_label.dtype))
        a = tf.argmax(self._val_out, 1)
        logging.warning(str(a.dtype))
        with self._network.graph.as_default():
            correct_pred = tf.equal(
                tf.argmax(self._val_out, 1), self._val_label)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float64))
        return accuracy

    def dataset(self, path, batch_size):
        filenames = glob.glob(os.path.join(path, '*.tfrecord'))
        # print(filenames)
        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=2)
        dataset = dataset.map(self._parse_dataset, num_parallel_calls=2)
        dataset = dataset.shuffle(
            buffer_size=self._config['dataset_buffer_size'])
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(1)
        iterator = dataset.make_initializable_iterator()
        return iterator

    def _parse_dataset(self, example_proto):
        depth_size = (self._config['im_height'],
                      self._config['im_width'], self._config['im_channels'])
        depth_len = self._config['im_height'] * self._config['im_width']
        feature = {'depth': tf.FixedLenFeature((depth_len,), tf.float32),
                   'pose': tf.FixedLenFeature((self._config['pose_len'],), tf.float32),
                   'label': tf.FixedLenFeature((1,), tf.float32)}
        parsed_features = tf.parse_single_example(example_proto, feature)
        depth = tf.reshape(parsed_features['depth'], depth_size)
        depth = depth - tf.constant(self._im_mean.astype('float32'))
        depth = depth / tf.constant(self._im_std.astype('float32'))
        pose = parsed_features['pose']
        pose = pose - tf.constant(self._pose_mean.astype('float32'))
        pose = pose / tf.constant(self._pose_std.astype('float32'))
        condition = tf.less(
            parsed_features['label'], self._config['metric_thresh'])
        label = tf.where(condition, tf.constant(
            [1], tf.int64), tf.constant([0], tf.int64))
        label = tf.reshape(label, ())
        return depth, pose, label

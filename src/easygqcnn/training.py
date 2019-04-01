import os
import glob
import logging
from datetime import datetime
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
        # 给tensorboard加上时间戳,参考https://blog.csdn.net/shahuzi/article/details/81223980
        summary_time = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        self._summary_dir = os.path.join(
            self._out_path, 'summary', summary_time)
        self._network = network
        # 先载入预处理生成的一些参数
        self.pre_load()
        # 创建数据集和神经网络配置
        self.creat_network()
        # 创建损失函数
        self._loss = self.creat_loss()
        # 创建优化器
        self._optimizer = self.create_optimizer()
        # 创建准确率计算
        self._val_accuracy = self.accuracy(self._val_label, self._val_out)
        with self._network.graph.as_default():
            self._train_softmax = tf.nn.softmax(self._train_out)
        self._train_accuracy = self.accuracy(
            self._train_label, self._train_softmax)
        # 创建tensorboard配置
        self.setup_tensorboard()

    def optimize(self, epoch_num):
        """ 进行模型训练
        """
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        step = 0
        mean_acc = 0
        mean_loss = 0
        with self._network.graph.as_default():
            init = tf.global_variables_initializer()
            saver = tf.train.Saver(max_to_keep=3)
            summary_writer = tf.summary.FileWriter(self._summary_dir)
            summary_writer.add_graph(self._network.graph)
        with tf.Session(graph=self._network.graph, config=gpu_config) as sess:
            sess.run(init)
            if self._config['fine_tune']:
                self._network.load_weights(sess, self.config['model_file'])
            # 训练过程
            self.validation(sess, 0, summary_writer)
            for epoch in range(epoch_num):
                logging.info("%d epoch training is start!" % (epoch + 1))
                sess.run(self._train_iterator.initializer)
                while True:
                    try:
                        _, acc, loss = sess.run(
                            [self._optimizer, self._train_accuracy, self._loss])
                        mean_acc += acc
                        mean_loss += loss
                        step += 1

                        if step % self._config['summary_step'] == 0:
                            mean_acc = mean_acc / self._config['summary_step']
                            mean_loss = mean_loss / self._config['summary_step']
                            summary = sess.run(self._train_merged, {self._train_mean_acc: mean_acc,
                                                                    self._train_mean_loss: mean_loss})
                            summary_writer.add_summary(summary, step)
                            logging.debug('step: %d, mean_loss: %.4f, mean_acc: %.4f' %
                                          (step, mean_loss, mean_acc))
                            mean_acc = 0
                            mean_loss = 0
                    except tf.errors.OutOfRangeError:
                        logging.info("%d epoch training is finish!" %
                                     (epoch + 1))
                        break
                self.validation(sess, epoch + 1, summary_writer)
                saver.save(sess, os.path.join(self._out_path, 'model.ckpt'), global_step=epoch)
            self._network.save_to_npz(sess, os.path.join(self._out_path, 'model.npz'))
            summary_writer.close()

    def validation(self, sess, i, writer):
        acc = 0
        con = 0
        sess.run(self._val_iterator.initializer)
        while True:
            try:
                acc += sess.run(self._val_accuracy)
                con += 1
            except tf.errors.OutOfRangeError:
                break
        final_acc = acc / con * 100
        summary = sess.run(self._val_merged, {self._validation_acc: final_acc})
        writer.add_summary(summary, i)
        logging.info("%d epoch validation is %.3f!" % (i, final_acc))

    def setup_tensorboard(self):
        with self._network.graph.as_default():
            with tf.name_scope('summary'):
                validation_acc = tf.placeholder(tf.float32, [])
                train_mean_acc = tf.placeholder(tf.float32, [])
                train_mean_loss = tf.placeholder(tf.float32, [])
                tf.summary.scalar('val_accuracy', validation_acc,
                                  collections=['val_summary'])
                tf.summary.scalar('train_accuracy', train_mean_acc,
                                  collections=['train_summary'])
                tf.summary.scalar('train_loss', train_mean_loss,
                                  collections=['train_summary'])
                train_merged = tf.summary.merge_all('train_summary')
                val_merged = tf.summary.merge_all('val_summary')
        self._train_merged = train_merged
        self._val_merged = val_merged
        self._validation_acc = validation_acc
        self._train_mean_acc = train_mean_acc
        self._train_mean_loss = train_mean_loss

    def pre_load(self):
        """ 预加载预处理好的数据信息
        datapoint个数, mean, std
        Note: 这个函数会修改6个私有属性
        """
        info = np.load(os.path.join(self._data_path, 'datapoint_info.npy'))
        self._train_num = info[0]
        self._val_num = info[1]
        self._im_mean = np.load(os.path.join(self._data_path, 'mean.npy'))
        self._im_std = np.load(os.path.join(self._data_path, 'std.npy'))
        self._pose_mean = np.load(os.path.join(
            self._data_path, 'pose_mean.npy'))
        self._pose_std = np.load(os.path.join(self._data_path, 'pose_std.npy'))

    def creat_loss(self):
        """ 创建损失，这里的损失定义为交叉熵损失加上L2正则化损失
        """
        with self._network.graph.as_default():
            # 计算所有变量的L2正则化损失
            with tf.name_scope('loss'):
                # 计算交叉熵损失，注意：这个函数的参数需要是1维的张量,所以_parse_dataset时需要reshape label
                cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self._train_label, logits=self._train_out))
                regularizer = tf.contrib.layers.l2_regularizer(
                    self._config['train_l2_regularizer'])
                for variable in self._network.get_variables('all', 'all'):
                    tf.add_to_collection('losses', regularizer(variable))
                loss = cross_entropy + tf.add_n(tf.get_collection('losses'))
        return loss

    def create_optimizer(self):
        """ 创建优化器
        """
        # 创建优化变量列表，如果是tuneing则按配置只优化全连接层
        var_list = self._network.get_variables('all', 'all')
        if self._config['fine_tune'] and self._config['update_fc_only']:
            var_list = self._network.get_variables('fc', 'all')
        with self._network.graph.as_default():
            with tf.name_scope('optimizer'):
                # 创建全局迭代次数变量
                global_step = tf.Variable(0, name='global_step')
                learning_rate = tf.train.exponential_decay(
                    self._config['base_lr'],                    # 基础学习率.
                    global_step,                                # 表示当前轮次的变量
                    # 多少轮衰减一次, 为训练集样本点数除以训练batch数
                    self._train_num / self._config['train_batch_size'],
                    self._config['decay_rate'],                     # 学习率衰减率.
                    staircase=True)
                # 配置优化器
                if self._config['optimizer'] == 'momentum':
                    optimizer = tf.train.MomentumOptimizer(
                        learning_rate, self._config['momentum_rate'])
                elif self._config['optimizer'] == 'adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate)
                elif self._config['optimizer'] == 'rmsprop':
                    optimizer = tf.train.RMSPropOptimizer(learning_rate)
                return optimizer.minimize(self._loss, global_step=global_step, var_list=var_list)

    def accuracy(self, label, out):
        with self._network.graph.as_default():
            with tf.name_scope('accuracy'):
                correct_pred = tf.equal(tf.argmax(out, 1), label)
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float64))
        return accuracy

    def creat_network(self):
        """ 初始化神经网络的配置
        Note: 这个函数会之间修改私有变量, 后面需要尽可能减少耦合
        """
        # 初始化训练数据和验证数据
        with self._network.graph.as_default():
            train_iterator = self.dataset(os.path.join(self._data_path, 'train'),
                                          self._config['train_batch_size'])
            val_iterator = self.dataset(os.path.join(self._data_path, 'validation'),
                                        self._config['val_batch_size'])
            train_im, train_pose, train_label = train_iterator.get_next()
            val_im, val_pose, val_label = val_iterator.get_next()
        train_out = self._network.inference(
            train_im, train_pose, add_softmax=False, drop_out=self._config['train_drop_out'])
        val_out = self._network.inference(val_im, val_pose, add_softmax=True)

        self._train_out = train_out
        self._val_out = val_out
        self._train_label = train_label
        self._val_label = val_label
        self._train_iterator = train_iterator
        self._val_iterator = val_iterator

    def dataset(self, path, batch_size):
        with tf.name_scope('dataset'):
            filenames = glob.glob(os.path.join(path, '*.tfrecord'))
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

import os
import logging
import unittest
import tensorflow as tf
from ruamel.yaml import YAML
from easygqcnn import NeuralNetWork

# 当前文件所在位置
file_path = os.path.split(__file__)[0]
ROOT_PATH = os.path.abspath(os.path.join(file_path, '..'))
MODEL_PATH = os.path.join(ROOT_PATH, 'data/test/models/ckpt/model.ckpt')
TEST_LOG_FILE = os.path.join(ROOT_PATH, 'tests/logs/test_network.log')
TEST_CFG_FILE = os.path.join(ROOT_PATH, 'config/test.yaml')
SAVE_PATH = os.path.join(ROOT_PATH, 'data/test/models/npz/model.npz')


def config_logging(file=None, level=logging.DEBUG):
    """ 配置全局的日志设置 """
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename=file, level=level,
                        format=LOG_FORMAT, filemode='w')


def load_config(file):
    """ 加载配置文件 """
    yaml = YAML(typ='safe')   # default, if not specfied, is 'rt' (round-trip)
    with open(file, 'r', encoding="utf-8") as f:
        config = yaml.load(f)
    return config


class NetworkTestCase(unittest.TestCase):
    @classmethod
    def setUp(cls):
        # 配置成gpu显存使用量按需求增长
        cls.gpu_config = tf.ConfigProto()
        cls.gpu_config.gpu_options.allow_growth = True
        cls.config = load_config(TEST_CFG_FILE)
        # 屏蔽TensorFlow输出的通知信息，参考https://blog.csdn.net/dcrmg/article/details/80029741
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    @classmethod
    def tearDown(cls):
        pass

    def test_create(self):
        gqcnn = NeuralNetWork(self.config)
        self.assertIsNotNone(gqcnn)

    # @unittest.skip('skip test_load_ckpt')
    def test_load_ckpt(self):
        gqcnn = NeuralNetWork(self.config)
        with tf.Session(graph=gqcnn.graph, config=self.gpu_config) as sess:
            gqcnn.load_weights(sess, MODEL_PATH, remap=True)

    # @unittest.skip('skip test_save')
    def test_save(self):
        gqcnn = NeuralNetWork(self.config)
        with tf.Session(graph=gqcnn.graph, config=self.gpu_config) as sess:
            gqcnn.load_weights(sess, MODEL_PATH, remap=True)
            gqcnn.save_to_npz(sess, SAVE_PATH)

    # @unittest.skip('skip test_load_npz')
    def test_load_npz(self):
        gqcnn = NeuralNetWork(self.config)
        with tf.Session(graph=gqcnn.graph, config=self.gpu_config) as sess:
            gqcnn.load_weights(sess, SAVE_PATH)

    # @unittest.skip('skip test_inference')
    def test_inference(self):
        gqcnn = NeuralNetWork(self.config)
        config_ = self.config['gqcnn_config']
        with gqcnn.graph.as_default():
            image = tf.placeholder(tf.float32, [None, config_['im_width'], config_['im_height'],
                                                config_['im_channels']], name='image_input')
            pose = tf.placeholder(
                tf.float32, [None, config_['pose_dim']], name='pose_input')
        out = gqcnn.inference(image, pose)
        self.assertIsNotNone(out)


if __name__ == "__main__":
    config_logging(TEST_LOG_FILE)
    try:
        unittest.main()
    except:
        pass

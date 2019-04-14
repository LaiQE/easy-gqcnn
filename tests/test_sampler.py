import os
import logging
import unittest
import numpy as np
from ruamel.yaml import YAML
from easygqcnn import ImageGraspSampler

file_path = os.path.split(__file__)[0]
ROOT_PATH = os.path.abspath(os.path.join(file_path, '..'))
TEST_LOG_FILE = os.path.join(ROOT_PATH, 'tests/logs/test_sampler.log')
TEST_CFG_FILE = os.path.join(ROOT_PATH, 'config/test.yaml')
IMAGE = os.path.join(ROOT_PATH, 'data/test/sampler/depth_0.npy')


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


class SamplerTestCase(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.config = load_config(TEST_CFG_FILE)
        cls.depth = np.load(IMAGE)
        cls.depth = np.squeeze(cls.depth)
        cls.roi = ((100, 200), (300, 400))

    @classmethod
    def tearDown(cls):
        pass

    # @unittest.skip('skip test_create')
    def test_create(self):
        sampler = ImageGraspSampler(self.depth, self.roi, self.config)
        self.assertIsNotNone(sampler)

    # @unittest.skip('skip test_process')
    def test_sample(self):
        sampler = ImageGraspSampler(self.depth, self.roi, self.config)
        grasps = sampler.sample(10)
        self.assertEqual(len(grasps), 10)


if __name__ == "__main__":
    config_logging(TEST_LOG_FILE)
    try:
        unittest.main()
    except:
        pass

import os
import logging
import unittest
from ruamel.yaml import YAML
from easygqcnn import NeuralNetWork, GQCNNTraing

file_path = os.path.split(__file__)[0]
ROOT_PATH = os.path.abspath(os.path.join(file_path, '..'))
TEST_LOG_FILE = os.path.join(ROOT_PATH, 'tests/logs/test_training.log')
TEST_CFG_FILE = os.path.join(ROOT_PATH, 'config/test.yaml')
DATA_PATH = os.path.join(ROOT_PATH, 'data/test/out_data')
OUT_PATH = os.path.join(ROOT_PATH, 'data/test/train_out')


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


class TrainingTestCase(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.config = load_config(TEST_CFG_FILE)

    @classmethod
    def tearDown(cls):
        pass

    # @unittest.skip('skip test_create')
    def test_create(self):
        network = NeuralNetWork(self.config, training=True)
        train = GQCNNTraing(self.config, network, DATA_PATH, OUT_PATH)
        train.optimize(5)
        self.assertIsNotNone(train)

    # @unittest.skip('skip test_process')
    # def test_process(self):
    #     processer = DataProcesser(self.config, RAW_PATH, OUT_PATH)
    #     processer.process()


if __name__ == "__main__":
    config_logging(TEST_LOG_FILE)
    try:
        unittest.main()
    except:
        pass

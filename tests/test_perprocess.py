import os
import logging
import unittest
from ruamel.yaml import YAML
from easygqcnn import DataProcesser

file_path = os.path.split(__file__)[0]
ROOT_PATH = os.path.abspath(os.path.join(file_path, '..'))
TEST_LOG_FILE = os.path.join(ROOT_PATH, 'tests/logs/test_preprocess.log')
TEST_CFG_FILE = os.path.join(ROOT_PATH, 'config/test.yaml')
RAW_PATH = os.path.join(ROOT_PATH, 'data/test/raw_data')
OUT_PATH = os.path.join(ROOT_PATH, 'data/test/out_data')


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


class ProcesserTestCase(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.config = load_config(TEST_CFG_FILE)

    @classmethod
    def tearDown(cls):
        pass

    @unittest.skip('skip test_create')
    def test_create(self):
        processer = DataProcesser(self.config, RAW_PATH, OUT_PATH)
        self.assertIsNotNone(processer)
    
    def test_process(self):
        processer = DataProcesser(self.config, RAW_PATH, OUT_PATH)
        processer.process()


if __name__ == "__main__":
    config_logging(TEST_LOG_FILE)
    # try:
    #     unittest.main()
    # except:
    #     pass
    raw = r'H:\Robot\Dex-Net\DataSet\mini_dexnet_all_trans_01_20_17'
    out = r'H:\Robot\template\out'
    config = load_config(TEST_CFG_FILE)
    processer = DataProcesser(config, raw, out)
    processer.process(is_dex=True)

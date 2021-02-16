import os
import sys
import logging
import tensorflow as tf
from shutil import rmtree, copyfile
from pathlib import Path
from ruamel.yaml import YAML

file_path = os.path.split(__file__)[0]
ROOT_PATH = os.path.abspath(os.path.join(file_path, '..'))
sys.path.append(os.path.join(ROOT_PATH, 'src'))
try:
    from easygqcnn import NeuralNetWork, GQCNNTraing
except Exception as e:
    raise e
TEST_LOG_FILE = os.path.join(ROOT_PATH, 'tools/logs/train_model.log')
TEST_CFG_FILE = os.path.join(ROOT_PATH, 'config/training.yaml')
GMDATA_PATH = Path.home().joinpath('Project/gmdata')
DATASET_PATH = GMDATA_PATH.joinpath('datasets/train_datasets')
INPUT_DATA_PATH = DATASET_PATH.joinpath('gq_data/small_data_train')
OUT_PATH = GMDATA_PATH.joinpath('datasets/models/gq/small_data')


def config_logging(file=None, level=logging.DEBUG):
    """ 配置全局的日志设置
    参考https://www.crifan.com/summary_python_logging_module_usage/
    """
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    # logging.basicConfig(filename=file, level=level,
    #                     format=LOG_FORMAT, filemode='w')
    logger = logging.getLogger('')
    logger.setLevel(level)
    rf_handler = logging.StreamHandler()  # 默认是sys.stderr
    # rf_handler.setLevel(logging.DEBUG)
    rf_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    f_handler = logging.FileHandler(file, mode='a')
    # f_handler.setLevel(logging.INFO)
    f_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    logger.addHandler(rf_handler)
    logger.addHandler(f_handler)


def load_config(file):
    """ 加载配置文件 """
    yaml = YAML(typ='safe')   # default, if not specfied, is 'rt' (round-trip)
    with open(file, 'r', encoding="utf-8") as f:
        config = yaml.load(f)
    return config


def main():
    config_logging(TEST_LOG_FILE)
    config = load_config(TEST_CFG_FILE)
    for t in 'gmd cor jaq'.split():
        network = NeuralNetWork(config, training=True)
        out_path = OUT_PATH.joinpath(t)
        input_path = INPUT_DATA_PATH.joinpath(t)
        if out_path.exists():
            rmtree(out_path)
        out_path.joinpath('summary').mkdir(parents=True)
        train = GQCNNTraing(config, network, input_path, out_path)
        train.optimize(50)
        for f in input_path.iterdir():
            if f.is_file():
                copyfile(f, out_path.joinpath(f.name))
    # with tf.Session(graph=train._network.graph) as sess:
    # init = tf.global_variables_initializer()
    # sess.run(init)
    # network.load_weights(sess, r'H:\Robot\GQ (from RSS 2017 paper)\model.ckpt', True)
    # result = train.validation(sess, None, None)
    # print(result)


if __name__ == "__main__":
    main()

import os
import shutil
import logging
from ruamel.yaml import YAML
import tensorflow as tf
from easygqcnn import NeuralNetWork, GQCNNTraing
from tensorflow import ConfigProto
from tensorflow import InteractiveSession

file_path = os.path.split(__file__)[0]
ROOT_PATH = os.path.abspath(os.path.join(file_path, '..'))
TEST_LOG_FILE = os.path.join(ROOT_PATH, 'tools/logs/train_model.log')
TEST_CFG_FILE = os.path.join(ROOT_PATH, 'config/training.yaml')
# DATA_PATH = r'/root/Project/gmdata/gq-data/ml-all-npz'
# OUT_PATH = r'/root/Project/gmdata/gq-data/models/ml-all-npz'
DATA_PATH = r'/root/Project/gmdata/gq-data/Cornell_Dataset'
OUT_PATH = r'/root/Project/gmdata/gq-data/models/Cornell_Dataset'
# DATA_PATH = r'/root/Project/gmdata/gq-data/ml-all-npz'
# OUT_PATH = r'/root/Project/gmdata/gq-data/models/ml-all-npz'

if os.path.exists(OUT_PATH):
    shutil.rmtree(OUT_PATH)

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)
    shutil.copyfile(os.path.join(DATA_PATH, 'mean.npy'), os.path.join(OUT_PATH, 'mean.npy'))
    shutil.copyfile(os.path.join(DATA_PATH, 'std.npy'), os.path.join(OUT_PATH, 'std.npy'))
    shutil.copyfile(os.path.join(DATA_PATH, 'pose_mean.npy'), os.path.join(OUT_PATH, 'pose_mean.npy'))
    shutil.copyfile(os.path.join(DATA_PATH, 'pose_std.npy'), os.path.join(OUT_PATH, 'pose_std.npy'))


if not os.path.exists(os.path.join(OUT_PATH, 'summary')):
    os.makedirs(os.path.join(OUT_PATH, 'summary'))

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

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
    network = NeuralNetWork(config, training=True)
    train = GQCNNTraing(config, network, DATA_PATH, OUT_PATH)
    train.optimize(30)
    # with tf.Session(graph=train._network.graph) as sess:
    # init = tf.global_variables_initializer()
    # sess.run(init)
    # network.load_weights(sess, r'H:\Robot\GQ (from RSS 2017 paper)\model.ckpt', True)
    # result = train.validation(sess, None, None)
    # print(result)


if __name__ == "__main__":
    main()

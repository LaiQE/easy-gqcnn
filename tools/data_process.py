import os
import sys
import shutil
import logging
from ruamel.yaml import YAML
# from easygqcnn import DataProcesser

file_path = os.path.split(__file__)[0]
ROOT_PATH = os.path.abspath(os.path.join(file_path, '..'))
sys.path.append(os.path.join(ROOT_PATH, 'src'))
try:
    from easygqcnn import DataProcesser
except Exception as e:
    raise e
LOG_FILE = os.path.join(ROOT_PATH, 'tools/logs/data_process.log')
CFG_FILE = os.path.join(ROOT_PATH, 'config/data_process.yaml')
# RAW_PATH = r'H:\Robot\Dex-Net\DataSet\mini_dexnet_all_trans_01_20_17'
# OUT_PATH = r'H:\Robot\template\out'
RAW_PATH = r'/root/Project/gmdata/gq-data/mix-dir-20x100'
OUT_PATH = r'/root/Project/gmdata/gq-data/mix-dir-20x100-recorder'
if os.path.exists(OUT_PATH):
    shutil.rmtree(OUT_PATH)
os.makedirs(OUT_PATH)


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

    f_handler = logging.FileHandler(file, mode='w')
    # f_handler.setLevel(logging.DEBUG)
    f_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    logger.addHandler(rf_handler)
    logger.addHandler(f_handler)


def initLogging(logFilename):
    """Init for logging
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format='LINE %(lineno)-4d  %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M',
        filename=logFilename,
        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter(
        'LINE %(lineno)-4d : %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def load_config(file):
    """ 加载配置文件 """
    yaml = YAML(typ='safe')   # default, if not specfied, is 'rt' (round-trip)
    with open(file, 'r', encoding="utf-8") as f:
        config = yaml.load(f)
    return config


def main():
    config_logging(LOG_FILE)
    config = load_config(CFG_FILE)
    processer = DataProcesser(config, RAW_PATH, OUT_PATH)
    processer.process(is_dex=False)


if __name__ == "__main__":
    main()

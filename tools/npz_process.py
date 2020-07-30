import os
import shutil
from easygqcnn.preprocess import NpzProcesser
from ruamel.yaml import YAML

file_path = os.path.split(__file__)[0]
ROOT_PATH = os.path.abspath(os.path.join(file_path, '..'))
RAW_PATH = r'/home/qianen/Project/gmdata/tree/ml-all-npz'
OUT_PATH = r'/home/qianen/Project/gmdata/gq-data/ml-all-npz'
CFG_FILE = os.path.join(ROOT_PATH, 'config/data_process_npz.yaml')

if os.path.exists(OUT_PATH):
    shutil.rmtree(OUT_PATH)
os.makedirs(OUT_PATH)


def load_config(file):
    """ 加载配置文件 """
    yaml = YAML(typ='safe')   # default, if not specfied, is 'rt' (round-trip)
    with open(file, 'r', encoding="utf-8") as f:
        config = yaml.load(f)
    return config


def main():
    config = load_config(CFG_FILE)
    processer = NpzProcesser(config, RAW_PATH, OUT_PATH)
    processer.process()


if __name__ == "__main__":
    main()

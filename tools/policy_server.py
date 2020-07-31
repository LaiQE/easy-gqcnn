import sys
import os
import Pyro4
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
from ruamel.yaml import YAML
from easygqcnn import GraspingPolicy


file_path = os.path.split(__file__)[0]
ROOT_PATH = os.path.abspath(os.path.join(file_path, '..'))
TEST_LOG_FILE = os.path.join(ROOT_PATH, 'tools/logs/policy_visual.log')
TEST_CFG_FILE = os.path.join(ROOT_PATH, 'config/policy.yaml')
IMAGE = os.path.join(ROOT_PATH, 'data/test/depth.npy')


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


@Pyro4.expose
class Planer(object):
    def __init__(self):
        config = load_config(TEST_CFG_FILE)
        config_logging(TEST_LOG_FILE)
        self.policy = GraspingPolicy(config)
        print('ok')

    def get_grasp(self, image, width):
        im = image.copy()
        i = 5
        q = 0
        while q < 0.9 and i:
            g, q = self.policy.action(im, None, width=width)
            i -= 1
        p0, p1 = g.endpoints
        return True, [p0, p1, g.depth, g.depth, q]


if __name__ == "__main__":
    pp = Planer()
    Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')
    Pyro4.Daemon.serveSimple({pp: 'Planer'}, ns=False, host='', port=6665)

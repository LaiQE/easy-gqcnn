import os
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
IMAGE = os.path.join(ROOT_PATH, 'data/test/depth_0.npy')


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


def plot_grasp(g, offset=[0, 0]):
    """ 使用plt在图像上展示一个夹爪 """
    def plot_2p(p0, p1, mode='r', width=None):
        p0 -= offset
        p1 -= offset
        x = [p0[0], p1[0]]
        y = [p0[1], p1[1]]
        plt.plot(x, y, mode, linewidth=width)

    def plot_center(center, axis, length, mode='r', width=2):
        axis = axis / np.linalg.norm(axis)
        p0 = center - axis * length / 2
        p1 = center + axis * length / 2
        plot_2p(p0, p1, mode, width)

    p0, p1 = g.endpoints
    axis = [g.axis[1], -g.axis[0]]
    plot_2p(p0, p1, 'r--')
    plot_center(p0, axis, g.width_px/2.5, width=3)
    plot_center(p1, axis, g.width_px/2.5, width=3)
    plt.plot(*(g.center - offset), 'bo')


def main():
    config = load_config(TEST_CFG_FILE)
    depth = np.load(IMAGE)
    depth = np.squeeze(depth)
    depth = depth[100:300, 200:400]
    # roi = ((100, 200), (300, 400))
    random.seed(15)
    np.random.seed(15)
    policy = GraspingPolicy(config)
    g, q = policy.action(depth, None)
    print(q)

    plt.figure()
    plt.imshow(depth)
    plt.colorbar()
    plot_grasp(g)
    plt.show()


if __name__ == "__main__":
    main()

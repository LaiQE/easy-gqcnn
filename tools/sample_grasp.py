import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from ruamel.yaml import YAML
from easygqcnn import ImageGraspSampler

file_path = os.path.split(__file__)[0]
ROOT_PATH = os.path.abspath(os.path.join(file_path, '..'))
TEST_LOG_FILE = os.path.join(ROOT_PATH, 'tools/logs/sampler.log')
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


def plot_grasp(grasp_2d, offset):
    def plot_2p(p0, p1):
        x = [p0[0], p1[0]]
        y = [p0[1], p1[1]]
        plt.plot(x, y)
    p0, p1 = grasp_2d.endpoints
    p0 = p0 - offset
    p1 = p1 - offset
    plot_2p(p0, p1)


def main():
    config = load_config(TEST_CFG_FILE)
    depth = np.load(IMAGE)
    depth = np.squeeze(depth)
    depth = depth[100:300, 200:400]
    # roi = ((100, 200), (300, 400))
    sampler = ImageGraspSampler(depth, None, config)
    grasps = sampler.sample(100)
    depth = np.squeeze(depth)

    plt.figure()
    plt.imshow(depth)
    plt.colorbar()
    for g in grasps[:]:
        plot_grasp(g[0], [0, 0])
        plt.plot(g[0].center[0], g[0].center[1], 'ro')
        plt.plot(g[1][0], g[1][1], 'yo')
        plt.plot(g[2][0], g[2][1], 'yo')
        print(g[1], g[2])
    plt.show()


if __name__ == "__main__":
    main()

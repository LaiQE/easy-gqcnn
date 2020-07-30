FROM tensorflow/tensorflow:1.15.0-gpu-py3-jupyter

ARG pip_source=https://pypi.tuna.tsinghua.edu.cn/simple
WORKDIR /root

RUN set -ex\
    && curl https://mirrors.ustc.edu.cn/repogen/conf/ubuntu-http-4-xenial > /etc/apt/sources.list \
    && apt-get update\
    && apt-get install -y libsm6 libxext6 libxrender-dev
RUN set -ex\  
    && pip install -i ${pip_source} --upgrade pip\
    && pip install --no-cache-dir -i ${pip_source} opencv-python==4.3.0.36 ruamel.yaml==0.16.5 scikit-image==0.15.0 scikit-learn==0.23.1 pandas==1.0.5\
    && apt-get autoremove

# docker run --gpus all -it --name gqcnn -v /home/qianen/Project/:/root/Project -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime laiqe/gqcnn:1.0 bash

CMD bash

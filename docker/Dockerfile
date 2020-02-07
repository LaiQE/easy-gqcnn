FROM tensorflow/tensorflow:1.13.2-gpu-py3

ARG pip_source=https://pypi.tuna.tsinghua.edu.cn/simple

ADD sources.list /etc/apt
ADD requirements.txt /root
ADD Shanghai /usr/share/zoneinfo/Asia/Shanghai
# 设置时区
ENV TZ=Asia/Shanghai

WORKDIR /root

RUN set -ex\
    && apt-get update\
    && apt-get install -y libsm6 libxext6 libxrender-dev\
    && pip install -i ${pip_source} --upgrade pip\
    && pip install --no-cache-dir -i ${pip_source} -r requirements.txt\
    && rm ./requirements.txt\
    && apt-get autoremove

EXPOSE 8888
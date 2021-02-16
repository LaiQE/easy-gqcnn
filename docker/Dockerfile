FROM tensorflow/tensorflow:1.13.2-gpu-py3
# 安装tzdata用,参考https://www.cnblogs.com/yahengwang/p/11072208.html
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /root
# 设置时区
ENV TZ=Asia/Shanghai
ENV LANG=C.UTF-8

RUN set -ex \
    && sed -i 's#http://archive.ubuntu.com#http://mirrors.aliyun.com#' /etc/apt/sources.list \
    && sed -i 's#http://security.ubuntu.com#http://mirrors.aliyun.com#' /etc/apt/sources.list \
    # 不然连接不到nvidia的服务器
    && rm -rf /etc/apt/sources.list.d/* \
    && apt-get update \
    # 设置用户
    # 参考https://github.com/cdr/code-server/blob/v3.8.0/ci/release-image/Dockerfile
    # 或者参考,这个可以设置用户账户密码
    # https://github.com/SuLingGG/OpenWrt-Env/blob/master/Dockerfile
    && apt-get install -qqy sudo curl zsh vim tree tzdata git \
    # 这样不用设置密码, 需要的时候用sudo passwd admin, 也不要设置主目录
    && adduser --gecos '' --disabled-password admin \
    && echo "admin ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/nopasswd \
    && HOME="/home/admin" \
    && curl -fsSL git.io/oh-my-zsh.sh | bash \
    && chown -R admin:admin /home/admin \
    && sed -i 's#antigen theme ys#antigen theme amuse#' /home/admin/.zshrc \
    # 安装fixuid用于启动时使用用户的uid, 参考https://github.com/boxboat/fixuid
    && ARCH="$(dpkg --print-architecture)" \
    && curl -fsSL "https://github.com/boxboat/fixuid/releases/download/v0.4.1/fixuid-0.4.1-linux-$ARCH.tar.gz" | tar -C /usr/local/bin -xzf - \
    && chown root:root /usr/local/bin/fixuid \
    && chmod 4755 /usr/local/bin/fixuid \
    && mkdir -p /etc/fixuid \
    && printf "user: admin\ngroup: admin\n" > /etc/fixuid/config.yml \
    && apt autoremove -y

ARG pip_source=https://pypi.douban.com/simple

ADD requirements.txt /root

RUN set -ex\
    && apt-get update -y \
    && apt-get install -y libsm6 libxext6 libxrender-dev freeglut3-dev\
    && pip install -i ${pip_source} --upgrade pip\
    && pip install --no-cache-dir -i ${pip_source} -r requirements.txt\
    && rm ./requirements.txt\
    && apt-get autoremove

EXPOSE 8888
WORKDIR /home/admin
USER admin:admin
RUN set -ex \
    # && /usr/bin/miniconda3/bin/conda init zsh \
    # 使sudo的时候环境变量不变,可以继续用python
    && echo "alias sudo='sudo env PATH=\$PATH'" >> .zshrc

ENTRYPOINT ["fixuid", "-q"]
CMD ["/bin/zsh"]
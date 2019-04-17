# easy-gqcnn
一种更为简单的GQCNN实现方式
基于dex-net2.0的论文，较为简单的适用于平行夹爪的GQCNN的实现  
对于伯克利的gqcnn实现，这里主要进行了如下改动
1. 将数据预处理为tfrecode进行模型训练，提高了训练速度
2. 可以同时兼容伯克利dex-net和easy-dexnet生成的数据
3. 所有程序全部基于python3.5  
除了以上的主要改动，由于这里是完全重写的程序，所有大部分的实现细节也都有改动

### 安装部署
easydexnet是基于python3.5编写，需要首先安装python3.5，另外Tensorflow需要另外安装  
> git clone https://github.com/LaiQE/easy-gqcnn.git  
cd easy-gqcnn  
python setup.py develop

### 使用
- 数据预处理, 将由dex-net或easy-dexnet生成的数据预处理为tfrecord  
参考tools/data_process.py  
需要改动config/data_process.yaml配置文件  
- 训练cnn网络模型  
参考tools/train_model.py  
需要改动config/training.yaml配置文件  
- 使用训练好的模型进行抓取位姿选择  
参考tools/policy_visual.py  
需要改动config/policy.yaml配置文件  
- 其他用法  
easy-gqcnn预计会在0.2版本增加ROS服务和相关的ROS接口

### 参考
http://berkeleyautomation.github.io/gqcnn/

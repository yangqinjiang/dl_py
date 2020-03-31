# coding: utf-8
import sys,os
sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 使用BatchNorm
network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100], output_size=10,
                              use_batchnorm=True)

x_batch = x_train[:1]
t_batch = t_train[:1]
grad_backprop = network.gradient(x_batch,t_batch)
grad_numerical = network.numerical_gradient(x_batch,t_batch)

#加入BatchNormal后, 比较梯度是否准确
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))
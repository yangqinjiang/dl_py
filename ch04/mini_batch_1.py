import sys,os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
# load_mnist函数是用于读入MNIST数据集的函数。
(x_train,t_train),(x_test,t_test) = load_mnist(normalize=True,one_hot_label=True)

print(x_train.shape)
print(t_train.shape)

#如何从这个训练数据中随机抽取10笔数据呢
train_size = x_train.shape[0]
batch_size=10
# 得到一个随机索引数组
batch_mask = np.random.choice(train_size,batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
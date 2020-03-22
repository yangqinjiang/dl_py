# coding: utf-8
import sys,os
sys.path.append(os.pardir)# 为了导入父目录的文件而进行的设定

import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax

print('一个一个的预测数据')
# 一个一个的预测数据

# 数据正规化,预处理
def get_data():
    (x_train, t_train),(x_test,t_test) = load_mnist(normalize=True,flatten=True,one_hot_label=False)
    return x_test,t_test


def init_network():
    # 加载预设的权重,偏置
    with open('sample_weight.pkl','rb') as f:
        network = pickle.load(f)
    return network    

# 预测 [测试集]
def predict(network,x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    # 1
    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)#这里的输入信息是 z1,即第一层隐藏层的输出
    # 2
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    # 3
    a3 = np.dot(z2,W3) + b3
    y =  softmax(a3) # 输出层

    return y

x,t = get_data()
print(x.shape)
print(t.shape)
network =  init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y=predict(network,x[i])
    # 获取概率最高的元素的索引
    p = np.argmax(y)
    # 如果预测正确,则得分+1
    if p == t[i]:
        accuracy_cnt += 1

# 得分
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))    
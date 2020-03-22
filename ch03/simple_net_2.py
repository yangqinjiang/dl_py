# coding: utf-8

import numpy as np
from sigmoid import sigmoid

def init_network():
    network ={}
    # 输入层与隐藏层一的权重,偏置
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    # 隐藏层一与隐藏层二的权重,偏置
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5] ,[0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    # 隐藏层二与输出层的权重,偏置
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])
    return network

# 前向传递
def forward(network,x):
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
    y =  a3 # 输出层,恒等值

    return y

if __name__ == '__main__':
    network = init_network()
    #输入信息
    x = np.array([1.0,0.5])
    y = forward(network ,x)
    print(y) #output: [ 0.31682708  0.69627909]
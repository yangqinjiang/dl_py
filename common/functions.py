# coding: utf-8
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid的反向传播梯度
def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def ReLU(x):
    return np.maximum(0,x)

def tanh(x):
    return np.tanh(x)

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))

# 均方误差
def mean_squared_error(y,t):
    return 0.5 * np.sum((y-t)**2)    


# 交叉熵误差(普通版)
def cross_entropy_error_0(y,t):
    detla = 1e-7  #避免出现 np.log(0)运算,变成负无限大的-inf
    return -np.sum(t * np.log(y + detla))


# mini-batch的交叉熵误差
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
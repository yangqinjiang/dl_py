# coding: utf-8

import numpy as np
from sigmoid import sigmoid

# 输入信息
X = np.array([1.0,0.5])

#第一层隐藏层的权重和参数
W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1 = np.array([0.1,0.2,0.3])
print( W1.shape)
print( X.shape)
print( B1.shape)

A1 = np.dot(X,W1) + B1 # 内积, 偏置
print(A1)
Z1 = sigmoid(A1)  #激活函数
print(Z1)


# 实现第1层到第2层的信号传递

#第二层隐藏层的权重和参数
W2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2 = np.array([0.1,0.2])

print(Z1.shape)
print(W2.shape)
print(B2.shape)
#这里的输入信息是 Z1,即第一层隐藏层的输出
A2 = np.dot(Z1,W2) + B2 #内积
Z2 = sigmoid(A2)  #激活函数
print(A2)
print(Z2)

# 恒等函数
def identity_function(x):
    return x

# 实现第2层隐藏层到输出层的信号传递

W3 = np.array([[0.1,0.3],[0.2,0.4]])
B3 = np.array([0.1,0.2])

A3 = np.dot(Z2 ,W3) + B3
Y  = identity_function(A3) # 或者 Y = A3
print(A3)
print(Y)
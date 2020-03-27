# coding: utf-8
import sys,os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:

    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        #初始化权重,随机值
        # 注意各层的矩阵形状不一致!!!
        self.params = {}
        # 第一层(输入层与隐藏层)的权重,偏置
        self.params['W1'] = weight_init_std * np.random.randn(input_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size) # b 初始化为 0

        # 第二层(隐藏层与输出层)的权重,偏置
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size) # b 初始化为 0

    def predict(self,x):
        W1,W2 = self.params['W1'],self.params['W2']
        b1,b2 = self.params['b1'],self.params['b2']

        #第一层(输入层与隐藏层)
        a1 = np.dot(x,W1) + b1
        z1 = sigmoid(a1)  #加权输入到激活函数

        # 第二层(隐藏层与输出层)
        a2 = np.dot(z1,W2) + b2
        y = softmax(a2) #加权输入到激活函数

        return y

    # x:输入数据, t: 监督数据
    def loss(self,x,t):
        y = self.predict(x)
        return cross_entropy_error(y,t)

    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        t = np.argmax(t,axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # 数值微分
    # x:输入数据, t: 监督数据
    def numerical_gradient(self,x,t):
        loss_W = lambda W: self.loss(x,t)
        grads = {}
        # 求偏导
        grads['W1'] = numerical_gradient(loss_W ,self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W ,self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W ,self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W ,self.params['b2'])

        return grads
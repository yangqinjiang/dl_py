# coding: utf-8
import sys,os
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from common.optimizer import *  # 导入四个常用 的梯度下降优化器


# 函数
def f(x,y):
    return x**2 / 20.0 + y**2

# 导数
def df(x,y):
    return x/10.0,2.0*y


init_pos = (-7.0,2.0)  # 梯度下降开始的点
params = {}
params['x'],params['y'] = init_pos[0] , init_pos[1]
grads = {}
grads['x'],grads['y'] = 0,0

# 优化器
optimizers = OrderedDict()
optimizers["SGD"] = SGD(lr=0.95)
optimizers['Momentum'] = Momentum(lr=0.1)
optimizers['AdaGrad'] = AdaGrad(lr=1.5)
optimizers['Adam'] = Adam(lr=0.3)

idx = 1

# 迭代所有优化器
for key in optimizers:
    optimizer = optimizers[key]
    x_history = []
    y_history = []
    params['x'] , params['y'] = init_pos[0],init_pos[1]

    # 每个优化器,迭代30次
    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])

        grads['x'],grads['y'] = df(params['x'],params['y'])
        optimizer.update(params,grads) # 调用优化器的update函数,更新params的值


    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)
    
    X, Y = np.meshgrid(x, y) 
    Z = f(X, Y)
    
    # for simple contour line  
    mask = Z > 7
    Z[mask] = 0
    
    # plot 
    plt.subplot(2, 2, idx)  # 画图
    idx += 1  # 索引+1
    plt.plot(x_history, y_history, 'o-', color="red")
    plt.contour(X, Y, Z)
    plt.ylim(-10, 10)
    plt.xlim(-10, 10)
    plt.plot(0, 0, '+')
    #colorbar()
    #spring()
    plt.title(key)  # 使用当前优化器的字典key
    plt.xlabel("x")
    plt.ylabel("y")
    
plt.show()
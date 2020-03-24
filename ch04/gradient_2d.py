# coding: utf-8

import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

# 求单变量的数值微分
def _numerical_gradient_no_batch(f,x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # 生成和x形状相同的数组
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val # 还原值
    
    return grad

# 数值微分
def numerical_gradient(f,X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f,X)
    else:
        grad = np.zeros_like(X)

        for idx,x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f,x)

        return grad


def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2,axis=1)

if __name__ == '__main__':
    #
    print(_numerical_gradient_no_batch(function_2,np.array([3.0,4.0])))
    print(_numerical_gradient_no_batch(function_2,np.array([0.0,2.0])))
    print(_numerical_gradient_no_batch(function_2,np.array([3.0,0.0])))


    x0 = np.arange(-2,2.5,0.25)
    x1= np.arange(-2,2.5,0.25)
    X ,Y = np.meshgrid(x0,x1)
    
    X = X.flatten()
    Y = Y.flatten()

    grad = numerical_gradient(function_2,np.array([X,Y]))
    #print(grad)

    plt.figure()
    plt.quiver(X,Y,-grad[0],-grad[1])
    plt.xlim([-2,2])
    plt.ylim([-2,2])
    plt.xlabel('x0')
    plt.xlabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()
    pass
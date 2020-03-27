# coding: utf-8
import numpy as np

# 适用于1d的数值微分
def _numerical_gradient_1d(f,x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] =  float(tmp_val) + h
        fxh1 = f(x) # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2 )/(2*h)

        x[idx] = tmp_val # 还原值
    
    return grad


# 适用于1d,2d的数值微分
def numerical_gradient_2d(f,X):
    if X.ndim == 1:
        return _numerical_gradient_1d(f,X)
    else:
        grad = np.zeros_like(X)
        for idx ,x in enumerate(X):
            grad[idx] = -_numerical_gradient_1d(f,x)

        return grad


# 适用于任意维度的数值微分
# 为了对应形状为多维数组的权重参数W，这里使用的 numerical_
# gradient()和之前的实现稍有不同。不过，改动只是为了对应多维
# 数组，所以改动并不大
def numerical_gradient(f,x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    # 迭代器
    it = np.nditer(x,flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val =  x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x) # f(x-h)
        grad[idx] = ( fxh1 - fxh2 ) / (2 * h)

        x[idx] = tmp_val # 还原值
        it.iternext()

    return grad
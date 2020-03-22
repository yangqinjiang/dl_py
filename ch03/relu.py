# coding: utf-8
import numpy as np
import matplotlib.pylab as plt

# relu函数
def relu(x):
    # 允许x参数取NumPy数组
    return np.maximum(x,0)  #公式

if __name__ == '__main__':
    #允许参数取NumPy数组
    X = np.arange(-5.0,5.0,0.1)
    Y = relu(X)
    plt.plot(X,Y)
    # 指定图中绘制的y轴的范围
    plt.ylim(-1.0,5.5)
    plt.show()    
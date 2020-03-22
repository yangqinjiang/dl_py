# coding: utf-8
import numpy as np
import matplotlib.pylab as plt

# sigmoid函数
def sigmoid(x):
    # 允许x参数取NumPy数组
    return 1 / (1 + np.exp(-x))  #公式

if __name__ == '__main__':
    #允许参数取NumPy数组
    X = np.arange(-5.0,5.0,0.1)
    Y = sigmoid(X)
    plt.plot(X,Y)
    # 指定图中绘制的y轴的范围
    plt.ylim(-0.1,1.1)
    plt.show()    
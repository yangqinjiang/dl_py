# coding: utf-8
import numpy as np
import matplotlib.pylab as plt

# 阶跃函数
def step_function(x):
    # 允许x参数取NumPy数组
    # 阈值是0
    return np.array(x>0,dtype=np.int)

if __name__ == '__main__':
    #允许参数取NumPy数组
    X = np.arange(-5.0,1.0,0.1)
    Y = step_function(X)
    plt.plot(X,Y)
    # 指定图中绘制的y轴的范围
    plt.ylim(-0.1,1.1)
    plt.show()    
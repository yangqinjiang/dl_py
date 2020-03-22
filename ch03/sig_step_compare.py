# coding: utf-8
import numpy as np
import matplotlib.pylab as plt

from step_function import step_function
from sigmoid import sigmoid


# 两种激活函数的比较
x = np.arange(-5.0,5.0,0.1)
y1 = sigmoid(x)
y2 = step_function(x)

plt.plot(x,y1)
plt.plot(x,y2,'k--')
plt.ylim(-0.1,1.1)
plt.show()
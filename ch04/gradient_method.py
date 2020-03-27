# coding: utf-8
import numpy as np
import matplotlib.pylab as plt
from gradient_2d import numerical_gradient


def gradient_descent(f,init_x,lr=0.01,step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())
        grad = numerical_gradient(f,x)
        x -= lr*grad

    return x,np.array(x_history)


def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0,4.0])

lr = 0.1
step_num = 20
x,x_history = gradient_descent(function_2,init_x,lr=lr,step_num=step_num)

plt.plot([-5,5],[0,0],'--b')
plt.plot([0,0],[-5,5],'--b')
plt.plot(x_history[:,0],x_history[:,1],'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()


#学习率过大或者过小都无法得到好的结果

# 学习率过大的例子：lr=10.0
init_x = np.array([-3.0,4.0])
x,x_history = gradient_descent(function_2,init_x,lr=10.0,step_num=100)
print(x)
print(x_history)


# 学习率过小的例子：lr=1e-10
init_x = np.array([-3.0,4.0])
x,x_history = gradient_descent(function_2,init_x,lr=1e-10,step_num=100)
print(x)
print(x_history)


# 实验结果表明，学习率过大的话，会发散成一个很大的值；反过来，学
# 习率过小的话，基本上没怎么更新就结束了。也就是说，设定合适的学习率
# 是一个很重要的问题。

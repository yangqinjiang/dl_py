# coding: utf-8

# 与门
def AND(x1,x2):
    # 初始化参数
    w1,w2,theta = 0.5,0.5,0.7
    tmp = x1*w1 + x2 * w2
    #当输入的加权总和,超过阈值时,返回1,否则返回0
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


import numpy as np
#
def AND2(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5]) # 权重
    b = -0.7 #偏置
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

if __name__ == '__main__':
    print AND(0,0)# 输出0
    print AND(1,0)# 输出0
    print AND(0,1)# 输出0
    print AND(1,1)# 输出1


    print AND2(0,0)# 输出0
    print AND2(1,0)# 输出0
    print AND2(0,1)# 输出0
    print AND2(1,1)# 输出1
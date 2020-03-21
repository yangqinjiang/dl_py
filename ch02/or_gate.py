# coding: utf-8

# 与非门

import numpy as np
#
def OR(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5]) # 权重
    b = -0.2 #偏置
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

if __name__ == '__main__':
    
    # 测试数据的列表
    for xs in [(0,0),(1,0),(0,1),(1,1)]:
        y = OR(xs[0],xs[1])
        print(str(xs) + " -> " + str(y))
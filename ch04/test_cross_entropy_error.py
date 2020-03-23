# coding: utf-8
import sys,os
sys.path.append(os.pardir)# 为了导入父目录的文件而进行的设定

import numpy as np
from common.functions import cross_entropy_error_0,cross_entropy_error
#  交叉熵误差
# 设 '2'为正确解
t = [0,0,1,0,0,0,0,0,0,0]
# '2' 的概率最高的情况为 0.6
y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
# 输出0.510825... ,损失函数的值更小
print(cross_entropy_error_0(np.array(y),np.array(t)))
print(cross_entropy_error(np.array(y),np.array(t)))


# 例2：“7”的概率最高的情况（0.6）
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
# 输出 2.3025 ,损失函数的值比较大
print(cross_entropy_error_0(np.array(y),np.array(t)))
print(cross_entropy_error(np.array(y),np.array(t)))
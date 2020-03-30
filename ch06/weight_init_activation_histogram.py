# coding: utf-8
import sys,os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from common.functions import sigmoid,ReLU,tanh

# 总结一下，当激活函数使用ReLU时，权重初始值使用He初始值，当
# 激活函数为sigmoid或tanh等S型曲线函数时，初始值使用Xavier初始值。
# 这是目前的最佳实践

# 1000个数据
input_data = np.random.randn(1000,100)
node_num = 100 #各隐藏层的节点(神经元)数
hidden_layer_size = 5 # 隐藏层有5层
activations = {} # 激活值的结果保存在这里

x = input_data
for i in range(hidden_layer_size):
    if i!=0:
        x = activations[i-1]

    #改变初始值进行实验!
    #w = np.random.randn(node_num,node_num) * 1
    # w = np.random.randn(node_num,node_num) * 0.01
    # w = np.random.randn(node_num,node_num) * np.sqrt(1.0 / node_num)
    w = np.random.randn(node_num,node_num) * np.sqrt(2.0 / node_num)

    a = np.dot(x,w)
    # 将激活函数的种类也改变,进行实验!
    #z = sigmoid(a)
    z = ReLU(a)
    # z = tanh(a) #如果用tanh
                #函数（双曲线函数）代替sigmoid函数，这个稍微歪斜的问题就能得
                #到改善。实际上，使用tanh函数后，会呈漂亮的吊钟型分布

    activations[i] = z


#绘制直方图
for i,a in activations.items():
    plt.subplot(1,len(activations),i+1)
    plt.title(str(i+1)+"-layer")
    if i!=0:plt.yticks([],[])
    #plt.xlim(0.1,1)
    #plt.ylim(0,7000)
    plt.hist(a.flatten(),30,range=(0,1))

plt.show()
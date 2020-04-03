# coding: utf-8
import sys,os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from deep_convnet import DeepConvNet
from dataset.mnist import load_mnist
# 这里我们实际看一下在什么样
# 的图像上发生了识别错误
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = DeepConvNet()
network.load_params('deep_convnet_params.pkl')
print("calcualting test accuracy...")

sampled = 1000
x_test = x_test[:sampled]
t_test = t_test[:sampled]

classified_ids = []

acc = 0.0
batch_size = 100
# 计算测试集的得分
for i in range(int(x_test.shape[0] / batch_size)):
    tx = x_test[i*batch_size:(i+1)*batch_size]
    tt = t_test[i*batch_size:(i+1)*batch_size]
    y = network.predict(tx,train_flg=False)
    y = np.argmax(y , axis=1)
    classified_ids.append(y)
    acc += np.sum(y == tt)

acc = acc / x_test.shape[0]
print("test accuracy: " + str(acc))

classified_ids = np.array(classified_ids)
classified_ids = classified_ids.flatten() # 压平

max_view = 20
current_view = 1

fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.2, wspace=0.2)

# 错误的识别示例
mis_pairs = {}
for i,val in enumerate(classified_ids == t_test):
    if not val: #检测不正确的
        ax = fig.add_subplot(4, 5, current_view, xticks=[], yticks=[])
        ax.imshow(x_test[i].reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
        mis_pairs[current_view] = (t_test[i], classified_ids[i])    
            
        current_view += 1
        if current_view > max_view:
            break

print("======= misclassified result =======")
print("{view index: (label, inference), ...}")
print(mis_pairs)

plt.show()
"""
察图8-2可知，这些图像对于我们人类而言也很难判断。实际上，这里
面有几个图像很难判断是哪个数字，即使是我们人类，也同样会犯“识别错误”。
比如，左上角的图像（正确解是“6”）看上去像“0”，它旁边的图像（正确解是
“3”）看上去像“5”。整体上，“1”和“7”、“0”和“6”、“3”和“5”的组合比较
容易混淆。通过这些例子，相信大家可以理解为何会发生识别错误了吧。
这次的深度CNN尽管识别精度很高，但是对于某些图像，也犯了和人
类同样的“识别错误”。从这一点上，我们也可以感受到深度CNN中蕴藏着
巨大的可能性。
"""
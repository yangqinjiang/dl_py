# coding: utf-8
import sys,os
sys.path.append(os.pardir)# 为了导入父目录的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

#加载数据
(x_train,t_train),(x_test,t_test) = load_mnist(flatten=True,normalize=False)
img = x_train[0]
label = t_train[0]
print(label)  # 5
print(img.shape) # (784 ,)
img = img.reshape(28,28)# 把图像的形状变成原来的尺寸
print(img.shape) #(28,28)
print('show image')

img_show(img)

# 调用img_show函数出错: 
#   FSPathMakeRef(/Applications/Preview.app) failed with error
# https://blog.csdn.net/qq_35793285/article/details/103028425
# 解决方式:
# ln -s /System/Applications/Preview.app /Applications/Preview.app



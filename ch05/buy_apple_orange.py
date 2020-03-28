# coding: utf-8
from layer_naive import *
# 购买2个苹果,3个orange,和消费税
# 计算图中层的实现（这里是加法层和乘法层）非常简单，使用这
# 些层可以进行复杂的导数计算
apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

#layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer() # 加法层
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple,apple_num) #(1)
orange_price = mul_orange_layer.forward(orange,orange_num) #(2)
all_price = add_apple_orange_layer.forward(apple_price,orange_price)
price = mul_tax_layer.forward(all_price,tax)#(4)

# backward
dprice = 1
dall_price ,dtax = mul_tax_layer.backward(dprice) #(4)
dapple_price,dorange_price = add_apple_orange_layer.backward(dall_price) #(3)
dorange,dorange_num = mul_orange_layer.backward(dorange_price) #(2)
dapple,dapple_num = mul_apple_layer.backward(dapple_price) #(1)

print("price:", int(price))
print("dApple:", dapple)
print("dApple_num:", int(dapple_num))
print("dOrange:", dorange)
print("dOrange_num:", int(dorange_num))
print("dTax:", dtax)
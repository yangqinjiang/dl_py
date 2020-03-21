# coding: utf-8

# 异或门
#复用与门, 或门, 与非门的代码
from and_gate import AND
from or_gate import OR
from nand_gate import NAND

def XOR(x1,x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y = AND(s1,s2)
    return y

if __name__ == '__main__':
    
    # 测试数据的列表
    for xs in [(0,0),(1,0),(0,1),(1,1)]:
        y = XOR(xs[0],xs[1])
        print(str(xs) + " -> " + str(y))


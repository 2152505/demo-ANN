import numpy as np


class NeuralNetwork:
     # 通过初始化的方式实现 输入‘x’及输出‘y’、以及网络权重（定义了二层4节点的网络结构） 的基本结构：
     def __init__(self, x, y):
         self.input = x
         self.weights1 = np.random.rand(self.input.shape[1],4) 
         # 这样的结构，意味着weights1的行数是输入变量x的列数，那么相乘时可能是与常规的 W·X 的方式相反，即 X·W
         # 因此，X的每一行，代表着一次x输入，对应着每一个输出y
         
         # 同上的，这样的结构形式定义，代表的表达式或应是：[(X·W1+b1)·W2]+b2
         ## 而在经过上述W1与W2的相乘后，每一行的x对应着一个y_i的值
         self.weights2 = np.random.rand(4,1)
         
         # 对应着y作为输出，定义self.output 的结构
         self.y = y
         self.output = np.zeros(y.shape)
 
     
     def feedforward(self): #定义前传方法：
         # 注意忽略了每层的偏差 b_i
         # 且转换函数使用 sigmoid()
         # 因没有相应的函数定义，将其中的 sigmoid() 函数进行展开
         # sigmoid(m)=1/(1 + np.exp(-m))        
        npDotW1_Inpt = np.dot(self.input, self.weights1)
        self.layer1 = 1/(1+np.exp(-1*npDotW1_Inpt))
         
        npDotL1_W2 = np.dot(self.layer1, self.weights2)
        self.output = 1/(1+np.exp(-1*npDotL1_W2))
         
     def backprop(self): # 定义后传方法：
         # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
         # 以上是原文注解，即在向后传播 backprop 时，使用链式法则获得相应权重值的损失函数
         # 因没有相应的函数定义，将原贴中的 sigmoid_derivative() 函数进行展开
         # sigmoid_derivative(m)=m * (1 - m)
 
         ## 此处采用的是 s_deriv(x) = x*(1-x) 的方式，如采用 exp(-x)/(1+exp(-x))^2) 的解析式也可以，本质相同：
         d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * (self.output*(1-self.output))))
         d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * (self.output*(1-self.output)), self.weights2.T) * (self.layer1*(1-self.layer1))))
     
         # update the weights with the derivative (slope) of the loss function
         # 通过上面的求导斜率，对W1和W2进行更新：
         self.weights1 += d_weights1
         self.weights2 += d_weights2
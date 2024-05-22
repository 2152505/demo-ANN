import ANN
import matplotlib.pyplot as plt
import numpy as np
# 建立输入和输出变量：
x_input = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y_obj = np.array([[0,1,1,0]]).T  
# 实例化 ANN：
nn = ANN.NeuralNetwork(x_input, y_obj)
  
# 进行计算，迭代次数为1500次：
m = 1500
loss = np.zeros(m)
 
for i in range(m):
    nn.feedforward() # 前传计算结果
    nn.backprop() # 后传更新权重 
    loss[i] = np.sum((nn.output-y_obj)**2) # 记录每次的结果偏差

# 绘制结果图形：
plt.plot(loss)
plt.xlabel('Iteration')
plt.ylabel('LossValue')
plt.grid(True)
plt.show()
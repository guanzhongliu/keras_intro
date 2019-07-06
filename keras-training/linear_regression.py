import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

# 线性回归

# 使用numpy生成100个随机点
x_data = np.random.rand(200)
noise = np.random.normal(0, 0.01, x_data.shape)
y_data = x_data * 0.1 + 0.2 + noise

# 显示随机点
plt.scatter(x_data, y_data)
plt.show()

# 构建一个顺序模型
model = Sequential()
# 在模型中添加一个全连接层，units输出维度，input_dim输入维度
model.add(Dense(units=1, input_dim=1))
# optimizer优化器
# sgd：Stochastic gradient descent随机梯度下降法
# mse：Mean Squared Error均方误差
model.compile(optimizer='sgd', loss='mse')

# 训练模型3001个批次
for step in range(5000):
    # 每次训练一个批次
    cost = model.train_on_batch(x_data, y_data)
    if step % 500 == 0:
        print('cost:', cost)
# 打印权值和偏置值
W, b = model.layers[0].get_weights()
print("W:", W, "b:", b)

# x_data输入网络中，得到预测值y_data
y_pred = model.predict(x_data)

# 显示随机点
plt.scatter(x_data, y_data)
# 显示预测结果
plt.plot(x_data, y_pred, 'r-', lw=2)
plt.show()

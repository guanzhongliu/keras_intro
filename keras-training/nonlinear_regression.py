import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.layers import Activation

x_data = np.linspace(-0.5, 0.5, 1000)
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

plt.scatter(x_data, y_data)
plt.show()

model = Sequential()
# 在模型中添加一个全连接层
# 1-10-1
# 隐藏层
model.add(Dense(units=10, input_dim=1))
# 假如未指定激活函数，则默认为线性
model.add(Activation('tanh'))
# 加入输出层，此时输入可省略
model.add(Dense(units=1))
model.add(Activation('tanh'))

# sgd默认学习率很小，此处应调高

# 定义优化算法
sgd = SGD(lr=0.3)
model.compile(optimizer=sgd, loss='mse')

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
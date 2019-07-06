# dropout应用

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD

# 载入数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# (60000, 28, 28)
print("x_shape: ", x_train.shape)
# (60000)
print("y_shape: ", y_train.shape)

# (60000, 28, 28) -> (60000, 784)
# -1会帮助你自动化生成， 255.0归一化
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

# 换one hot格式
# np_utils.to_categorical转标签格式
# 手写识别10个类，num_classes = 10
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 创建模型，输入784个神经元，输出10个神经元
# tanh双曲正切
model = Sequential([
    # 加入多个隐藏层
    Dense(units=200, input_dim=784, bias_initializer='one', activation='tanh'),
    Dense(units=100, bias_initializer='one', activation='tanh'),
    Dense(units=10, bias_initializer='one', activation='softmax'),
])
# 定义优化器, loss fuction,同时在计算时得到准确率
sgd = SGD(lr=0.2)

# 此处将loss函数更改为交叉熵，使模型收敛速度更快
model.compile(
    optimizer=sgd,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 训练模型
# epochs为迭代周期，把所有图片训练一次为一个周期
model.fit(x_train, y_train, batch_size=32, epochs=10)
# 评估模型
loss, accuracy = model.evaluate(x_test, y_test)

print('\ntest loss: ', loss)
print('test accuracy:', accuracy)

loss, accuracy = model.evaluate(x_train, y_train)

print('\ntrain loss: ', loss)
print('train accuracy:', accuracy)

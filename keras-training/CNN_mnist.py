# CNN应用于手写数字识别

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, Convolution2D, MaxPool2D, Flatten
from keras.regularizers import l2
from keras.optimizers import SGD, Adam

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 四个参数：数量， 长度， 宽度， 深度（黑白为1，彩色为3）

x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

model = Sequential()
# 第一个卷积层
# input_shape输入平面
# filters卷积核/滤波器个数
# kernel_size卷积窗口大小
# strides 步长
# padding padding方式 same/valid
# activation激活函数
model.add(Convolution2D(
    input_shape=(28, 28, 1),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',
    activation='relu'
))
# 第一个池化层 特征图变为14*14（pool_size）
model.add(MaxPool2D(
    pool_size=2,
    strides=2,
    padding='same',
))
# 第二个卷积层（filter = 64， kernel_size = 5）
model.add(Convolution2D(64, 5, strides=1, padding='same', activation='relu'))
# 第二个池化层
model.add(MaxPool2D(2, 2, 'same'))
# 把第二个池化层的输出扁平化为1维
model.add(Flatten())
# 第一个全连接层
model.add(Dense(1024, activation='relu'))
# Dropout
model.add(Dropout(0.5))
# 第二个全连接层
model.add(Dense(10, activation='softmax'))
# 定义优化器
adam = Adam(lr=1e-4)

# 定义优化器， loss function， 训练过程中计算准确率
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, batch_size=64, epochs=10)

loss, accuracy = model.evaluate(x_train, y_train)

print('train loss: ', loss)
print('train accuracy:', accuracy)

# 评估模型
loss, accuracy = model.evaluate(x_test, y_test)

print(('test_loss: ', loss))
print('test accuracy： ', accuracy)



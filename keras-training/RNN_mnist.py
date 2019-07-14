# RNN应用于手写数字识别

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.recurrent import SimpleRNN
from keras.layers import Dense
from keras.optimizers import Adam

# 数据长度 一行有28个数据
input_size = 28
# 序列长度 一共有28行
time_steps = 28
# 隐藏层cell个数
cell_size = 50

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

model = Sequential()

model.add(SimpleRNN(
    units=cell_size,
    input_shape=(time_steps, input_size),  # 输入
))

# 输出层
model.add(Dense(10, activation='softmax'))

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

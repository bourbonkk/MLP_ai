# coding=utf-8

from keras.models import Model
from keras.layers import Input, Dense, LSTM, Bidirectional
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt


# 학습 데이터를 생성한다.
# ex: data = [1,2,3,4,5,6,7,8,9,10]가 주어졌을 때 generateX(data, 5)를 실행하면
# 아래와 같은 학습데이터 변환한다.
#
# x                      y
# ---------              -
# 1,2,3,4,5              6
# 2,3,4,5,6              7
# 3,4,5,6,7              8
# ...
def generateX(a, n):
    x_train = []
    y_train = []
    for i in range(len(a)):
        x = a[i:(i + n)]
        if (i + n) < len(a):
            x_train.append(x)
            y_train.append(a[i + n])
        else:
            break
    return np.array(x_train), np.array(y_train)


# Sine 함수에 노이즈를 섞은 데이터로 학습 데이터 100개를 생성한다
data = np.sin(2 * np.pi * 0.03 * np.arange(0, 100)) + np.random.random(100)
x, y = generateX(data, 10)
x = x.reshape(-1, 10, 1)
y = y.reshape(-1, 1)

# 학습용 데이터와 시험용 데이터
x_train = x[:70, :, :]
y_train = y[:70:, :]
x_test = x[70:, :, :]
y_test = y[70:, :]

# 2층-양방항 구조의 LSTM 모델을 생성한다.
K.clear_session()  # 모델 생성전에 tensorflow의 graph 영역을 clear한다.
xInput = Input(batch_shape=(None, x_train.shape[1], x_train.shape[2]))
xLstm_1 = LSTM(10, return_sequences=True)(xInput)
xLstm_2 = Bidirectional(LSTM(10))(xLstm_1)
xOutput = Dense(1)(xLstm_2)

model = Model(xInput, xOutput)
model.compile(loss='mse', optimizer='adam')

# 학습
model.fit(x_train, y_train, epochs=500, batch_size=20, verbose=1)

# 예측
y_hat = model.predict(x_test, batch_size=1)

# 예측 결과 시각화
a_axis = np.arange(0, len(y_train))
b_axis = np.arange(len(y_train), len(y_train) + len(y_hat))

plt.figure(figsize=(10, 6))
plt.plot(a_axis, y_train.reshape(70, ), 'o-')
plt.plot(b_axis, y_hat.reshape(20, ), 'o-', color='red', label='Predicted')
plt.plot(b_axis, y_test.reshape(20, ), 'o-', color='green', alpha=0.2, label='Actual')
plt.legend()
plt.show()

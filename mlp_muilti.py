# coding=utf-8

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

hidden_node_num = [10, 9, 8]  # 층별 노드의 갯수
hidden_layer_num = len(hidden_node_num)  # 히든 레이어 층 수

# load data
data_file_name = 'x_square.txt'
xy = np.genfromtxt(data_file_name, dtype='float32')

temp_x = xy[:, 0]
temp_y = xy[:, 1]

x_data = [temp_x]
y_data = [temp_y]  # ndarray를 List 형태로 변환

x = tf.placeholder(dtype=tf.float32)  # tensor placeholder 생성 (값 저장소), 매개변수
y = tf.placeholder(dtype=tf.float32)

w = []
b = []
layer = []
x_data_len = len(x_data)

# first layer
w.append(tf.Variable(tf.random_normal([hidden_node_num[0], x_data_len]), name="w0"))  # 10, 1 인 2차원 텐서 생성
b.append(tf.Variable(tf.random_normal([hidden_node_num[0], 1]), name="b0"))

# add hidden layers (variable number)
for i in range(1, hidden_layer_num):
    wName = "w" + str(i)
    bName = "b" + str(i)
    w.append(tf.Variable(tf.random_normal([hidden_node_num[i], hidden_node_num[i - 1]]), name=wName))
    b.append(tf.Variable(tf.random_normal([hidden_node_num[i], 1]), name=bName))

# add final layer
wName = "w" + str(hidden_layer_num)
bName = "b" + str(hidden_layer_num)
w.append(tf.Variable(tf.random_normal([1, hidden_node_num[-1]]), name=wName))
b.append(tf.Variable(tf.random_normal([1], 1), name=bName))

# define model
layer.append(tf.nn.sigmoid(tf.matmul(w[0], x) + b[0]))
for i in range(1, hidden_layer_num):
    layer.append(tf.nn.sigmoid(tf.matmul(w[i], layer[i - 1]) + b[i]))
y_out = tf.matmul(w[-1], layer[-1]) + b[-1]

# setup cost function and optimizer
# cost = tf.reduce_mean(tf.square(y_out - y))
# opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
cost = tf.nn.l2_loss(y_out - y)
opt = tf.train.AdamOptimizer(0.1)  # GradientDescent 보다 성능이 좋은 옵티마이저

train = opt.minimize(cost)

# training model
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(5000):
        sess.run(train, feed_dict={x: x_data, y: y_data})
    # generate test data
    x_temp = np.linspace(0, 20, 50)
    # y = linspace(x1,x2)는 x1과 x2 사이에서 균일한 간격의 점 100개로 구성된 행 벡터를 반환합니다.
    # y = linspace(x1,x2,n)은 n개의 점을 생성합니다. 점 사이의 간격은 (x2-x1)/(n-1)입니다.
    #
    # linspace는 콜론 연산자 “:”과 유사하지만, 점 개수를 직접 제어할 수 있으며 항상 끝점을 포함합니다.
    # 이름 “linspace”의 “lin”은 선형 간격 값을 생성하는 것을 나타내며,
    # 이는 로그 간격 값을 생성하는 형제 함수 logspace와 대조됩니다.

    x_test = [x_temp]
    y_test = sess.run(y_out, feed_dict={x: x_test})

    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state('.\\model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess)
    else:
        sess.run(tf.global_variables_initializer())

    saver.save(sess, '.\\model\\dnn.chpt')

print(len(x_temp))
print(x_test)
print(y_test)

# design graph
plt.plot(x_data, y_data, 'ro', alpha=0.05)
plt.plot(x_test, y_test, 'h', alpha=1)
plt.show()

# Various line types, plot symbols and colors may be obtained with
#     plot(X,Y,S) where S is a character string made from one element
#     from any or all the following 3 columns:
#              b     blue          .     point              -     solid
#              g     green         o     circle             :     dotted
#              r     red           x     x-mark             -.    dashdot
#              c     cyan          +     plus               --    dashed
#              m     magenta       *     star             (none)  no line
#              y     yellow        s     square
#              k     black         d     diamond
#              w     white         v     triangle (down)
#                                  ^     triangle (up)
#                                  <     triangle (left)
#                                  >     triangle (right)
#                                  p     pentagram
#                                  h     hexagram

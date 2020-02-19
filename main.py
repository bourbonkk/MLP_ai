# coding=utf-8

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# https://twinw.tistory.com/242?category=543725
# load data
data_file_name = 'x_square.txt'
xy = np.genfromtxt(data_file_name, dtype='float32')

temp_x = xy[:, 0]
temp_y = xy[:, 1]

# reshape data
x_data = np.reshape(temp_x, [1, -1])
y_data = np.reshape(temp_y, [1, -1])

# setup input layer
x = tf.placeholder(dtype=tf.float32, shape=[1, None])
y = tf.placeholder(dtype=tf.float32, shape=[1, None])

number_of_hidden = 10

# setup hidden layer
w1 = tf.Variable(tf.random_normal([number_of_hidden, 1]))
b1 = tf.Variable(tf.random_normal([number_of_hidden, 1]))
layer1_out = tf.nn.sigmoid(tf.matmul(w1, x) + b1)

# setup output layer
w2 = tf.Variable(tf.random_normal([1, number_of_hidden]))
b2 = tf.Variable(tf.random_normal([1, 1]))
y_out = tf.matmul(w2, layer1_out) + b2

# setup cost
cost = tf.nn.l2_loss(y_out - y)

# setup optimizer
optimizer = tf.train.AdamOptimizer(0.01)
do_train = optimizer.minimize(cost)

# training model
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(5000):
        sess.run(do_train, feed_dict={x: x_data, y: y_data})
    # generate test data
    x_temp = np.linspace(0, 20, 50)
    x_test = [x_temp]
    y_test = sess.run(y_out, feed_dict={x: x_test})

# design graph
plt.plot(x_data, y_data, 'ro', alpha=0.05)
plt.plot(x_test, y_test, 'b^', alpha=1)
plt.show()

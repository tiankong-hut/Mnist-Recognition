# -*- coding: utf-8 -*-
# http://blog.csdn.net/u010089444/article/details/60963053
# python3.4

import tensorflow as tf
from tensorflow.contrib import rnn

import numpy as np
# import input_data
from tensorflow.examples.tutorials.mnist import input_data


# configuration variables
input_vec_size = lstm_size = 28 # 输入向量的维度
time_step_size = 28 # 循环层长度

batch_size = 128
test_size = 256

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, W, B, lstm_size):
    # X, input shape: (batch_size, time_step_size, input_vec_size)
    # XT shape: (time_step_size, batch_size, input_vec_size)
    XT = tf.transpose(X, [1, 0, 2])  # permute time_step_size and batch_size,[28, 128, 28]

    # XR shape: (time_step_size * batch_size, input_vec_size)
    XR = tf.reshape(XT, [-1, lstm_size]) # each row has input for each lstm cell (lstm_size=input_vec_size)

    # Each array shape: (batch_size, input_vec_size)
    X_split = tf.split(XR, time_step_size, 0) # split them to time_step_size (28 arrays),shape = [(128, 28),(128, 28)...]


    # Make lstm with lstm_size (each input vector size). num_units=lstm_size; forget_bias=1.0
    lstm = rnn.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=True)

    # Get lstm cell output, time_step_size (28) arrays with lstm_size output: (batch_size, lstm_size)
    # rnn..static_rnn()的输出对应于每一个timestep，如果只关心最后一步的输出，取outputs[-1]即可
    outputs, _states = rnn.static_rnn(lstm, X_split, dtype=tf.float32)  # 时间序列上每个Cell的输出:[... shape=(128, 28)..]

    # Linear activation
    # Get the last output
    return tf.matmul(outputs[-1], W) + B, lstm.state_size # State size to initialize the stat
# 载入数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) # 读取数据
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# mnist.train.images是一个55000 * 784维的矩阵, mnist.train.labels是一个55000 * 10维的矩阵
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# 将每张图用一个28x28的矩阵表示,(55000,28,28,1)
trX = trX.reshape(-1, 28, 28)
teX = teX.reshape(-1, 28, 28)

X = tf.placeholder("float", [None, 28, 28])
Y = tf.placeholder("float", [None, 10])

# get lstm_size and output 10 labels
W = init_weights([lstm_size, 10])  # 输出层权重矩阵28×10
B = init_weights([10])  # 输出层bais

py_x, state_size = model(X, W, B, lstm_size)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

session_conf = tf.ConfigProto()
session_conf.gpu_options.allow_growth = True

# Launch the graph in a session
with tf.Session(config=session_conf) as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(100):
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1, batch_size)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

        test_indices = np.arange(len(teX))  # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX[test_indices]})))



        # configuration
        #                        O * W + b -> 10 labels for each image, O[? 28], W[28 10], B[10]
        #                       ^ (O: output 28 vec from 28 vec input)
        #                       |
        #      +-+  +-+       +--+
        #      |1|->|2|-> ... |28| time_step_size = 28
        #      +-+  +-+       +--+
        #       ^    ^    ...  ^
        #       |    |         |
        # img1:[28] [28]  ... [28]
        # img2:[28] [28]  ... [28]
        # img3:[28] [28]  ... [28]
        # ...
        # img128 or img256 (batch_size or test_size 256)
        #      each input size = input_vec_size=lstm_size=28


# 没有rnn模块      #python2.7-tensorflow-V0.80


# /usr/bin/python3.4 /home/chuwei/PycharmProjects/MNIST/Mnist-12.py
# Extracting MNIST_data/train-images-idx3-ubyte.gz
# Extracting MNIST_data/train-labels-idx1-ubyte.gz
# Extracting MNIST_data/t10k-images-idx3-ubyte.gz
# Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
# 2017-11-28 10:47:04.917418: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
# 2017-11-28 10:47:04.917494: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
# 2017-11-28 10:47:04.917511: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
# 0 0.57421875
# 1 0.84375
# 2 0.84765625
# 3 0.921875
# 4 0.92578125
# 5 0.921875
# 6 0.94921875
# 7 0.95703125
# 8 0.953125
# 9 0.9375
# 10 0.94921875
# 11 0.9375
# 12 0.9453125
# 13 0.96484375
# 14 0.9609375
# 15 0.95703125
# 16 0.96875
# 17 0.95703125
# 18 0.96484375
# 19 0.953125
# 20 0.9765625
# 21 0.9609375
# 22 0.96484375
# 23 0.9765625
# 24 0.97265625
# 25 0.98046875
# 26 0.97265625
# 27 0.94921875
# 28 0.98046875
# 29 0.98828125
# 30 0.96875
# 31 0.97265625
# 32 0.9765625
# 33 0.96875
# 34 0.98046875
# 35 0.96875
# 36 0.9765625
# 37 0.99609375
# 38 0.98046875
# 39 0.984375
# 40 0.98828125
# 41 0.97265625
# 42 0.984375
# 43 0.9765625
# 44 0.96875
# 45 0.96875
# 46 0.984375
# 47 0.9609375
# 48 0.984375
# 49 0.9765625
# 50 0.9765625
# 未完待续.....
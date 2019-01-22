# http://blog.sina.com.cn/s/blog_4b0020f30102wv4l.html
#coding=UTF-8

from tensorflow.examples.tutorials.mnist import input_data
# import input_data

mnist = input_data.read_data_sets('./MNIST_data/', one_hot=True)

import tensorflow as tf
import numpy as np
import time

# parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 32
display_step = 10

# network parameters
n_input = 28  # 特征维度，字符图片的每一行看成输入特征
n_steps = 28  # 每张字符图片共有28行
n_hidden = 128  # 隐藏单元个数
n_classes = 10  # 类别总数,10个字符，因此类别为10
n_layers = 3  # RNN中有多少个cell

# tf Graph input
x = tf.placeholder("float32", [None, n_steps, n_input])
# rnn中的中间状态变量，包含cell 的状态（c_t）和每个cell 的输出状态(h_t) 对应lstm的输出公式[ht=o*tanh(ct)]
# 本例中的多层RNN中，每一个CELL中的状态数目相等，因此输入状态变量是2*n_hidden*n_layers]，实际上是可以不相等的
# 另外，可以提供初始状态，也可以不提供，让tf自动初始化
istate = tf.placeholder("float32", [None, 2 * n_hidden * n_layers])
y = tf.placeholder("float32", [None, n_classes])

# define weights, 设置weights 和biases为tf全局变量，weigths['hidden']  biases['hidden']参数代表对输入数据先进行一次线性变换(可选)，weigths['out']  biases['out']代表了从RNN状态到字符类别的线性连接层的参数，在训练的过程中，weigths，biases会持续变化
weights = {  # 'hidden':tf.Variable(tf.random_normal([n_input,n_hidden])),
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))}
biases = {  # 'hidden':tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))}


# define the LSTM network very simple,one cell
# 基于一个基本LSTM cell的RNN网络
def RNN(_X, _istate, _weights, _biases):
    # 将输入数据由[ batch_size,nsteps,n_input] 变为 [ nsteps，batch_size,n_input]
    _X = tf.transpose(_X, [1, 0, 2])
    _X = tf.reshape(_X, [-1, n_input])
    # linear activation, not neccessary for the lstm model,can be ommited
    # _X = tf.matmul(_X,_weights['hidden'])+_biases['hidden']  #输入rnn之前先加一层线性变换，可选
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    _X = tf.split(0, n_steps, _X)  # input a length T list of tensors
    outputs, states = tf.nn.rnn(lstm_cell, _X,
                                initial_state=_istate)  # 由于_X是list,输出ouputs也是lists, 长度为T，元素为[batch_size,hidden_units]的tensor，因此后面可以使用-1索引
    return tf.matmul(outputs[-1], _weights['out']) + _biases['out']  # rnn's ouput is a list of tensors


# lstm based on dyrnn
def DRNN(_X, _istate, _weights, _biases):
    _X = tf.transpose(_X, [1, 0, 2])  # because the input format is batch_size*nsteps_n_input
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    multi_cell = tf.nn.rnn_cell.MultiRNNCell(
        [lstm_cell] * n_layers)  # n_layers lstm cells，该函数接受[cell1,cell2,cell3] cell列表为参数，构建一个多层的RNN模型
    # print(multi_cell.state_size)  #可以用rnn_cell的state_size获取rnn_cell的大小
    # different from tf.nn.rnn(),input must be a tensor or a tuple of tensors
    outputs, states = tf.nn.dynamic_rnn(multi_cell, _X, initial_state=_istate, time_major=True)
    # if not set initial state, dtype must be set
    # outputs,states = tf.nn.dynamic_rnn(multi_cell,_X,dtype=tf.float32,time_major=True)
    # print(outputs.get_shape())
    # outputs,states = tf.nn.dynamic_rnn(multi_cell,_X,dtype=tf.float32,time_major=True) #if not set initial state, must set dtype
    last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)  # dynamic_rnn's output is a tensor
    return tf.matmul(last, _weights['out']) + _biases['out']


# RNN
if n_layers <= 1:
    pred = RNN(x, istate, weights, biases)
else:
    # DRNN
    pred = DRNN(x, istate, weights, biases)

# softmax交叉熵值损失
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 计算准确率
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    step = 1
    total_time = 0.0
    start_time = time.time()
    while step*batch_size:

        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape((batch_size, n_steps, n_input))  # numpy nd array, the feed_dict cannot be tensors

        # note that, the feed_dict should not contain any tensor, but should be nd-array
        sess.run(optimizer,
                 feed_dict={x: batch_xs, y: batch_ys, istate: np.zeros((batch_size, 2 * n_hidden * n_layers))})
        if step % display_step == 0:
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys,
                                             istate: np.zeros((batch_size, 2 * n_hidden * n_layers))})
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,
                                                istate: np.zeros((batch_size, 2 * n_hidden * n_layers))})
            print
            "Iter " + str(step * batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + \
            ", Training Accuracy= " + "{:.5f}".format(acc)
        step = step + 1
    total_time = total_time + (time.time() - start_time)
    print("Optimization %d iterations, Finished in %.4f seconds!" % (training_iters, total_time))

    # Doing some tests: Calculate accuracy for 256 mnist test images
    test_len = 256
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print
    "Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                                       istate: np.zeros((test_len, 2 * n_hidden * n_layers))})


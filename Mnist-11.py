#coding=UTF-8
# http://blog.csdn.net/mebiuw/article/details/52705731
# https://segmentfault.com/a/1190000008346992
# python2.7

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 在这里做数据加载，还是使用那个MNIST的数据，以one_hot的方式加载数据，记得目录可以改成之前已经下载完成的目录
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# mnist = input_data.read_data_sets("/home/chuwei/PycharmProjects/MNIST/MNIST_data/", one_hot=True)  #效果同下
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)   #默认与Mnist-11在一个文件夹: /home/chuwei/PycharmProjects/MNIST/MNIST_data/

# Training set images: train-images-idx3-ubyte.gz (9.9 MB, 解压后 47 MB, 包含 60,000 个样本)
# Training set labels: train-labels-idx1-ubyte.gz (29 KB, 解压后 60 KB, 包含 60,000 个标签)
# Test set images: t10k-images-idx3-ubyte.gz (1.6 MB, 解压后 7.8 MB, 包含 10,000 个样本)
# Test set labels: t10k-labels-idx1-ubyte.gz (5KB, 解压后 10 KB, 包含 10,000 个标签)
# training = mnist.train.images
# trainlabel = mnist.train.labels
# testing = mnist.test.images
# testlabel = mnist.test.labels
'''
MNIST的数据是一个28*28的图像，这里RNN测试，把他看成一行行的序列（28维度（28长的sequence）*28行）
'''

# RNN学习时使用的参数
learning_rate = 0.001        #学习率
training_iters = 1000         #训练次数
batch_size =128              #每轮训练数据大小

# 神经网络的参数
n_input = 28    # 输入层的n,28步
n_steps = 28    # 28长度,每一步
n_hidden = 100  # 隐藏层神经元个数
n_classes = 10  # 输出的数量，因为是分类问题，0~9个数字，这里一共有10个

# 构建tensorflow的输入X的placeholder(占位符)
x = tf.placeholder("float", [None, n_steps, n_input])
# tensorflow里的LSTM需要两倍于n_hidden的长度的状态，一个state和一个cell
# Tensorflow LSTM cell requires 2x n_hidden length (state & cell)
istate = tf.placeholder("float", [None, 2 * n_hidden])
# 输出Y
y = tf.placeholder("float", [None, n_classes])

# 随机初始化每一层的权值和偏置
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])),  #Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
          }
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
         }

'''
构建RNN
'''
def RNN(_X, _istate, _weights, _biases):
    # 规整输入的数据
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size

    _X = tf.reshape(_X, [-1, n_input])  # (n_steps*batch_size, n_input) ,-1表示根据N_input来调整。
    # 输入层到隐含层，第一次是直接运算
    _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']
    # 之后使用LSTM
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # 28长度的sequence，所以是需要分解位28次
    _X = tf.split(0, n_steps, _X)   #对第0维切割， n_steps * (batch_size, n_hidden); tf.split(dimension, num_split, input)
    # _X = tf.split(_X, n_steps, 0)     #Tensorflow > 1.0

    # 开始跑RNN那部分
    outputs, states = tf.nn.rnn(lstm_cell, _X, initial_state=_istate)


    # 输出层
    return tf.matmul(outputs[-1], _weights['out']) + _biases['out']


pred = RNN(x, istate, weights, biases)

# 定义损失函数和优化方法，其中损失函数为softmax交叉熵，优化方法为Adam
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))  # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Adam Optimizer

# 进行模型的评估，argmax是取出取值最大的那一个的标签作为输出
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))  #判断预测值索引与实际值索引是否相等,返回true or false
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  #tf.cast:将correct_pred的数据格式转化成float32

# 初始化
init = tf.initialize_all_variables()
# 即initialize_variables(all_variables())

list1=[]    # 在循环之前先创建好列表,存储数据画图
list2=[]
# 开始运行
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # 持续迭代
    while step * batch_size < training_iters:
        # 随机抽出这一次迭代训练时用的数据
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)   #通过next_batch()就可以一个一个batch的拿数据
        # 对数据进行处理，使得其符合输入
        batch_xs = batch_xs.reshape((batch_size, n_steps, n_input))
        # 迭代
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,
                                       istate: np.zeros((batch_size, 2 * n_hidden))})
        sess.run(correct_pred, feed_dict={x: batch_xs, y: batch_ys,
                                       istate: np.zeros((batch_size, 2 * n_hidden))})
        acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,
                                       istate: np.zeros((batch_size, 2 * n_hidden))})
        Acc = acc
        list1.append(step * batch_size)  # 添加到列表中,画图用
        list2.append(Acc)
        print (step * batch_size)
        print ("Acc:",Acc)

        # 在特定的迭代回合进行数据的输出
        if step % 1 == 0:              #每20次输出1次准确率
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,
                                                istate: np.zeros((batch_size, 2 * n_hidden))})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys,
                                             istate: np.zeros((batch_size, 2 * n_hidden))})
            print ("Iter " + str(step * batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                  ", Training Accuracy= " + "{:.5f}".format(acc))

        step += 1

    print ("Optimization Finished!")

    # 画出准确率的图
    plt.plot(list1, list2)
    plt.xlabel('Cycle times')
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.show()

   # 载入测试集进行测试
    batch_size = 1000
    # Calculate accuracy for 128 mnist test images
    test_len = batch_size
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    # print  sess.run(accuracy, feed_dict={x: test_data, y: test_label,
    #                istate: np.zeros((batch_size, 2 * n_hidden))})  #掉了一个标点符号“）”，导致非法错误
    print ("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                                             istate: np.zeros((batch_size, 2 * n_hidden))}))


    # "Testing Accuracy:",
    # # 载入测试集进行测试
    # test_len = 256
    # test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    # test_label = mnist.test.labels[:test_len]
    #
    # optimization = sess.run(optimizer, feed_dict={x: test_data, y: test_label,
    #                                               istate: np.zeros((test_size, 2 * n_hidden))})
    # Testing_acc = sess.run(accuracy, feed_dict={x: test_data, y: test_label,
    #                                             istate: np.zeros((test_size, 2 * n_hidden))})
    # print "Testing_acc:", Testing_acc



    # MNIST的数据集，共有三部分：
    # 55, 000
    # points
    # 训练集(mnist.train), 10, 000
    # points
    # 测试集(mnist.test), 5, 000
    # 验证集 （mnist.validation).
    # 与之同时，由于其是一个有监督的学习任务，所以还有对应的标签（也就是图像对应的真实数字），这部分位于（mnist.train.labels），
    # 标签也是以one - hot的方式表示，即这个向量共有10维，第I个位1就是证明这个Label是I


# /usr/bin/python2.7 /home/chuwei/PycharmProjects/MNIST/Mnist-11.py
# Extracting MNIST_data/train-images-idx3-ubyte.gz
# Extracting MNIST_data/train-labels-idx1-ubyte.gz
# Extracting MNIST_data/t10k-images-idx3-ubyte.gz
# Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
# Iter 1280, Minibatch Loss= 1.824641, Training Accuracy= 0.34375
# Iter 2560, Minibatch Loss= 1.726485, Training Accuracy= 0.38281
# Iter 3840, Minibatch Loss= 1.373146, Training Accuracy= 0.53125
# Iter 5120, Minibatch Loss= 1.121431, Training Accuracy= 0.59375
# Iter 6400, Minibatch Loss= 1.105026, Training Accuracy= 0.57031
# Iter 7680, Minibatch Loss= 1.064457, Training Accuracy= 0.65625
# Iter 8960, Minibatch Loss= 0.855956, Training Accuracy= 0.71094
# Iter 10240, Minibatch Loss= 0.691251, Training Accuracy= 0.79688
# Iter 11520, Minibatch Loss= 0.469006, Training Accuracy= 0.88281
# Iter 12800, Minibatch Loss= 0.827366, Training Accuracy= 0.70312
# .......
# Iter 93440, Minibatch Loss= 0.153870, Training Accuracy= 0.95312
# Iter 94720, Minibatch Loss= 0.058097, Training Accuracy= 0.98438
# Iter 96000, Minibatch Loss= 0.173155, Training Accuracy= 0.94531
# Iter 97280, Minibatch Loss= 0.155955, Training Accuracy= 0.95312
# Iter 98560, Minibatch Loss= 0.063680, Training Accuracy= 0.97656
# Iter 99840, Minibatch Loss= 0.222029, Training Accuracy= 0.93750
# Optimization Finished!

# Process finished with exit code 0
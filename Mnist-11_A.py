#coding=UTF-8
# http://blog.csdn.net/mebiuw/article/details/52705731
# https://segmentfault.com/a/1190000008346992
# python2.7

from __future__ import division     #必须放在最前面，division进行精确的除法运算
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# import Mnist-11_B.py
# # from Mnist-11_B.py __import classify()
# if __name__ == "__main__":
#     Mnist-11_B.classify()

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)   #默认与Mnist-11在一个文件夹: /home/chuwei/PycharmProjects/MNIST/MNIST_data/

'''
MNIST的数据是一个28*28的图像，这里RNN测试，把他看成一行行的序列（28维度（28长的sequence）*28行）
'''

# RNN学习时使用的参数
learning_rate = 0.001        #学习率
training_iters = 80000       #训练次数
batch_size =100              #每轮训练数据大小

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
    _X = tf.split(0, n_steps, _X)  # n_steps * (batch_size, n_hidden)
    # _X = tf.split(_X, n_steps, _X)  # n_steps * (batch_size, n_hidden)    #Tensorflow > 0.12.0
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
arg_max_pre = tf.argmax(pred, 1)
arg_max_y   = tf.argmax(y, 1)



# print "correct_pred_1: ",correct_pred    #未进行训练，所以没有结果
#      correct_pred_1:  Tensor("Equal:0", shape=(?,), dtype=bool)

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  #tf.cast:将correct_pred的数据格式转化成float32
# print "accuracy_1: ",accuracy
#      accuracy_1:  Tensor("Mean_1:0", shape=(), dtype=float32)

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
        print "batch_xs_1: ",batch_xs.shape
        print "batch_ys:   ",batch_ys.shape
        # 对数据进行处理，使得其符合输入
        batch_xs = batch_xs.reshape((batch_size, n_steps, n_input))
        print "batch_xs_2: ",batch_xs.shape
        # 迭代
        optimization=sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,
                                       istate: np.zeros((batch_size, 2 * n_hidden))})
        print "optimization:",optimization    #optimizer无返回值?

        #不能用和后面一样的字母，因为第二轮要重复赋值
        arg_pre =sess.run(arg_max_pre, feed_dict={x: batch_xs, y: batch_ys,
                                          istate: np.zeros((batch_size, 2 * n_hidden))})
        arg_y =sess.run(arg_max_y, feed_dict={x: batch_xs, y: batch_ys,
                                          istate: np.zeros((batch_size, 2 * n_hidden))})
        print "arg_pre:\n ",arg_pre     #100个数
        print "arg_y：\n ", arg_y       #100个数
        # print "arg_pre.shape:\n ",arg_pre.shape

        cor_pre =sess.run(correct_pred, feed_dict={x: batch_xs, y: batch_ys,
                                       istate: np.zeros((batch_size, 2 * n_hidden))})
        # print "cor_pre:\n ",cor_pre



        acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,
                                       istate: np.zeros((batch_size, 2 * n_hidden))})
        Acc = acc
        list1.append(step * batch_size)  # 添加到列表中,画图用
        list2.append(Acc)
        print "训练次数：",(step * batch_size)
        print "Acc:",Acc

        # 在特定的迭代回合进行数据的输出
        if step % 10 == 0:              #每20次输出1次准确率
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,
                                                istate: np.zeros((batch_size, 2 * n_hidden))})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys,
                                             istate: np.zeros((batch_size, 2 * n_hidden))})
            print ("Iter " + str(step * batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                  ", Training Accuracy= " + "{:.5f}".format(acc))
        step += 1


    print "arg_pre.shape: ",arg_pre.shape

    # 初始化计数器
    A0_to_0 = 0
    A0_to_other = 0
    other_to_0 = 0

    A1_to_1 = 0
    A1_to_other = 0
    other_to_1 = 0

    A2_to_2 = 0
    A2_to_other = 0
    other_to_2 = 0

    A3_to_3 = 0
    A3_to_other = 0
    other_to_3 = 0


    A4_to_4 = 0
    A4_to_other = 0
    other_to_4 = 0

    A5_to_5 = 0
    A5_to_other = 0
    other_to_5 = 0

    A6_to_6 = 0
    A6_to_other = 0
    other_to_6 = 0

    A7_to_7 = 0
    A7_to_other = 0
    other_to_7 = 0

    A8_to_8 = 0
    A8_to_other = 0
    other_to_8 = 0

    A9_to_9 = 0
    A9_to_other = 0
    other_to_9 = 0

    for i in range(0,100) :        #for循环：对每个元素都执行相同的操作
        # print "i=: ",i
        #所有数字:0-9
        if arg_pre[i] == arg_y[i] :
            print "正确识别的数字:-- ", arg_pre[i]

        #数字：0
        if arg_y[i] == arg_pre[i] == 0:
            A0_to_0 = A0_to_0+1
            # print "正确的数字: ", arg_pre[i]
            # print "A0_to_0--正确的个数: ", A0_to_0
        elif arg_y[i] != 0 and arg_pre[i] ==0 :
            other_to_0+=1
            # print "other_to_0--误识的个数: ", other_to_0
        elif arg_y[i] == 0 and arg_pre[i] != 0 :
            A0_to_other+=1
            # print "A0_to_other--拒识的个数: ", A0_to_other

        #数字：1
        if  arg_pre[i] == arg_y[i] == 1:
            A1_to_1 = A1_to_1 + 1
        elif arg_y[i] != 1 and arg_pre[i] == 1:
            other_to_1 += 1
        elif arg_y[i] == 1 and arg_pre[i] != 1:
            A1_to_other += 1

        #数字：2
        if  arg_pre[i] == arg_y[i] == 2:
            A2_to_2 = A2_to_2 + 1
        elif arg_y[i] != 2 and arg_pre[i] == 2:
            other_to_2 += 1
        elif arg_y[i] == 2 and arg_pre[i] != 2:
            A2_to_other += 1

        #数字：3
        if  arg_pre[i] == arg_y[i] == 3:
            A3_to_3 = A3_to_3 + 1
        elif arg_y[i] != 3 and arg_pre[i] == 3:
            other_to_3 += 1
        elif arg_y[i] == 3 and arg_pre[i] != 3:
            A3_to_other += 1

        #数字：4
        if  arg_pre[i] == arg_y[i] == 4:
            A4_to_4 += 1
        elif arg_y[i] != 4 and arg_pre[i] == 4:
            other_to_0 += 1
        elif arg_y[i] == 4 and arg_pre[i] != 4:
            A4_to_other += 1

        #数字：5
        if  arg_pre[i] == arg_y[i] == 5:
            A5_to_5 += 1
        elif arg_y[i] != 5 and arg_pre[i] == 5:
            other_to_5 += 1
        elif arg_y[i] == 5 and arg_pre[i] != 5:
            A5_to_other += 1

        # 数字：6
        if  arg_pre[i] == arg_y[i] == 6:
            A6_to_6 += 1
        elif arg_y[i] != 6 and arg_pre[i] == 6:
            other_to_6 += 1
        elif arg_y[i] == 6 and arg_pre[i] != 6:
            A6_to_other += 1

        #数字：7
        if  arg_pre[i] == arg_y[i] == 7:
            A7_to_7 +=1
        elif arg_y[i] != 7 and arg_pre[i] == 7:
            other_to_7 += 1
        elif arg_y[i] == 7 and arg_pre[i] != 7:
            A7_to_other += 1

        #数字：8
        if  arg_pre[i] == arg_y[i] == 8:
            A8_to_8 += 1
        elif arg_y[i] != 8 and arg_pre[i] == 8:
            other_to_8 += 1
        elif arg_y[i] == 8 and arg_pre[i] != 8:
            A8_to_other += 1

        # 数字：9
        if arg_pre[i] == arg_y[i] == 9:
            A9_to_9 += 1
        elif arg_y[i] != 9 and arg_pre[i] == 9:
            other_to_9 += 1
        elif arg_y[i] == 9 and arg_pre[i] != 9:
            A9_to_other += 1

    #每个数字的总量
    A0 = A0_to_0 + other_to_0 + A0_to_other
    A1 = A1_to_1 + other_to_1 + A1_to_other
    A2 = A2_to_2 + other_to_2 + A2_to_other
    A3 = A3_to_3 + other_to_3 + A3_to_other
    A4 = A4_to_4 + other_to_4 + A4_to_other
    A5 = A5_to_5 + other_to_5 + A5_to_other
    A6 = A6_to_6 + other_to_6 + A6_to_other
    A7 = A7_to_7 + other_to_7 + A7_to_other
    A8 = A8_to_8 + other_to_8 + A8_to_other
    A9 = A9_to_9 + other_to_9 + A9_to_other
    # print "A9--",A9
    # print "A4--", A4

    print " "
    print "A0_to_0--正确识别率: %.3f "  %(A0_to_0 / A0)
    print "other_to_0--误识率: %.3f "   %(other_to_0 / A0)
    print "A0_to_other--拒识率: %.3f"   %(A0_to_other / A0)
    print " "
    print "A1_to_1--正确识别率: %.3f "  %(A1_to_1 / A1)
    print "other_to_1--误识率: %.3f "   %(other_to_1 / A1)
    print "A1_to_other--拒识率: %.3f"   %(A1_to_other / A1)
    print " "
    print "A2_to_2--正确识别率: %.3f "  %(A2_to_2 / A2)
    print "other_to_2--误识率: %.3f "   %(other_to_2 / A2)
    print "A2_to_other--拒识率: %.3f "  %(A2_to_other / A2)
    print " "
    print "A3_to_3--正确识别率:%.3f  "   %(A3_to_3 / A3)
    print "other_to_3--误识率: %.3f "   %(other_to_3 / A3)
    print "A3_to_other--拒识率:%.3f  "  %(A3_to_other / A3)
    print " "
    print "A4_to_4--正确识别率: %.3f "   %(A4_to_4 / A4)
    print "other_to_4--误识率: %.3f "   %(other_to_4 / A4)
    print "A4_to_other--拒识率: %.3f "  %(A4_to_other / A4)
    print " "
    print "A5_to_5--正确识别率:  %.3f"   %(A5_to_5 / A5)
    print "other_to_5--误识率:  %.3f"   %(other_to_5 / A5)
    print "A5_to_other--拒识率:  %.3f"  %(A5_to_other / A5)
    print " "
    print "A6_to_6--正确识别率: %.3f"    %( A6_to_6 / A6)
    print "other_to_6--误识率:  %.3f"   %(other_to_6 / A6)
    print "A6_to_other--拒识率: %.3f"   %(A6_to_other / A6)
    print " "
    print "A7_to_7--正确识别率: %.3f "   %(A7_to_7 / A7)
    print "other_to_7--误识率: %.3f "   %(other_to_7 / A7)
    print "A7_to_other--拒识率: %.3f"   %(A7_to_other / A7)
    print " "
    print "A8_to_8--正确识别率: %.3f "   %(A8_to_8 / A8)
    print "other_to_8--误识率:  %.3f"   %(other_to_8 / A8)
    print "A8_to_other--拒识率: %.3f "  %(A8_to_other / A8)
    print " "
    print "A9_to_9--正确识别率: %.3f "   %(A9_to_9 / A9)
    print "other_to_9--误识率:  %.3f"   %(other_to_9 / A9)
    print "A9_to_other--拒识率: %.3f"   %(A9_to_other / A9)
    print " "

    print ("Optimization Finished!")

    # 画出准确率的图
    # plt.plot(list1, list2)
    # plt.xlabel('Cycle times')
    # plt.ylabel("Accuracy")
    # plt.title("Training Accuracy")
    # plt.show()

#    载入测试集进行测试
    batch_size = 1000
    # Calculate accuracy for 128 mnist test images
    test_len = batch_size
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    # print  sess.run(accuracy, feed_dict={x: test_data, y: test_label,
    #                istate: np.zeros((batch_size, 2 * n_hidden))})  #掉了一个标点符号“）”，导致非法错误
    print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                                             istate: np.zeros((batch_size, 2 * n_hidden))})


    # "Testing Accuracy:",
    # # 载入测试集进行测试
    # test_len = 256
    # test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    # test_label = mnist.test.labels[:test_len]




    # MNIST的数据集，共有三部分：
    # 55, 000
    # points
    # 训练集(mnist.train), 10, 000
    # points
    # 测试集(mnist.test), 5, 000
    # 验证集 （mnist.validation).
    # 与之同时，由于其是一个有监督的学习任务，所以还有对应的标签（也就是图像对应的真实数字），这部分位于（mnist.train.labels），
    # 标签也是以one - hot的方式表示，即这个向量共有10维，第I个位1就是证明这个Label是I


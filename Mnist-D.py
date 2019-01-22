# coding: utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# In[2]:

sess = tf.InteractiveSession()

# In[3]:

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# In[4]:

learning_rate = 0.001
batch_size = 128

n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])


# In[5]:

def RNN(x, weight, biases):
    # x shape: (batch_size, n_steps, n_input)
    # desired shape: list of n_steps with element shape (batch_size, n_input)
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(0, n_steps, x)
    outputs = list()
    lstm = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    state = (tf.zeros([n_steps, n_hidden]),) * 2
    sess.run(state)
    with tf.variable_scope("myrnn2") as scope:      # tf.variable_scope:在一个作用域 scope 内共享一些变量
        for i in range(n_steps - 1):
            if i > 0:
                scope.reuse_variables()
            output, state = lstm(x[i], state)
            outputs.append(output)
    final = tf.matmul(outputs[-1], weight) + biases
    return final


# In[6]:

def RNN(x, n_steps, n_input, n_hidden, n_classes):
    # Parameters:
    # Input gate: input, previous output, and bias
           # tf.truncated_normal(shape, mean, stddev):shape表示生成张量的维度，
           # mean是均值，stddev是标准差。这个函数产生正态分布，均值和标准差自己设定
    ix = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.1, 0.1))
    im = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.1, 0.1))
    ib = tf.Variable(tf.zeros([1, n_hidden]))
    # Forget gate: input, previous output, and bias
    fx = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.1, 0.1))
    fm = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.1, 0.1))
    fb = tf.Variable(tf.zeros([1, n_hidden]))
    # Memory cell: input, state, and bias
    cx = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.1, 0.1))
    cm = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.1, 0.1))
    cb = tf.Variable(tf.zeros([1, n_hidden]))
    # Output gate: input, previous output, and bias
    ox = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.1, 0.1))
    om = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.1, 0.1))
    ob = tf.Variable(tf.zeros([1, n_hidden]))
    # Classifier weights and biases
    w = tf.Variable(tf.truncated_normal([n_hidden, n_classes]))
    b = tf.Variable(tf.zeros([n_classes]))

    # Definition of the cell computation
    def lstm_cell(i, o, state):
        input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
        forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
        update = tf.tanh(tf.matmul(i, cx) + tf.matmul(o, cm) + cb)
        state = forget_gate * state + input_gate * update
        output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
        return output_gate * tf.tanh(state), state

    # Unrolled LSTM loop
    outputs = list()
    state = tf.Variable(tf.zeros([batch_size, n_hidden]))
    output = tf.Variable(tf.zeros([batch_size, n_hidden]))

    # x shape: (batch_size, n_steps, n_input)
    # desired shape: list of n_steps with element shape (batch_size, n_input)
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(0, n_steps, x)
    for i in x:
        output, state = lstm_cell(i, output, state)
        outputs.append(output)
    logits = tf.matmul(outputs[-1], w) + b
    return logits


# In[7]:

pred = RNN(x, n_steps, n_input, n_hidden, n_classes)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
# init = tf.global_variables_initializer()
init = tf.initialize_all_variables()
# In[8]:
list1 = []  # 在循环之前先创建好列表,存储数据画图
list2 = []
# Launch the graph
sess.run(init)
for step in range(2000):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    batch_x = batch_x.reshape((batch_size, n_steps, n_input))
    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

    acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
    list1.append(step)   # 添加到列表中,画图用
    Acc = acc
    list2.append(Acc)
    print step
    print Acc
    # print list1
    # print list2

    if step % 50 == 0:
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
        loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        print "Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(
               loss) + ", Training Accuracy= " + "{:.5f}".format(acc)

print "Optimization Finished!"

#画出准确率的图
plt.plot(list1, list2 )
plt.xlabel('Cycle times')
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.show()


# Calculate accuracy for 128 mnist test images
test_len = batch_size
test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
test_label = mnist.test.labels[:test_len]
print ("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))



# coding=utf-8
# http://blog.csdn.net/xxzhangx/article/details/54563574
#加载tensorflow包
import tensorflow as tf
#加载读取函数
from tensorflow.examples.tutorials.mnist import input_data
#读数据，one_hot表示将矩阵处理为行向量，即28*28 => 1*784
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#预定义X
x = tf.placeholder("float", [None, 784])
#预定义W
W = tf.Variable(tf.zeros([784,10]))
#预定义b
b = tf.Variable(tf.zeros([10]))
#模型
y = tf.nn.softmax(tf.matmul(x,W) + b)
#预定义真实值
y_ = tf.placeholder("float", [None,10])
#交叉熵损失函数
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#基于梯度的优化
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#初始化模型的所有变量
init = tf.global_variables_initializer()
#启动一个模型，并初始化
sess = tf.Session()
sess.run(init)
#开始训练模型，训练1000次，每次抽取100个批数据
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#真实与预测
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#输出结果
print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
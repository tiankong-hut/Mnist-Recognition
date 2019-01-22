#coding=utf-8
#https://www.cnblogs.com/denny402/p/5853538.html
#https://www.baidu.com/link?url=8-IYA6LiH3bUaQoL3p1eLYnztDaEsb-I2R5mgY3jKuOLjl7FOJbHFqqw-iGlVfcV-pm4TdPX1etrTy1HNu_zFvE9zrHdbH7C6xc7zqOkQjO&wd=&eqid=84a76c71000018f8000000065ba25388

import tensorflow as tf 
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)     #下载并加载mnist数据
x = tf.placeholder(tf.float32, [None, 784])                        #输入的数据占位符
y_actual = tf.placeholder(tf.float32, shape=[None, 10])            #输入的标签占位符

batch_size = 50

#定义四个函数，分别用于初始化权值W，初始化偏置项b, 构建卷积层和构建池化层。
#定义一个函数，用于初始化所有的权值 W
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#定义一个函数，用于初始化所有的偏置项 b
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
#定义一个函数，用于构建卷积层
# x-- input-- shape[batch, in_height, in_width, in_channels]
# w-- filter-- shape为[filter_height, filter_width, in_channels, out_channels] 
# strides=[1, 1, 1, 1] -- 水平方向卷积步长为第二个参数1，垂直方向步长为第三个参数1
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#定义一个函数，用于构建池化层
#ksize-- （滤波器）池化窗口的大小，通常取[1, height, width, 1]，因为不想在batch和channels上做池化，所以这两个维度设为了1
#strides=[1, 2, 2, 1] -- 水平步长为第二个参数2，垂直步长为第三个参数2
def max_pool(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
  # return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

#接下来构建网络。整个网络由两个卷积层（包含激活层和池化层），一个全连接层，一个dropout层和一个softmax层组成。 
#构建网络
# x--input--[batch, in_height, in_width, in_channels]
x_image = tf.reshape(x, [-1,28,28,1])         #转换输入数据shape,以便于用于网络中
W_conv1 = weight_variable([5, 5, 1, 32])      # w--filter--[filter_height, filter_width, in_channels, out_channels] 
b_conv1 = bias_variable([32])       
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)     #第一个卷积层 ，(28-5+2*1)/1+1=26。不足一列时，SAME比VALID的填充方式多了一列
h_pool1 = max_pool(h_conv1)                                  #第一个池化层 ，(26-2+2*1)/2+1=14。卷积核里最大值

W_conv2 = weight_variable([5, 5, 32, 64])    
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)      #第二个卷积层,(14-5+2*1)/1+1=12
h_pool2 = max_pool(h_conv2)                                   #第二个池化层,(12-2+2*1)/2+1=7

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])              #reshape成向量
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)    #第一个全连接层

keep_prob = tf.placeholder("float") 
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)                  #dropout层

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_predict=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)   #softmax层

#网络构建好后，就可以开始训练了。
cross_entropy = -tf.reduce_sum(y_actual*tf.log(y_predict))     #交叉熵
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)    #梯度下降法
correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_actual,1))    
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))                 #精确度计算
sess=tf.InteractiveSession()                          
sess.run(tf.initialize_all_variables())
for i in range(20000):
  batch = mnist.train.next_batch(60)   # batch_size=50 ,设置为100时效果差。
  if i%100 == 0:                       # 训练100次，验证一次
    train_acc = accuracy.eval(feed_dict={x:batch[0], y_actual: batch[1], keep_prob: 1.0})
    print ("step %d, training accuracy %g"%(i,train_acc))
    train_step.run(feed_dict={x: batch[0], y_actual: batch[1], keep_prob: 0.5})  #keep_prob: 0.5 -- dropout防止过拟合

test_acc=accuracy.eval(feed_dict={x: mnist.test.images, y_actual: mnist.test.labels, keep_prob: 1.0})
print ("test accuracy %g"%test_acc)


"""
#Tensorflow依赖于一个高效的C++后端来进行计算。与后端的这个连接叫做session。一般而言，使用TensorFlow程序的流程是先创建一个图，然后在session中启动它。
#这里，我们使用更加方便的InteractiveSession类。通过它，你可以更加灵活地构建你的代码。它能让你在运行图的时候，插入一些计算图，这些计算图是由某些操作(operations)构成的。这对于工作在交互式环境中的人们来说非常便利，比如使用IPython。
#训练20000次后，再进行测试，测试精度可以达到99%。
"""
"""
卷积和池化函数
（1）tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)

介绍参数：
input：指卷积需要输入的参数，具有这样的shape[batch, in_height, in_width, in_channels]，分别是[batch张图片, 每张图片高度为in_height, 每张图片宽度为in_width, 图像通道为in_channels]。
filter：指用来做卷积的滤波器，当然滤波器也需要有相应参数，滤波器的shape为[filter_height, filter_width, in_channels, out_channels]，分别对应[滤波器高度, 滤波器宽度, 接受图像的通道数, 卷积后通道数]，其中第三个参数 in_channels需要与input中的第四个参数 in_channels一致，out_channels第一看的话有些不好理解，如rgb输入三通道图，我们的滤波器的out_channels设为1的话，就是三通道对应值相加，最后输出一个卷积核。
strides:代表步长，其值可以直接默认一个数，也可以是一个四维数如[1,2,1,1]，则其意思是水平方向卷积步长为第二个参数2，垂直方向步长为1.
padding：代表填充方式，参数只有两种，SAME和VALID，SAME比VALID的填充方式多了一列，比如一个3*3图像用2*2的滤波器进行卷积，当步长设为2的时候，会缺少一列，则进行第二次卷积的时候，VALID发现余下的窗口不足2*2会直接把第三列去掉，SAME则会填充一列，填充值为0。
use_cudnn_on_gpu：bool类型，是否使用cudnn加速，默认为true。大概意思是是否使用gpu加速，还没搞太懂。
name：给返回的tensor命名。给输出feature map起名字。

（2）tf.nn.max_pool(value, ksize, strides, padding, name=None)

value：池化的输入，一般池化层接在卷积层的后面，所以输出通常为feature map。feature map依旧是[batch, in_height, in_width, in_channels]这样的参数。
ksize：池化窗口的大小，参数为四维向量，通常取[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1。ps：估计面tf.nn.conv2d中stries的四个取值也有              相同的意思。
stries：步长，同样是一个四维向量。
padding：填充方式同样只有两种不重复了。
"""
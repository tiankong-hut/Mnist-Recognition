#coding=utf-8
#各种函数解析

import numpy as np
import tensorflow as tf

# pow()用法:幂次方
# import math
# a = math.pow(2,10)   #pow(x,y) 等价于 x**y:arange
# print a

# python中的list是python的内置数据类型，list中的数据类不必相同的，
# 而array的中的类型必须全部相同。在list中的数据类型保存的是数据的存放的地址，
# 简单的说就是指针，并非数据，这样保存一个list就太麻烦了
# list1 = [1, 2, 3, 'a']
# print list1
# a = np.array([1, 2, 3, 4, 5])
# b = np.array([[1, 2, 3], [4, 5, 6]])
# c = list(a)  # array到list的转换
# print a, np.shape(a)
# print b, np.shape(b)
# print c, np.shape(c)
## shape函数是numpy.core.fromnumeric中的函数，它的功能是读取矩阵的长度，
# 比如shape([0])就是读取矩阵第一维度的长度。
# print np.shape([2])

## python range(): 创建一个整数列表，一般用在 for 循环中。
# x= range(1,10,3)
# print x

## random() 方法返回随机生成的一个实数，它在[0,1)范围内。
# import random
# # 生成第一个随机数
# print "random() : ", random.random()
# # 生成第二个随机数
# print "random() : ", random.random()

# seed() 方法改变随机数生成器的种子，可以在调用其他随机模块函数之前调用此函数
# import random
# random.seed( 10 )
# print "Random number with seed 10 : ", random.random()
# # 生成同一个随机数
# random.seed( 10 )
# print "Random number with seed 10 : ", random.random()
# # 生成同一个随机数
# random.seed( 10 )
# print "Random number with seed 10 : ", random.random()
# # 生成不同随机数
# random.seed(5)
# print random.random()

# a = np.array(([1, 2, 3, 4],[4, 5, 6, 7], [7, 8, 9, 10]))
# print a
# print  np.shape([a])
# # a.shape

# from numpy import *
# num=0
# random.seed(5)     #不同随机数
# while(num<5):
#     random.seed(5)   #相同随机数
#     print(random.random())
#     num+=1


#  randint() 函数来生成随机数该函数的语法为：random.randint(a,b)
# 函数返回数字N,N为a到b之间的数字（a <= N <= b),包含a和b。每次执行后都返回不同的数字
# import random
# print (random.randint(0,10))

# #shape和reshape输出数组的行和列数
# import numpy as np
# x = np.array([[1,2,5],[2,3,5],[3,4,5],[2,3,6]])
# # 输出数组的行和列数
# print x.shape  # (4, 3)
# # 只输出行数
# print x.shape[0] # 4
# # 只输出列数
# print x.shape[1] # 3
# # 将矩阵变成2行6列,确保总元素不变
# print x.reshape([2,6])
# # 4列，行数自动生成
# print x.reshape([-1,4])
# 大意是说，数组新的shape属性应该要与原来的配套，如果等于-1的话,
# 那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
# z.reshape(-1, 2) :newshape等于-1，列数等于2，行数未知
# z.reshape(-1, 1) :也就是说，先前我们不知道z的shape属性是多少，但是想让z变成只有一列，行数不知道多少
# z.reshape(-1)    :z.reshape(1,-1),一行，自动计算列

#矩阵转置T与transpose
# 1 .T,适用于一、二维数组
# arr = np.arange(15).reshape(3,5)#生成一个4行5列的数组
# print arr
# print arr.T #求转置
# 2 对于高维数组，transpose需要用到一个由轴编号组成的元组，才能进行转置。
# arr1 = np.arange(24).reshape(2,3,4)    #2个3*4的矩阵
# print arr1
# print arr1.shape #看形状
# #  形状	索引
# #   2	0
# #   2	1
# #   3	2
# # 所以说，transpose参数的真正意义在于这个shape元组的索引
# arr2 = arr1.transpose(( 0,2,1))
# print arr2
# print arr2.shape

# #零矩阵
# print np.zeros(9)
# print np.zeros([2,5])

# # append() 方法用于在列表末尾添加新的对象
# aList = [123, 'abc'];
# aList.append(2009);
# print "New List : ", aList;

# # zeros_like：返回和输入大小相同，类型相同，用0填满的数组
# a=np.array([2,3])
# print np.zeros_like(a)
# #二维
# a=np.array([[2,3],[3,4]])
# print np.zeros_like(a)

# import copy
# print copy.int2binary[4]  ??

# # round() 方法返回浮点数x的四舍五入值。
# # round( x [, n]  )
# print  round(80.25456, 1)
# print "round(100.000056, 3) : ", round(100.521656, 3)
# print "round(-100.000056, 3) : ", round(-100.521656, 3)
# print round(12.556)

# # abs() 函数返回数字的绝对值
# # abs( x )
# print  abs(-45)
# print  abs(-100.12)

# #矩阵乘法  dot()
# print  np.dot(7,3)

# b = a: 赋值引用，a 和 b 都指向同一个对象
# b = a.copy(): 浅拷贝, a 和 b 是一个独立的对象，但他们的子对象还是指向统一对象（是引用）
# b = copy.deepcopy(a): 深度拷贝, a 和 b 完全拷贝了父对象及其子对象，两者是完全独立的

#unpackbits-转化为二进制数
# a = np.array([[2],[7],[23]] ,dtype = np.uint8)
# print a
# b = np.unpackbits(a, axis = 1)
# print b      #[2]--[0 0 0 0 0 0 1 0]

# atleast_2d()作用
# print np.atleast_2d(3.0)   #array([[ 3.]])
# x = np.arange(3.0)
# print np.atleast_2d(x)     #array([[ 0.,  1.,  2.]])
# print np.atleast_2d(x).base is x  #True

# enumerate:枚举类型
# enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，
# 同时列出数据和数据下标，一般用在 for 循环当中。
# seasons = ['Spring', 'Summer', 'Fall', 'Winter']
# print list(enumerate(seasons))             # [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
# print list(enumerate(seasons, start=1))    # 小标从 1 开始  # [(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]
# # for循环
# seq = ['one', 'two', 'three']
# for i, element in enumerate(seq):
#       print i, seq[i]


# reversed 函数返回一个反转的迭代器。
#!/usr/bin/env python3
## 字符串
# seqString = 'Runoob'
# print(list(reversed(seqString)))
# print list(reversed(seqString))
# # 元组
# seqTuple = ('R', 'u', 'n', 'o', 'o', 'b')
# print(list(reversed(seqTuple)))
# print list(reversed(seqTuple))
# # range
# seqRange = range(5, 9)      #创建列表
# print(list(reversed(seqRange)))
# # 列表
# seqList = [1, 2, 4, 3, 5]
# print(list(reversed(seqList)))

# str(): 将对象转化为适于人阅读的形式
# s = 'RUNOOB'
# print str(s)      #'RUNOOB'
# dict = {'runoob': 'runoob.com', 'google': 'google.com','baidu':'baidu.com'};
# print str(dict)      #"{'google': 'google.com', 'runoob': 'runoob.com'}"

# 二进制和十进制互换int(),bin()
# last_mask_str = '11110000'
# last_mask_str = str(int(last_mask_str, 2))
# print last_mask_str   # 240
# print bin(int(last_mask_str))  # 0b11110000


# 用OneHotEncode编码把这些整数转化为二进制。每个特征用一个二进制数字来表示。例如，特征A分配的数值为7，
# 那么one-hot编码为它分配的二进制数字的第七位就是1，其余位为0.
# one hot编码的优点：
# 1.能够处理非连续型数值特征。
# 2.在一定程度上也扩充了特征。比如性别本身是一个特征，经过one hot编码以后，就变成了男或女两个特征。
# 注意：此处一定将训练特征和测试特征一起转化，因为转化之后数组的维度将会发生变化，有一个不转化，就会出错。


# os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
# 这个列表以字母顺序。 它不包括 '.' 和'..' 即使它在文件夹中。
# import os, sys
# # 打开文件
# path = '/home/chuwei/PycharmProjects/MNIST'
# dirs = os.listdir( path )
# # 输出所有文件和文件夹
# for file in dirs:
#    print file


# a=tf.constant([-1.0, 2.0, 3.0, 4.0])
# with tf.Session() as sess:
#     b=tf.nn.dropout(a, 0.5, noise_shape=[1,4])
#     print (sess.run(b))

#python3或者python2下运行
# a=tf.constant([[1.0,2.0],[1.0,2.0],[1.0,2.0]])
# sess=tf.Session()
# print (sess.run(tf.sigmoid(a)))


# cast(x, dtype, name=None)   将x的数据格式转化成dtype
# a = tf.Variable([1,0,1,1])
# b = tf.cast(a,dtype=tf.bool)
# sess = tf.Session()
# sess.run(tf.initialize_all_variables())
# print(sess.run(b))
#[ True False  True  True]


# tf.argmax(input, axis=None, name=None, dimension=None)
# 此函数是对矩阵按行或列计算最大值
# input：输入Tensor
# axis：0表示按列，1表示按行
# name：名称
# dimension：和axis功能一样，默认axis取值优先。新加的字段
# 返回：Tensor  一般是行或列的最大值下标向量(cc索引)
# b=tf.argmax(input=a,axis=0)

# tf.matmul(a, b, transpose_a=False, transpose_b=False, adjoint_a=False,
#           adjoint_b=False, a_is_sparse=False, b_is_sparse=False, name=None)
# 将矩阵a 乘于 矩阵b

# tf.Variable（initializer， name）：initializer是初始化参数，
# 可以有tf.random_normal，tf.constant，tf.constant等
# a1 = tf.Variable(tf.random_normal(shape=[2, 3], mean=0, stddev=1), name='a1')
# a2 = tf.Variable(tf.constant(1), name='a2')
# a3 = tf.Variable(tf.ones(shape=[2, 3]), name='a3')
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     print sess.run(a1)
#     print sess.run(a2)
#     print sess.run(a3)

# tf.random_normal使用方法
# tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
#     从正态分布中输出随机值
#     shape: 一维的张量，也是输出的张量。
#     mean: 正态分布的均值。
#     stddev: 正态分布的标准差。
#     dtype: 输出的类型。
#     seed: 一个整数，当设置之后，每次生成的随机数都一样。
#     name: 操作的名字。

# tf.truncated_normal使用方法
# tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
# 从截断的正态分布中输出随机值。
# 生成的值服从具有指定平均值和标准偏差的正态分布，如果生成的值大于平均值2个标准偏差的值则丢弃重新选择。
#     shape: 一维的张量，也是输出的张量。
#     mean: 正态分布的均值。
#     stddev: 正态分布的标准差。
#     dtype: 输出的类型。
#     seed: 一个整数，当设置之后，每次生成的随机数都一样。
#     name: 操作的名字

# tf.split(dimension, num_split, input)：dimension的意思就是输入张量的哪一个维度，
# 如果是0就表示对第0维度进行切割。num_split就是切割的数量，如果是2就表示输入张量被切成2份，
# 每一份是一个列表。
# A = [[1, 2, 3], [4, 5, 6]]
# x = tf.split(1, 3, A)
# with tf.Session() as sess:
#     c = sess.run(x)
#     for ele in c:
#         print ele

# 求最大值tf.reduce_max(input_tensor, reduction_indices=None, keep_dims=False, name=None)
# 求平均值tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None)
# 参数1--input_tensor:待求值的tensor。
# 参数2--reduction_indices:在哪一维上求解。
# 参数（3）（4）可忽略

# tf.equal(A, B):对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，
# 反正返回False，返回的值的矩阵维度和A是一样的

# 占位符placeholder和feed_dict:
# 创建了各种形式的常量和变量后，但TensorFlow 同样还支持占位符。占位符并没有初始值，
# 它只会分配必要的内存。在会话中，占位符可以使用feed_dict馈送数据。
# feed_dict是一个字典，在字典中需要给出每一个用到的占位符的取值。
# 在训练神经网络时需要每次提供一个批量的训练样本，如果每次迭代选取的数据要通过常量表示，
# 那么TensorFlow 的计算图会非常大。因为每增加一个常量，TensorFlow都会在计算图中增加一个结点。
# 所以说拥有几百万次迭代的神经网络会拥有极其庞大的计算图，而占位符却可以解决这一点，
# 它只会拥有占位符这一个结点。





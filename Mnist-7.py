#coding=utf-8
# 通过玩具代码一边学习一边调试能达到最好的学习效果。
# 本文通过一个简单的python实现，教会你循环神经网络。
# RNN-LSTM

import copy, numpy as np
np.random.seed(0)    #固定随机数生成器,seed( ) 用于指定随机数生成时所用算法开始的整数值


def sigmoid(x):  #激活函数
    output = 1/(1+np.exp(-x))
    return output

def sigmoid_output_to_derivative(output):  #激活函数的导数
    return output*(1-output)
#训练数据
int2binary = {}  #整数到其二进制的映射
binary_dim = 8   #256内的加法
#计算0-256的二进制表示
largest_number = pow(2,binary_dim)  #pow(x,y)表示（x的y次方） 2的8次方
binary = np.unpackbits(np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]
#输入变量
alpha = 0.1   #学习速率
input_dim = 2    #维度为2,输入2个数
hidden_dim = 16  #隐藏层的神经元节点数
output_dim = 1   #输出是一个数
#初始化权重
synapse_0 = 2*np.random.random((input_dim, hidden_dim))-1  #输入层到隐藏层的转化矩阵，维度为2*16， 2是输入维度，16是隐藏层维度
synapse_1 = 2*np.random.random((hidden_dim, output_dim))-1
synapse_h = 2*np.random.random((hidden_dim, hidden_dim))-1
#三个矩阵变化
synapse_0_update = np.zeros_like(synapse_0)  #synapse 突触
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

#训练,学习10000个例子
for j in range(10000):
    #随机产生两个0-128的数字,并查出他们的二进制表示
    a_int = np.random.randint(largest_number/2)   #randint:生成随机数
    a = int2binary[a_int]  #二进制编码
    b_int = np.random.randint(largest_number/2)
    b = int2binary[b_int]
    # print b
    # print b.shape

    c_int = a_int + b_int
    c = int2binary[c_int]
    # print c
    # print c.shape[0]   #矩阵的行数
    #存储神经网络的预测值
    d = np.zeros_like(c)   #zeros_like：返回和输入大小相同，类型相同，用0填满的数组
    overallError = 0       #总误差清零

    layer_2_deltas = list()  #存储每个时间的输出层的误差
    layer_1_values = list()  #存储每个时间的隐藏层的值
    layer_1_values.append(np.zeros(hidden_dim))  #开始没有隐含层,所以都是零  #append在列表未添加对象,zeros表示零矩阵
    # print  layer_1_values

    for position in range(binary_dim):

        x = np.array([a[binary_dim - position-1], b[binary_dim - position-1]])   #从右到左，每次去两个输入数字的一个bit位
        y = np.array([c[binary_dim - position-1]]).T    #正确答案
        # hidden layer (新的隐藏层)
        layer_1 = sigmoid(np.dot(x, synapse_0) + np.dot(layer_1_values[-1], synapse_h)) #（输入层 + 之前的隐藏层）-> 新的隐藏层，这是体现循环神经网络的最核心的地方
        # output layer (新的输出层)
        layer_2 = sigmoid(np.dot(layer_1, synapse_1))   #隐藏层 * 隐藏层到输出层的转化矩阵synapse_1 -> 输出层

        layer_2_error = y - layer_2  #预侧误差
        layer_2_deltas.append((layer_2_error) * sigmoid_output_to_derivative(layer_2)) #记录每个时间点的误差导数
        overallError += np.abs(layer_2_error[0])  #总误差

        d[binary_dim - position-1] = np.round(layer_2[0][0])  #记录每一个预测bit位  ??

        layer_1_values.append(copy.deepcopy(layer_1))  #记录隐藏层的值

    future_layer_1_delta = np.zeros(hidden_dim)    #delta:变量增量

        # 前面代码我们完成了所有时间点的正向传播以及计算最后一层的误差，现在我们要做的是反向传播，从最后一个时间点到第一个时间点














# coding=utf-8
# https://www.cnblogs.com/zhang-yd/p/7850344.html
# mnist的数据预处理
# mnist数据集在现在的image classification起的影响越来越小的。因为其数据量小，类别少，分类简单，
# 一直没法能够作为算法比较的有效对比数据集。但是这个算法在debug 的时候还是有着很重要的角色。

import numpy as np
import struct

from PIL import Image
import os


def train():
    data_file = './train-images.idx3-ubyte'
    data_file_size = 47040016
    data_file_size = str(data_file_size - 16) + 'B'

    data_buf = open(data_file, 'rb').read()

    magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', data_buf, 0)

    datas = struct.unpack_from('>' + data_file_size, data_buf, struct.calcsize('>IIII'))

    datas = np.array(datas).astype(np.uint8).reshape(numImages, 1, numRows, numColumns)

    label_file = '/home/chuwei/PycharmProjects/MNIST/train-labels.idx1-ubyte'

    label_file_size = 60008
    label_file_size = str(label_file_size - 8) + 'B'

    label_buf = open(label_file, 'rb').read()

    magic, numLabels = struct.unpack_from('>II', label_buf, 0)
    labels = struct.unpack_from('>' + label_file_size, label_buf, struct.calcsize('>II'))
    labels = np.array(labels).astype(np.int64)

    datas_root = './mnist_train'
    if not os.path.exists(datas_root):
        os.mkdir(datas_root)

    for i in range(10):
        file_name = datas_root + os.sep + str(i)
        if not os.path.exists(file_name):
            os.mkdir(file_name)

    train_x = []
    train_y = []

    for ii in range(numLabels):
        img = Image.fromarray(datas[ii, 0, 0:28, 0:28])
        label = labels[ii]
        file_name = datas_root + os.sep + str(label) + os.sep + 'mnist_train_' + str(ii) + '.png'
        img.save(file_name)
        train_x.append(file_name)
        train_y.append(label)

    with open('./mnist_train.txt', 'w') as f:
        for i in range(len(train_x)):
            f.write(str(train_x[i]) + '\t' + str(train_y[i]) + '\n')
    print('Done')


def test():
    data_file = './t10k-images.idx3-ubyte'

    data_file_size = 7840016
    data_file_size = str(data_file_size - 16) + 'B'

    data_buf = open(data_file, 'rb').read()

    magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', data_buf, 0)
    datas = struct.unpack_from('>' + data_file_size, data_buf, struct.calcsize('>IIII'))
    datas = np.array(datas).astype(np.uint8).reshape(numImages, 1, numRows, numColumns)

    label_file = './t10k-labels.idx1-ubyte'

    label_file_size = 10008
    label_file_size = str(label_file_size - 8) + 'B'

    label_buf = open(label_file, 'rb').read()

    magic, numLabels = struct.unpack_from('>II', label_buf, 0)
    labels = struct.unpack_from('>' + label_file_size, label_buf, struct.calcsize('>II'))
    labels = np.array(labels).astype(np.int64)

    datas_root = './mnist_test'

    if not os.path.exists(datas_root):
        os.mkdir(datas_root)

    for i in range(10):
        file_name = datas_root + os.sep + str(i)
        if not os.path.exists(file_name):
            os.mkdir(file_name)

    test_x, test_y = [], []

    for ii in range(numLabels):
        img = Image.fromarray(datas[ii, 0, 0:28, 0:28])
        label = labels[ii]
        file_name = datas_root + os.sep + str(label) + os.sep + 'mnist_test_' + str(ii) + '.png'
        img.save(file_name)
        test_x.append(file_name)
        test_y.append(label)

    with open('./mnist_test.txt', 'w') as f:
        for i in range(len(test_x)):
            f.write(str(test_x[i]) + '\t' + str(test_y[i]) + '\n')

    print('Done')


if __name__ == '__main__':
    train()
    test()
    print('Done')
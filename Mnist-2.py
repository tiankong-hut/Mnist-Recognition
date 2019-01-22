# coding=utf-8
# https://www.cnblogs.com/x1957/archive/2012/06/02/2531503.html
#读取mnist图片的方法
'''
import numpy as np
import struct
import matplotlib.pyplot as plt  #matplotlib在python3中安装了
#使用二进制方式读取文件
filename = 'train-images.idx3-ubyte'
binfile  = open(filename,'rb')
buf = binfile.read()
#使用struc.unpack_from
#Python的值根据格式符转换为字符串:struct.unpack(fmt, string)
index = 0
magic, numImages, numRows, numColumns = \
    struct.unpack_from('>IIII', buf, index)   #'>IIII'是说使用大端法读取4个unsinged int32
index += struct.calcsize('>IIII')

# 测试是否读取成功
im = struct.unpack_from('>784B',buf,index)  # '>784B'的意思就是用大端法读取784个unsigned byte
index += struct.calcsize('>784B')      # 计算格式字符串所对应的结果的长度

im = np.array(im)             #array()排序
im = im.reshape(28,28)        #reshape重新调整矩阵的行数、列数、维数

fig = plt.figure()
plotwindow = fig.add_subplot(221)  #参数111的意思是：分割成1行1列，在从左到右从上到下的第1块
plt.imshow(im,cmap='gray')         #或fig.add_subplot(1,1,1)
plt.show()
'''

# http://blog.csdn.net/opipa/article/details/51882875
#标签文件的读取方法：
import struct
from array import array
with open("train-labels.idex1-ubyte","rb") as f:
    magic, size = struct.unpack(">II", f.read(8))
    labels = array("B", f.read())
    print magic, size, labels

# 图片文件的读取方法（ipython下）

import struct
from array import array

with open("t10k-images.idx3-ubyte","rb") as f:
    magic, size, rows, cols = struct.unpack(">IIII", f.read(16))  #unpack(fmt, string)
    print magic, size, rows, cols
    image_data = array("B", f.read())
    images = []
    for i in range(size):
        images.append([0]*rows*cols)

    for i in range(size):
        images[i][:] = image_data[i*rows*cols:(i+1)*rows*cols]

# 显示前72幅图片：
import numpy as np
from PIL import Image             #PIL用于图像操作
import  matplotlib.pyplot as plt
# matlotlib inline

for i, img in enumerate(images):
    if i < 72:
        plt.subplot(9,8,i+1)  #9行8列
        img = np.array(img)
        img = img.reshape(rows, cols)
        img = Image.fromarray(img)        # ????有问题
        plt.imshow(img, cmap='gray')
        plt.axis("off")
    else:
        break



# import numpy as np
# from PIL import Image
#
# img = Image.new('L', (100, 28))
# img.putpixel((5, 3), 17)
#
# matrix = np.array(img)
# # img = Image.fromarray(img)
# print matrix[5, 3] #This returns 0
#
# print matrix[3, 5] #This returns 17
#
# matrix = matrix.transpose()
# print matrix[5, 3] #This returns 17








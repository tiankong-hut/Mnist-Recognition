#coding=utf-8
# http://blog.csdn.net/u010194274/article/details/50817999
# train_image的过程
# 使用转换mnist数据库保存为bmp图片

import struct
import numpy as np
import  matplotlib.pyplot as plt
import Image
import os

#二进制的形式读入
filename='train-labels.idx1-ubyte'
binfile=open(filename,'rb')
buf=binfile.read()
#大端法读入4个unsigned int32
#struct用法参见网站 http://www.cnblogs.com/gala/archive/2011/09/22/2184801.html
index=0
magic,numImages,numRows,numColumns=struct.unpack_from('>IIII',buf,index)
print "magic,numImages,numRows,numColumns:\n", magic,numImages,numRows,numColumns
# 2051 60000 28 28
index+=struct.calcsize('>IIII')

#创建存放图片的文件夹
path = 'Train-labels'
Is_exist = os.path.exists(path)
if not Is_exist:
    # 如果不存在则创建目录
    os.makedirs(path)

#将每张图片按照格式存储到对应位置
for image in range(0,200):
    im=struct.unpack_from('>784B',buf,index)
    index+=struct.calcsize('>784B')
   #这里注意 Image对象的dtype是uint8，需要转换
    im=np.array(im,dtype='uint8')
    im=im.reshape(28,28)

    im = Image.fromarray(im)
    im.save('Train-labels/train-label_%s.bmp' % image)
    # im.save('path:/train_%s.bmp' % image, 'bmp')
    # print "im:",im

   # fig=plt.figure()
   # plotwindow=fig.add_subplot(111)
   # plt.imshow(im,cmap='gray')
   # plt.show()





# coding=UTF-8

# from sklearn import preprocessing
# enc = preprocessing.OneHotEncoder()
# print enc
# OneHotEncoder(categorical_features=all, dtype=<type 'float'>, n_values=auto)
# enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
# enc.transform([[0, 1, 3]]).toarray()
# 输出结果：array([[ 1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.]])


# http://blog.csdn.net/a595130080/article/details/64442800
#数字0-9 one-hot编码
# import numpy as np
# labels = np.arange(0,10)
# one_hot_labels = []
# for num in labels:
#     print num
#     one_hot = [0] * 10     #全变为零,总共10位
#     # print one_hot
#     one_hot[num] = 1
#     print one_hot

    # if num == 0:
    #    one_hot[0] = 1.0
    # else:
    #    one_hot[num] = 1.0
    # one_hot_labels.append(one_hot)
    # print one_hot_labels


# http://blog.csdn.net/vange/article/details/5395771
# http://blog.csdn.net/gzlaiyonghao/article/details/1852726
# 图像处理库是 PIL(Python Image Library)
# Python下利用PIL实现可设定阈值的二值图像转换
import Image
# load a color image   加载一个彩色图像
im = Image.open("picture_2.jpg")
# convert to grey level image
Lim = im.convert("L")         # L表示灰度，1表示二值图模式
Lim.save("pic_2_gray.jpg")

# setup a converting table with constant threshold  设置一个具有恒定阈值的转换表
threshold = 130
table = []
for i in range( 256 ):
      if i < threshold:
         table.append(0)      # append() 方法用于在列表末尾添加新的对象。
      else:
         table.append(1)

 #  convert to binary image by the table  转换为二进制图像的表格
bim = Lim.point(table, "1")     # L表示灰度，1表示二值图模式
bim.save("pic_2_binary.jpg")

# convert函数将灰度图转换为二值图时，是采用固定的阈值127来实现的，即灰度高于127的像素值为1，而灰度低于127的像素值为0。
# 为了能够通过自定义的阈值实现灰度图到二值图的转换，就要用到Image.point函数,用到Image.point(table, mode)。






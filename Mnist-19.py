#coding=utf-8
# http://www.jb51.net/article/102981.htm

# 在 python 中除了用 opencv，也可以用 matplotlib 和 PIL 这两个库操作图片。
# 本人偏爱 matpoltlib，因为它的语法更像 matlab。
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
import numpy as np

# 1. 显示图片
lena = mpimg.imread('/home/chuwei/PycharmProjects/Neural network/figure_2.png')
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
print lena.shape  # (512, 512, 3)
plt.imshow(lena)  # 显示图片
plt.axis('off')   # 不显示坐标轴
plt.show()

# 3. 将 RGB 转为灰度图
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
gray = rgb2gray(lena)
# 也可以用 plt.imshow(gray, cmap = plt.get_cmap('gray'))
plt.imshow(gray, cmap='Greys_r')
plt.axis('off')
plt.show()



# https://www.cnblogs.com/denny402/p/5121897.html
# 从外部读取图片并显示
# from skimage import io
# img=io.imread('home/chuwei/PycharmProjects/Neural network/figure_2.png')
# io.imshow(img)
# 读取单张灰度图片
# from skimage import io
# img=io.imread('d:/dog.jpg',as_grey=True)
# io.imshow(img)







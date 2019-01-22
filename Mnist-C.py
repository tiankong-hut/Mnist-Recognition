# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt


# x = np.linspace(0, 3, 100)    #其中x1、x2、N分别为起始值、终止值、元素个数
# plt.plot(x, np.exp(x/3))
# plt.xlabel('Time(s)')
# plt.ylabel("Volt")
# plt.title("Python")
# plt.xlabel('循环次数')   #中文不能显示
# plt.ylabel("准确率")
# plt.title("准确率")
# plt.show()


# plt.figure(1)  # 创建图表1
# plt.figure(2)  # 创建图表2
# ax1 = plt.subplot(211)  # 在图表2中创建子图1
# ax2 = plt.subplot(212)  # 在图表2中创建子图2
# x = np.linspace(0, 3, 100)
# for i in xrange(5):
#     plt.figure(1)  # ❶ # 选择图表1
#     plt.plot(x, np.exp(i * x / 3))
#     plt.sca(ax1)  # ❷ # 选择图表2的子图1
#     plt.plot(x, np.sin(i * x))
#     plt.sca(ax2)  # 选择图表2的子图2
#     plt.plot(x, np.cos(i * x))
# plt.show()


# x=np.linspace(0,10,100)
# y=np.sin(x)
# y1=np.cos(x)
# plt.figure(figsize=(8,4)) #整个现实图（框架）的大小
# plt.plot(x,y,'r-o',label="$sin(x)$",linewidth=1)
# plt.plot(x,y1,'b-o',label="$cose(x)$",linewidth=1)
# plt.xlabel('Time(s)')
# plt.ylabel("Volt")
# plt.title("Python chart")
# plt.show()


list = []   #在循环之前先创建好列表
# for i in range(1,11):
a = 1 #获得你要的数
list.append(a)#添加到列表中
a = 6  # 获得你要的数
list.append(a)  # 添加到列表中
print list

# list1 = ['physics', 'chemistry', 1997, 2000];
# list2 = [1, 2, 3, 4, 5, 6, 7 ];
# print "list1[0]: ", list1[0]
# print "list2[1:5]: ", list2[1:5]
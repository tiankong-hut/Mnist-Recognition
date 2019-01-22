# -*- coding:utf-8 -*-
# http://blog.csdn.net/xiaopangxia/article/details/59177385
# SVM对sklearn自带手写数字数据集进行分类
# sklearn自带一些数据集，其中手写数字数据集可通过load_digits加载


# from sklearn.datasets import load_digits  # 加载手写数字识别数据
# import pylab as pl
# digits = load_digits()
# # 数据纬度，1797幅图，8*8
# # 显示一副图片
# pl.gray()
# pl.matshow(digits.images[0])
# pl.show()


import sys
from sklearn.datasets import load_digits  # 加载手写数字识别数据
import pylab as pl
from sklearn.cross_validation import train_test_split  # 训练测试数据分割
from sklearn.preprocessing import StandardScaler  # 标准化工具
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report  # 预测结果分析工具

print load_digits()           #打印出矩阵

reload(sys)
sys.setdefaultencoding('utf-8')

digits = load_digits()
# 数据纬度，1797幅图，8*8
print digits.data.shape        #打印出行数和列数!

# 分割数据
X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=33)

ss = StandardScaler()
# fit是实例方法，必须由实例调用
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

lsvc = LinearSVC()
lsvc.fit(X_train, Y_train)

Y_predict = lsvc.predict(X_test)

print classification_report(Y_test, Y_predict, target_names=digits.target_names.astype(str))


# 需要记住几个性能指标的含义
# precision为精确率，是真阳结果与所有被判为阳性结果的比。
# recall为召回率， 是真阳结果与所有实际为阳性数据的比。
# 真阳者，实际为真且判为真。
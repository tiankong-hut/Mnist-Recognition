#coding=UTF-8
# https://www.cnblogs.com/lutingting/p/5159193.html
# Python如何读取指定文件夹下的所有图像

import numpy as np
import  os
# import Image
from PIL import Image

def load_Img(imgDir,imgFoldName):
    imgs = os.listdir(imgDir+imgFoldName)
    imgNum = len(imgs)
    data = np.empty((imgNum,1,12,12),dtype="float32")
    label = np.empty((imgNum,),dtype="uint8")
    for i in range (imgNum):
        img = Image.open(imgDir+imgFoldName+"/"+imgs[i])
        arr = np.asarray(img,dtype="float32")
        data[i,:,:,:] = arr
        label[i] = int(imgs[i].split('.')[0])
    return data,label      #这里得到的data和label都是ndarray数据

#调用方式
craterDir = "/home/chuwei/PycharmProjects/MNIST/"
foldName = "Picture"
data, label = load_Img(craterDir,foldName)


'''
  Load the image files form the folder
  input:
      imgDir: the direction of the folder
      imgName:the name of the folder
  output:
     data:the data of the dataset
     label:the label of the datset
'''
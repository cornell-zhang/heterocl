from __future__ import print_function
#import keras
#from keras.datasets import cifar100
#from keras.preprocessing.image import ImageDataGenerator
#from keras.models import Model, Sequential
#from keras.layers import Dense, Dropout, Activation, Flatten, Input
#from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import random
import cv2
import matplotlib.pyplot as plot
import os
import tarfile
import torchvision
# set your path to the data set
#root = ".../imagenet_2012/images/val"
val_tag = []
val_img = []
j=0;
#input required number of samples per class
samples_per_class = 1
#input required input size of image
input_shape = (299,299)
folder_list = os.listdir(root)
for i in range(len(folder_list)):
    folder_path = root + "/" + folder_list[i]
    img_list = os.listdir(folder_path)
    for img_name in img_list:
        j+=1
        print("{} out of {} \r".format(j,samples_per_class*1000),end="")
        img_path = folder_path+"/"+img_name
        file=cv2.imread(img_path)
        file=cv2.resize(file,input_shape)
        file=cv2.cvtColor(file, cv2.COLOR_BGR2RGB)
        file=np.array(file).reshape((3,input_shape[0],input_shape[1]))
        val_img.append(file)
        val_tag.append(i)
        if(j%samples_per_class==0):
            break
# set output path of numpy files
#dest = "...imagenet_numpy/images/{val,train}"
np.save(dest+"x_test_xception.npy",val_img)
np.save(dest+"y_test_xception.npy",val_tag)

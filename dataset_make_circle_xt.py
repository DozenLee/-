# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np
import glob
import random
import cv2
import matplotlib.pyplot as plt
import math
def f():
    datasets_img= []
    datasets_label= []
    for j in range(5000):#10000
    #for i in range(1000):
        i=j+5000
        img = cv2.imread("./new_data/img_withcirclenoise/image%05d.PNG" %(i))#circle noise
        #img = cv2.imread("./new_data/img_withpixelnoise/image%05d.PNG" %(i))#circle noise
        img = cv2.resize(img,(32,32))
        
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        x = np.array(img,dtype=np.float32)
        x /=255
        
        x = x.reshape(1,32,32)
        datasets_img.append(x)
        
        #with open('./data/test.txt','r') as x:
        with open('./new_data/train_circle_XYj.txt','r') as x:
        #with open('./new_data/train_pixel_XYj.txt','r') as x:
            labels =x.readlines()
            t = labels[i]
            t = t.split()
            t = t[0:2] #teacher:xy
            #t = t[2] #teacher: angel
            #t = math.radians(t)
            t = np.array(t,dtype=np.float32)#teacher:xy
            #t = np.array(([math.cos(t),math.sin(t),dtype=np.float32)
            #t /= 360
            t /= 256
            #print t
            #print t.shape
            #print type(t)
        #print t
        datasets_label.append(t)
    #random.shuffle(datasets_)
    train_X = datasets_img
    train_Y = datasets_label
    #test = datasets[9:10]
    #test = datasets[7000:8317]
    #return train,test
    #print train
    return train_X,train_Y
    
    
    
    
#train_X,train_Y = f()
#print type(train_X)
#print train_X





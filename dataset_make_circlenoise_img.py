# -*- coding: utf-8 -*-
from numpy import *
from scipy import * 
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import os


for w in range(2):
   
    img_circle=np.zeros((256,256,3),np.uint8)+255

                
    j = random.randint(0,360)#0-360度転動
    
    X = random.randint(40,216)
    Y = random.randint(40,216)
    

    for k in range(10):
        x = random.randint(10,246)
        y = random.randint(10,246)
        img1_circle = cv2.circle(img_circle,(x,y),10,(random.randint(0,255),random.randint(0,255),random.randint(0,255)),-1)
 
         
        #k = k+1
    cv2.imshow("1",img1_circle)
    cv2.waitKey(0) 
    img1_maru=cv2.ellipse(img1_circle, (X,Y), (40, 30), j, 0, 180, (255,0,0), -1) #画椭圆
    img2_maru=cv2.ellipse(img1_maru, (X,Y), (40, 30), j, 180, 360, (0,255,0),-1)
    img3_maru=cv2.ellipse(img2_maru, (X , Y), (20, 15), j, 0, 360, (0,0,255),-1)
    img_withcirclenoise= cv2.cvtColor(img3_maru, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("./new/image%05d.PNG" %(w),img_withcirclenoise)
       
 
    """
    f = open("./new_data/train_pixel_XYj.txt",'a')
    #f = open("./test_XYj.txt",'a')
    f.writelines(str(X) + " " + str(Y) + " " + str(j) + '\n')
    f.close
    print (w)
    """
    w = w + 1
        
        
        
      

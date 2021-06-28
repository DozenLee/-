# -*- coding: utf-8 -*-
import datetime
import numpy as np
import matplotlib
import matplotlib.pylab as plt
from sklearn import model_selection, metrics
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import dataset_make_combine_xt_angel
import dataset_make_circle_xt_angel
import dataset_make_rect_xt_angel
from chainer import Link, Chain, ChainList, cuda, optimizers, serializers
import copy
import train_xy_weight_angel
from scipy import * 
import cv2
import random
import os
from PIL import Image
#--------------------------------------------------------#
imshow = 0 #  1:show img   0: caculate mean loss
use_newnoise_img = 0   # 2:use all combine noise 1:use all rect noise img   0:use 0.2 test_x
use_input_I = 1   #  1:predict by using I   0:predict by using I_ 
#------------------------------------------------------#

model = train_xy_weight_angel.JoinedModel()

#------------------------------------------------------#

gpu = 0#use  gpu
cuda.check_cuda_available()#
xp = cuda.cupy
#-----------------------------------------------------------#
#データセットを読み込む
#-------------------------------------------------------------#

#train_X,train_Y=dataset_make_pixel_xt.f() # input pixelnoise img
#
if use_newnoise_img == 0:
    train_X,train_Y=dataset_make_circle_xt_angel.f() # input circlenoise img
    train_x, test_x, train_y, test_y = model_selection.train_test_split(train_X, train_Y, test_size=0.2)
    train_x = np.array(train_x,dtype=np.float32)
    test_x = np.array(test_x,dtype=np.float32)
    train_y = np.array(train_y,dtype=np.float32)
    test_y = np.array(test_y,dtype=np.float32)

    train_x = chainer.cuda.to_gpu(train_x)
    test_x = chainer.cuda.to_gpu(test_x)
    train_y = chainer.cuda.to_gpu(train_y)
    
    test_y = chainer.cuda.to_gpu(test_y)
if use_newnoise_img == 1 :
    train_X,train_Y=dataset_make_rect_xt_angel.f()# input rectnoise img
    test_x = np.array(train_X,dtype=np.float32)
    test_y = np.array(train_Y,dtype=np.float32)
    
    test_x = chainer.cuda.to_gpu(test_x)
    test_y = chainer.cuda.to_gpu(test_y)
    
if use_newnoise_img == 2 :
    train_X,train_Y=dataset_make_combine_xt_angel.f()# input combinenoise img
    test_x = np.array(train_X,dtype=np.float32)
    test_y = np.array(train_Y,dtype=np.float32)
    
    test_x = chainer.cuda.to_gpu(test_x)
    test_y = chainer.cuda.to_gpu(test_y)                        
    
#print test_x.shape
#-----------------------------------------------------------#
if gpu >= 0:
        chainer.cuda.get_device(gpu).use()
        model.to_gpu(gpu)



serializers.load_npz("./load_model/joinedmodel_npz_model0300_final",model)





loss = xp.zeros(())
total_loss = np.zeros(())

loss_wrong_len = []

for index in range(len(test_x)):    
    

    y_weight = model.model_weight(x=test_x,train=False)

    
    y_weight = y_weight[index].data
    
    
    y_weight = chainer.cuda.to_cpu(y_weight)
    
    
    
    
    
    img_y_weight =y_weight.reshape(32,32) 
    
#--------------------------------------------------------#
    x_ = (test_x[index])
    x_ = x_.reshape(1,1,32,32)
    if use_input_I == 1:
        y_xy = model.model_xy(x = x_,t = None,train = False)  #  Iを用いてＸＹを予測
    
    x_ = chainer.cuda.to_cpu(x_)
    
    img_x = x_.reshape(32,32)
    
    
#--------------------------------------------------------#
#make N,N_,I_ 
#---------------------------------------------------    
    I = img_x*255
    I_ =np.zeros((32,32),dtype = np.float32)
    R = img_y_weight
      
    N = np.random.uniform(0,255,(32,32))
    N = np.array(N,dtype=np.float32)
    
    #prepare for new noise img     
    N_ = np.zeros((32,32),dtype = np.float32)
    #N_ = np.zeros((32,32),dtype = np.float32)+255#  I + 255*R
    
    R_ = 0
    #print R
    #R_max = R.max()
    
    lenth_0 = []
 
    for Ri in range(32):
        for Rj in range(32):
            if R[Ri][Rj] < 0.2:
                lenth_0.append(R[Ri][Rj])
            R_ = (R[Ri][Rj]) 
           
            N_[Ri][Rj] = (N[Ri][Rj]) * R_
            I_[Ri][Rj] = (I[Ri][Rj] + N_[Ri][Rj])
            #I_[Ri][Rj] = (I[Ri][Rj] + N_[Ri][Rj]*R_)
            if I_[Ri][Rj]  > 255:
                I_[Ri][Rj] = 255
           
    img_I_ = I_        
            
            
    #print len(lenth_0)
    
    I_  = np.array(I_,dtype=np.float32)
    #print I_ 
    I_ = I_.reshape(1,1,32,32)
    I_  = chainer.cuda.to_gpu(I_)
    
    
    if use_input_I == 0:
        y_xy = model.model_xy(x = I_/255,t = None,train = False) #  I_を用いてＸＹを予測
    
    y_xy = y_xy[0].data
   
    y_xy = chainer.cuda.to_cpu(y_xy) 
    #print y_xy      
#--------------------------------------------------------#
#draw xy img
#--------------------------------------------------- 
    img=np.zeros((32,32,3),np.uint8)+255#设置背
 #------------------------------------#
# cosA_t = t[0] sinA_t =t[1]
#----------------------------------------#    
    
    t = test_y[index]#正解の座標を読み込む
    cosA_t = t[0]
    sinA_t = t[1]
    t_list=t.tolist()
    y_list=y_xy.tolist()
    print(t_list[1],y_list[1])
    """
    o_rad = math.atan2(sinA_t,cosA_t)
    if o_rad > 0:
        o_deg = math.degrees(o_rad)
    if o_rad < 0:
        o_deg = math.degrees(o_rad) + 360
    """
    #print o_deg
    
    if cosA_t >= 0 and sinA_t >= 0:
        A_rad_t = math.asin(sinA_t)
        A_deg_t = math.degrees(A_rad_t) 
    if cosA_t <= 0 and sinA_t >= 0:
        A_rad_t = math.pi - math.asin(sinA_t)
        A_deg_t = math.degrees(A_rad_t)
    if cosA_t <= 0 and sinA_t <= 0:
        A_rad_t = math.pi - math.asin(sinA_t)
        A_deg_t = math.degrees(A_rad_t)
    if cosA_t >= 0 and sinA_t <= 0:   
        A_rad_t = math.asin(sinA_t)+2*math.pi
        A_deg_t = math.degrees(A_rad_t)
        
    #print (A_deg_t)        
#------------------------------------#
# cosA_y = y[0] sinA_y =y[1]
#----------------------------------------#    
    cosA_y = y_xy[0]
    sinA_y = y_xy[1]
    
    #print (cosA_t,sinA_t)
    #print (cosA_y,sinA_y)
    
    if cosA_y >1:
        cosA_y = 1
    if cosA_y <-1:
        cosA_y = -1
    if sinA_y > 1:
        sinA_y = 1
    if sinA_y <-1:
        sinA_y = -1    
    
    
    
    if cosA_y >= 0 and sinA_y >= 0:
        A_rad_y = math.asin(sinA_y)
        A_deg_y = math.degrees(A_rad_y) 
    if cosA_y <= 0 and sinA_y >= 0:
        A_rad_y = math.pi - math.asin(sinA_y)
        A_deg_y = math.degrees(A_rad_y)
    if cosA_y <= 0 and sinA_y <= 0:
        A_rad_y = math.pi - math.asin(sinA_y)
        A_deg_y = math.degrees(A_rad_y)
    if cosA_y >= 0 and sinA_y <= 0:   
        A_rad_y = math.asin(sinA_y)+2*math.pi
        A_deg_y = math.degrees(A_rad_y)
        
    
    
    
   
#元画像の楕円中心を対応する必要があれば：        
    if use_newnoise_img == 1:    
        with open('./new_data/train_rect_XYj.txt','r') as k:
                labels =k.readlines()
                j = labels[index]
                j = j.split()
                j = j[0:2]
                j = np.array(j,dtype=np.float32)
                j_X = int((j[0])/8)  # 元画像の楕円中心　Ｘ座標 
                J_Y = int((j[1])/8)# 元画像の楕円中心　Ｙ座標 
                
            
    
    A_deg_t_int = int(A_deg_t)
    A_deg_y_int = int(A_deg_y)
#中心を（１６，１６）に固定する       
    img_t_xy = cv2.ellipse(img, (16, 16), (12, 6), A_deg_t_int, 0, 180, (255,0,0), -1)
    img_t_xy= cv2.ellipse(img_t_xy, (16, 16), (12, 6), A_deg_t_int, 180, 360, (0,255,0), -1)
    img_y_xy = cv2.ellipse(img_t_xy, (16, 16), (8, 4), A_deg_y_int, 0, 180, (100,0,0), -1)
    img_y_xy = cv2.ellipse(img_y_xy, (16, 16), (8, 4), A_deg_y_int, 180, 360, (0,100,0), -1)           
   
#--------------------------------------------#
#imshow
#-------------------------------------------#
    img_I = np.array(img_x*255,dtype = np.uint8)#ori img I
  
    #print R
    img_weight = np.array(R*255,dtype = np.uint8)#weight img  R(x,y) = R(I)
    
    
    img_N = N
    
    img_N = np.array(img_N,dtype = np.uint8)
   
    img_N_ = np.array(N_,dtype = np.uint8)#new noise  N_ = N * R(x,y)
    
    
    
    I_ = chainer.cuda.to_cpu(I_)
    I_ = I_.reshape(32,32)
    
    
    
    #img_I_ = np.array(((I_+255)/765)*255,dtype = np.uint8)#ori + new noise    I_ = I + N_
    img_I_ = np.array(I_,dtype = np.uint8)
    
    
    #print img_x
    #print img_y
    img_I = cv2.resize(img_I,(240,240),0,0,cv2.INTER_NEAREST)# I
    
    img_weight = cv2.resize(img_weight,(240,240),0,0,cv2.INTER_NEAREST)
    
    img_N = cv2.resize(img_N,(240,240),0,0,cv2.INTER_NEAREST)
    img_N_ = cv2.resize(img_N_,(240,240),0,0,cv2.INTER_NEAREST)#
    
    img_I_ = cv2.resize(img_I_,(240,240),0,0,cv2.INTER_NEAREST)
    
    img_y_xy = cv2.resize(img_y_xy,(240,240),0,0,cv2.INTER_NEAREST) # predict xy
    
    
    if imshow == 0:
       
        if abs(A_deg_t-A_deg_y) >= 180:
            abs_ty = 360-(abs(A_deg_t-A_deg_y))
            T = np.array((abs_ty),dtype = np.float32)
            Y = np.array((0),dtype = np.float32)
            print ("!",A_deg_t,A_deg_y)
        else :    
            T = np.array((A_deg_t),dtype = np.float32)
            Y = np.array((A_deg_y),dtype = np.float32)
            print (A_deg_t,A_deg_y)
        loss = F.mean_squared_error(T,Y)    
        """
        #t = np.array(([cosA_t,sinA_t]),dtype = np.float32)
        #y_xy = np.array(([cosA_y,sinA_y]),dtype = np.float32)    
        loss = F.mean_squared_error(t,y_xy)
        """
        
        loss = loss.data
        #print loss
        total_loss += loss
        
        if loss >400:
            loss_wrong_len.append(loss)
        
        if loss >400:
            
            
            cv2.imwrite("./pict/angel_I_image%02d.PNG" %(index),img_I)
            cv2.imwrite("./pict/angel_weight_image%02d.PNG" %(index),img_weight)
            cv2.imwrite("./pict/angel_F_image%02d.PNG" %(index),img_y_xy)
            
            
            cv2.imshow("I",img_I)
            cv2.imshow("weight_R",img_weight)
            cv2.imshow("N",img_N)
            cv2.imshow("N_",img_N_)
            cv2.imshow("newimg_I_",img_I_)
            cv2.imshow("predict_xy",img_y_xy)
        
    
        
            x= 200
            y = 170
            cv2.moveWindow("I",x,y+30)
            cv2.moveWindow("weight_R",x,y+330)
            cv2.moveWindow("N",x+300,y+30)
            cv2.moveWindow("N_",x+600,y+30)
            cv2.moveWindow("newimg_I_",x+600,y+330)
            cv2.moveWindow("predict_xy",x+600,y+630)
        
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    if imshow == 1:
    
        cv2.imshow("I",img_I)
        cv2.imshow("weight_R",img_weight)
        cv2.imshow("N",img_N)
        cv2.imshow("N_",img_N_)
        cv2.imshow("newimg_I_",img_I_)
        cv2.imshow("predict_xy",img_y_xy)
    
    
    
        x= 200
        y = 170
        cv2.moveWindow("I",x,y+30)
        cv2.moveWindow("weight_R",x,y+330)
        cv2.moveWindow("N",x+300,y+30)
        cv2.moveWindow("N_",x+600,y+30)
        cv2.moveWindow("newimg_I_",x+600,y+330)
        cv2.moveWindow("predict_xy",x+600,y+630)
    
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if imshow == 0:
        print index
if 1:
    print len(loss_wrong_len)
    print total_loss/len(test_x)        
   
"""
  with open('./new_data/train_circle_XYj.txt','r') as x:#use circlenoise img
    #with open('./new_data/train_rect_XYj.txt','r') as x:#use rectnoise img
        labels =x.readlines()
        j = labels[index]
        j = j.split()
        j = j[0:2]
        j = np.array(j,dtype=np.float32)
    #print j
    X_t = int((j[0])/8)
    Y_t = int((j[1])/8)
        
    X_y = int((j[0])/8)
    Y_y = int((j[1])/8)
    
"""


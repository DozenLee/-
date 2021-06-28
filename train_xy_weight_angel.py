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
import dataset_make_circle_xt_angel
from chainer import Link, Chain, ChainList, cuda, optimizers, serializers
import copy
import cv2



#------------------------------------------------------#
gpu = 0#cpu or gpu
cuda.check_cuda_available()#add by zhao v2
xp = cuda.cupy
#------------------------------------------------------=

#-----------------------------------------------------------#
#データセットを読み込む
#-------------------------------------------------------------#
#train_X,train_Y=dataset_make_circle_xt.f()# teacher:xy
train_X,train_Y=dataset_make_circle_xt_angel.f()# teacher : angel

train_x, test_x, train_y, test_y = model_selection.train_test_split(train_X, train_Y, test_size=0.2)
train_x = np.array(train_x,dtype=np.float32)
test_x = np.array(test_x,dtype=np.float32)
train_y = np.array(train_y,dtype=np.float32)
test_y = np.array(test_y,dtype=np.float32)

train_x = chainer.cuda.to_gpu(train_x)
test_x = chainer.cuda.to_gpu(test_x)
train_y = chainer.cuda.to_gpu(train_y)
test_y = chainer.cuda.to_gpu(test_y)




#N = np.load("noise.npy")   
#N = np.array(N,dtype=np.float32)
#N_ = np.zeros((64,64),dtype = np.float32)

#-----------------------------------------------------------#
# xyを予測するCNNモデルクラス定義
#-----------------------------------------------------------#
class CNN_xy(chainer.Chain):
    def __init__(self):
        super(CNN_xy, self).__init__(
             conv1 = L.Convolution2D(None, 20, 5),
             conv2 = L.Convolution2D(20, 50, 5),
             l1 = L.Linear(None, 500),
             l2 = L.Linear(500, 500),
             l3 = L.Linear(500, 2, initialW=np.zeros((2, 500), dtype=np.float32))
             
        )    
    def __call__(self,x,t = None,train=False):
        
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)

        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        #y = F.tanh(self.l3(h))
        y = (self.l3(h))
        #y = y.reshape(50,)
        
        return y
      
    def reset(self):
        self.zerograds() # 勾配の初期化          
#--------------------------------------------------
#重み画像を生成モデルクラスの定義
class CNN_weight(chainer.Chain):
    def __init__(self):
        super(CNN_weight, self).__init__(
             conv1 = L.Convolution2D(None, 20, 5),
             conv2 = L.Convolution2D(20, 50, 5),
             #conv3 = L.Convolution2D(50, 100, 5),
             
             l1 = L.Linear(None, 500),
             l2 = L.Linear(500, 500),
             l3 = L.Linear(500, 1024, initialW=np.zeros((1024,500), dtype=np.float32))
             #l3 = L.Linear(500, 1024)
        )    
    def __call__(self,x,train=False):
        #x = chainer.Variable(x)
        #if train:
            #t = chainer.Variable(t)
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        #h = F.max_pooling_2d(F.relu(self.conv3(h)), 2)

        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        y = F.sigmoid(self.l3(h))
        #y =(self.l3(h))
         
        y = y.reshape(len(x),1,32,32)
        #print y.shape
        
        return y
      
    def reset(self):
        self.zerograds() # 勾配の初期化
        
class JoinedModel(chainer.Chain):
    def __init__(self):
        super(JoinedModel,self).__init__()
        with self.init_scope():
            self.model_xy = CNN_xy()
            self.model_weight = CNN_weight() 
       
                             
#---------------------------------------------#
#元出力　Ｉ　を　変換ノイズ画像を加える　Ｉ’を作る
#(model,input)
#---------------------------------------------#
def input_plus_transformednoise(model_weight,x):
   
    img_N = np.random.uniform(-255,255,(len(x), 1, 32,32))
       
    img_N = np.array(img_N,dtype=np.float32)
   
    img_N = chainer.cuda.to_gpu(img_N)  
    
    
    
    with chainer.using_config('enable_backprop', True):
        R = model_weight(x = x,train = True)
        
        N_ = F.prod(F.stack((img_N,R)),axis=0)
        
        I_ = chainer.Variable(x) + N_/255
        
        I_.to_gpu()
        train_x_noise_batch = I_
        
        
    
        return train_x_noise_batch    
                    
#---------------------------------------------#
#compute_xy_loss(model, x, t, train=True)
#---------------------------------------------#
def compute_xy_loss(model, x, t, train=True):
    
    with chainer.using_config('enable_backprop', train):
        #x_= chainer.Variable(x)
        
        t_ = chainer.Variable(t)
        y_ = model(x=x,t=t_,train=train)
      
    return F.mean_squared_error(y_,t_)
    
#---------------------------------------------#
#compute_weight_loss(model, x, train=True)
#---------------------------------------------#
def compute_weight_loss(model, x, train=True):
   
    #p = 0.1# 全画像の中に物体領域の割
  
    batchsize = x.shape[0]
    p = xp.array([0.95] * batchsize,dtype = np.float32)
    p = chainer.Variable(p)
    
    
    with chainer.using_config('enable_backprop', train):
        x_= chainer.Variable(x) #Variable(x)
    
        y_ = model(x=x,train=train) #Variable(xp.array((50, 1, 32, 32)))
        
        y_integral_mean = chainer.Variable(xp.zeros((), dtype=np.float32)) #Variable(xp.zeros())
        
        y_integral_mean = F.sum(y_, axis=(1,2,3))/1024
        #print p.shape
        print y_integral_mean
            
        return F.mean_squared_error(y_integral_mean,p)
  
# ------------------------------------------------------------------- #
# outputLogFile(fp, list_log)
# 内容 : ログをファイルに出力する。配列形式で渡されたログをスペース区切りで出力
# 引数 : fp	  出力ファイルのポインタ
#        list_log 出力ログ(配列形式)
# 戻り値: なし
# ------------------------------------------------------------------- #
def outputLogFile(fp, list_log):
    l = len(list_log)
    for i in range(l-1):
        fp.write( str(list_log[i]) + " ")

    # 最後のみ別処理
    fp.write( str(list_log[l-1]) + "\n")

# ------------------------------------------------------------------- #
# evaluate(model, skeleton, scene, batchsize)
# 内容 : 学習中モデルの評価を行う。
# ------------------------------------------------------------------- #
def evaluate(model_xy,model_weight, t, x, batchsize):
    #loss function: L = a*|F(I)-P|**2 + b*|F(I_)-P|**2 + r|Area_R - k|**2
    a = 0.3
    b = 0.4
    r = 10
    
    #for epoch in range(epoch_num):
    N_len=len(x)
    perm = np.random.permutation(N_len)
    total_loss = xp.zeros(())
    
    loss_xy_log = xp.zeros(())
    loss_xy_weight_log = xp.zeros(())
    loss_weight_log = xp.zeros(())
    
    for i in range(0, N_len, batch_size):
        x_batch = (x[perm[i:i+batch_size]])
        t_batch = (t[perm[i:i+batch_size]])
        train_x_noise_batch = input_plus_transformednoise(model_weight,x_batch)
        
        loss_xy = compute_xy_loss(model_xy,x_batch, t_batch, train=False)#   |F(I)-P|**2           
        loss_weight = compute_weight_loss(model_weight,x_batch, train=False)# |Area_R - k|**2
        loss_xy_weight = compute_xy_loss(model_xy,train_x_noise_batch, t_batch, train=False)#|F(I_)-P|**2
        
        loss_xy_log += loss_xy.data.reshape(())
        loss_xy_weight_log += loss_xy_weight.data.reshape(())
        loss_weight_log += loss_weight.data.reshape(())
        
        
        loss = a*(loss_xy.data) + b*(loss_xy_weight.data) + r*(loss_weight.data)
        total_loss += loss.reshape(())
        #total_loss += loss
    
    #print (cuda.to_cpu(total_loss)/N_len)
    #print (cuda.to_cpu(loss_xy_log)/N_len)
    #print (cuda.to_cpu(loss_weight_log)/N_len)  
    return cuda.to_cpu(total_loss)/N_len, total_loss , cuda.to_cpu(loss_xy_log)/N_len ,cuda.to_cpu(loss_xy_weight_log)/N_len,cuda.to_cpu(loss_weight_log)/N_len
        


   

#------------------------------------------------------------#
# train用関数
#------------------------------------------------------------#    

def run():
    global epoch_num
    global batch_size
    global N
    global st
    epoch_num = 301
    batch_size = 50
    N = train_x.shape[0]
    st = datetime.datetime.now()
    
    
#--------------------------------#
#joinedmodel
#------------------------------------#
    
    model = JoinedModel()
    serializers.load_npz("./load_model/angel_joinedmodel_npz_model0200_final",model)#use learned params


    if gpu >= 0:
        chainer.cuda.get_device(gpu).use()
        model.to_gpu(gpu)
       
    optimizer = chainer.optimizers.Adam()
    
    optimizer.setup(model)
    
    

    # 学習状況出力用ファイル
    f_trainLog = open("./log/joinedmodel/joinedmodel_log_perp_train.txt", "w")
    f_testLog  = open("./log/joinedmodel/joinedmodel_log_perp_test.txt" , "w")
    
    f_train_xy_log = open("./log/joinedmodel/joinedmodel_log_perp_train_xy.txt" , "w")
    f_train_weight_log = open("./log/joinedmodel/joinedmodel_log_perp_train_weight.txt" , "w")
    f_train_xy_weight_log = open("./log/joinedmodel/joinedmodel_log_perp_train_xy_weight.txt" , "w")
    
    f_test_xy_log = open("./log/joinedmodel/joinedmodel_log_perp_test_xy.txt" , "w")
    f_test_weight_log = open("./log/joinedmodel/joinedmodel_log_perp_test_weight.txt" , "w")
    f_test_xy_weight_log = open("./log/joinedmodel/joinedmodel_log_perp_test_xy_weight.txt" , "w")
    # 累計誤差を格納する変数
    accum_loss = chainer.Variable(xp.zeros((), dtype=np.float32))
            
             
    base = 1# モデルを保存するタイミングを調整する用の変数
    
    
    a = 0.3
    b = 0.4
    r = 10
    
    
    for epoch in range(epoch_num):
        model.model_xy.reset()
        model.model_weight.reset()
        perm = np.random.permutation(N)
   
        for i in range(0, N, batch_size):
            x_batch = (train_x[perm[i:i+batch_size]])
            t_batch = (train_y[perm[i:i+batch_size]])
            train_x_noise_batch = input_plus_transformednoise(model.model_weight,x_batch)
            
            loss_xy_accum = compute_xy_loss(model.model_xy,x_batch, t_batch, train=True)#   |F(I)-P|**2           
            loss_weight_accum = compute_weight_loss(model.model_weight,x_batch, train=True)# |Area_R - k|**2
            loss_xy_weight_accum = compute_xy_loss(model.model_xy,train_x_noise_batch, t_batch, train=True)#|F(I_)-P|**2
            #print loss_xy.data
            
            #print loss_xy_weight.data
            
            #print loss_weight.data
            #accum_loss += a*(loss_xy) + b*(loss_xy_weight) + r*(loss_weight)
            accum_loss = a*(loss_xy_accum) + b*(loss_xy_weight_accum) + r*(loss_weight_accum)
            
            
            
        
            
            # lossに基づいてモデルの重み更新
            #if (i + 1) % bprop_len == 0:
            optimizer.target.zerograds()
            accum_loss.backward() # ロス値の勾配を求める
            optimizer.update()

            #accum_loss.unchain_backward()  # truncate (グラフから余分な部分を消し去る）
            # 累計誤差をリセットする
            accum_loss = chainer.Variable(xp.zeros((), dtype=np.float32)) 
            
            
        # epoch毎の処理
        # 端数分に対しても重み更新する
        #optimizer.target.zerograds()
        #accum_loss.backward() # ロス値の勾配を求める
        #optimizer.update()

        #accum_loss.unchain_backward()  # truncate (グラフから余分な部分を消し去る）
        # 累計誤差をリセットする
        #accum_loss = chainer.Variable(xp.zeros((), dtype=np.float32))
        
        eva_trainSample_ = evaluate(copy.deepcopy(model.model_xy),copy.deepcopy(model.model_weight),train_y, train_x, batch_size)
        eva_testSample_ = evaluate(copy.deepcopy(model.model_xy),copy.deepcopy(model.model_weight),test_y, test_x, batch_size)
 
        # ログファイルに評価結果を出力
        outputLogFile(f_trainLog, eva_trainSample_[0:2])
        outputLogFile(f_testLog, eva_testSample_[0:2])
        
        
        outputLogFile(f_train_xy_log, eva_trainSample_[2:3])
        outputLogFile(f_train_weight_log, eva_trainSample_[4:])
        outputLogFile(f_train_xy_weight_log, eva_trainSample_[3:4])
        
        outputLogFile(f_test_xy_log, eva_testSample_[2:3])
        outputLogFile(f_test_weight_log, eva_testSample_[4:])
        outputLogFile(f_test_xy_weight_log, eva_testSample_[3:4])
       
        
        ed = datetime.datetime.now()
        
        #コンソールに評価結果を出力
        print "epoch: ", epoch
        print "Train: loss_all: {:.9f}".format(eva_trainSample_[0])
        print "Test : loss_all: {:.9f}".format(eva_testSample_[0])
        print ("Time : loss_all: "+str(ed-st))
        
        
        print "Train_xy: loss_all: {:.9f}".format(eva_trainSample_[2])
        print "Test_xy : loss_all: {:.9f}".format(eva_testSample_[2])
        
        print "Train_xy_weight: loss_all: {:.9f}".format(eva_trainSample_[3])
        print "Test_xy_weight : loss_all: {:.9f}".format(eva_testSample_[3])
       
        print "Train_weight: loss_all: {:.9f}".format(eva_trainSample_[4])
        print "Test_weight : loss_all: {:.9f}".format(eva_testSample_[4])
        st = datetime.datetime.now()
    #-------------------------------------------------------------------------------------------
   
    # モデルの保存

        if epoch % base == 0:
            if  gpu >= 0:
                model.to_gpu()
                #serializers.save_npz('./model/xy_model/npz_model%04d_final' % (epoch),model)
                serializers.save_npz('./model/joinedmodel/joinedmodel_npz_model%04d_final' % (epoch),model)
                                                                                 
        if epoch / base == 10:
            base *= 10
        
if __name__ == '__main__':
    print "start"
    run()        















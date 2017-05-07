
# coding: utf-8

# In[1]:

#51CTO课程频道：http://edu.51cto.com/lecturer/index/user_id-12330098.html
#优酷频道：http://i.youku.com/sdxxqbf
#微信公众号：深度学习与神经网络
#Github：https://github.com/Qinbf


# In[2]:

import numpy as np


# In[3]:

#输入数据
X = np.array([[1,0,0],
              [1,0,1],
              [1,1,0],
              [1,1,1]])
#标签
Y = np.array([[0,1,1,0]])
#权值初始化，取值范围-1到1
V = np.random.random((3,4))*2-1 
W = np.random.random((4,1))*2-1
print(V)
print(W)
#学习率设置
lr = 0.11

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(x):
    return x*(1-x)

def update():
    global X,Y,W,V,lr
    
    L1 = sigmoid(np.dot(X,V))#隐藏层输出(4,4)
    L2 = sigmoid(np.dot(L1,W))#输出层输出(4,1)
    
    L2_delta = (Y.T - L2)*dsigmoid(L2)
    L1_delta = L2_delta.dot(W.T)*dsigmoid(L1)
    
    W_C = lr*L1.T.dot(L2_delta)
    V_C = lr*X.T.dot(L1_delta)
    
    W = W + W_C
    V = V + V_C


# In[4]:

for i in range(20000):
    update()#更新权值
    if i%500==0:
        L1 = sigmoid(np.dot(X,V))#隐藏层输出(4,4)
        L2 = sigmoid(np.dot(L1,W))#输出层输出(4,1)
        print('Error:',np.mean(np.abs(Y.T-L2)))
        
L1 = sigmoid(np.dot(X,V))#隐藏层输出(4,4)
L2 = sigmoid(np.dot(L1,W))#输出层输出(4,1)
print(L2)

def judge(x):
    if x>=0.5:
        return 1
    else:
        return 0

for i in map(judge,L2):
    print(i)


# In[ ]:




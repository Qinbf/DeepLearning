
# coding: utf-8

# In[1]:

#51CTO课程频道：http://edu.51cto.com/lecturer/index/user_id-12330098.html
#优酷频道：http://i.youku.com/sdxxqbf
#微信公众号：深度学习与神经网络
#Github：https://github.com/Qinbf


# In[2]:

import numpy as np
import matplotlib.pyplot as plt


# In[3]:

#输入数据
X = np.array([[1,0,0,0,0,0],
              [1,0,1,0,0,1],
              [1,1,0,1,0,0],
              [1,1,1,1,1,1]])
#标签
Y = np.array([-1,1,1,-1])
#权值初始化，1行3列，取值范围-1到1
W = (np.random.random(6)-0.5)*2
print(W)
#学习率设置
lr = 0.11
#计算迭代次数
n = 0
#神经网络输出
O = 0

def update():
    global X,Y,W,lr,n
    n+=1
    O = np.dot(X,W.T)
    W_C = lr*((Y-O.T).dot(X))/int(X.shape[0])
    W = W + W_C


# In[4]:

for _ in range(100000):
    update()#更新权值
    #-0.1,0.1,0.2,-0.2
    #-1,1,1,-1


#正样本
x1 = [0,1]
y1 = [1,0]
#负样本
x2 = [0,1]
y2 = [0,1]

def calculate(x,root):
    a = W[3]
    b = W[2]+x*W[4]
    c = W[0]+x*W[1]+x*x*W[3]
    if root==1:
        return (-b+np.sqrt(b*b-4*a*c))/(2*a)
    if root==2:
        return (-b-np.sqrt(b*b-4*a*c))/(2*a)
    

xdata = np.linspace(-1,2)

plt.figure()

plt.plot(xdata,calculate(xdata,1),'r')
plt.plot(xdata,calculate(xdata,2),'r')

plt.plot(x1,y1,'bo')
plt.plot(x2,y2,'yo')
plt.show()


# In[5]:

O = np.dot(X,W.T)
print(O)


# In[ ]:





# coding: utf-8

# In[1]:

#51CTO课程频道：http://edu.51cto.com/lecturer/index/user_id-12330098.html
#优酷频道：http://i.youku.com/sdxxqbf
#微信公众号：深度学习与神经网络
#Github：https://github.com/Qinbf


# In[2]:

'''
异或
0^0 = 0
0^1 = 1
1^0 = 1
1^1 = 0
'''


# In[3]:

import numpy as np
import matplotlib.pyplot as plt


# In[4]:

#输入数据
X = np.array([[1,0,0],
              [1,0,1],
              [1,1,0],
              [1,1,1]])
#标签
Y = np.array([-1,1,1,-1])
#权值初始化，1行3列，取值范围-1到1
W = (np.random.random(3)-0.5)*2
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
    O = np.sign(np.dot(X,W.T))
    W_C = lr*((Y-O.T).dot(X))/int(X.shape[0])
    W = W + W_C


# In[5]:

for _ in range(1000):
    update()#更新权值
    print(W)#打印当前权值
    print(n)#打印迭代次数
    O = np.sign(np.dot(X,W.T))#计算当前输出
    if(O == Y.T).all(): #如果实际输出等于期望输出，模型收敛，循环结束
        print('Finished')
        print('epoch:',n)
        break

#正样本
x1 = [0,1]
y1 = [1,0]
#负样本
x2 = [0,1]
y2 = [0,1]

#计算分界线的斜率以及截距
k = -W[1]/W[2]
d = -W[0]/W[2]
print('k=',k)
print('d=',d)

xdata = np.linspace(-2,5)

plt.figure()
plt.plot(xdata,xdata*k+d,'r')
plt.plot(x1,y1,'bo')
plt.plot(x2,y2,'yo')
plt.show()


# In[ ]:




# In[ ]:




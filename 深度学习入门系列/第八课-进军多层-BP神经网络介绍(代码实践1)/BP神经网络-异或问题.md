

```python
#51CTO课程频道：http://edu.51cto.com/lecturer/index/user_id-12330098.html
#优酷频道：http://i.youku.com/sdxxqbf
#Github：https://github.com/Qinbf
```


```python
import numpy as np
```


```python
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
```

    [[-0.79306585  0.93936225 -0.60257721 -0.79228528]
     [-0.33840192 -0.52791209  0.55944743 -0.40831393]
     [ 0.96380363 -0.3276025  -0.5034575  -0.09247931]]
    [[-0.17293012]
     [ 0.93869591]
     [ 0.44443866]
     [ 0.65668778]]
    


```python
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
```

    Error: 0.499047148968
    Error: 0.499857996791
    Error: 0.499614219706
    Error: 0.499316268255
    Error: 0.498682783133
    Error: 0.496841542713
    Error: 0.489461839163
    Error: 0.449512073597
    Error: 0.313798464967
    Error: 0.197238982213
    Error: 0.142094417235
    Error: 0.113103606623
    Error: 0.0954231534527
    Error: 0.0834680939614
    Error: 0.0747931048371
    Error: 0.0681739585616
    Error: 0.0629319246265
    Error: 0.0586602932735
    Error: 0.0551001604186
    Error: 0.0520785809769
    Error: 0.0494754177282
    Error: 0.0472044694425
    Error: 0.0452021783992
    Error: 0.0434205838824
    Error: 0.0418227653044
    Error: 0.0403798032797
    Error: 0.0390686980356
    Error: 0.0378709096924
    Error: 0.0367713131431
    Error: 0.0357574358059
    Error: 0.0348188923908
    Error: 0.0339469594524
    Error: 0.0331342507934
    Error: 0.0323744667475
    Error: 0.0316621983325
    Error: 0.0309927726809
    Error: 0.0303621298827
    Error: 0.0297667239918
    Error: 0.0292034428078
    Error: 0.0286695423777
    [[ 0.03097444]
     [ 0.97259213]
     [ 0.97276123]
     [ 0.02703325]]
    0
    1
    1
    0
    


```python

```

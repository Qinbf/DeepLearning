
# coding: utf-8

# In[1]:

#51CTO课程频道：http://edu.51cto.com/lecturer/index/user_id-12330098.html
#优酷频道：http://i.youku.com/sdxxqbf
#微信公众号：深度学习与神经网络
#Github：https://github.com/Qinbf


# In[2]:

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import  metrics,linear_model
from sklearn.neural_network import BernoulliRBM
from sklearn.datasets import load_digits
from sklearn.pipeline import Pipeline


# In[3]:

digits = load_digits()#载入数据
X = digits.data#数据
Y = digits.target#标签
#输入数据归一化
X -= X.min()
X /= X.max()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2,random_state=0)


#创建RBM模型
logistic = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose=True)
classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])


# In[4]:

#设置学习率
rbm.learning_rate = 0.06
#设置迭代次数
rbm.n_iter = 20
#设置隐藏层单元
rbm.n_components = 200
logistic.C = 6000.0
#训练模型
classifier.fit(X_train, Y_train)


# In[5]:

print()
print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        classifier.predict(X_test))))


# In[ ]:




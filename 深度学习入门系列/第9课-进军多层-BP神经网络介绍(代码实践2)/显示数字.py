
# coding: utf-8

# In[1]:

#51CTO课程频道：http://edu.51cto.com/lecturer/index/user_id-12330098.html
#优酷频道：http://i.youku.com/sdxxqbf
#微信公众号：深度学习与神经网络
#Github：https://github.com/Qinbf


# In[2]:

from sklearn.datasets import load_digits
import pylab as pl

digits = load_digits()#载入数据集
print(digits.data.shape)

pl.gray()#灰度化图片
pl.matshow(digits.images[0])
pl.show()


# In[ ]:




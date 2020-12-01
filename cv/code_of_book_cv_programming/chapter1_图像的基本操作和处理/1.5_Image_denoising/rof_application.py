
# coding: utf-8

# In[26]:


from numpy import *
from numpy import random
from scipy.ndimage import filters
import rof


# In[27]:


# 使用噪声创建合成图像
im = zeros((500,500))
im[100:400,100:400] = 128
im[200:300,200:300] = 255
im = im + 30*random.standard_normal((500,500))

# rof除噪
rof.denoise(im,im)
# 高斯模糊处理
# G = filters.gaussian_filter(im,10)


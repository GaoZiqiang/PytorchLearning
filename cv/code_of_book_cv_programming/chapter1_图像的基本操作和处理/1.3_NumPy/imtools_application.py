
# coding: utf-8

# In[10]:


from PIL import Image
from numpy import *# import方式有两种，第一种import ** as @@，在调用**的属性和方法时需要使用点标记法引用
#from numpy import *# 使用from ** import *这种方法就不需要使用点标记法显式引用了
import imtools


# In[9]:


#im = array(Image.open('demo.jpg'))
pil_im = Image.open('demo.jpg')
im = array(Image.open('demo.jpg').convert('L'))


# In[11]:


im2,cdf = imtools.histeq(im)


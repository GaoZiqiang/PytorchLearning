
# coding: utf-8

# In[17]:


from PIL import Image
from numpy import *
from pylab import *
import imtools
from IPython import embed


# In[19]:


# 读取文件列表
path = '/home/gaoziqiang/tempt/data'
imlist = imtools.get_imlist(path)


# In[20]:


im = array(Image.open(imlist[0]))# 通过打开一张图像，获取图像的尺寸大小
m,n = im.shape[0:2]# 获取图像的大小
imnbr = len(imlist)# 获取图像的数目


# In[21]:


# 创建矩阵，保存所有压平后的图像数据
embed()
immatrix = array([array(Image.open(im)).flatten() for im in imlist],'f')
# embed()


# In[22]:


# 执行PCA操作
V,S,immean = imtools.pca(immatrix)


# In[ ]:


# 显示一些图像(均值图像和前７个模式)
figure()
gray()
subplot(2,4,1)
imshow(immean.reshape(m,n))
for i in range(7):
    subplot(2,4,i+2)
    imshow(V[i].reshape(m,n))
    
show()


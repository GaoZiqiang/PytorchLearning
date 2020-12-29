#!/usr/bin/env python
# coding: utf-8

# In[32]:


import sift
from scipy.ndimage import filters
from numpy import *
from PIL import Image


# In[33]:


imname = 'empire.jpg'
im1 = array(Image.open(imname).convert('L'))
sift.process_image(imname,'empire.sift')
l1,d1 = sift.read_features_from_file('empire.sift')


# In[6]:


figure()
gray()
sift.plot_features(im1,l1,circle=True)
show()


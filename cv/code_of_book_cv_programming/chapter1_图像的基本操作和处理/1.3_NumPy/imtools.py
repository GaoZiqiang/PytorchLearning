
# coding: utf-8

# In[6]:


from PIL import Image
from numpy import *
import os
def imresize(im,sz):
    '''
    使用PIL对象重新定义图像数组的大小
    @param im:输入的图像array
    @param sz:resize的大小，是一个tuple
    '''
    pil_im = Image.fromarray(uint8(im))
    
    return array(pil_im.resize(sz))


# In[2]:


# 直方图均衡化
def histeq(im,nbr_bins=256):
    '''
    对一幅灰度图像进行直方图均值化
    @param nbr_bins:直方图中使用小区间的数目
    
    @return 直方图均衡化后的图像以及用来做像素值映射的累积分布函数
    '''
    # 计算图像的直方图
    imhist,bins = histogram(im.flatten(),nbr_bins,normed=True)
    cdf = imhist.cumsum()# 累积分布函数
    cdf = 255*cdf/cdf[-1]#归一化
    # 使用累积分布函数的线性插值，计算新的像素值
    im2 = interp(im.flatten(),bins[:-1],cdf)
    
    return im2.reshape(im.shape),cdf


# In[5]:


# 计算平均图像
def compute_average(imlist):
    '''
    计算图像列表的平均图像
    '''
    # 打开第一幅图像，将其存储在浮点型数组中
    averageim = array(Image.open(imlist[0],'f'))
    
    for imname in imlist[1:]:
        try:
            averageim += array(Image.open(imname))
        except:
            print(imname + '...skipped')
    averageim /= len(imlist)
    
    # 返回uint8类型的平均图像
    return array(avrageim,'uint8')


# In[7]:


# 图像的主成分分析
def pca(X):
    '''
    主成分分析：
    input:矩阵X，其中该矩阵中存储训练数据，每一行为一条训练数据
    return:投影矩阵（按照维度的重要性排序）、方差和均值
    '''
    # 获取维数
    num_data,dim = X.shape
    # 数据中心化
    mean_X = X.mean(axis=0)
    X = X - mean_X
    
    # 数据个数小于数据维度
    if dim > num_data:
        # PCA-使用紧致技巧
        M = dot(X,X.T)# 协方差矩阵
        e,EV = linalg.eigh(M)# 特征值和特征向量
        tmp = dot(X.T,EV).T# 这就是紧致技巧
        V = tmp[::-1]
        S = sqrt(e)[::-1]# 对特征值进行逆转
        for i in range(V.shape[1]):
            V[:,i] /= S
    else:
        # PCA-使用SVD方法
        U,S,V = linalg.svd(X)
        V = V[:num_data]# 仅仅返回前num_data维的数据
    
    # 返回投影矩阵、方差和均值
    return V,S,mean_X


# In[9]:


# 获取文件名列表　
def get_imlist(path):
    '''返回列表中所有JPG图像的文件名列表'''
    return [os.path.join(path,f) for f in  os.listdir(path) if f.endswith('.jpg')]


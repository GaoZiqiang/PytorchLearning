## 1.1 PIL：Python图像处理类库
PIL（Python Imaging Library，图像处理类库）提供了通用的图像处理功能，以及大量有用的基本图像操作，比如图像缩放、裁剪、旋转、颜色转换等。
利用 PIL 中的函数，我们可以从大多数图像格式的文件中读取数据，然后写入最常见的图像格式文件中。PIL 中最重要的模块为 Image。
打开图像：
from PIL import Image

pil_im = Image.open('demo.png')

// 进行颜色转换--灰度图像
pil_im.convert('L')

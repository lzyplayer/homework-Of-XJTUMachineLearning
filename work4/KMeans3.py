from scipy.cluster.vq import *
from scipy.misc import imresize

from pylab import *

from PIL import Image


# steps*steps像素聚类
def clusterpixels_square(infile, k, steps):
    currim = array(Image.open(infile))

    # im.shape[0] 高 im.shape[1] 宽
    dx = currim.shape[0] // steps
    dy = currim.shape[1] // steps
    # 计算每个区域的颜色特征
    features = []
    for x in range(steps):
        for y in range(steps):
            R = mean(currim[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 0])
            G = mean(currim[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 1])
            B = mean(currim[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 2])
            features.append([R, G, B])
    features = array(features, 'f')  # 变为数组
    # 聚类， k是聚类数目
    centroids, variance = kmeans(features, k)
    code, distance = vq(features, centroids)

    # 用聚类标记创建图像
    codeim = code.reshape(steps, steps)
    codeim = imresize(codeim, currim.shape[:2], 'nearest')
    return codeim


# stepsX*stepsY像素聚类
def clusterpixels_rectangular(infile, k, stepsX):
    currim = array(Image.open(infile))

    stepsY = stepsX * currim.shape[1] // currim.shape[0]

    # im.shape[0] 高 im.shape[1] 宽
    dx = currim.shape[0] // stepsX
    dy = currim.shape[1] // stepsY
    # 计算每个区域的颜色特征
    features = []
    for x in range(stepsX):
        for y in range(stepsY):
            R = mean(currim[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 0])
            G = mean(currim[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 1])
            B = mean(currim[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 2])
            features.append([R, G, B])
    features = array(features, 'f')  # 变为数组
    # 聚类， k是聚类数目
    centroids, variance = kmeans(features, k)
    code, distance = vq(features, centroids)
    # 用聚类标记创建图像
    codeim = code.reshape(stepsX, stepsY)
    codeim = imresize(codeim, currim.shape[:2], 'nearest')
    return codeim


# 计算最优steps 为保证速度以及减少噪点 最大值为maxsteps 其值为最接近且小于maxsteps 的x边长的约数
def getfirststeps(img, maxsteps):
    msteps = img.shape[0]

    n = 2

    while msteps > maxsteps:
        msteps = img.shape[0] // n
        n = n + 1

    return msteps


# Test


# 图像文件 路径
infile = './pic/milano3.jpg'
infile2= './pic/milano4.jpg'
im = array(Image.open(infile))
im2 = array(Image.open(infile2))
# 参数
m_k = 3
b_k = 10
m_maxsteps = 128

# 显示原图empire.jpg
figure()

subplot(251)

title('milano1')

imshow(im)

# 用改良矩形块对图片的像素进行聚类
codeim = clusterpixels_rectangular(infile, m_k, getfirststeps(im, m_maxsteps))

subplot(252)

title('New steps = ' + str(getfirststeps(im, m_maxsteps)) + ' K = ' + str(m_k))

imshow(codeim)

subplot(253)
title('New steps = ' + str(getfirststeps(im, m_maxsteps)) + ' K = ' + str(b_k))
codeim = clusterpixels_rectangular(infile, b_k, getfirststeps(im, m_maxsteps))
imshow(codeim)
# 方形块对图片的像素进行聚类
st = 100
codeim = clusterpixels_square(infile, m_k, st)

subplot(254)

title('Old steps = 200,K = ' + str(m_k))

imshow(codeim)

subplot(255)
title('Old steps = 200,K = ' + str(b_k))
codeim = clusterpixels_square(infile, b_k, st)
imshow(codeim)

subplot(256)
title('milano2')
imshow(im2)

subplot(257)
title('New steps = ' + str(getfirststeps(im2, m_maxsteps)) + ' K = ' + str(m_k))
codeim = clusterpixels_rectangular(infile2, m_k, getfirststeps(im2, m_maxsteps))
imshow(codeim)

subplot(258)

title('New steps = ' + str(getfirststeps(im2, m_maxsteps)) + ' K = ' + str(b_k))
codeim = clusterpixels_rectangular(infile2, b_k, getfirststeps(im2, m_maxsteps))
imshow(codeim)

subplot(259)
title('Old steps = 200,K = ' + str(m_k))
codeim = clusterpixels_square(infile2, m_k, st)
imshow(codeim)

subplot(2,5,10)
title('Old steps = 200,K = ' + str(b_k))
codeim = clusterpixels_square(infile2, b_k, st)
imshow(codeim)


show()

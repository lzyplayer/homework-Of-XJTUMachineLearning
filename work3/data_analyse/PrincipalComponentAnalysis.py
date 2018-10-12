# -*- coding: utf-8 -*-
# @Time     : 2018/10/10 9:47
# @Author   : vickylzy
import numpy as np


def normalized(msg):
    dataNormed = (msg - msg.mean(0)) / msg.std(0)
    return dataNormed


# @para msg_received: shape(a,b)
# @para keep 保留维度数量
def pri_com_ana(msg_received, keep):
    component = keep
    selected = range(-component, 0)  # 降维保留的维度

    # normalized
    msgNormalized = normalized(msg_received)  # shape = (a, b)
    # Covariance matrix
    covMatrix = np.cov(msgNormalized.T)  # shape = (b, b)
    print(covMatrix)
    # 提取前n个特征值特征向量
    eigVal, eigVector = np.linalg.eig(covMatrix)  # eigVal.shape = （b,1） eigVector.shape = (b,b)
    sortList = np.argsort(eigVal)
    print(sortList[selected])
    selectedVector = eigVector[:, sortList[selected]]  # selectedVector.shape=(b,keep)
    # 产生主成分结果
    return np.dot(msg_received, selectedVector)  # shape = (a, keep)

    # PCA by sklearn
    # pca = PCA(n_components=2)
    # fitResult = pca.fit_transform(msg_received)
    # fitResult = np.array(fitResult, dtype=np.float16)
    # print(fitResult)

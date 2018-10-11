# -*- coding: utf-8 -*-
# @Time     : 2018/10/10 9:47
# @Author   : vickylzy
from sklearn.decomposition import PCA
from readDataClass import read_data_class
import numpy as np


def normalized(msg):
    dataNormed = (msg - msg.mean(0)) / msg.std(0)
    return dataNormed


# if __name__ == '__main__':
msgReceived = read_data_class('./data_analyse/data_class1.txt')  # ./data_analyse
component = 2
selected = range(-component, 0)
print(selected)
# delete
msgReceived = np.array([[1, -1, 2],
                        [2, 0, 0],
                        [0, 1, -1]])

# PCA by sklearn
pca = PCA(n_components=2)
fitResult = pca.fit_transform(msgReceived)
fitResult = np.array(fitResult, dtype=np.float16)
print(fitResult)

# normalized
msgNormalized = normalized(msgReceived)
# Covariance matrix
covMatrix = np.cov(msgNormalized)
print(covMatrix)
# 提取前n个特征值特征向量
eigVal, eigVector = np.linalg.eig(covMatrix)
sortList = np.argsort(eigVal)
print(sortList[selected])
selectedVector = eigVector[:, sortList[selected]]
# 产生主成分结果
primalComp = np.dot(msgReceived, selectedVector)

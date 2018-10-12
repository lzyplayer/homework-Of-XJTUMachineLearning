# -*- coding: utf-8 -*-
# @Time     :  8:37
# @Author   : vickylzy

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import datasets
import readDataClass
import numpy as np

# iris = datasets.load_iris()
#
# X = iris.data
# y = iris.target
lda = LinearDiscriminantAnalysis(n_components=2)
# result = lda.fit_transform(X,y)

msg = readDataClass.read_data_class('./data_analyse/data_class1.txt')
# msg = np.array(msg, dtype=np.float64)
target1 = np.ones([256, 1])
target2 = np.zeros([768, 1])
target = np.append(target1, target2)

result = lda.fit_transform(msg, target)

type(msg)

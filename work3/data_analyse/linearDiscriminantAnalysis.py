# -*- coding: utf-8 -*-
# @Time     :  8:37
# @Author   : vickylzy

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import numpy as np


# para: msg: shape=(a,b)
def lin_dis_ana(msg):
    lda = LinearDiscriminantAnalysis(n_components=2)
    target = np.zeros((1024,))  # shape=(a,1)
    for i in range(0, 1024):
        target[i] = i % 4
    return lda.fit_transform(msg, target)  # shape=(a,2)

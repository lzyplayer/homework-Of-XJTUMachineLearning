# -*- coding: utf-8 -*-
# @Time     : 2018/10/12 12:19
# @Author   : vickylzy

from sklearn.cross_decomposition import CCA


def can_cor_ana(msg1, msg2):
    cca = CCA(n_components=1)
    cca.fit(msg1, msg2)
    w, v = cca.transform(msg1, msg2)
    return w, v

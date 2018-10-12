# -*- coding: utf-8 -*-
# @Time     : 2018/10/10 9:46
# @Author   : vickylzy

from readDataClass import read_data_class
from PrincipalComponentAnalysis import pri_com_ana
from linearDiscriminantAnalysis import lin_dis_ana
from canonicalCorrelationAnalysis import can_cor_ana

if __name__ == '__main__':
    msg1 = read_data_class('data_class1.txt')
    msg2 = read_data_class('data_class2.txt')
    # 主成分分析
    print(pri_com_ana(msg1, 4))
    # 线性判别分析
    print(lin_dis_ana(msg2))
    # 典型相关分析
    w, v = can_cor_ana(msg1, msg2)
    print(w, v)

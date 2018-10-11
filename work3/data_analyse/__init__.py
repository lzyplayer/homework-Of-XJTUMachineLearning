# -*- coding: utf-8 -*-
# @Time     : 2018/10/10 9:46
# @Author   : vickylzy

from readDataClass import read_data_class
from PrincipalComponentAnalysis import pri_com_ana

if __name__ == '__main__':
    msg1 = read_data_class('data_class1.txt')
    msg2 = read_data_class('data_class2.txt')

    print(pri_com_ana(msg1, 4))
    print(pri_com_ana(msg2, 4))

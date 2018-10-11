# -*- coding: utf-8 -*-
# @Time     : 2018/10/10 9:46
# @Author   : vickylzy


import struct
import numpy


def read_data_class(path):
    with open(path, mode='rb') as t:
        # t.seek(0, 0)
        res = numpy.zeros([1024, 8])

        for i in range(0, 1024):
            for j in range(0, 8):
                bytes_read = t.read(4)
                f_value, = struct.unpack("f", bytes_read)
                res[i][j] = f_value
        print(res.shape)
        return res

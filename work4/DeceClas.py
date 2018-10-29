# -*- coding: utf-8 -*-
# @Time     : 2018/10/29 19:47
# @Author   : vickylzy
from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np

cMessage = np.array([[0, 0, 0, 3],
                     [0, 0, 1, 3],
                     [1, 0, 0, 3],
                     [1, 0, 1, 2],
                     [0, 1, 0, 1],
                     [0, 1, 1, 1],
                     [1, 1, 0, 3],
                     [1, 1, 1, 2],
                     [0, 2, 0, 1],
                     [0, 2, 1, 1],
                     [1, 2, 0, 2],
                     [1, 2, 1, 2]])
cMessage = np.array([['girl', 'young', 'false', 'C'],
                     ['girl', 'young', 'true', 'C'],
                     ['boy', 'young', 'false', 'C'],
                     ['boy', 'young', 'true', 'B'],
                     ['girl', 'middle', 'false', 'A'],
                     ['girl', 'middle', 'true', 'A'],
                     ['boy', 'middle', 'false', 'C'],
                     ['boy', 'middle', 'true', 'B'],
                     ['girl', 'old', 'false', 'A'],
                     ['girl', 'old', 'true', 'A'],
                     ['boy', 'old', 'false', 'B'],
                     ['boy', 'old', 'true', 'B']])
cData = cMessage[:, 0:2]
cTarget = cMessage[:, 3]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(cData, cTarget)

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

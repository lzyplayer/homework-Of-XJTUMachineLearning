# -*- coding: utf-8 -*-
# @Time     : 2018/10/29 19:47
# @Author   : vickylzy
from sklearn.datasets import load_iris
from sklearn import tree
import graphviz
import numpy as np
import os

os.environ["PATH"] += os.pathsep + 'C:/ProgramData/Anaconda3/pkgs/graphviz-2.38.0-4/Library/bin/graphviz/'

cMessage = np.array([[0, 0, 0, 'C'],
                     [0, 0, 1, 'C'],
                     [1, 0, 0, 'C'],
                     [1, 0, 1, 'B'],
                     [0, 1, 0, 'A'],
                     [0, 1, 1, 'A'],
                     [1, 1, 0, 'C'],
                     [1, 1, 1, 'B'],
                     [0, 2, 0, 'A'],
                     [0, 2, 1, 'A'],
                     [1, 2, 0, 'B'],
                     [1, 2, 1, 'B']])
# cMessage = np.array([['girl', 'young', 'false', 'C'],
#                      ['girl', 'young', 'true', 'C'],
#                      ['boy', 'young', 'false', 'C'],
#                      ['boy', 'young', 'true', 'B'],
#                      ['girl', 'middle', 'false', 'A'],
#                      ['girl', 'middle', 'true', 'A'],
#                      ['boy', 'middle', 'false', 'C'],
#                      ['boy', 'middle', 'true', 'B'],
#                      ['girl', 'old', 'false', 'A'],
#                      ['girl', 'old', 'true', 'A'],
#                      ['boy', 'old', 'false', 'B'],
#                      ['boy', 'old', 'true', 'B']])
cFeatureName = ['sex', 'age', 'marriage']
cTargetName = ['C', 'B', 'A']
cData = cMessage[:, 0:3]
cTarget = cMessage[:, 3]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(cData, cTarget)
dotData = tree.export_graphviz(clf, out_file=None, feature_names=cFeatureName, filled=True, special_characters=True,class_names=cTargetName)
graph = graphviz.Source(dotData)
graph.render('assure')

# iris = load_iris()
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(iris.data, iris.target)

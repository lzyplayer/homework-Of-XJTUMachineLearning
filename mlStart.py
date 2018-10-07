import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
# from sklearn import datasets
# from sklearn import svm
import pandas as pd


career = pd.read_csv("D:/workSpace_PY/career_data.csv")
careerNP = np.array(career)
data=careerNP[:,0:3]
target=careerNP[:,3]
careerNP[careerNP=='Yes']=1
careerNP[careerNP=='No']=0
careerNP[careerNP=='master']=1
careerNP[careerNP=='bachlor']=0
careerNP[careerNP=='phd']=2
careerNP[careerNP=='C++']=0
careerNP[careerNP=='Java']=1
careerNP=np.array(careerNP,dtype=int)
data=careerNP[:,0:3]
target=careerNP[:,3]
# import csv
# with open('D:/workSpace_PY/career_data.csv',newline='') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#     for row in spamreader:
#         print(".".join(row))
# yes=1 no=0 bachlor=0 master = 1 c++=0 java=1 
# data = np.array([[1,0,0],
#                 [1,0,1],
#                 [0,1,1],
#                 [0,1,0],
#                 [1,0,1],
#                 [0,1,0],
#                 [1,1,1],
#                 [1,2,0],
#                 [0,2,1],
#                 [0,0,1]])
# target = np.array([0,1,1,0,1,0,1,1,1,0])  
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.4, random_state=0)
data_train
data_test
target_train
target_test
# iris = datasets.load_iris()
gnb=GaussianNB()
model=gnb.fit(data_train,target_train)
predictResult=model.predict(data_test)
print()
print("the predict result:\n",predictResult)
print("the actual result:\n",target_test)

# -*- coding: utf-8 -*-

# __author__='vickylzy'

import jieba
import jieba.analyse
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB 
from sklearn.model_selection import train_test_split

#提取关键词
def extractDirWords(path):
    # path='./hw2data/女性/'
    print('loading '+ path )
    textNum=len(os.listdir(path))
    wordslist=list()
    for i in range(1,textNum+1):  #textNum
        with open(path+str(i)+'.txt','r',errors='ignore') as txt:
            currline=txt.readline()
            # print(currline)
            keyWordList=jieba.analyse.extract_tags(currline,topK=10,allowPOS=['a','c','d','e','f','i','n','nr','ns','nt','v','vn'])# nz:其他专有名称 p:介词 r:代词])
            line=str()
            for keyw in keyWordList:
                line=line+' '+keyw
            wordslist.append(line)
    return wordslist


if __name__ == '__main__':
    # print(os.path.abspath('.')+'\data')
    girlText=extractDirWords('./data/女性/')
    sportText=extractDirWords('./data/体育/')
    collageText=extractDirWords('./data/校园/')
    labelGirl=['女性']*len(girlText)
    labelSport=['体育']*len(sportText)
    labelCollage=['校园']*len(collageText)

    #构造数据集
    dataSet=girlText+sportText+collageText
    labelSet=labelGirl+labelSport+labelCollage
    print('dataset generated completed!' )
    
    #分选测试集训练集
    data_train, data_test, target_train, target_test = train_test_split(dataSet, labelSet, test_size=0.2, random_state=0)

    #汉语词组特征化
    count_ver= CountVectorizer( max_df = 0.8, decode_error = 'ignore') 
    trainVector=count_ver.fit_transform(data_train)
    tfidTer=TfidfTransformer()
    tfidVector=tfidTer.fit(trainVector).transform(trainVector)
    
    count_ver_t= CountVectorizer(vocabulary=count_ver.vocabulary_, max_df = 0.8, decode_error = 'ignore') 
    trainVector_t=count_ver_t.fit_transform(data_test)
    tfidVector_t=tfidTer.fit(trainVector_t).transform(trainVector_t)
    print('vectorizering completed!' )
  
    #train
    muNB=MultinomialNB(alpha =0.01)  #0.01?
    muNB.fit(tfidVector,target_train)
    print('training completed!' )   

    #predict
    print('predicting result...' ) 
    predict_result=muNB.predict(tfidVector_t)

    #
    correct = [target_test[i]==predict_result[i] for i in range(len(predict_result))]
    r = len(predict_result)
    t = correct.count(True)
    f = correct.count(False)
    print('predict times: '+str(r)+'\ncorrect times: '+str(t)+'\ncorr pencentage: '+str(float(t/r)))
# print(",".join(tags))
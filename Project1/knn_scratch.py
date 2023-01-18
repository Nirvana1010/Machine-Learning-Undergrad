# -*- coding:utf-8 -*-
import os
import numpy as np
from sklearn.model_selection import LeaveOneOut

# Get label
def getLabel(hot,size):
    for i in range(size):
        if hot[i]=='1':
            return i
    return -1

# load data & normalization
def loadData():
    # training set
    train=[]
    with open(os.path.dirname(__file__)+'/semeion_train.csv','r') as train_file:
        for row in train_file:
            line=row.strip().split(' ')
            train.append(line[:-10]+[getLabel(line[-10:],10)])
    # normalization
    train=np.array(train,dtype=float)
    m,n=np.shape(train)
    data=train[:,0:n-1]
    min_data = data.min(0)
    max_data = data.max(0)
    data = (data - min_data) / (max_data - min_data)
    train[:,0:n-1]=data
    #print(train)
    # test set
    test=[]
    with open(os.path.dirname(__file__)+'/semeion_test.csv','r') as test_file:
        for row in test_file:
            line=row.strip().split(' ')
            test.append(line[:-10]+[getLabel(line[-10:],10)])
    # normalization
    test = np.array(test, dtype=float)
    m, n = np.shape(test)
    data = test[:, 0:n - 1]
    min_data = data.min(0)
    max_data = data.max(0)
    data = (data - min_data) / (max_data - min_data)
    test[:, 0:n - 1] = data
    #print(test)
    return train,test

# euclidean distance calculation
def euclidean_Dist(data1,data2):
    det=(data1-data2)[:,0:n-1]
    return np.linalg.norm(det,axis=1)

# get k nearest neighbors and predicted label for test samples
def getKNNPredictedLabel(train_data,test_data,train_label,test_label,k):
    correct_cnt = 0
    pre_label=[]
    i = 0
    for test in test_data:
        dist = euclidean_Dist(train_data, test)

        distSorted=np.argsort(dist)
        classCount={}
        for num in range(k):
            voteLabel=train_label[distSorted[num]]
            classCount[voteLabel]=classCount.get(voteLabel,0)+1
        sortedClassCount=sorted(classCount.items(),key=lambda x:x[1],reverse=True)

        predictedLabel=sortedClassCount[0][0]
        pre_label.append(predictedLabel)
        if (predictedLabel==test_label[i]):
            correct_cnt+=1
        i=i+1
    return correct_cnt,pre_label

if __name__ == '__main__':
    (train_data,test_data)=loadData()
    loo = LeaveOneOut();
    m,n=np.shape(train_data)
    train_label=train_data[:,-1]
    test_label=test_data[:,-1]

    for k in (1, 3, 5, 7):
        correct = 0
        # LOO CV
        for train, valid in loo.split(train_data):
            correct_cnt, pre_label = getKNNPredictedLabel(train_data[train, :], train_data[valid],
                                                          train_label[train], train_label[valid], k)
            correct += correct_cnt
        acc = correct / np.shape(train_data[train, :])[0]
        print('k = ', k, ', No. of samples are correctly classified', correct, ',accuracy: %.2f' % (acc * 100), '%')
    
# -*- coding: UTF-8 -*-
import math
import numpy as np  

f = open('wine.data','r')
types = [[],[],[]]                      
test_data = [[],[],[]]
train_data = [[],[],[]]
confusion = np.zeros((3,3))             # confusion matrix
data_num = 0                            # no. of samples
test_len = []                           # no. of samples in each category in test set
means = [[],[],[]]                      # mean for each category
std = [[],[],[]]                        # std for each category
myline = '1'
while myline:
    myline = f.readline().split(',')
    if len(myline) != 14:
        break
    for t in range(len(myline)):
        if t == 0:
            myline[t] = int(myline[t])
        else:
            myline[t] = float(myline[t])
    temp = myline.pop(0)
    types[temp - 1].append(myline)
test_len = [round(len(types[i]) / 10) for i in range(3)]
data_num = sum([len(types[i]) for i in range(3)])

def bayes_classificate():
    for i in range(3):
        means[i] = np.mean(train_data[i],axis=0)        # mean
        std[i] = np.std(train_data[i],axis=0)           # std
    wrong_num = 0
    for i in range(3):
        for t in test_data[i]:                  # get each sample from each category
            my_type = []
            for j in range(3):
                # MLE via gaussian distribution
                temp = np.log((2*math.pi) ** 0.5 * std[j])
                temp += np.power(t - means[j], 2) / (2 * np.power(std[j], 2))
                temp = np.sum(temp)
                temp = -1*temp+math.log(len(types[j])/data_num)
                my_type.append(temp)                        # save score
            pre_type = my_type.index(max(my_type))          # prediction: category with max score
            if pre_type != i:                               # no. of wrong predictions
                wrong_num+=1
                confusion[i][pre_type] += 1
            else:
                confusion[i][i] += 1
    #print(confusion)
    return wrong_num

def cross_check():
    wrong_num = 0
    confusion_type = np.zeros((3,4))
    for i in range(10):        # 10-fold CV
        for j in range(3):
            if (i+1)*test_len[j]>len(types[j]):
                test_data[j] = np.mat(types[j][i*test_len[j]:])
                train_data[j] = np.mat(types[j][:i*test_len[j]])
            else:
                test_data[j] = np.mat(types[j][i*test_len[j]:(i+1)*test_len[j]])
                train_data[j] = np.mat(types[j][:i*test_len[j]]+types[j][(i+1)*test_len[j]:])
        wrong_num+=bayes_classificate()

    global confusion
    confusion = confusion.T
    print(confusion)
    for i in range(3):
        confusion_type[i][0] = confusion[i][i]                  #TP
        Y = confusion[i]
        P = confusion[:,i]
        Y = np.delete(Y, i, axis=0)
        P = np.delete(P, i, axis=0)
        confusion_type[i][1] = sum(Y)                           #FP
        confusion_type[i][2] = sum(P)                           #FN
        temp = np.delete(confusion, i, axis=0)
        temp = np.delete(temp, i, axis=1)
        confusion_type[i][3] = sum(temp[0]) + sum(temp[1])      #TN
        print("Category %d :" % (i+1))
        print("confusion matrix:")
        print(str(confusion_type[i][0]) + "  " + str(confusion_type[i][1]))
        print(str(confusion_type[i][2]) + "  " + str(confusion_type[i][3]))
        precision = confusion_type[i][0] / (confusion_type[i][0]+confusion_type[i][1])
        print("precision: %f" % precision)
        recall = confusion_type[i][0] / (confusion_type[i][0]+confusion_type[i][2])
        print("recall: %f" % recall)
        F_measure = 2 / (1/precision + 1/recall)
        print("F_score: %f" % F_measure)
    print("accuracy:"+str(1-wrong_num/data_num))

if __name__=='__main__':
    cross_check()
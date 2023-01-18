import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import KFold
import math


def loadData():
    # load data
    data=pd.read_csv('HWData3.csv')
    data=np.array(data)
    m, n = data.shape
    X = data[:, 0:n - 1]
    Y = data[:, -1]
    # normaliaztion
    std = StandardScaler()
    #std = MinMaxScaler()
    X = std.fit_transform(X)
    return X,Y

# parameter estimation
def gaussian_param(sample):
    # get average
    shape = sample.shape
    mu = np.average(sample,axis=0)
    # X - E(X)
    sub = sample-mu
    # get covariance matrix
    cov = np.empty((shape[1],shape[1]))
    for i in range(shape[1]):
        for j in range(i+1):
            cov[i,j]=cov[j,i]=np.matmul(sub[:,i],sub[:,j])/shape[0]

    return mu,cov

# get probabilities based on parameters
def gaussian(mu,cov,sample):
    shape=sample.shape[0]
    sub=(sample.copy()-mu).reshape(shape,1)
    y=1/(math.pow(2*math.pi,shape/2)*math.sqrt(np.linalg.det(cov)))
    y=y*math.exp(-1/2*np.matmul(np.matmul(sub.T,np.linalg.inv(cov)),sub))
    return y

def cross_validate_p(X,Y):
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        params=[]
        # get mean and convariance for each category
        for i in range(1,4):
            index = np.where(y_train==i)
            params.append(gaussian_param(x_train[index]))
        right_cnt = 0
        # get probability for test sample under 3 distribution
        for x,y in zip(x_test,y_test):
            possible=[gaussian(mu,cov,x) for mu,cov in params]
            ret,result=0,0
            # prediction: result with max probability
            for i in range(3):
                if ret<possible[i]:
                    ret,result=possible[i],i+1
            if y == result:
                right_cnt+=1
        yield(right_cnt/y_test.shape[0])


if __name__ == '__main__':
    (X,Y)= loadData()
    result = list(cross_validate_p(X,Y))
    print(sum(result) / 10)




import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import KFold


def loadData():
    # load data
    data=pd.read_csv('HWData3.csv')
    data=np.array(data)
    m, n = data.shape
    X = data[:, 0:n - 1]
    Y = data[:, -1]
    # normalization
    std = StandardScaler()
    X = std.fit_transform(X)
    # 10-fold CV
    return X,Y

# sklearn: kernels
kernels=('gaussian','tophat','epanechnikov','exponential','linear')

def cross_validate_np(X,Y,kernel,bandwidth):
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        # get models based on data distribution
        pattern=[]
        for i in range(1,4):
            index = np.where(y_train==i)
            pattern.append(KernelDensity(kernel=kernel,bandwidth=bandwidth)
                .fit(x_train[index]))
        right_cnt=0
        # make predictions
        possible=np.array([pat.score_samples(x_test) for pat in pattern]).T
        for pos,label in zip(possible,y_test):
            ret,result=pos[0],1
            # prediction: result with max probability
            for i in (1,2):
                if ret<pos[i]:
                    ret,result=pos[i],i+1
            if label == result:
                right_cnt+=1
        yield(right_cnt/y_test.shape[0])

def fit_bandwidth(X,Y,kernel):
    ret,result=0,[] # result: best bandwidth
    acc_list,bands=[],[round(i,2) for i in np.arange(0.01,2.01,0.01)]
    for i in bands:
        acc=np.array(list(cross_validate_np(X,Y,kernel,i))).mean()
        acc_list.append(acc) # accuracy with each bandwidth
        if ret<acc: # update result
            ret,result=acc,[i]
        elif ret==acc:
            result.append(i)

    return ret,result

if __name__=='__main__':
    # load data
    (X,Y)= loadData()
    # for each kernel function, use 10-fold CV
    for kernel in kernels:
        print('kernel=%s,'%kernel,fit_bandwidth(X,Y,kernel))

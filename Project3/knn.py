import numpy as np
import pandas as pd
import math
import scipy.special as gamma
from sklearn.preprocessing import StandardScaler
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

def euDist(X, Y):
    m, n = np.shape(X)
    res = (X-Y)[:, 0:m-1]
    return np.linalg.norm(res, axis=1)

def knn(X, Y, k):
    kf = KFold(n_splits=10)
    corr_rate = []
    for train_index, test_index in kf.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        right_cnt = 0
        for x,y in zip(x_test, y_test):
            possible = []
            for i in range(1,4):
                index = np.where(y_train == i)
                dist = euDist(x_train[index], x)
                distSort = np.argsort(dist)
                r = dist[distSort[k-1]]
                p = k/(150 * r * math.sqrt(math.pi) / math.factorial(2))
                possible.append(p)
            ret, result = possible[0], 1
            for i in (1,2):
                if ret < possible[i]:
                    ret, result = possible[i], i+1
            if y == result:
                right_cnt += 1
        rate = right_cnt/15
        corr_rate.append(rate)
    return corr_rate


if __name__ == "__main__":
    (X, Y) = loadData()
    for k in (1, 3, 5, 7):
        result = []
        result = knn(X, Y, k)
        # print(result)
        print('k: %d' %k, sum(result)/10)


from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os

np.random.seed(0)


# get label
def getLabel(hot, size):
    for i in range(size):
        if hot[i] == '1':
            return i
    return -1


# load data & normalization
def loadData():
    # training set
    train = []
    with open(os.path.dirname(__file__) + '/semeion_train.csv', 'r') as train_file:
        for row in train_file:
            line = row.strip().split(' ')
            train.append(line[:-10] + [getLabel(line[-10:], 10)])
    # normalization
    train = np.array(train, dtype=float)
    m, n = np.shape(train)
    data = train[:, 0:n - 1]
    min_data = data.min(0)
    max_data = data.max(0)
    data = (data - min_data) / (max_data - min_data)
    train[:, 0:n - 1] = data
    # print(train)
    # test set
    test = []
    with open(os.path.dirname(__file__) + '/semeion_test.csv', 'r') as test_file:
        for row in test_file:
            line = row.strip().split(' ')
            test.append(line[:-10] + [getLabel(line[-10:], 10)])
    # normalization
    test = np.array(test, dtype=float)
    m, n = np.shape(test)
    data = test[:, 0:n - 1]
    min_data = data.min(0)
    max_data = data.max(0)
    data = (data - min_data) / (max_data - min_data)
    test[:, 0:n - 1] = data
    # print(test)
    return train, test


if __name__ == '__main__':
    (train_data, test_data) = loadData()
    train_label = train_data[:, -1]
    test_label = test_data[:, -1]

    for k in (1, 3, 5, 7):
        # KNN classifier
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
        # fit model
        knn.fit(train_data, train_label)
        # make predictions
        predict = knn.predict(test_data)
        # get probablities
        probility = knn.predict_proba(test_data)
        # get nearest neighbors for the last sample
        neighbor = knn.kneighbors(test_data[-1:], k, False)
        # classification accuracy
        score = knn.score(test_data, test_label, sample_weight=None)
        print('k为', k , '时， accuracy: %.2f' % (score * 100), '%')


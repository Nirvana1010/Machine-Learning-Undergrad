import scipy
from scipy import io
import numpy as np
from random import randrange
from sklearn.datasets import make_blobs
from sklearn.preprocessing import normalize, label_binarize
from sklearn import neighbors, model_selection, naive_bayes, svm, ensemble
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt


def distanceNorm(Norm, D_value):
    # initialization

    # Norm for distance
    if Norm == '1':
        counter = np.absolute(D_value)
        counter = np.sum(counter)
    elif Norm == '2':
        counter = np.power(D_value, 2)
        counter = np.sum(counter)
        counter = np.sqrt(counter)
    elif Norm == 'Infinity':
        counter = np.absolute(D_value)
        counter = np.max(counter)
    else:
        raise Exception('We will program this later......')

    return counter


def fit(features, labels, iter_ratio, k, norm):
    # initialization
    (n_samples, n_features) = np.shape(features)
    distance = np.zeros((n_samples, n_samples))
    weight = np.zeros(n_features)
    labels = list(labels)

    # compute distance
    for index_i in range(n_samples):
        for index_j in range(index_i + 1, n_samples):
            D_value = features[index_i] - features[index_j]
            distance[index_i, index_j] = distanceNorm(norm, D_value)
    distance += distance.T

    # start iteration
    for iter_num in range(int(iter_ratio * n_samples)):
        # random extract a sample
        index_i = randrange(0, n_samples, 1)
        self_features = features[index_i]

        # initialization
        nearHit = list()
        nearMiss = dict()
        n_labels = list(set(labels))
        termination = np.zeros(len(n_labels))
        del n_labels[n_labels.index(labels[index_i])]
        for label in n_labels:
            nearMiss[label] = list()
        distance_sort = list()

        # search for nearHit and nearMiss
        distance[index_i, index_i] = np.max(distance[index_i])  # filter self-distance
        for index in range(n_samples):
            distance_sort.append([distance[index_i, index], index, labels[index]])

        distance_sort.sort(key=lambda x: x[0])

        for index in range(n_samples):
            # search nearHit
            if distance_sort[index][2] == labels[index_i]:
                if len(nearHit) < k:
                    nearHit.append(features[distance_sort[index][1]])
                else:
                    termination[distance_sort[index][2] - 1] = 1
            # search nearMiss
            elif distance_sort[index][2] != labels[index_i]:
                if len(nearMiss[distance_sort[index][2]]) < k:
                    nearMiss[distance_sort[index][2]].append(features[distance_sort[index][1]])
                else:
                    termination[distance_sort[index][2] - 1] = 1

            if list(termination).count(0) == 0:
                break

        # update weight
        nearHit_term = np.zeros(n_features)
        for x in nearHit:
            nearHit += np.abs(np.power(self_features - x, 2))
        nearMiss_term = np.zeros((len(list(set(labels))), n_features))
        for index, label in enumerate(nearMiss.keys()):
            for x in nearMiss[label]:
                nearMiss_term[index] += np.abs(np.power(self_features - x, 2))
            weight += nearMiss_term[index] / (k * len(nearMiss.keys()))
        weight -= nearHit_term / k

        # print weight/(iter_ratio*n_samples);
    return weight / (iter_ratio * n_samples)


def test():
    (features, labels) = make_blobs(n_samples=500, n_features=10, centers=4)
    features = normalize(X=features, norm='l2', axis=0)
    for x in range(1, 11):
        weight = fit(features=features, labels=labels, iter_ratio=1, k=5, norm='2')
        print(weight)


if __name__ == '__main__':
    # test()

    features_struct = scipy.io.loadmat('urban.mat')
    features_X = features_struct['X']
    features_Y = features_struct['Y']
    features_Y = np.reshape(features_Y, (1, -1))

    for x in range(10):
        weight = fit(features=features_X, labels=features_Y[0], iter_ratio=1, k=5, norm='2')

    weightSort = np.argsort(weight, axis=0)

    knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights="distance")
    nb = naive_bayes.GaussianNB()
    svm = svm.SVC(kernel="linear", C=0.025, probability=True)
    rf = ensemble.RandomForestClassifier()

    # kNN
    knn_score = []
    knn_auc = []
    for x in range(1,6):
        ran = x * 25
        attr = weightSort[0:ran]
        X = features_X[:, attr]
        Y = features_Y[0]
        n_class = 9
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=0.7, random_state=0)
        y_one_hot = label_binarize(Y_test, np.arange(n_class))
        knn.fit(X_train, Y_train)
        Y_predict = knn.predict(X_test)
        Y_score = knn.predict_proba(X_test)
        score = accuracy_score(Y_test, Y_predict)
        # precision = metrics.precision_score(Y_test, Y_predict, average='micro')
        auc = metrics.roc_auc_score(y_one_hot, Y_score, average='micro')
        knn_score.append(score)
        knn_auc.append(auc)
        print("kNN %d/6  accuracy: %.2f  auc: %.2f" % (x, score, auc))

    # NB
    nb_score = []
    nb_auc = []
    for x in range(1,6):
        ran = x * 25
        attr = weightSort[0:ran]
        X = features_X[:, attr]
        Y = features_Y[0]
        n_class = 9
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=0.7, random_state=0)
        y_one_hot = label_binarize(Y_test, np.arange(n_class))
        nb.fit(X_train, Y_train)
        Y_predict = nb.predict(X_test)
        Y_score = nb.predict_proba(X_test)
        score = accuracy_score(Y_test, Y_predict)
        auc = metrics.roc_auc_score(y_one_hot, Y_score, average='micro')
        # precision = metrics.precision_score(Y_test, Y_predict, average='micro')
        nb_score.append(score)
        nb_auc.append(auc)
        print("NB %d/6  accuracy: %.2f  auc: %.2f" % (x, score, auc))

    # SVM
    svm_score = []
    svm_auc = []
    for x in range(1, 6):
        ran = x * 25
        attr = weightSort[0:ran]
        X = features_X[:, attr]
        Y = features_Y[0]
        n_class = 9
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=0.7, random_state=0)
        y_one_hot = label_binarize(Y_test, np.arange(n_class))
        svm.fit(X_train, Y_train)
        Y_predict = svm.predict(X_test)
        Y_score = svm.predict_proba(X_test)
        score = accuracy_score(Y_test, Y_predict)
        auc = metrics.roc_auc_score(y_one_hot, Y_score, average='micro')
        # precision = metrics.precision_score(Y_test, Y_predict, average='micro')
        svm_score.append(score)
        svm_auc.append(auc)
        print("SVM %d/6  accuracy: %.2f  auc: %.2f" % (x, score, auc))

    # Random Forests
    rf_score = []
    rf_auc = []
    for x in range(1, 6):
        ran = x * 25
        attr = weightSort[0:ran]
        X = features_X[:, attr]
        Y = features_Y[0]
        n_class = 9
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=0.7, random_state=0)
        y_one_hot = label_binarize(Y_test, np.arange(n_class))
        rf.fit(X_train, Y_train)
        Y_predict = rf.predict(X_test)
        Y_score = rf.predict_proba(X_test)
        score = accuracy_score(Y_test, Y_predict)
        auc = metrics.roc_auc_score(y_one_hot, Y_score, average='micro')
        # precision = metrics.precision_score(Y_test, Y_predict, average='micro')
        rf_score.append(score)
        rf_auc.append(auc)
        print("Random Forests %d/6  accuracy: %.2f  auc: %.2f" % (x, score, auc))

    # plot
    plot_x = [1/6, 2/6, 3/6, 4/6, 5/6]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(plot_x, knn_score, label="kNN")
    ax.plot(plot_x, nb_score, label="NB")
    ax.plot(plot_x, svm_score, label="SVM")
    ax.plot(plot_x, rf_score, label="Random Forests")
    plt.xlabel('ratio of features')
    plt.ylabel('precision')
    ax.set_title('precison')
    ax.legend()
    # plt.show()
    plt.savefig('relief-F score.png')

    plot_x = [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(plot_x, knn_auc, label="kNN")
    ax.plot(plot_x, nb_auc, label="NB")
    ax.plot(plot_x, svm_auc, label="SVM")
    ax.plot(plot_x, rf_auc, label="Random Forests")
    plt.xlabel('ratio of features')
    plt.ylabel('precision')
    ax.set_title('auc')
    ax.legend()
    # plt.show()
    plt.savefig('relief-F auc.png')


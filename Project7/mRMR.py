import warnings
from scipy import io
import numpy as np
import scipy
from sklearn import neighbors, model_selection, naive_bayes, svm, ensemble
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.preprocessing import normalize, label_binarize
import matplotlib.pyplot as plt


class MRMR():
    def __init__(self, feature_num):
        """
        mRMR is a feature selection which maximises the feature-label correlation and minimises
        the feature-feature correlation. this implementation can only applied for numeric values,
        read more about mRMR, please refer :ref:`https://blog.csdn.net/littlely_ll/article/details/71749776`.

        :param feature_num: selected number of features
        """
        self.feature_num = feature_num
        self._selected_features = []


    def _check_array(self, data):
        if isinstance(data, list):
            data = np.asarray(data)
        assert isinstance(data, np.ndarray), "input should be an array!"
        return data


    def fit(self, X, y):
        """
        fit an array data

        :param X: a numpy array

        :param y: the label, a list or one dimension array

        :return:
        """
        X = self._check_array(X)
        y = self._check_array(y)
        assert X.shape[0] == len(y), "X and y not in the same length!"

        if self.feature_num > X.shape[1]:
            self.feature_num = X.shape[1]
            warnings.warn("The feature_num has to be set less or equal to {}".format(X.shape[1]), UserWarning)

        MIs = self.feature_label_MIs(X, y)
        max_MI_arg = np.argmax(MIs)

        selected_features = []

        MIs = list(zip(range(len(MIs)), MIs))
        selected_features.append(MIs.pop(int(max_MI_arg)))

        while True:
            max_theta = float("-inf")
            max_theta_index = None

            for mi_outset in MIs:
                ff_mis = []
                for mi_inset in selected_features:
                    ff_mi = self.feature_feature_MIs(X[:, mi_outset[0]], X[:, mi_inset[0]])
                    ff_mis.append(ff_mi)
                theta = mi_outset[1] - 1 / len(selected_features) * sum(ff_mis)
                if theta >= max_theta:
                    max_theta = theta
                    max_theta_index = mi_outset
            selected_features.append(max_theta_index)
            MIs.remove(max_theta_index)

            if len(selected_features) >= self.feature_num:
                break

        self._selected_features = [ind for ind, mi in selected_features]

    def transform(self, X):
        return X[:, self._selected_features]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def entropy(self, c):
        """
        entropy calculation

        :param c:

        :return:
        """
        c_normalized = c / float(np.sum(c))
        c_normalized = c_normalized[np.nonzero(c_normalized)]
        H = -sum(c_normalized * np.log2(c_normalized))
        return H

    def feature_label_MIs(self, arr, y):
        """
        calculate feature-label mutual information

        :param arr:

        :param y:

        :return:
        """
        m, n = arr.shape
        MIs = []
        p_y = np.histogram(y)[0]
        h_y = self.entropy(p_y)

        for i in range(n):
            p_i = np.histogram(arr[:, i])[0]
            p_iy = np.histogram2d(arr[:, 0], y)[0]

            h_i = self.entropy(p_i)
            h_iy = self.entropy(p_iy)

            MI = h_i + h_y - h_iy
            MIs.append(MI)
        return MIs

    def feature_feature_MIs(self, x, y):
        """
        calculate feature-feature mutual information

        :param x:

        :param y:

        :return:
        """
        p_x = np.histogram(x)[0]
        p_y = np.histogram(y)[0]
        p_xy = np.histogram2d(x, y)[0]

        h_x = self.entropy(p_x)
        h_y = self.entropy(p_y)
        h_xy = self.entropy(p_xy)

        return h_x + h_y - h_xy

    @property
    def important_features(self):
        return self._selected_features


if __name__ == '__main__':
    features_struct = scipy.io.loadmat('urban.mat')
    features_X = features_struct['X']
    features_Y = features_struct['Y']
    features_Y = np.reshape(features_Y, (1, -1))

    knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights="distance")
    nb = naive_bayes.GaussianNB()
    svm = svm.SVC(kernel="linear", C=0.025, probability=True)
    rf = ensemble.RandomForestClassifier()

    mrmr = MRMR(125)
    mrmr.fit(features_X, features_Y[0])

    # kNN
    knn_score = []
    knn_auc = []
    for x in range(1, 6):
        ran = x * 25
        attr = mrmr.important_features[0:ran]
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
    for x in range(1, 6):
        ran = x * 25
        attr = mrmr.important_features[0:ran]
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
        attr = mrmr.important_features[0:ran]
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
        attr = mrmr.important_features[0:ran]
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
    plot_x = [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]
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
    plt.savefig('mRMR score.png')

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
    plt.savefig('mRMR auc.png')

    print('finish')
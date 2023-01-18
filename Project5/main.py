from matplotlib import pyplot as plt
import numpy as np
import os
from sklearn.datasets.samples_generator import make_blobs

import cluster


def create_data(centers, num=100, std=0.7):
    '''
    Generate data for clustering
    :param centers: center for each cluster
    :param num: no. of samples
    :param std: std in each cluster
    :return: data for clustering
    '''
    X, labels_true = make_blobs(n_samples=num, centers=centers, cluster_std=std)
    return X, labels_true


def plot_data(*data):
    '''
    Plotting data for clustering
    :param data: dataset, labels
    :return: None
    '''
    X, labels_true, labels_predict, cnt = data
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = 'rgbyckm'  # different color for each cluster
    markers = 'o^sP*DX'
    for i in range(len(labels_true)):
        predict = labels_predict[i]
        ax.scatter(X[i, 0], X[i, 1], label="cluster %d" % labels_true[i],
                   color=colors[predict % len(colors)], marker=markers[labels_true[i] % len(markers)], alpha=0.5)
        # color: prediciton, shape: true label

    # ax.legend(loc="best",framealpha=0.5)
    ax.set_xlabel("X[0]")
    ax.set_ylabel("X[1]")
    ax.set_title("data")
    plt.savefig(os.path.dirname(__file__) + '/images/p%d4' % cnt)

METHOD_APPLY = [cluster.singleLinkage,cluster.completeLinkage,cluster.averageLinkage]

# generate data

centers=[[1,1,1],[3,5,7],[5,4,5],[2,2,3]] # center for each cluster
# centers = [[1, 1, 1], [4, 8, 4]]
X, labels_true = create_data(centers, 2000, 0.5)  # generate data for clustering
np.savetxt(os.path.dirname(__file__) + '/data.dat', X)
np.savetxt(os.path.dirname(__file__) + '/label.dat', labels_true)

print("generate data finish!")

cnt = 0
for method in METHOD_APPLY:
    model = cluster.AgglomerativeClustering()
    model.fit(X, method)
    k = 4
    plot_data(X, labels_true, model.label(k), cnt)
    cnt += 1

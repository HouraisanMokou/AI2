import time

import numpy as np
# from sklearn.manifold import TSNE
from openTSNE.sklearn import TSNE
from sklearn.cluster import KMeans, DBSCAN, MeanShift, SpectralClustering
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
from collections import Counter

from loguru import logger


def show_VAE_by_TSNE(latent_space, labels, num_classes=10):
    tsne = TSNE(n_components=2)
    tsne_space = tsne.fit_transform(latent_space)

    for i in range(num_classes):
        group = tsne_space[labels == i, ::]
        # group_labels = np.ones(group.shape[0])*i
        plt.scatter(group[:, 0], group[:, 1], s=2)

    plt.legend([str(_) for _ in range(num_classes)])
    plt.show()


def show_VAE_by_multi_TSNE(latent_space, labels, num_classes=10):
    tsne = TSNE(n_components=2)
    for i in range(num_classes):
        group = tsne.fit_transform(latent_space[labels == i, ::])
        plt.scatter(group[:, 0], group[:, 1], s=2)
        plt.show()


def my_cluster(data, eps=0.2):
    st = data[0, :]
    cluster1 = []
    left = set(map(tuple, list(data.copy())))
    while True:
        left_numpy = np.asarray(list(left))
        dis = np.linalg.norm(left_numpy - st, 2, axis=1)
        neigh = left_numpy[dis < eps, ::]
        if len(neigh) == 0:
            break
        cluster1 += map(tuple, list(neigh))
        left = left.difference(set(map(tuple, list(neigh))))
        st = np.asarray(list(left))[0, :]
    cluster1 = set(cluster1)
    data_set = set(map(tuple, list(data.copy())))
    cluster2 = data_set.difference(cluster1)
    return np.asarray(list(cluster1)), np.asarray(list(cluster2))


def show_VAE_by_single_TSNE(latent_space, labels=None):
    tsne = TSNE(n_components=2)
    group = tsne.fit_transform(latent_space)
    logger.info(f'has tune to tsne view')
    if labels is None:
        plt.scatter(group[:, 0], group[:, 1], s=2)
        plt.show()
    else:
        plt.figure(figsize=(5, 10))
        plt.subplot(2, 1, 1)
        for i in range(10):
            plt.scatter(group[labels == i, 0], group[labels == i, 1], s=2)
        plt.legend([f'c{_}' for _ in range(10)])
        plt.subplot(2, 1, 2)
        group = (group - np.mean(group)) / np.std(group)
        # c1, c2 = my_cluster(group)
        # plt.scatter(c1[:, 0], c1[:, 1], s=2)
        # plt.scatter(c2[:, 0], c2[:, 1], s=2)
        # plt.show()
        cluster = DBSCAN(eps=0.12, min_samples=5)
        clustered_label = cluster.fit_predict(group).squeeze()
        max_label = np.max(clustered_label).astype(int)
        cur_cnts = np.asarray([np.sum(clustered_label == i) for i in range(max_label + 1)])
        logger.info(f'cluster number: {cur_cnts}')
        main_cluster = np.argmax(cur_cnts)
        real_clustered_label = np.zeros(clustered_label.shape)
        sorted_cnts = np.sort(cur_cnts)
        if max_label > 1:
            clustered_label[clustered_label > 1] = 0
        else:
            if main_cluster == 1:
                clustered_label[clustered_label == 1] = 2
                clustered_label[clustered_label == 0] = 1
                clustered_label[clustered_label == 2] = 0

        max_label = np.max(clustered_label).astype(int)
        for i in range(max_label + 1):
            plt.scatter(group[clustered_label == i, 0], group[clustered_label == i, 1], s=2)
        plt.savefig(f'./{time.time()}.png')
        return clustered_label == 0


def cluster_single_class(latent_space, labels=None):
    tsne = TSNE(n_components=2)
    group = tsne.fit_transform(latent_space)
    logger.info(f'has tune to tsne view')
    if labels is None:
        plt.scatter(group[:, 0], group[:, 1], s=2)
        plt.show()
    else:
        plt.figure(figsize=(5, 10))
        plt.subplot(2, 1, 1)
        for i in range(10):
            plt.scatter(group[labels == i, 0], group[labels == i, 1], s=2)
        plt.legend([f'c{_}' for _ in range(10)])
        plt.subplot(2, 1, 2)
        group = (group - np.mean(group)) / np.std(group)
        cluster = DBSCAN(eps=0.12, min_samples=5)
        clustered_label = cluster.fit_predict(group).squeeze()
        max_label = np.max(clustered_label).astype(int)
        ub = max_label + 1

        cluster_res = []
        for i in range(ub):
            cluster_idx = np.asarray(np.where(clustered_label == i)).squeeze()
            cluster_res.append(tuple(list(cluster_idx)))
        cluster_res.sort(key=lambda x: len(x), reverse=True)
        if len(cluster_res) > 2:
            for i in range(len(cluster_res)):
                if i != 0 and i != 1:
                    cluster_res[0] = tuple(list(cluster_res[0]) + list(cluster_res[i]))
            tmp_cluster = []
            thres = 100
            for i in range(len(cluster_res)):
                if len(cluster_res[i]) > thres:
                    tmp_cluster.append(cluster_res[i])
            cluster_res = tmp_cluster
        for i in range(len(cluster_res)):
            plt.scatter(group[cluster_res[i], 0], group[cluster_res[i], 1], s=2)
        plt.savefig(f'./{time.time()}.png')
        # plt.show()
        return cluster_res

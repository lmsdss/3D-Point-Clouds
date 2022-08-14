import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csgraph
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons
from sklearn.metrics import pairwise_distances


class SpectralClustering(object):

    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, **kwargs):
        self.__K = n_clusters
        self.__labels = None

    def fit(self, data):
        N, _ = data.shape
        # 计算两个矩阵的成对距离
        A = pairwise_distances(data)
        gamma = np.var(A) / 4
        A = np.exp(-A ** 2 / (2 * gamma ** 2))
        # 拉普拉斯矩阵
        L = csgraph.laplacian(A, normed=True)
        # 谱分解
        eigval, eigvec = np.linalg.eig(L)
        idx_k_smallest = np.where(eigval < np.partition(eigval, self.__K)[self.__K])
        features = np.hstack([eigvec[:, i] for i in idx_k_smallest])
        # 使用KMeans++聚类
        k_means = KMeans(init='k-means++', n_clusters=self.__K, tol=1e-6)
        k_means.fit(features)
        # 获得类别编号
        self.__labels = k_means.labels_

    def predict(self, data):
        return np.copy(self.__labels)


def generate_dataset(N=200, noise=0.07, random_state=50, visualize=False):
    X, y = make_moons(N, noise=noise, random_state=random_state)

    # 是否可视化原始数据
    if visualize:
        plt.title('Raw Dataset')
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='rainbow')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    return X


if __name__ == '__main__':
    # 生成数据
    X = generate_dataset(visualize=True)

    K = 2
    spectral_clustering = SpectralClustering(n_clusters=K)
    spectral_clustering.fit(X) # fit
    category = spectral_clustering.predict(X)  # predict

    # visualize:
    # 可视化:
    color = ['green', 'blue', 'red', 'pink']
    labels = ['cluster0', 'cluster1', 'cluster2', 'cluster3']

    for k in range(K):
        plt.scatter(X[category == k][:, 0], X[category == k][:, 1], c=color[k], label=labels[k])

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Spectral Clustering')
    plt.show()

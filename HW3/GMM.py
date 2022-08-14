# 文件功能：实现 GMM 算法

import numpy as np
from numpy import *
import pylab
import random, math
import KMeans
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal

plt.style.use('seaborn')


class GMM(object):
    def __init__(self, n_clusters, max_iter=50):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.k = n_clusters
        # 更新Mu
        self.mu = None  # mean
        # 更新Var
        self.cov = None  # 协方差矩阵
        # 更新先验概率
        self.prior = None  # 权重
        # 更新后验概率
        self.posteriori = None

    def fit(self, data):
        # 作业3
        # 屏蔽开始
        # 1.初始化
        # 使用K-means中心点初始化Mu
        k_means = KMeans.K_Means(n_clusters=self.k)
        k_means.fit(data)
        # 将Mu的初始值设为k-means的中心点
        self.mu = np.asarray(k_means.centers_)
        # 将cov初始化K*2*2的单位矩阵
        self.cov = np.asarray([eye(2, 2)] * self.k)
        # 对先验概率进行均等分
        self.prior = np.asarray([1 / self.k] * self.k).reshape(self.k, 1)
        # 将后验概率初始化为K*N的矩阵
        self.posteriori = np.zeros((self.k, len(data)))

        for _ in range(self.max_iter):
            # 2.E-step算出后验概率p，每个点属于哪个类的概率
            for k in range(self.k):
                # 提取每个点的概率密度分布
                self.posteriori[k] = multivariate_normal.pdf(x=data, mean=self.mu[k], cov=self.cov[k])
            # diag方便进行对应的数据乘法运算 ravel将矩阵里所有元素变为列表
            self.posteriori = np.dot(diag(self.prior.ravel()), self.posteriori)
            # 归一化
            self.posteriori /= np.sum(self.posteriori, axis=0)
            # 3.M-step使用MLE算出高斯模型的参数
            self.Nk = np.sum(self.posteriori, axis=1)
            self.mu = np.asarray([np.dot(self.posteriori[k], data) / self.Nk[k] for k in range(self.k)])
            self.cov = np.asarray([np.dot((data - self.mu[k]).T,
                                          np.dot(np.diag(self.posteriori[k].ravel()), data - self.mu[k])) / self.Nk[k]
                                   for k in range(self.k)])
            self.prior = np.asarray([self.Nk / self.k]).reshape(self.k, 1)
        self.fitted = True
        # 屏蔽结束

    def predict(self, data):
        # 屏蔽开始
        result = []
        if not self.fitted:
            print('Unfitted')
            return result
        # 每个点的后验概率的最大值
        result = np.argmax(self.posteriori, axis=0)
        return result
        # 屏蔽结束


# 生成仿真数据
def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()
    return X


if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)
    # 分类数
    K = 3
    gmm = GMM(n_clusters=K)
    gmm.fit(X)
    category = np.array(gmm.predict(X))

    # 可视化:
    color = ['green', 'blue', 'red', 'pink']
    labels_ = ['cluster0', 'cluster1', 'cluster2', 'cluster3']

    for k in range(K):
        plt.scatter(X[category == k][:, 0], X[category == k][:, 1], c=color[k], label=labels_[k])

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('GMM')
    plt.show()

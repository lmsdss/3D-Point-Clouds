# 文件功能： 实现 K-Means 算法

import random
import matplotlib.pyplot as plt
import numpy as np


class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    def fit(self, data):
        # 作业1
        # 计算模型参数
        # 屏蔽开始
        # 随机选取K个中心点
        self.centers_ = data[random.sample(range(data.shape[0]), self.k_)]
        # 深拷贝中心点
        old_centers_ = np.copy(self.centers_)
        # 分类点标签
        labels = [[] for i in range(self.k_)]

        for _ in range(self.max_iter_):
            # E-Step 计算每个点所属点类别
            for idx, point in enumerate(data):
                # 使用二范数求解每个点对旧的聚类中心的距离
                diff = np.linalg.norm(old_centers_ - point, axis=1)
                labels[np.argmin(diff)].append(idx)
            # M-Step 根据相应类别里的点重新计算出聚类中心
            for i in range(self.k_):
                points = data[labels[i], :]
                self.centers_[i] = points.mean(axis=0)

            if np.sum(np.abs(self.centers_ - old_centers_)) < self.tolerance_ * self.k_:
                break
            old_centers_ = np.copy(self.centers_)

        self.fitted = True
        # 屏蔽结束

    def predict(self, p_datas):
        result = []
        # 作业2
        # 计算每个点的类别
        # 屏蔽开始
        if not self.fitted:
            print('Unfitted')
            return result
        for point in p_datas:
            # 使用二范数求解每个点对新的聚类中心的距离
            diff = np.linalg.norm(self.centers_ - point, axis=1)
            result.append(np.argmin(diff))
        # 屏蔽结束
        return result


if __name__ == '__main__':
    # 分类数
    K = 2
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = K_Means(n_clusters=K)
    k_means.fit(x) # fit:
    category = np.array(k_means.predict(x))    # predict:
    print(category)

    # 可视化:
    color = ['green', 'blue', 'red', 'pink']
    labels_ = ['cluster0', 'cluster1', 'cluster2', 'cluster3']

    for i in range(K):
        plt.scatter(x[category == i][:, 0], x[category == i][:, 1], c=color[i], label=labels_[i])

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('KMeans')
    plt.show()

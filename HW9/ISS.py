import pandas as pd
import open3d as o3d
import numpy as np
import heapq
import open3d
import numpy


class ISS_detector:
    def __init__(self) -> None:
        self.__point_cloud = None
        self.__radius = 1
        self.__gamma_32 = 1.2
        self.__gamma_21 = 1.2
        self.__l3_min = 0.0001
        self.__k_min = 3
        self.__feature_index = None

        # 小顶堆，用于非极大值抑制
        self.__pq = []

        # 保存被抑制的点的集合
        self.__suppressed = set()

        # 记录每个点的邻居点，用于进行极大值抑制
        self.__neighbors_every_point = []

    def set_pointcloud(self, pointcloud: open3d.geometry.PointCloud) -> None:
        self.__point_cloud = pointcloud
        # 构建搜索树
        self.__search_tree = o3d.geometry.KDTreeFlann(self.__point_cloud)

    def set_attribute(self, radius: float, gamma_32: float, gamma_21: float, l3_min: float, k_min: int) -> None:
        self.__radius = radius
        self.__gamma_32 = gamma_32
        self.__gamma_21 = gamma_21
        self.__l3_min = l3_min
        self.__k_min = k_min

    def __get_cov_matrix(self, center, pointclouds, neigbor_idx, weight):

        # 获取邻点
        neighbors = pointclouds[neigbor_idx]

        # 获取距离
        distance = neighbors - center

        # 加权权重
        weight = np.asarray(weight)
        weight = 1.0 / weight
        weight = np.reshape(weight, (-1,))

        # 协方差矩阵
        cov = 1.0 / weight.sum() * np.dot(distance.T, np.dot(np.diag(weight), distance))

        return cov

    def __non_maximum_suppression(self) -> None:

        # 进行非极大值抑制
        while (self.__pq):
            # 取出l3最大的点
            _, idx_centor = heapq.heappop(self.__pq)

            # 判断取出的点是否已经被抑制
            if not idx_centor in self.__suppressed:
                # 取出当前点对应的邻居
                neighbor = self.__neighbors_every_point[idx_centor]
                # 排除自身
                neighbor = neighbor[1:]
                # 加入被抑制点的集合
                for _i in neighbor:
                    self.__suppressed.add(_i)

            else:
                continue

    def __filter_by_param(self, df_data: dict) -> None:
        df_data = pd.DataFrame.from_dict(df_data)

        # 排除非极大值点
        df_data = df_data.loc[df_data["id"].apply(lambda id: not id in self.__suppressed), df_data.columns]

        # 排除不符合协方差要求的点
        df_data = df_data.loc[(df_data["l1"] > df_data["l2"] * self.__gamma_21) \
                              & (df_data["l2"] > df_data["l3"] * self.__gamma_32) \
                              & (df_data["l3"] > self.__l3_min), df_data.columns]

        self.__feature_index = df_data["id"].values

    def __eig_and_sort(self, cov: numpy.ndarray) -> numpy.ndarray:
        # 对协方差矩阵进行特征值分解
        eigen_values, _ = np.linalg.eig(cov)

        # 对特征值进行排序
        eigen_values = eigen_values[np.argsort(eigen_values)[::-1]]

        return eigen_values

    def detect(self) -> None:

        # 保存每个点的id lamda1，2，3
        df_data = {
            "id": [],
            "l1": [],
            "l2": [],
            "l3": [],
        }

        pointclouds = np.asarray(self.__point_cloud.points)

        # 保存每个点的邻居点个数的缓存，用于加速计算
        num_neighbor_cach = np.zeros((pointclouds.shape[0], 1))

        # 遍历每一个点
        for idx_centor, center in enumerate(pointclouds):

            # 计算对应的协方差矩阵
            # 计算给定半径内的邻点
            [k, idx, _] = self.__search_tree.search_radius_vector_3d(center, self.__radius)
            self.__neighbors_every_point.append(idx)

            # 排除邻点个数小于给定阈值的点
            if k < self.__k_min:
                # 构造小顶堆，用于非极大值抑制，注意这里是-l3
                heapq.heappush(self.__pq, (-0.0, idx_centor))

                df_data["id"].append(idx_centor)
                df_data["l1"].append(0.0)
                df_data["l2"].append(0.0)
                df_data["l3"].append(0.0)

                continue

            # 权重矩阵
            w = []

            # 遍历每个邻点，并记录每个点的邻点个数
            for index in idx:

                # 如果没有被计算过
                if (num_neighbor_cach[index] == 0):
                    [k_, _, _] = self.__search_tree.search_radius_vector_3d(pointclouds[index], self.__radius)
                    num_neighbor_cach[index] = k_

                # 保存当前判定点对应的每个邻居的邻居个数，用于计算权重矩阵
                w.append(num_neighbor_cach[index])

            # 获取协方差矩阵
            cov = self.__get_cov_matrix(center, pointclouds, idx, w)

            # 获取排序后的特征值
            eigen_values = self.__eig_and_sort(cov)

            # 构造小顶堆，用于非极大值抑制，注意这里是-l3
            heapq.heappush(self.__pq, (-eigen_values[2], idx_centor))

            df_data["id"].append(idx_centor)
            df_data["l1"].append(eigen_values[0])
            df_data["l2"].append(eigen_values[1])
            df_data["l3"].append(eigen_values[2])

        # 非极大值抑制
        self.__non_maximum_suppression()

        # 根据参数进行过滤并获取特征点对应的索引
        self.__filter_by_param(df_data)

    def get_feature_index(self) -> numpy.ndarray:
        return self.__feature_index

    def get_feature_points(self) -> numpy.ndarray:
        return np.asarray(self.__point_cloud.points)[self.__feature_index, :]

import numpy as np
import open3d as o3d

import numpy
import open3d


class FPFH_decriptor:
    def __init__(self) -> None:
        self.__B = 11
        self.__pointclouds = None
        self.__feature_index = None
        self.__seacrch_tree = None
        self.__radius = 1
        self.__keypoints = None
        self.__descriptors = None
        self.__search_tree = None
        pass

    def set_pointclouds(self, pointclouds: open3d.geometry.PointCloud) -> None:
        self.__pointclouds = pointclouds
        self.__search_tree = o3d.geometry.KDTreeFlann(self.__pointclouds)

    def set_attribute(self, B: int, radius: float) -> None:
        self.__B = B
        self.__radius = radius

    def set_keypoints(self, keypoints: numpy.ndarray) -> None:
        self.__keypoints = keypoints

    def __SFPH(self, keypoint: numpy.ndarray) -> None:
        # 获取点云
        points = np.asarray(self.__pointclouds.points)
        normals = np.asarray(self.__pointclouds.normals)

        # 获取邻居节点
        [k, idx_neighbors, _] = self.__search_tree.search_radius_vector_3d(keypoint, self.__radius)

        # 获取n1
        n1 = normals[idx_neighbors[0]]

        # 移除关键点本身
        idx_neighbors = idx_neighbors[1:]

        # 计算 (p2-p1)/norm(p2-p1)
        diff = points[idx_neighbors] - keypoint
        diff = diff / np.reshape(np.linalg.norm(diff, ord=2, axis=1), (k - 1, 1))

        u = n1
        v = np.cross(u, diff)
        w = np.cross(u, v)

        # 计算n2
        n2 = normals[idx_neighbors]

        # 计算alpha
        alpha = np.reshape((v * n2).sum(axis=1), (k - 1, 1))

        # 计算phi
        phi = np.reshape((u * diff).sum(axis=1), (k - 1, 1))

        # 计算 theta
        theta = np.reshape(np.arctan2((w * n2).sum(axis=1), (u * n2).sum(axis=1)), (k - 1, 1))

        # 计算相应的直方图
        alpha_hist = np.reshape(np.histogram(alpha, self.__B, range=[-1.0, 1.0])[0], (1, self.__B))
        phi_hist = np.reshape(np.histogram(phi, self.__B, range=[-1.0, 1.0])[0], (1, self.__B))
        theta_hist = np.reshape(np.histogram(theta, self.__B, range=[-3.14, 3.14])[0], (1, self.__B))

        # 组成描述子
        fpfh = np.hstack((alpha_hist, phi_hist, theta_hist))

        return fpfh

    def describe(self) -> None:
        pointclouds = np.asarray(self.__pointclouds.points)

        self.__descriptors = numpy.ndarray((0, 3 * self.__B))

        for keypoint in self.__keypoints:
            # 寻找keypoint的邻居点
            [k, idx_neighbors, _] = self.__search_tree.search_radius_vector_3d(keypoint, self.__radius)

            # 移除关键点本身
            idx_neighbors = idx_neighbors[1:]

            # 计算权重
            w = 1.0 / np.linalg.norm(keypoint - pointclouds[idx_neighbors], ord=2, axis=1)

            # 计算邻居的SPFH
            neighbors_SPFH = np.reshape(np.asarray([self.__SFPH(pointclouds[i]) for i in idx_neighbors]),
                                        (k - 1, 3 * self.__B))

            # 计算自身的描述子
            self_SFPH = self.__SFPH(keypoint)

            # 计算最终的FPFH
            neighbors_SPFH = 1.0 / (k - 1) * np.dot(w, neighbors_SPFH)

            # 获取描述子并归一化
            finial_FPFH = self_SFPH + neighbors_SPFH
            finial_FPFH = finial_FPFH / np.linalg.norm(finial_FPFH)

            self.__descriptors = numpy.vstack((self.__descriptors, finial_FPFH))

    def get_descriptors(self) -> list:
        return self.__descriptors

import copy
import concurrent.futures

import numpy as np
from scipy.spatial.distance import pdist
import open3d as o3d


class RANSAC_ICP:
    def __init__(self, builder) -> None:
        self.max_works = builder.max_works
        self.num_samples = builder.num_samples
        self.max_correspondence_distance = builder.max_correspondence_distance
        self.max_iteration = builder.max_iteration
        self.max_validation = builder.max_validation
        self.max_edge_length_ratio = builder.max_edge_length_ratio
        self.normal_angle_threshold = builder.normal_angle_threshold

        self.__pcd_source = None
        self.__pcd_target = None
        self.__source_features = None
        self.__target_features = None
        # 为目标点云构建搜索树:
        self.__search_tree_target = None

    def __str__(self) -> str:
        info = ("RANSAC_ICP:" + "\n" + "max_works:{}".format(self.max_works) + "\n" +
                "num_samples:{}".format(self.num_samples) + "\n" +
                "max_correspondence_distance:{}".format(self.max_correspondence_distance) + "\n" +
                "max_iteration:{}".format(self.max_iteration) + "\n" +
                "max_validation:{}".format(self.max_validation) + "\n" +
                "max_edge_length_ratio:{}".format(self.max_edge_length_ratio) + "\n" +
                "normal_angle_threshold:{}".format(self.normal_angle_threshold))

        return info

    def ransac_match(self) -> np.matrix:
        # 获取对应的特征空间匹配对:
        matches = self.__get_potential_matches()

        # RANSAC:
        N, _ = matches.shape
        idx_matches = np.arange(N)

        # SE3
        T = None

        # 构造一个生成器，用来随机选取四个点
        proposal_generator = (
            matches[np.random.choice(idx_matches, self.num_samples, replace=False)] for _ in iter(int, 1)
        )

        # 验证器，用来根据给定的检查参数获取符合要求的匹配对:
        validator = lambda proposal: self.__is_valid_match(proposal)

        # 并行执行,找到一个满足要求的初始解
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_works) as executor:
            for T in map(validator, proposal_generator):
                if not (T is None):
                    break

        print('[RANSAC ICP]: Get first valid proposal:\n')
        print(T)

        print("Start registration...")
        # ICP迭代一次，获取用于迭代的初始解
        best_result = self.__icp_iter(T)

        # RANSAC验证:这步根据实践可以省略
        num_validation = 0
        for i in range(self.max_iteration):
            # get proposal:
            T = validator(next(proposal_generator))

            # 检查有效性
            if (not (T is None)) and (num_validation < self.max_validation):
                num_validation += 1

                # 优化所有关键点的估计
                result = self.__icp_iter(T)

                # 更新最好的结果:
                best_result = best_result if best_result.fitness > result.fitness else result

                if num_validation == self.max_validation:
                    break

        return best_result

    def __icp_iter(self, T):

        # source点云中的点数:
        N = len(self.__pcd_source.points)

        # 评估两次迭代之间的相对变化用于提前停止:
        result_prev = result_curr = o3d.registration.evaluate_registration(
            self.__pcd_source, self.__pcd_target, self.max_correspondence_distance, T
        )

        for _ in range(self.max_iteration):
            # TODO: transform is actually an in-place operation. deep copy first otherwise the result will be WRONG
            pcd_source_current = copy.deepcopy(self.__pcd_source)
            # transform:
            pcd_source_current = pcd_source_current.transform(T)

            # 查找相当的
            matches = []
            for n in range(N):
                query = np.asarray(pcd_source_current.points)[n]
                _, idx_nn_target, dis_nn_target = self.__search_tree_target.search_knn_vector_3d(query, 1)

                if dis_nn_target[0] <= self.max_correspondence_distance:
                    matches.append(
                        [n, idx_nn_target[0]]
                    )
            matches = np.asarray(matches)

            if len(matches) >= 4:
                # ICP:
                P = np.asarray(self.__pcd_source.points)[matches[:, 0]]
                Q = np.asarray(self.__pcd_target.points)[matches[:, 1]]
                T = self.__solve_icp(P, Q)

                # 评估
                result_curr = o3d.registration.evaluate_registration(
                    self.__pcd_source, self.__pcd_target, self.max_correspondence_distance, T
                )

                # 如果没有显著改善
                if self.__shall_terminate(result_curr, result_prev):
                    print('[RANSAC ICP]: Early stopping.')
                    break

        return result_curr

    def __shall_terminate(self, result_curr, result_prev):
        # 相对适应性提高:
        relative_fitness_gain = result_curr.fitness / result_prev.fitness - 1

        return relative_fitness_gain < 0.01

    # 用来检测给定给的匹配对在给定参数下是否有效
    def __is_valid_match(self, proposal):
        idx_source, idx_target = proposal[:, 0], proposal[:, 1]

        # 1.法向量方向检查
        if not self.normal_angle_threshold is None:
            # 获取对应的法向量匹配对:
            normals_source = np.asarray(self.__pcd_source.normals)[idx_source]
            normals_target = np.asarray(self.__pcd_target.normals)[idx_target]

            # 检测对应特征点法向量之间的差距:
            normal_cos_distances = (normals_source * normals_target).sum(axis=1)
            is_valid_normal_match = np.all(normal_cos_distances >= np.cos(self.normal_angle_threshold))

            if not is_valid_normal_match:
                return None

        # 获取相关点
        points_source = np.asarray(self.__pcd_source.points)[idx_source]
        points_target = np.asarray(self.__pcd_target.points)[idx_target]

        # 2.几何相似性检查:
        pdist_source = pdist(points_source)
        pdist_target = pdist(points_target)
        is_valid_edge_length = np.all(
            np.logical_and(
                pdist_source > self.max_edge_length_ratio * pdist_target,
                pdist_target > self.max_edge_length_ratio * pdist_source
            )
        )
        if not is_valid_edge_length:
            return None

        # 3.相关距离检查
        T = self.__solve_icp(points_source, points_target)  # 两对点之间的变换矩阵
        R, t = T[0:3, 0:3], T[0:3, 3]
        # 判断经过变换之后的距离
        deviation = np.linalg.norm(
            points_target - np.dot(points_source, R.T) - t,
            axis=1
        )
        is_valid_correspondence_distance = np.all(deviation <= self.max_correspondence_distance)

        # 有效则返回初步的变换矩阵，否则返回None
        return T if is_valid_correspondence_distance else None

    # 闭式求解icp
    def __solve_icp(self, source, target):
        # 计算均值:
        up = source.mean(axis=0)
        uq = target.mean(axis=0)

        # 去重心化后的点云:
        P_centered = source - up
        Q_centered = target - uq

        # SVD分解求R和T
        U, s, V = np.linalg.svd(np.dot(Q_centered.T, P_centered), full_matrices=True, compute_uv=True)
        R = np.dot(U, V)
        t = uq - np.dot(R, up)

        # 将R和T变换为变换矩阵的格式
        T = np.zeros((4, 4))
        T[0:3, 0:3] = R
        T[0:3, 3] = t
        T[3, 3] = 1.0

        return T

    def set_source_pointscloud(self, pointscloud: o3d.geometry.PointCloud) -> None:
        self.__pcd_source = pointscloud

    def set_target_pointscloud(self, pointscloud: o3d.geometry.PointCloud) -> None:
        self.__pcd_target = pointscloud
        self.__search_tree_target = o3d.geometry.KDTreeFlann(self.__pcd_target)

    def set_source_features(self, source_features: np.ndarray) -> None:
        self.__source_features = source_features

    def set_target_features(self, target_features: np.ndarray) -> None:
        self.__target_features = target_features

    def __get_potential_matches(self) -> np.ndarray:
        # 在高维空间构建对应的搜索树
        search_tree = o3d.geometry.KDTreeFlann(self.__target_features)

        # 为原始点云中的每一个点寻找对应的特征点
        _, N = self.__source_features.shape
        matches = []
        for i in range(N):
            query = self.__source_features[:, i]
            _, idx_nn_target, _ = search_tree.search_knn_vector_xd(query, 1)
            matches.append(
                [i, idx_nn_target[0]]
            )

        # 结果为N*2的数组
        matches = np.asarray(matches)

        return matches

    # 建造者模式
    class Builder:
        def __init__(self) -> None:
            self.max_works = 16
            self.num_samples = 4
            self.max_correspondence_distance = 1.5
            self.max_iteration = 1000
            self.max_validation = 500
            self.max_edge_length_ratio = 0.9
            self.normal_angle_threshold = None

        def set_max_works(self, max_works: int):
            self.max_works = max_works
            return self

        def set_num_samples(self, num_samples: int):
            self.num_samples = num_samples
            return self

        def set_max_correspondence_distance(self, max_correspondence_distance: float):
            self.max_correspondence_distance = max_correspondence_distance
            return self

        def set_max_iteration(self, max_iteration: int):
            self.max_iteration = max_iteration
            return self

        def set_max_validation(self, max_validation: int):
            self.max_validation = max_validation
            return self

        def set_max_edge_length_ratio(self, max_edge_length_ratio: float):
            self.max_edge_length_ratio = max_edge_length_ratio
            return self

        def set_normal_angle_threshold(self, normal_angle_threshold: float):
            self.normal_angle_threshold = normal_angle_threshold
            return self

        def build(self):
            return RANSAC_ICP(self)

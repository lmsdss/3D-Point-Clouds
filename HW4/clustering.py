# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类
import math
import os
import random
import struct
import sys
import time

import numpy as np
import open3d
from scipy.spatial import KDTree


# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)


# 假设地面为水平，使用LeastSquare与RANSAC进行平面拟合，取出平面附近的点作为地面
def LSQ(X: np.ndarray):
    # 模型：z=ax+by+c
    A = X.copy()
    b = np.expand_dims(X[:, 2], axis=1)
    A[:, 2] = 1

    # 由X=(AT*A)-1*AT*b直接求解
    A_T = A.T
    A1 = np.dot(A_T, A)
    A2 = np.linalg.inv(A1)
    A3 = np.dot(A2, A_T)
    x = np.dot(A3, b)
    return x


# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     segmented_cloud: 删除地面点之后的点云
def ground_segmentation(data: np.ndarray, tao=0.35, e=0.4, N_regular=100):
    # 作业1
    # 屏蔽开始
    s = data.shape[0]
    count = 0
    dic = {}
    p = 0.99  # 希望得到正确模型的概率

    # 计算迭代次数
    if math.log(1 - (1 - e) ** s) < sys.float_info.min:
        N = N_regular
    else:
        N = math.log(1 - p) / math.log(1 - (1 - e) ** s)

    # 开始迭代
    while count < N:

        ids = random.sample(range(0, s), 3)
        p1, p2, p3 = data[ids]
        # 判断是否共线
        L = p1 - p2
        R = p2 - p3
        if 0 in L or 0 in R:
            continue
        # 如果是一条直线，那么X Y Z应当满足比例关系
        else:
            if L[0] / R[0] == L[1] / R[1] == L[2] / R[2]:
                continue

        # 计算平面参数，求出平面的方程
        a = (p2[1] - p1[1]) * (p3[2] - p1[2]) - (p2[2] - p1[2]) * (p3[1] - p1[1])
        b = (p2[2] - p1[2]) * (p3[0] - p1[0]) - (p2[0] - p1[0]) * (p3[2] - p1[2])
        c = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        d = 0 - (a * p1[0] + b * p1[1] + c * p1[2])

        # 求所有点到平面点距离
        distance = abs(a * data[:, 0] + b * data[:, 1] + c * data[:, 2] + d) / (a ** 2 + b ** 2 + c ** 2) ** 0.5

        id_set = []
        for i, d in enumerate(distance):
            if d < tao:
                id_set.append(i)

        # 再使用最小二乘法，LSQ可以让RANSAC的分割结果更加精确
        p = LSQ(data[id_set])
        print(p)
        a, b, c, d = p[0], p[1], -1, p[2]

        dic[len(id_set)] = [a, b, c, d]

        if len(id_set) > s * (1 - e):
            break

        count += 1

    parm = dic[max(dic.keys())]
    a, b, c, d = parm

    # 求所有点到平面点距离
    distance = abs(a * data[:, 0] + b * data[:, 1] + c * data[:, 2] + d) / (a ** 2 + b ** 2 + c ** 2) ** 0.5

    id_set = []
    for i, d in enumerate(distance):
        if d < tao:
            id_set.append(i)

    return np.array(id_set)


def clustering(X: np.ndarray, r: float, min_samples: int):
    # 作业2
    # 屏蔽开始
    point_number = X.shape[0]
    # 未访问的点的集合
    unvisited_list = [i for i in range(point_number)]
    # 未访问的点的个数
    unvisited_number = point_number
    # 访问过的点的集合
    visited_list = []
    clusters_index = []
    # 建立KDTree树
    tree = KDTree(X)

    # 未访问的点的个数大于0
    while unvisited_number > 0:
        # 从未访问的点随机选择一个
        random_id = random.choice(unvisited_list)
        # 访问过的random_id点进入visited_list
        visited_list.append(random_id)
        # 从未访问过的点中移除
        unvisited_list.remove(random_id)
        # 未访问的点个数减一
        unvisited_number -= 1

        # 对该点进行r临近搜索
        N = tree.query_ball_point(X[random_id], r)

        # 若该点r临近的个数N小于min_samples，则标记为噪声点
        if len(N) < min_samples:
            continue
        # 大于等于min_samples则标记为核心点
        else:
            clusters = []
            # 添加随机选择的初始点
            clusters.append(random_id)
            # 在该r领近内移除该点，因为该点已经标记为一类
            N.remove(random_id)
            # 对该点r领域的其他点进行核心点判断
            while len(N) > 0:
                # 删除最后一个
                p = N.pop()
                if p in unvisited_list:
                    # 访问过的p点进入visited_list
                    visited_list.append(p)
                    # 从未访问过的点中移除
                    unvisited_list.remove(p)
                    # 未访问的点个数减一
                    unvisited_number -= 1
                    # 添加p点
                    clusters.append(p)
                    pN = tree.query_ball_point(X[p], r)
                    # 临近点个数大于等于min_samples
                    if len(pN) >= min_samples:
                        pN.remove(p)
                        # 扩展同一个类的点的数目
                        N = N + pN
            # 将属于同一个类的所有点保存起来
            clusters_index.append(clusters)
    # 屏蔽结束
    return clusters_index


def main():
    root_dir = ''  # 数据集路径
    cat = os.listdir(root_dir)
    cat = cat[0:]
    iteration_num = len(cat)

    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        print('clustering pointcloud file:', filename)

        origin_data = read_velodyne_bin(filename)
        print('origin data points num:', origin_data.shape[0])
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(origin_data)
        # 显示原始点云
        open3d.visualization.draw_geometries([pcd])

        points = np.array(pcd.points)
        # 调用voxel滤波函数，实现滤波
        filtered_cloud = voxel_filter(points, 0.05, "random")  # centroid or random
        print('voxel data points num:', filtered_cloud.shape[0])
        points = filtered_cloud

        # 用滤波后的点进行地面分割
        ground_begin_t = time.time()
        ground_points_idx = ground_segmentation(data=points)
        ground_end_t = time.time()
        ground_time_sum = ground_end_t - ground_begin_t
        print("地面分割的时间:", ground_time_sum)

        ground_data = points[ground_points_idx]
        print('ground data points num:', ground_data.shape[0])
        ground_pcd = open3d.geometry.PointCloud()
        ground_pcd.points = open3d.utility.Vector3dVector(ground_data)

        # 地面为蓝色
        c = [0, 0, 255]
        colors = np.tile(c, (ground_data.shape[0], 1))
        ground_pcd.colors = open3d.utility.Vector3dVector(colors)

        # 地面以外点点为要分类的点
        segmented_points_idx = []
        for i in range(points.shape[0]):
            # 用索引序号判断
            if i not in ground_points_idx:
                segmented_points_idx.append(i)

        # 滤除地面后的点云
        segmented_data = points[segmented_points_idx]
        print('segmented data points num:', segmented_data.shape[0])
        segmented_pcd = open3d.geometry.PointCloud()
        segmented_pcd.points = open3d.utility.Vector3dVector(segmented_data)

        # 地面以外为红色
        c = [255, 0, 0]
        colors = np.tile(c, (segmented_data.shape[0], 1))
        segmented_pcd.colors = open3d.utility.Vector3dVector(colors)

        # 可视化
        open3d.visualization.draw_geometries([ground_pcd, segmented_pcd])

        # 从点云中提取聚类
        cluster_begin_t = time.time()
        cluster_index = clustering(segmented_data, 0.3, 10)
        cluster_end_t = time.time()
        cluster_time_sum = cluster_end_t - cluster_begin_t
        print("聚类的时间:", cluster_time_sum)

        draw = []
        # Debug发现cluster_index的标签有200多，后来让每个类依次使用以下5种颜色，相同的类会是同一颜色。
        # 绿色 青色 黄色 紫色 深红
        color_set = [[0, 255, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [255, 0, 255]]

        for i in range(len(cluster_index)):
            segmented_result = open3d.geometry.PointCloud()
            segmented_result.points = open3d.utility.Vector3dVector(segmented_data[cluster_index[i]])

            # 依次使用5种颜色
            c = color_set[i % 5]

            cs = np.tile(c, (segmented_data[cluster_index[i]].shape[0], 1))
            segmented_result.colors = open3d.utility.Vector3dVector(cs)
            draw.append(segmented_result)

        draw.append(ground_pcd)
        open3d.visualization.draw_geometries(draw)


if __name__ == '__main__':
    main()

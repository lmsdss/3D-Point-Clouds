# 对数据集中的点云，批量执行构建树和查找，包括kdtree和octree，并评测其运行时间

import os
import struct
import time

import numpy as np
from scipy import spatial

import kdtree as kdtree
import octree as octree
from result_set import KNNResultSet, RadiusNNResultSet


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


def main():
    # configuration
    leaf_size = 32  # 每个leaf最多有32个点
    min_extent = 0.0001  # octant的最小尺寸
    k = 8  # KNN搜索的临近点
    radius = 1  # radius NN的半径

    root_dir = '/home/ustc-swf/point_cloud/Homework/HW2/data'  # 数据集路径
    cat = os.listdir(root_dir)
    iteration_num = len(cat)

    print("------------- octree --------------")
    # 时间初始化
    construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0

    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        db_np = read_velodyne_bin(filename)

        # Octree统计构造时间
        begin_t = time.time()
        root = octree.octree_construction(db_np, leaf_size, min_extent)
        construction_time_sum += time.time() - begin_t

        query = db_np[0, :]

        # Octree KNN search
        begin_t = time.time()
        result_set = KNNResultSet(capacity=k)
        octree.octree_knn_search(root, db_np, result_set, query)
        knn_time_sum += time.time() - begin_t

        # Octree Radius search
        begin_t = time.time()
        result_set = RadiusNNResultSet(radius=radius)
        octree.octree_radius_search_fast(root, db_np, result_set, query)
        radius_time_sum += time.time() - begin_t

        # brute search
        begin_t = time.time()
        diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
        nn_idx = np.argsort(diff)  # 排序不是必要的
        nn_dist = diff[nn_idx]
        brute_time_sum += time.time() - begin_t
    print("Octree: build %.3f, knn %.3f, radius %.3f, brute %.3f" % (construction_time_sum * 1000 / iteration_num,
                                                                     knn_time_sum * 1000 / iteration_num,
                                                                     radius_time_sum * 1000 / iteration_num,
                                                                     brute_time_sum * 1000 / iteration_num))

    print("------------- kdtree --------------")
    # 时间初始化
    construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0

    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        db_np = read_velodyne_bin(filename)

        # kdtree统计构造时间
        begin_t = time.time()
        root = kdtree.kdtree_construction(db_np, leaf_size)
        construction_time_sum += time.time() - begin_t

        query = db_np[0, :]

        # kdtree KNN search
        begin_t = time.time()
        result_set = KNNResultSet(capacity=k)
        kdtree.kdtree_knn_search(root, db_np, result_set, query)
        knn_time_sum += time.time() - begin_t

        # kdtree Radius search
        begin_t = time.time()
        result_set = RadiusNNResultSet(radius=radius)
        kdtree.kdtree_radius_search(root, db_np, result_set, query)
        radius_time_sum += time.time() - begin_t

        # brute search
        begin_t = time.time()
        diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
        nn_idx = np.argsort(diff)  # 排序不是必要的
        nn_dist = diff[nn_idx]
        brute_time_sum += time.time() - begin_t
    print("Kdtree: build %.3f, knn %.3f, radius %.3f, brute %.3f" % (construction_time_sum * 1000 / iteration_num,
                                                                     knn_time_sum * 1000 / iteration_num,
                                                                     radius_time_sum * 1000 / iteration_num,
                                                                     brute_time_sum * 1000 / iteration_num))
    print("-------------scipy.kdtree -------------")
    # 时间初始化
    construction_time_sum = 0
    knn_time_sum = 0

    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        db_np = read_velodyne_bin(filename)

        # scipy.spatital.kdtree统计构造时间
        begin_t = time.time()
        scipy_kdtree = spatial.KDTree(db_np)
        construction_time_sum += time.time() - begin_t

        query = db_np[0, :]

        # spatial.KDTree knn search
        begin_t = time.time()
        d, i = scipy_kdtree.query(query, k=8)
        knn_time_sum += time.time() - begin_t

    print("scipy.Kdtree: build %.3f, knn %.3f" % (construction_time_sum * 1000 / iteration_num,
                                                           knn_time_sum * 1000 / iteration_num))


if __name__ == '__main__':
    main()

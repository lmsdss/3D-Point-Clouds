# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d
import os
import numpy as np
from pyntcloud import PyntCloud
from pandas import DataFrame
import matplotlib.pyplot as plt
import random
import time


# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
def voxel_filter(point_cloud, size, filter_mode, is_Hash_Table):
    filtered_points = []
    # 作业3
    # 屏蔽开始
    # step1 计算边界点
    x_max, y_max, z_max = np.amax(point_cloud, axis=0)  # 计算 x,y,z三个维度的最值
    x_min, y_min, z_min = np.amin(point_cloud, axis=0)
    # step2 确定体素的尺寸
    size_r = size
    # step3 计算每个volex的维度
    Dx = (x_max - x_min) / size_r
    Dy = (y_max - y_min) / size_r
    Dz = (z_max - z_min) / size_r

    # 排序算法
    if (is_Hash_Table == False):
        start_time = time.time()
        # 计算每个点在volex grid内每一个维度的值
        h = list()
        for i in range(len(point_cloud)):
            hx = np.floor((point_cloud[i][0] - x_min) / size_r)
            hy = np.floor((point_cloud[i][1] - y_min) / size_r)
            hz = np.floor((point_cloud[i][2] - z_min) / size_r)
            h.append(hx + hy * Dx + hz * Dx * Dy)
        # step5 对h值进行排序
        h = np.array(h)
        h_indice = np.argsort(h)  # 提取索引 从小到大
        h_sorted = h[h_indice]  # 升序

        count = 0  # 用于维度的累计
        # 将h值相同的点放入到同一个grid中，并进行筛选
        for i in range(len(h_sorted) - 1):
            if h_sorted[i] == h_sorted[i + 1]:  # 当前的点与后面的相同，放在同一个volex grid中
                continue
            else:
                if (filter_mode == "centroid"):  # 均值滤波
                    point_idx = h_indice[count: i + 1]
                    filtered_points.append(np.mean(point_cloud[point_idx], axis=0))  # 取同一个grid的均值
                    count = i
                elif (filter_mode == "random"):  # 随机滤波
                    point_idx = h_indice[count: i + 1]
                    filtered_points.append(random.choice(point_cloud[point_idx]))
                    count = i

        end_time = time.time()
        print('Sort:', end_time - start_time)
        # Sort: 0.052618980407714844

    # TODO:Hash算法实现可能有问题，反而速度变慢
    elif (is_Hash_Table == True):
        start_time = time.time()

        voxel = [[] for _ in range(int(Dx * Dy * Dz))]

        container_size = len(voxel)

        for i in range(len(point_cloud)):
            hx = np.floor((point_cloud[i][0] - x_min) / size_r)
            hy = np.floor((point_cloud[i][1] - y_min) / size_r)
            hz = np.floor((point_cloud[i][2] - z_min) / size_r)
            h = (hx + hy * Dx + hz * Dx * Dy) % container_size

            voxel[int(h)].append(i)

        for v in voxel:
            if len(v) == 0:
                continue
            else:
                if (filter_mode == "centroid"):  # 均值滤波
                    filtered_points.append(np.sum(point_cloud[v], axis=0) / len(v))

                elif (filter_mode == "random"):  # 随机滤波
                    filtered_points.append(random.choice(point_cloud[v]))

        end_time = time.time()
        print('Hash:', end_time - start_time)
        # Hash: 0.05608105659484863

    # 屏蔽结束
    # 把点云格式改成array
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points


def main():
    category_index = 17  # 物体编号，范围是0-39，即对应数据集中40个物体
    txt_index = 10  # 每个种类的txt文件编号, 范围是0-XXX

    # 数据集路径
    root_dir = '/home/ustc-swf/point_cloud/Homework/modelnet40_normal_resampled'
    category = os.listdir(root_dir)
    category.sort()

    # 数据集类别路径
    category_filename = os.path.join(root_dir, category[category_index])
    txt = os.listdir(category_filename)
    txt.sort()

    # 每个类别的不同txt文件路径
    txt_filename = os.path.join(category_filename, txt[txt_index])

    # 加载原始点云 txt文件处理
    point_cloud_data = np.genfromtxt(txt_filename, delimiter=",")  # 为 xyz的 N*3矩阵
    point_cloud_data = DataFrame(point_cloud_data[:, 0:3])  # 选取每一列的第0个元素到第二个元素   [0,3)
    point_cloud_data.columns = ['x', 'y', 'z']  # 给选取到的数据附上标题
    point_cloud_pynt = PyntCloud(point_cloud_data)  # 将points的数据存到结构体中

    # 转成open3d能识别的格式
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)  # to_instance实例化
    point_cloud_o3d_filter = o3d.geometry.PointCloud()
    # o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    points = np.array(point_cloud_o3d.points)

    # 调用voxel滤波函数，实现滤波
    filtered_cloud = voxel_filter(points, 0.05, "centroid", is_Hash_Table=False)  # centroid or random
    point_cloud_o3d_filter.points = o3d.utility.Vector3dVector(filtered_cloud)
    # 显示滤波后的点云
    o3d.visualization.draw_geometries([point_cloud_o3d])
    o3d.visualization.draw_geometries([point_cloud_o3d_filter])


if __name__ == '__main__':
    main()

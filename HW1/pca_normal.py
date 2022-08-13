# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证

import open3d as o3d
import os
import numpy as np
from pyntcloud import PyntCloud
from pandas import DataFrame


# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#     correlation：分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量
#

def PCA(data, correlation=False, sort=True):
    # 作业1
    # 屏蔽开始
    # 求均值
    data_mean = np.mean(data, axis=0)
    # 归一化
    data_normalized = data - data_mean
    # 协方差矩阵H
    H = np.dot(data_normalized.T, data_normalized)
    # SVD求解特征值、特征向量
    eigenvectors, eigenvalues, eigenvectors_T = np.linalg.svd(H)
    # 屏蔽结束

    if sort:
        # 降序排列
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors


def main():
    # 物体种类编号，范围是0-39，即对应数据集中40个物体
    category_index = 2 # 0 1 2
    # 每个种类的txt文件编号, 范围是0-XXX
    txt_index = 1

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

    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    o3d.visualization.draw_geometries([point_cloud_o3d])  # 显示原始点云

    # 从点云中获取点，只对点进行处理 print(point_cloud_pynt)
    points = point_cloud_pynt.points
    print('total points number is:', points.shape[0])  # (10000, 3)

    # 用PCA分析点云主方向
    w, v = PCA(points, correlation=False)
    # 点云主方向对应的向量
    point_cloud_vector = v[:, 0]
    print('the main orientation of this pointcloud is: ', point_cloud_vector)
    # 在原点云中画图
    # 画点：原点、第一主成分、第二主成分
    point = [[0, 0, 0], v[:, 0], v[:, 1]]
    # 画出三点之间两条连线  原点和两个成分向量
    lines = [[0, 1], [0, 2]]
    # 不同的线不同的颜色
    colors = [[1, 0, 0], [0, 0, 0]]

    # 构造open3d中的LineSet对象，用于主成分显示
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(point), lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    # 显示原始点云和PCA后的连线
    o3d.visualization.draw_geometries([point_cloud_o3d, line_set])

    # 循环计算每个点的法向量
    # 找出每一点的knn个临近点，拟合成曲面，然后PCA找到特征向量最小的值，作为法向量
    # 建立kd树方便搜索
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
    # 法向量
    normals = []
    # 作业2
    # 屏蔽开始
    for i in range(point_cloud_data.shape[0]):  # (10000, 3)
        # search_knn_vector_3d(self, query, knn)
        # Returns:
        # Tuple[int, open3d.utility.IntVector, open3d.utility.DoubleVector]
        [_, idx_int, idx_double] = pcd_tree.search_knn_vector_3d(point_cloud_o3d.points[i], 16)
        # asarray将点云从oepn3d形式转换为矩阵形式
        # asarray和array一样 但是array会copy出一个副本，asarray不会，节省内存
        k_nearest_point = np.asarray(point_cloud_o3d.points)[idx_int, :]

        # print(k_nearest_point)
        w, v = PCA(k_nearest_point)
        # 法向量是最小特征值
        normals.append(v[:, 2])

    # 由于最近邻搜索是第二章的内容，所以此处允许直接调用open3d中的函数
    # 屏蔽结束
    normals = np.array(normals, dtype=np.float64)
    # TODO: 此处把法向量存放在了normals中
    point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)

    o3d.visualization.draw_geometries([point_cloud_o3d, line_set])


if __name__ == '__main__':
    main()

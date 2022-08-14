import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd


# 读取点云文件
def read_point_cloud(file_name):
    df = pd.read_csv(file_name, header=None)
    df.columns = ["x", "y", "z", "nx", "ny", "nz"]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(df[["x", "y", "z"]].values)
    pcd.normals = o3d.utility.Vector3dVector(df[["nx", "ny", "nz"]].values)

    return pcd


# 获取每个节点的SFPH描述子
def spfh(pcd, search_tree, key_point, radius, b):
    points = np.asarray(pcd.points)   # 获取点云
    normals = np.asarray(pcd.normals)

    [k, idx_neighbor, _] = search_tree.search_radius_vector_3d(key_point, radius)  # 获取邻居节点
    n1 = normals[idx_neighbor[0]]  # 获取n1
    idx_neighbor = idx_neighbor[1:]  # 移除关键点本身

    diff = points[idx_neighbor] - key_point   # 计算 (p2-p1)/norm(p2-p1)
    diff = diff / np.reshape(np.linalg.norm(diff, ord=2, axis=1), (k - 1, 1))

    u = n1
    v = np.cross(u, diff)  # 向量叉乘
    w = np.cross(u, v)

    n2 = normals[idx_neighbor]  # 计算n2
    alpha = np.reshape((v * n2).sum(axis=1), (k - 1, 1)) # 计算alpha
    phi = np.reshape((u * diff).sum(axis=1), (k - 1, 1))  # 计算phi
    theta = np.reshape(np.arctan2((w * n2).sum(axis=1), (u * n2).sum(axis=1)), (k - 1, 1))   # 计算 theta

    # 计算相应的直方图
    alpha_hist = np.reshape(np.histogram(alpha, b, range=[-1.0, 1.0])[0], (1, b))
    phi_hist = np.reshape(np.histogram(phi, b, range=[-1.0, 1.0])[0], (1, b))
    theta_hist = np.reshape(np.histogram(theta, b, range=[-3.14, 3.14])[0], (1, b))

    # 组成描述子
    fpfh = np.hstack((alpha_hist, phi_hist, theta_hist))

    return fpfh


# 计算FPFH描述子
def description_fpfh(pcd, search_tree, key_point, radius, b):
    points = np.asarray(pcd.points)   # 点云
    # 寻找keypoint的邻居点
    [k, idx_neighbor, _] = search_tree.search_radius_vector_3d(key_point, radius)
    if k <= 1:
        return None

    idx_neighbor = idx_neighbor[1:]  # 移除关键点本身
    w = 1.0 / np.linalg.norm(key_point - points[idx_neighbor], ord=2, axis=1) # 计算权重

    # 计算邻居的SPFH
    neighbor_spfh = np.reshape(np.asarray([spfh(pcd, search_tree, points[i], radius, b) for i in idx_neighbor]),
                               (k - 1, 3 * b))

    self_spfh = spfh(pcd, search_tree, key_point, radius, b) # 计算自身的描述子
    neighbor_spfh = 1.0 / (k - 1) * np.dot(w, neighbor_spfh)   # 计算最终的FPFH

    fpfh = self_spfh + neighbor_spfh

    return fpfh


if __name__ == "__main__":
    file_name = "airplane_0001.txt"    # 提取描述子的的点云文件路径
    radius = 0.05   # 最近邻搜索半径
    b = 15  # 每个直方图bin的个数

    pcd = read_point_cloud(file_name)   # 读取点云
    search_tree = o3d.geometry.KDTreeFlann(pcd)   # 构建搜索树

    # 其中 1 和 2 为相似特征点， 3为其余特征点
    key_point1 = np.array([-0.1641, -0.1407, 0.3181])
    key_point2 = np.array([0.1623, -0.1364, 0.3173])
    key_point3 = np.array([-0.004708, -0.3359, -0.597])

    fpfh1 = description_fpfh(pcd, search_tree, key_point1, radius, b)
    fpfh2 = description_fpfh(pcd, search_tree, key_point2, radius, b)
    fpfh3 = description_fpfh(pcd, search_tree, key_point3, radius, b)

    fpfh1 = fpfh1 / np.linalg.norm(fpfh1)
    fpfh2 = fpfh2 / np.linalg.norm(fpfh2)
    fpfh3 = fpfh3 / np.linalg.norm(fpfh3)

    plt.plot(range(3 * b), fpfh1.T, ls="-", color="c", marker=",", lw=2, label="keypoint1")
    plt.plot(range(3 * b), fpfh2.T, ls="-", color="b", marker=",", lw=2, label="keypoint2")
    plt.plot(range(3 * b), fpfh3.T, ls="-", color="y", marker=",", lw=2, label="keypoint3")

    plt.legend()
    plt.show()

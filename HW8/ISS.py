import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree  # KDTree 进行搜索


def compute_cov_eigenvalue(point_cloud):
    x = np.asarray(point_cloud[:, 0])
    y = np.asarray(point_cloud[:, 1])
    z = np.asarray(point_cloud[:, 2])
    m = np.vstack((x, y, z))  # (3*n) 每行代表一个属性， 每列代表x y z(一个点)
    cov = np.cov(m)  # 求解每个点坐标的协方差矩阵
    eigenvalue, eigen_vector = np.linalg.eigh(cov)  # 求解三个特征值
    eigenvalue = eigenvalue[np.argsort(-eigenvalue)]  # 降序排列特征值
    return eigenvalue  # 返回特征值


def iss(data):
    eigen_values = []
    feature = []
    key_point = set()  # 关键点的集合

    # 构建 kd_tree
    leaf_size = 4
    radius = 0.1
    tree = KDTree(data, leaf_size)
    # 使用radiusNN得到每个点的邻近点
    nearest_idx = tree.query_radius(data, radius)  # (10000)

    # 求解每个点在各自的radius范围内的特征值
    for i in range(len(nearest_idx)):
        eigen_values.append(compute_cov_eigenvalue(data[nearest_idx[i]]))
    eigen_values = np.asarray(eigen_values)

    t1 = 0.45  # 𝛾21阈值
    t2 = 0.45  # 𝛾32阈值
    for i in range(len(nearest_idx)):
        # 𝜆2/𝜆1 < 𝛾21   𝜆3/𝜆2 < 𝛾32
        if eigen_values[i, 1] / eigen_values[i, 0] < t1 and eigen_values[i, 2] / eigen_values[i, 1] < t2:
            key_point.add(i)  # 获得初始关键点的索引

    # 𝜆3NMS非极大抑制
    unvisited = key_point  # 未访问集合
    while len(key_point):
        unvisited_old = unvisited
        core = list(key_point)[np.random.randint(0, len(key_point))]  # 从关键点集中随机选取一个核心点core
        visited = [core]  # 把核心点标记为visited
        unvisited = unvisited - {core}  # 从未访问集合中剔除

        while len(visited):  # 遍历所有初始关键点
            new_core = visited[0]
            if new_core in key_point:
                # 当前关键点的范围内所包含的其他关键点
                other_key = unvisited & set(nearest_idx[new_core])
                visited += (list(other_key))
                unvisited = unvisited - other_key
            visited.remove(new_core)  # new_core已被访问
        overlap_point = unvisited_old - unvisited  # 有重叠的关键点群
        key_point = key_point - overlap_point  # 求差集

        cluster = []
        for i in list(overlap_point):
            cluster.append(eigen_values[i][2])  # 获取每个关键点的最小特征值
        nms_output = np.argmax(np.asarray(cluster))  # 特征值最大的为关键点
        feature.append(list(overlap_point)[nms_output])  # 添加到feature特征点数组中

    return feature


if __name__ == '__main__':
    point_cloud = np.genfromtxt(r"airplane_0001.txt", delimiter=",")  # bed_0001.txt airplane_0001.txt
    point_cloud = point_cloud[:, 0:3]  # x y z

    # 计算特征点的序号
    feature_idx = iss(point_cloud)
    # 特征点
    feature_point = point_cloud[feature_idx]

    # pcd类型的数据
    pcd_point = o3d.geometry.PointCloud()
    pcd_feature = o3d.geometry.PointCloud()

    # 将点云转换成open3d中的数据形式并用pcd来保存，以方便用open3d处理
    pcd_point.points = o3d.utility.Vector3dVector(point_cloud)
    pcd_feature.points = o3d.utility.Vector3dVector(feature_point)

    # 指定原始点显示为蓝色
    pcd_point.paint_uniform_color([0, 0, 1])
    # 指定特征点显示为红色
    pcd_feature.paint_uniform_color([1, 0, 0])

    # 将点云从open3d形式转换为矩阵形式
    np.asarray(pcd_point.points)
    np.asarray(pcd_feature.points)

    print(np.asarray(pcd_feature.points))

    # 用open3d可视化生成的点云
    # o3d.visualization.draw_geometries([pcd_point, pcd_feature])

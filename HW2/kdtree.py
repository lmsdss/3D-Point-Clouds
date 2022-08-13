# kdtree的具体实现，包括构建和查找

import math

import numpy as np

from result_set import KNNResultSet, RadiusNNResultSet


# Node类，Node是tree的基本组成元素
class Node:
    def __init__(self, axis, value, left, right, point_indices):
        self.axis = axis
        self.value = value
        self.left = left
        self.right = right
        self.point_indices = point_indices

    def is_leaf(self):
        if self.value is None:
            return True
        else:
            return False

    def __str__(self):
        output = ''
        output += 'axis %d, ' % self.axis
        if self.value is None:
            output += 'split value: leaf, '
        else:
            output += 'split value: %.2f, ' % self.value
        output += 'point_indices: '
        output += str(self.point_indices.tolist())
        return output


# 功能：构建树之前需要对value进行排序，同时对一个的key的顺序也要跟着改变
# 输入：
#     key：键
#     value:值
# 输出：
#     key_sorted：排序后的键
#     value_sorted：排序后的值
def sort_key_by_vale(key, value):
    # 确保输入的点的索引和点的值维度相同
    assert key.shape == value.shape  # assert 断言操作，用于判断一个表达式，在表达式条件为false的时候触发异常
    assert len(key.shape) == 1  # numpy是多维数组
    sorted_idx = np.argsort(value)  # value是一个列表，不是numpy 对value值进行排序
    key_sorted = key[sorted_idx]
    value_sorted = value[sorted_idx]  # 进行升序排序
    return key_sorted, value_sorted


# 用于X Y轴的轮换  改变分割维度
def axis_round_robin(axis, dim):
    if axis == dim - 1:
        return 0
    else:
        return axis + 1

# 哪个轴上的数据分布方差大，就垂直于哪个轴分割维度
def axis_variance(leaf_point):
    arr_var = np.var(leaf_point,axis=0)       #求方差
    arr_axis_max = max(arr_var[0],arr_var[1],arr_var[2])     #选取方差较大的进行轴进行切割
    if( arr_axis_max == arr_var[0]):
        return 0       #axis= 0
    elif ( arr_axis_max == arr_var[1]):
        return 1       #axis = 1
    else:
        return 2       #axis = 2


# 功能：通过递归的方式构建树
# 输入：
#     root: 树的根节点
#     db: 点云数据
#     point_indices：排序后的键
#     axis: scalar
#     leaf_size: scalar
# 输出：
#     root: 即构建完成的树
def kdtree_recursive_build(root, db, point_indices, axis, leaf_size):
    if root is None:
        root = Node(axis, None, None, None, point_indices)  # 实例化Node

    # determine whether to split into left and right
    if len(point_indices) > leaf_size:  # 判断是否需要进行分割
        # --- get the split position ---
        point_indices_sorted, _ = sort_key_by_vale(point_indices, db[point_indices, axis])  # 对点进行排列，dp存储信息

        # 作业1
        # 屏蔽开始
        # 取排序后中间点
        middle_left_idx = math.ceil(point_indices_sorted.shape[0] / 2) - 1  # ceil()函数用于从上取整  计算出左边有多少个点
        # 取排序后中间点的原始索引
        middle_left_point_idx = point_indices_sorted[middle_left_idx]  # 左边节点的最大值
        # 取排序后中间点的值
        middle_left_point_value = db[middle_left_point_idx, axis]

        # 取右边一个节点的上述信息
        middle_right_idx = middle_left_idx + 1  # 右边的点数
        middle_right_point_idx = point_indices_sorted[middle_right_idx]  # 右边节点的最小值
        middle_right_point_value = db[middle_right_point_idx, axis]  # 提取值

        # 根节点的赋值为上述两个点的平均值
        root.value = (middle_right_point_value + middle_left_point_value) * 0.5  # 取middle为root的值
        # === get the split position ===
        # 进行递归分割 小值放左边，构建左子树
        root.left = kdtree_recursive_build(root.left,
                                           db,
                                           point_indices_sorted[0:middle_right_idx],
                                           # 方差建轴
                                           axis_variance(db[point_indices_sorted[0:middle_right_idx]]),leaf_size)


        # 大值放右边，构建右子树
        root.right = kdtree_recursive_build(root.right,
                                            db,
                                            point_indices_sorted[middle_right_idx:],
                                            # 方差建轴
                                            axis_variance(db[point_indices_sorted[0:middle_right_idx]]),leaf_size)
        # 屏蔽结束
    return root


# 功能：翻转一个kd树
# 输入：
#     root：kd树
#     depth: 当前深度
#     max_depth：最大深度
def traverse_kdtree(root: Node, depth, max_depth):  # 计算kdtree的深度
    depth[0] += 1
    if max_depth[0] < depth[0]:
        max_depth[0] = depth[0]

    if root.is_leaf():  # 打印最后的叶子节点
        print(root)
    else:
        traverse_kdtree(root.left, depth, max_depth)  # 累加计算深度
        traverse_kdtree(root.right, depth, max_depth)

    depth[0] -= 1


# 功能：构建kd树（利用kdtree_recursive_build功能函数实现的对外接口）
# 输入：
#     db_np：原始数据
#     leaf_size：scale
# 输出：
#     root：构建完成的kd树
def kdtree_construction(db_np, leaf_size):
    N, dim = db_np.shape[0], db_np.shape[1]  # (64, 3)

    # build kd_tree recursively
    root = None
    root = kdtree_recursive_build(root,
                                  db_np,
                                  np.arange(N),
                                  axis=0,
                                  leaf_size=leaf_size)
    return root


# 功能：通过kd树实现knn搜索，即找出最近的k个近邻
# 输入：
#     root: kd树
#     db: 原始数据
#     result_set：搜索结果 #KNNResultSet 继承二叉树的结果集
#     query：索引信息
# 输出：
#     搜索失败则返回False
def kdtree_knn_search(root: Node, db: np.ndarray, result_set: KNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf():  # 如果搜索到是叶子节点，直接进行暴力搜索
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)  # 求距离
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False

    # 作业2
    # 提示：仍通过递归的方式实现搜索
    # 屏蔽开始
    if query[root.axis] <= root.value:  # 如果当前目标点在对应axis上的值小于根节点的值，则对左子树进行搜索
        kdtree_knn_search(root.left, db, result_set, query)
        # 如果目标点离轴虚线的距离小于worst_dist 继续搜寻节点的右边
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_knn_search(root.right, db, result_set, query)
    else:
        kdtree_knn_search(root.right, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_knn_search(root.left, db, result_set, query)
    # 屏蔽结束

    return False


# 功能：通过kd树实现radius搜索，即找出距离radius以内的近邻
# 输入：
#     root: kd树
#     db: 原始数据
#     result_set:搜索结果
#     query：索引信息
# 输出：
#     搜索失败则返回False
def kdtree_radius_search(root: Node, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf():
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False

    # 作业3
    # 提示：通过递归的方式实现搜索
    # 屏蔽开始
    if query[root.axis] <= root.value:
        kdtree_radius_search(root.left, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_radius_search(root.right, db, result_set, query)
    else:
        kdtree_radius_search(root.right, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_radius_search(root.left, db, result_set, query)
    # 屏蔽结束

    return False


def main():
    # configuration
    db_size = 64
    dim = 3
    leaf_size = 4
    k = 1

    # N*3维的随机数据点
    db_np = np.random.rand(db_size, dim)  # db_np.shape -> (64, 3)

    root = kdtree_construction(db_np, leaf_size=leaf_size)

    # 遍历当前树
    depth = [0]
    max_depth = [0]
    traverse_kdtree(root, depth, max_depth)
    # 获取每一个叶子节点的信息
    print("tree max depth: %d" % max_depth[0])

    # Knn搜索测试
    print("knn search:")
    query = np.asarray([0, 0, 0])
    result_set = KNNResultSet(capacity=k)
    kdtree_knn_search(root, db_np, result_set, query)
    print(result_set)

    # 暴力搜索测试
    print("brute search:")
    diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    nn_idx = np.argsort(diff)
    nn_dist = diff[nn_idx]
    # 索引
    print(nn_idx[0:k])
    # 距离
    print(nn_dist[0:k])

    # Radius 搜索测试
    print("Radius search:")
    query = np.asarray([0, 0, 0])
    result_set = RadiusNNResultSet(radius=0.5)
    kdtree_radius_search(root, db_np, result_set, query)
    print(result_set)


if __name__ == '__main__':
    main()

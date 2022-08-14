import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


# 读取点云数据点x y z坐标
def read_pcd_from_file(file):
    np_pts = np.zeros(0)
    with open(file, 'r') as f:
        pts = []
        for line in f:
            one_pt = list(map(float, line[:-1].split(',')))
            pts.append(one_pt[:3])
        np_pts = np.array(pts)
    return np_pts


# 读取文件名
def read_file_names_from_file(file):
    with open(file, 'r') as f:
        files = []
        for line in f:
            files.append(line.split('\n')[0])
    return files


# 点云归一化，以centroid为中心，半径为1
def normalize(point_cloud):
    # 求均值
    centroid = np.mean(point_cloud, axis=0)
    # 点云平移
    point_cloud = point_cloud - centroid
    # 计算到原点的最远距离
    m = np.max(np.sqrt(np.sum(point_cloud ** 2, axis=1)))
    point_cloud = point_cloud / m
    return point_cloud


class PointNetDataset(Dataset):
    def __init__(self, root_dir, train):
        super(PointNetDataset, self).__init__()

        self._train = train
        self._classes = []

        self._features = []
        self._labels = []

        self.load(root_dir)

    def classes(self):
        return self._classes

    def __len__(self):
        return len(self._features)

    def __getitem__(self, idx):
        feature, label = self._features[idx], self._labels[idx]

        # TODO: normalize feature
        feature = normalize(feature)
        # TODO: rotation to feature
        # 构造旋转矩阵
        theta = np.random.uniform(0, np.pi * 2)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        feature[:, [0, 2]] = feature[:, [0, 2]].dot(rotation_matrix)  # Z轴旋转

        # jitter
        feature += np.random.normal(0, 0.02, size=feature.shape)
        feature = torch.Tensor(feature.T)

        l_lable = [0 for _ in range(len(self._classes))] # 长为40的列表
        l_lable[self._classes.index(label)] = 1 # 预测的类别为1，其余为0
        label = torch.Tensor(l_lable)
        label_p = label.argmax()  # 返回预测类别的编号
        return feature, label, label_p

    def load(self, root_dir):
        things = os.listdir(root_dir)
        # ['sofa', 'piano', 'modelnet10_train.txt', 'modelnet40_train.txt', 'airplane', 'sink']
        files = []
        for f in things:
            if self._train == 0:
                if f == 'modelnet40_train.txt':
                    files = read_file_names_from_file(root_dir + '/' + f)
            elif self._train == 1:
                if f == 'modelnet40_test.txt':
                    files = read_file_names_from_file(root_dir + '/' + f)
            if f == "modelnet40_shape_names.txt":
                self._classes = read_file_names_from_file(root_dir + '/' + f)
                # print(self._classes) # ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf'......] 所有类别
        tmp_classes = []
        for file in files:
            # modelnet40_test.txt:airplane_0627  bathtub_0107
            num = file.split("_")[-1]  # 0627  0107
            kind = file.split("_" + num)[0]  # airplane bathtub
            if kind not in tmp_classes:
                tmp_classes.append(kind)

            pcd_file = root_dir + '/' + kind + '/' + file + '.txt'
            np_pts = read_pcd_from_file(pcd_file)  # (10000, 3) 数据前三列x y z
            self._features.append(np_pts)
            self._labels.append(kind)  # 对应的类别 airplane airplane bench bench
            # print(self._labels)

        if self._train == 0:
            print("There are " + str(len(self._labels)) + " trian files.")
        elif self._train == 1:
            print("There are " + str(len(self._labels)) + " test files.")


if __name__ == "__main__":

    train_data = PointNetDataset("/home/ustc-swf/point_cloud/Homework/modelnet40_normal_resampled", train=1)
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
    cnt = 0
    for pts, label, label_p in train_loader:

        # print(label_p)
        # print("pts.shape", pts.shape)  # torch.Size([2, 3, 10000]) 因为batch_size=2
        # print("label.shape:", label.shape)  # torch.Size([2, 40]) 因为batch_size=2
        cnt += 1
        if cnt > 0:  # 3
            break

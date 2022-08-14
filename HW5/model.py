import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# T-Net 3×3 transform
class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        # MLP
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        # 全链接层
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)  # 9=3*3

        # BatchNormalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        # 激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch_size = x.size()[0]  # 3
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        # 转换维度1024
        x = x.view(-1, 1024)  # torch.Size([3, 1024])

        x = F.relu(self.bn4(self.fc1(x)))  # torch.Size([3, 512])
        x = F.relu(self.bn5(self.fc2(x)))  # torch.Size([3, 256])
        x = self.fc3(x)

        # 展平的对角矩阵
        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batch_size, 1)

        # 将单位矩阵送入GPU
        if x.is_cuda:
            iden = iden.cuda()

        x = x + iden
        x = x.view(-1, 3, 3)  # batch_size x 3 x 3

        return x


# T-Net 64*64 transform K默认为64
class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batch_size = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batch_size, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden  # 仿射变换
        x = x.view(-1, self.k, self.k)
        return x


# PointNet编码器
class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)  # T-Net 3×3 transform
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat  # 全局特征
        self.feature_transform = feature_transform  # 特征转换
        if self.feature_transform:
            self.fstn = STNkd(k=64)  # T-Net 64*64 transform

    def forward(self, x):
        B, D, N = x.size()  # B:batch_size D: 维度 3(xyz坐标) 6(xyz坐标+法向量)  N:一个物体所取点的数目
        trans = self.stn(x)
        x = x.transpose(2, 1)  # 交换一个tensor的两个维度
        if D > 3:
            feature = x.split(3, dim=2)
        # 两个三维张量相乘 b*n*m  *  b*m*p  =  b*n*p
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)  # STNkd T-Net
            x = x.transpose(2, 1)  # 对输入的点云进行 feature transform
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x  # 局部特征
        x = F.relu(self.bn2(self.conv2(x)))  # MLP
        x = self.bn3(self.conv3(x))  # MLP
        x = torch.max(x, 2, keepdim=True)[0]  # Max pooling
        x = x.view(-1, 1024)
        if self.global_feat:  # 需要返回的是否是全局特征？
            return x, trans, trans_feat  # 返回全局特征、坐标变换矩阵、特征变换矩阵
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            # 返回局部特征与全局特征的拼接
            return torch.cat([x, pointfeat], 1), trans, trans_feat


# 对特征转换矩阵做正则化，正交矩阵不会损失特征信息
def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]  # 返回一个2维张量,对角线位置全1,其它位置全0
    if trans.is_cuda:
        I = I.cuda()

    # 正则化损失函数
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


class PointNet(nn.Module):
    # k：类别数  normal_channel：是否使用法向量信息
    def __init__(self, k=40, normal_channel=False):
        super(PointNet, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)  # 计算对数概率
        return x, trans_feat


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        # NLLloss的输入是一个对数概率向量和一个目标标签，它不会计算对数概率，适合网络最后一层是log_softmax
        loss = F.nll_loss(pred, target)  # 分类损失

        # 特征变换正则化损失
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        # 总的损失函数
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale  # 主要强调分类的损失
        return total_loss


if __name__ == "__main__":
    net = PointNet()
    sim_data = Variable(torch.rand(2, 3, 10000))
    out, trans_feat = net(sim_data)

    print(out)
    print(type(out))  # <class 'torch.Tensor'>
    print(out.size())  # torch.Size([3, 40])

    criterion = get_loss()

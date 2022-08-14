import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import PointNetDataset
from model import PointNet

SEED = 9
gpus = [0]
batch_size = 8
ckp_path = '../output200/latest.pth'


def load_ckp(ckp_path, model):
    state = torch.load(ckp_path)
    model.load_state_dict(state['state_dict'])
    print("model load from %s" % ckp_path)


if __name__ == "__main__":
    torch.manual_seed(SEED)
    device = torch.device(f'cuda:{gpus[0]}' if torch.cuda.is_available() else 'cpu')
    print("Loading test dataset...")
    test_data = PointNetDataset("/home/ustc-swf/point_cloud/Homework/modelnet40_normal_resampled", train=1)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    model = PointNet().to(device=device)
    if ckp_path:
        load_ckp(ckp_path, model)
        model = model.to(device)


    model.eval() # 作用是不启用Batch Normalization和Dropout。

    with torch.no_grad():
        accs = []

        for x, y, gt in test_loader:
            x = x.to(device)
            gt = gt.to(device)  # label_p
            # TODO: put x into network and get out
            out, _ = model(x)

            # TODO: get pred_y from out
            pred_y = out.data.max(1)[1]  # max(1)返回每一行中的最大值及索引,[1]取出代表类别的索引
            print("pred[" + str(pred_y) + "] gt[" + str(gt) + "]")

            # TODO: calculate acc from pred_y and gt
            correct = pred_y.eq(gt.data).cpu().sum()  # 判断是否匹配，并计算匹配的数量
            acc = correct.item() / batch_size
            accs.append(acc)

        print("final acc is: " + str(np.mean(accs)))

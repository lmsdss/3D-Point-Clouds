import datetime
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import PointNetDataset
from model import PointNet

SEED = 13
batch_size = 6
epochs = 200
decay_lr_factor = 0.95
decay_lr_every = 2
lr = 0.01
gpus = [0]
global_step = 0
show_every = 1
val_every = 3
date = datetime.date.today()
save_dir = "../output200"


# 保存 ckpt 文件
def save_ckp(ckp_dir, model, optimizer, epoch, best_acc, date):
    os.makedirs(ckp_dir, exist_ok=True)
    state = {
        'state_dict': model.state_dict(),  # 模型架构信息,它包括每个图层的参数矩阵。
        'optimizer': optimizer.state_dict()
    }
    ckp_path = os.path.join(ckp_dir, f'date_{date}-epoch_{epoch}-maxacc_{best_acc:.3f}.pth')
    torch.save(state, ckp_path)
    torch.save(state, os.path.join(ckp_dir, f'latest.pth'))
    print('model saved to %s' % ckp_path)


# 加载模型
def load_ckp(ckp_path, model, optimizer):
    state = torch.load(ckp_path)  # 被保存的 ckpt 的位置
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print("model load from %s" % ckp_path)


def softXEnt(prediction, real_class):
    # TODO: return loss here
    # NLL_loss的输入是一个对数概率向量和一个目标标签，它不会计算对数概率，适合网络最后一层是log_softmax
    loss = F.nll_loss(prediction, real_class)  # 分类损失
    return loss


def get_eval_acc_results(model, data_loader, device):

    model.eval()  # 不启用Batch Normalization和Dropout。
    with torch.no_grad():
        accs = []
        for x, y, gt in data_loader:
            x = x.to(device)
            gt = gt.to(device) # label_p 预测的类别

            # TODO: put x into network and get out
            out, _ = model(x)

            # TODO: get pred_y from out
            pred_y = out.data.max(1)[1]  # max(1)返回每一行中的最大值及索引,[1]取出代表类别的索引

            # TODO: calculate acc from pred_y and gt
            correct = pred_y.eq(gt.data).cpu().sum()  # 判断是否匹配，并计算匹配的数量
            acc = correct.item() / batch_size
            accs.append(acc)

        return np.mean(accs)


if __name__ == "__main__":
    # 定义logs文件位置
    writer = SummaryWriter('./output/runs/tersorboard')
    torch.manual_seed(SEED)
    device = torch.device(f'cuda:{gpus[0]}' if torch.cuda.is_available() else 'cpu')

    print("Loading train dataset...")
    train_data = PointNetDataset("/home/ustc-swf/point_cloud/Homework/modelnet40_normal_resampled", train=0)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    print("Loading valid dataset...")
    val_data = PointNetDataset("/home/ustc-swf/point_cloud/Homework/modelnet40_normal_resampled", train=1)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    print("Set model and optimizer...")
    model = PointNet().to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=decay_lr_every, gamma=decay_lr_factor)

    best_acc = 0.0
    # 作用是启用batch normalization和dropout
    model.train()
    print("Start trainning...")
    for epoch in range(epochs):
        acc_loss = 0.0
        num_samples = 0
        start_tic = time.time()
        for x, y, y_p in train_loader:

            x = x.to(device)
            y = y.to(device)
            y_p = y_p.to(device)  # label_p

            # TODO: set grad to zero
            optimizer.zero_grad()  # 避免backward时梯度累加
            # TODO: put x into network and get out
            out, trans_feat = model(x)  # out: 模型预测输出  [batch_size, 40]

            loss = softXEnt(out, y_p.long())  # (batch_size, n, C)  (batch_size, n)

            # TODO: loss backward
            loss.backward()
            # TODO: update network's param
            optimizer.step()

            acc_loss += batch_size * loss.item()
            num_samples += y.shape[0]
            global_step += 1
            acc = np.sum(
                np.argmax(out.cpu().detach().numpy(), axis=1) == np.argmax(y.cpu().detach().numpy(), axis=1)) / len(y)
            # print('acc: ', acc)
            if (global_step + 1) % show_every == 0:
                # ...log the running loss
                writer.add_scalar('training loss', acc_loss / num_samples, global_step)
                writer.add_scalar('training acc', acc, global_step)
                # print( f"loss at epoch {epoch} step {global_step}:{loss.item():3f}, lr:{optimizer.state_dict()['param_groups'][0]['lr']: .6f}, time:{time.time() - start_tic: 4f}sec")
        scheduler.step()
        print(
            f"loss at epoch {epoch}:{acc_loss / num_samples:.3f}, lr:{optimizer.state_dict()['param_groups'][0]['lr']: .6f}, time:{time.time() - start_tic: 4f}sec")

        if (epoch + 1) % val_every == 0:

            acc = get_eval_acc_results(model, val_loader, device)
            print("eval at epoch[" + str(epoch) + f"] acc[{acc:3f}]")
            writer.add_scalar('validing acc', acc, global_step)

            if acc > best_acc:
                best_acc = acc
                save_ckp(save_dir, model, optimizer, epoch, best_acc, date)

                example = torch.randn(1, 3, 10000).to(device)
                """
                利用Tracing将模型转换为TorchScript
                TorchScript是一种从PyTorch代码创建可序列化和可优化模型的方法，
                用TorchScript编写的任何代码都可以从Python进程中保存并加载到没有Python依赖关系的进程中。
                要通过tracing来将PyTorch模型转换为Torch脚本，必须将模型的实例以及样本输入传递给torch.jit.trace函数，
                这将生成一个torch.jit.ScriptModule对象，最后ScriptModule序列化为一个文件（如：model.pt），模型固化就结束了.
                """
                traced_script_module = torch.jit.trace(model, example)
                traced_script_module.save("../output200/traced_model.pt")

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from cal_ce_mse import cross_entropy_mse
import os

sample_num = 1000  # 样本数量
min_cls_num = 2  # 最小类别数
max_cls_num = 100  # 最大类别数
times = 100  # 计算不同准确率循环的次数

for cls_num in range(min_cls_num, max_cls_num + 1):
    print('cls num:', cls_num)

    acc_list = []
    mse_list = []
    ce_list = []

    # 每个类别循环多次，计算不同准确率下的交叉熵损失和均方差损失
    for i in range(times):
        pred = torch.randn(sample_num, cls_num)  # 随机生成预测值
        pred_index = torch.argmax(F.softmax(pred, dim=1), dim=1)  # 预测的类别

        # 准确率为0
        if i == 0:
            error = pred_index + 1
            error[error == cls_num] = 0
            target = error

        # 准确率为1
        elif i == times - 1:
            target = pred_index

        # 准确率在0-1之间
        else:
            error = pred_index + 1
            error[error == cls_num] = 0
            target = torch.cat([pred_index[:i], error[i:]], dim=0)

        ce_loss, mse_loss, accuracy = cross_entropy_mse(pred, target)
        ce_list.append(ce_loss)
        mse_list.append(mse_loss)
        acc_list.append(accuracy)

    # 作图
    plt.plot(acc_list, ce_list, label=f'ce: {min(ce_list)}-{max(ce_list)} -> {round(max(ce_list) - min(ce_list), 3)}')
    plt.plot(acc_list, mse_list,
             label=f'mse: {min(mse_list)}-{max(mse_list)} -> {round(max(mse_list) - min(mse_list), 3)}')

    plt.legend()
    plt.xlabel('Accuracy')
    plt.ylabel('Loss')
    plt.title(f"cls_num={cls_num}, sample_num={sample_num}")

    os.makedirs('img', exist_ok=True)
    plt.savefig(f"img/cls_{cls_num}.jpg")
    plt.clf()

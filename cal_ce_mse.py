"""
计算交叉熵和均方差损失
"""

import torch
import torch.nn.functional as F


def cross_entropy_mse(pred, target):
    """
    :param pred:预测值
    :param target: 标签
    :return: 交叉熵损失，均方差损失，准确率
    """
    ce_loss = F.cross_entropy(pred, target.long()).item()  # 计算交叉熵损失

    mse_loss = F.mse_loss(F.softmax(pred, dim=1), F.one_hot(target, pred.shape[1])).item()  # 计算均方差损失

    ce_loss = round(ce_loss, 3)
    mse_loss = round(mse_loss, 3)

    # 准确率
    accuracy = torch.sum(torch.eq(target, torch.argmax(F.softmax(pred, dim=1), dim=1))) / target.shape[0]

    return ce_loss, mse_loss, accuracy

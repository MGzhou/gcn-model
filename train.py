#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Time :2024/06/13 10:07:26
@Desc :模型训练，测试
'''

import itertools
import os
import pickle
import urllib
from collections import namedtuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

from tqdm import tqdm


from data import Data, CoraData
from gcn import GcnModel

from utils import plot_loss_with_acc



def tensor_from_numpy(x, device):
    # 将向量由numpy格式转为torch.tensor张量
    return torch.from_numpy(x).to(device)


def read_cora(data_root, device):
    # 加载数据，并转换为torch.Tensor
    dataset = CoraData(data_root=data_root).data

    node_feature = dataset.x / dataset.x.sum(1, keepdims=True)  # 归一化特征数据，使得每一行和为1
    # 节点数量 ， 输入维度
    num_nodes, input_dim = node_feature.shape  # 2708,1433

    tensor_x = tensor_from_numpy(node_feature, device)  # (2708,1433)
    tensor_y = tensor_from_numpy(dataset.y, device)  #(2708,)
    # bool 数组， 可以用于从 tensor_x 中获取为 True 对应索引的值
    tensor_train_mask = tensor_from_numpy(dataset.train_mask, device) #前140个为True
    tensor_val_mask = tensor_from_numpy(dataset.val_mask, device)  # 140 - 639  500个
    tensor_test_mask = tensor_from_numpy(dataset.test_mask, device) # 1708 - 2707 1000个

    normalize_adjacency = CoraData.normalization(dataset.adjacency)   # 规范化邻接矩阵 计算 L=D^-0.5 * (A+I) * D^-0.5

    # 原始创建coo_matrix((data, (row, col)), shape=(4, 4)) indices为index复数
    indices = torch.from_numpy(np.asarray([normalize_adjacency.row, 
                                        normalize_adjacency.col]).astype('int64')).long()
    values = torch.from_numpy(normalize_adjacency.data.astype(np.float32))
    #构造邻居矩阵向量,构造出来的张量是 (2708,2708)
    tensor_adjacency = torch.sparse.FloatTensor(indices, values, 
                                                (num_nodes, num_nodes)).to(device)
    
    return input_dim, tensor_x, tensor_y, tensor_train_mask, tensor_val_mask, tensor_test_mask, tensor_adjacency



def test(model, adjacency, x, y, mask):
    """
    测试和评估函数
    """
    model.eval()  # 表示将模型转变为evaluation（测试）模式，这样就可以排除BN和Dropout对测试的干扰
    with torch.no_grad():  # 显著减少显存占用
        logits = model(adjacency, x) #(N,16)->(N,7) N节点数
        test_mask_logits = logits[mask]  # 矩阵形状和mask一样
        predict_y = test_mask_logits.max(1)[1]  # 返回每一行的最大值中索引（返回最大元素在各行的列索引）
        accuarcy = torch.eq(predict_y, y[mask]).float().mean()
    return accuarcy, test_mask_logits.cpu().numpy(), y[mask].cpu().numpy()



def main(config, is_draw_loss=True):
    """
    主函数
    Args:
        -------
        config: 参数字典
        is_draw_lossP: 是否画loss和acc的曲线图，默认画
    """
    
    # 获取训练测试数据
    input_dim, x, y, train_mask, val_mask, test_mask, adjacency = read_cora(data_root=config['data_path'], device=config['device'])

    # 模型定义：Model, Loss, Optimizer
    model = GcnModel(input_dim).to(config['device'])
    criterion = nn.CrossEntropyLoss().to(config['device'])  # criterion评判标准
    optimizer = optim.Adam(model.parameters(),    # optimizer 优化程序 ，使用Adam 优化方法，权重衰减https://blog.csdn.net/program_developer/article/details/80867468
                        lr=config['lr'], 
                        weight_decay=config['weight_decay'])

    #------------------------- 训练 ----------------------------#
    loss_history = []
    val_acc_history = []
    model.train()
    train_y = y[train_mask]  # shape=（140，）不是（2708，）了
    # 共进行200次训练
    for epoch in range(config['epochs']):
        logits = model(adjacency, x)  # 前向传播
        train_mask_logits = logits[train_mask]   # 只选择训练节点进行监督 (140,)
        
        loss = criterion(train_mask_logits, train_y)    # 计算损失值  
        optimizer.zero_grad()  # 梯度归零
        loss.backward()     # 反向传播计算参数的梯度
        optimizer.step()    # 使用优化方法进行梯度更新
        
        # 评估
        train_acc, _, _ = test(model, adjacency, x, y, train_mask)     # 计算当前模型训练集上的准确率  调用test函数
        val_acc, _, _ = test(model, adjacency, x, y, val_mask)     # 计算当前模型在验证集上的准确率
        
        # 记录训练过程中损失值和准确率的变化，用于画图
        loss_history.append(loss.item())
        val_acc_history.append(val_acc.item())
        print("Epoch {:03d}/{}: Loss {:.4f}, TrainAcc {:.4}, ValAcc {:.4f}".format(
            epoch, config['epochs'], loss.item(), train_acc.item(), val_acc.item()))
    
    #------------------------- 测试 ----------------------------#
    test_acc, _, _ = test(model, adjacency, x, y, test_mask)
    print("Test accuarcy: ", test_acc.item())

    if is_draw_loss:
        plot_loss_with_acc(loss_history, val_acc_history)


if __name__=="__main__":
    # 设置随机种子
    np.random.seed(1203)
    torch.manual_seed(1203)
    # 超参数定义
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu" #设备, 如果当前显卡忙于其他工作，可以设置为 DEVICE = "cpu"，使用cpu运行
    config = {
        "lr":0.1,               # 学习率
        "weight_decay":5e-4,    # 权重衰减
        "epochs":200,           # 训练轮次
        "device": DEVICE,       # 训练设备
        "data_path": "/data/zmp/progect/task4_keytech/test/gcn-model/data/cora"  # 数据集路径
    }

    main(config=config, is_draw_loss=True)

    
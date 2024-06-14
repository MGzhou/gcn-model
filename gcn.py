#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Time :2024/06/13 10:07:10
@Desc :模型文件
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """一层图卷积网络
        
        完整GCN函数
        f = sigma(D^-1/2 A D^-1/2 * H * W)
        卷积公式 = D^-1/2 A D^-1/2 * H * W

        adjacency = D^-1/2 A D^-1/2 已经经过归一化，标准化的拉普拉斯矩阵
        
        Args:
        ----------
            input_dim: int
                节点输入特征的维度
            output_dim: int
                输出特征维度
            use_bias : bool, optional
                是否使用偏置
        """
        super(GraphConvolution, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        
        # 定义GCN层的 W 权重形状
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        
        #定义GCN层的 b 权重矩阵
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    # 这里才是声明初始化 nn.Module 类里面的W,b参数
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法
    
        Args: 
        -------
            adjacency: torch.sparse.FloatTensor
                邻接矩阵
            input_feature: torch.Tensor
                输入特征
        """
        support = torch.mm(input_feature, self.weight)  # 矩阵相乘 m由matrix缩写
        output = torch.sparse.mm(adjacency, support)  # sparse 稀疏的
        if self.use_bias:
            output += self.bias  # bias 偏置
        return output
    
    # 一般是为了打印类实例的信息而重写的内置函数
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.input_dim) + ' -> ' \
            + str(self.output_dim) + ')'



class GcnModel(nn.Module):
    """
    定义一个包含两层GraphConvolution的模型
    """
    def __init__(self, input_dim=1433):
        super(GcnModel, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, 16) #(N,1433)->(N,16)
        self.gcn2 = GraphConvolution(16, 7) #(N,16)->(N,7)
    
    def forward(self, adjacency, feature):
        h = F.relu(self.gcn1(adjacency, feature)) #(N,1433)->(N,16),经过relu函数
        logits = self.gcn2(adjacency, h) #(N,16)->(N,7)
        return logits

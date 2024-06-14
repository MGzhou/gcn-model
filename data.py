#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Time :2024/06/13 10:06:40
@Desc :读取和处理数据集的类文件
'''

import itertools
import os
import pickle
import urllib
from collections import namedtuple

import numpy as np
import scipy.sparse as sp

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


Data = namedtuple('Data', ['x', 'y', 'adjacency',
                           'train_mask', 'val_mask', 'test_mask'])


class CoraData(object):
    """
    读取Cora数据集类
    """
    # 文件名称
    filenames = ["ind.cora.{}".format(name) for name in
                 ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']]

    def __init__(self, data_root="./data/cora", rebuild=False):
        """Cora数据，包括数据处理，加载等功能
        当数据的缓存文件存在时，将使用缓存文件，否则将进行处理，并将处理中间结果缓存

        处理之后的数据可以通过属性 .data 获得，它将返回一个数据对象，包括如下几部分：
            * x: 节点的特征，维度为 2708 * 1433，类型为 np.ndarray
            * y: 节点的标签，总共包括7个类别，类型为 np.ndarray
            * adjacency: 邻接矩阵，维度为 2708 * 2708，类型为 scipy.sparse.coo.coo_matrix
            * train_mask: 训练集掩码向量，维度为 2708，当节点属于训练集时，相应位置为True，否则False
            * val_mask: 验证集掩码向量，维度为 2708，当节点属于验证集时，相应位置为True，否则False
            * test_mask: 测试集掩码向量，维度为 2708，当节点属于测试集时，相应位置为True，否则False

        Args:
        -------
            data_root: string, optional
                存放数据的目录，原始数据路径: {data_root}
                缓存数据路径: {data_root}/processed_cora.pkl
            rebuild: boolean, optional
                是否需要重新构建数据集，默认将[x, y, adjacency, train_mask, val_mask, test_mask]缓存, 当设为True时, 如果存在缓存数据也会重建数据

        """
        self.data_root = data_root
        save_file = os.path.join(self.data_root, "processed_cora.pkl")
        if os.path.exists(save_file) and not rebuild:
            print("Using Cached file: {}".format(save_file))
            self._data = pickle.load(open(save_file, "rb"))
        else:
            # 处理数据集
            self._data = self.process_data()
            with open(save_file, "wb") as f:
                pickle.dump(self.data, f)
            print("Cached file: {}".format(save_file))
    
    @property
    def data(self):
        """返回Data数据对象，包括x, y, adjacency, train_mask, val_mask, test_mask"""
        return self._data

    def process_data(self):
        """
        处理数据，得到节点特征X 和标签Y，邻接矩阵adjacency，训练集train_mask、验证集val_mask 以及测试集test_mask
        """
        print("Process data ...")
        _, tx, allx, y, ty, ally, graph, test_index = [self.read_data(
            os.path.join(self.data_root, name)) for name in self.filenames]
        
        # 测试test_index的形状（1000，），如果那里不明白可以测试输出一下矩阵形状
        print('test_index shape',test_index.shape)
        
        train_index = np.arange(y.shape[0])  # [0,...139] 140个元素
        print('train_index shape',train_index.shape) 
        
        val_index = np.arange(y.shape[0], y.shape[0] + 500) # [140 - 640] 500个元素
        print('val_index shape',val_index.shape)  
        
        sorted_test_index = sorted(test_index)  # #test_index就是随机选取的下标,排下序

        x = np.concatenate((allx, tx), axis=0)  # 1708 +1000 =2708 特征向量
        y = np.concatenate((ally, ty), axis=0).argmax(axis=1) # 把最大值的下标重新作为一个数组 标签向量
        
        x[test_index] = x[sorted_test_index]  # 打乱顺序,单纯给test_index 的数据排个序,不清楚具体效果
        y[test_index] = y[sorted_test_index]
        num_nodes = x.shape[0]  #2078
        
        train_mask = np.zeros(num_nodes, dtype=np.bool_)  # 生成零向量
        val_mask = np.zeros(num_nodes, dtype=np.bool_)
        test_mask = np.zeros(num_nodes, dtype=np.bool_)
        train_mask[train_index] = True  # 前140个元素为训练集
        val_mask[val_index] = True  # 140 -639 500个
        test_mask[test_index] = True  # 1708-2708 1000个元素
        
        #下面两句是我测试用的，本来代码没有
        #是为了知道使用掩码后，y_train_mask 的形状，由输出来说是（140，）
        # 这就是后面划分训练集的方法
        y_train_mask = y[train_mask]
        print('y_train_mask shape',y_train_mask.shape)
        
        #构建邻接矩阵
        adjacency = self.build_adjacency(graph)
        print("Node's feature shape: ", x.shape)
        print("Node's label shape: ", y.shape)
        print("Adjacency's shape: ", adjacency.shape)
        print("Number of training nodes: ", train_mask.sum())
        print("Number of validation nodes: ", val_mask.sum())
        print("Number of test nodes: ", test_mask.sum())

        return Data(x=x, y=y, adjacency=adjacency,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    @staticmethod
    def build_adjacency(adj_dict):
        """根据邻接表创建邻接矩阵"""
        edge_index = []
        num_nodes = len(adj_dict)
        # print('num_nodes:\n',num_nodes)
        for src, dst in adj_dict.items():  # 格式为 {index：[index_of_neighbor_nodes]}
            edge_index.extend([src, v] for v in dst)  # 
            edge_index.extend([v, src] for v in dst)
            
        # 去除重复的边
        edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))  # 以轮到的元素为k,每个k对应一个数组，和k相同放进数组，不
                                                                                # 同再生成一个k,sorted()是以第一个元素大小排序
        
        edge_index = np.asarray(edge_index)
        
        #稀疏矩阵 存储非0值 节省空间
        adjacency = sp.coo_matrix((np.ones(len(edge_index)), 
                                   (edge_index[:, 0], edge_index[:, 1])),
                    shape=(num_nodes, num_nodes), dtype="float32")
        return adjacency

    @staticmethod
    def read_data(path):
        """使用不同的方式读取原始数据以进一步处理"""
        name = os.path.basename(path)
        if name == "ind.cora.test.index":
            out = np.genfromtxt(path, dtype="int64")
            return out
        else:
            out = pickle.load(open(path, "rb"), encoding="latin1")
            out = out.toarray() if hasattr(out, "toarray") else out
            return out

    @staticmethod
    def normalization(adjacency):
        """计算 L=D^-0.5 * (A+I) * D^-0.5"""
        adjacency += sp.eye(adjacency.shape[0])    # 增加自连接
        degree = np.array(adjacency.sum(1))
        d_hat = sp.diags(np.power(degree, -0.5).flatten())
        return d_hat.dot(adjacency).dot(d_hat).tocoo()  #返回稀疏矩阵的coo_matrix形式

if __name__=="__main__":
    # 这样可以单独测试Process_data函数
    a = CoraData(rebuild=True)

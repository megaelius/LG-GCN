import numpy as np
import pandas as pd
import h5py
import torch
from torch._C import device
from architecture import CustomDenseDeepGCN

from numpy import random
from gcn_lib.dense.torch_edge import DenseKnnGraph

from modelnet40 import ModelNet40
from architecture import ClassificationGraphNN
from types import SimpleNamespace
import networkx as nx

import argparse

from tqdm import tqdm
from gcn_lib.dense import pairwise_distance

def dense_knn_to_set(knn_index):
    edges = set()
    for i in range(knn_index.shape[2]):
        for j in range(knn_index.shape[3]):
            l = list(knn_index[:,:,i,j])
            tuple = (int(l[0]),int(l[1]))
            edges.add(tuple)
    return edges

def scaled_stress(feats1,feats2):
    f1 = feats1.transpose(2, 1).squeeze(-1)
    f2 = feats2.transpose(2, 1).squeeze(-1)
    d1squared = pairwise_distance(f1)
    d2squared = pairwise_distance(f2)
    #d1 = torch.sqrt(d1squared/f1.shape[2] + 1)
    #d2 = torch.sqrt(d2squared/f2.shape[2] + 1)
    d1 = torch.sqrt(d1squared)
    d2 = torch.sqrt(d2squared)
    #remove nans
    d1[d1!=d1]=0
    d2[d2!=d2]=0
    #print(d1[0,0,0])
    crit = torch.nn.MSELoss(reduction='none')
    #print(d1.shape, ((d1squared.view(f1.shape[0],-1)).sum(1, keepdim=True).unsqueeze(-1)).shape)
    #Shapes are B,N,N and B,1,1
    scaled_se = 2*crit(d1,d2)/((d1squared.view(f1.shape[0],-1)).sum(1, keepdim=True).unsqueeze(-1))
    #print(scaled_se.shape)
    flat_scaled_se = scaled_se.view(f1.shape[0],-1)
    #print(flat_scaled_se.shape)
    return torch.sqrt(flat_scaled_se.sum(dim=1).sum(dim=0)/2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='model weigths')
    parser.add_argument('-d', type = int, default=3, help='d_graph')
    args = parser.parse_args()

    dataset = ModelNet40(1024,'test')
    knn = DenseKnnGraph()
    device = torch.device('cpu')

    total_average_shortest_path = 0
    total_weakly_conected_components = 0
    count = 0
    count_sp = 0
    for pc, target in tqdm(dataset):
        with torch.no_grad():
            #forward pointcloud and obtain graphs
            points = torch.Tensor(pc).unsqueeze(0).unsqueeze(-1).transpose(2,1)

            knn_index_xyz = knn(points)
            edges_xyz = dense_knn_to_set(knn_index_xyz)

            #Create graphs
            nodelist = list(range(pc.shape[0]))

            DG_xyz = nx.DiGraph()
            DG_xyz.add_nodes_from(nodelist)
            DG_xyz.add_edges_from(edges_xyz)

            #Number of Weakly connected components
            cc = nx.number_weakly_connected_components(DG_xyz)
            total_weakly_conected_components += cc

            if cc==1:
                #Compute average shortest path
                asp_xyz = nx.average_shortest_path_length(DG_xyz)
                #asp_xyz += nx.average_shortest_path_length(DG_xyz)
                total_average_shortest_path += asp_xyz
                count_sp += 1

            count += 1

    filename = 'results/results_baseline.txt'
    with open(filename,'w') as f:
        f.write(f'baseline:\n')
        f.write(f'Average shortest path length: {total_average_shortest_path/count_sp}\n')
        f.write(f'Average number of weakly connected components {total_weakly_conected_components/count}\n')

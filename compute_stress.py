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
    opt = SimpleNamespace(n_filters = 64, 
                        k= 16,
                        act='relu',
                        norm = 'batch',
                        bias = True,
                        dropout = 0.3,
                        n_blocks = 4,
                        graph = 'KNN',
                        knn_criterion = 'MLP',
                        graph_feats = args.d,
                        in_channels = 3,
                        graph_dropout = 0,
                        n_classes = 40)

    model = ClassificationGraphNN(opt)
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict['state_dict'])
    model.eval()
    
    total_stress = 0
    total_average_shortest_path = 0
    total_percentage_shared_edges = 0
    total_weakly_conected_components = 0
    count = 0
    count_sp = 0
    for pc, target in tqdm(dataset):
        with torch.no_grad():
            #forward pointcloud and obtain graphs
            points = torch.Tensor(pc).unsqueeze(0).unsqueeze(-1).transpose(2,1)
            pred, edge_features = model(points.to(device))

            knn_index_mlp = knn(edge_features)
            knn_index_xyz = knn(points)
            edges_mlp = dense_knn_to_set(knn_index_mlp)
            edges_xyz = dense_knn_to_set(knn_index_xyz)
            
            #Compute stress for this pointcloud
            pc_stress = scaled_stress(points,edge_features)
            total_stress += pc_stress

            #Compute percentage of shared edges
            pse = len(edges_xyz.intersection(edges_mlp))/len(edges_xyz)
            total_percentage_shared_edges +=pse

            count += 1
    with open(f'{args.weights[:-4]}.txt','w') as f:
        f.write(f'{args.weights[:-4]}:\n')
        f.write(f'Average Stress: {total_stress/count}\n')
        #f.write(f'Average shortest path length: {total_average_shortest_path/count_sp}\n')
        f.write(f'Average percentage of shared edges with regular knn: {total_percentage_shared_edges/count}\n')
        #f.write(f'Average number of weakly connected components {total_weakly_conected_components/count}\n')
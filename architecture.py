import torch
from torch.nn.modules.activation import ReLU
from gcn_lib.dense import BasicConv, GraphConv2d, PlainDynBlock2d, ResDynBlock2d, DenseDynBlock2d, DenseDilatedKnnGraph, DenseKnnGraph, PlainBlock2d, ResBlock2d, DenseBlock2d, SimGraph, batched_index_select
from torch.nn import Sequential as Seq

class CustomDenseDeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(CustomDenseDeepGCN, self).__init__()
        channels = opt.n_filters
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        self.dropout = opt.dropout
        epsilon = opt.epsilon
        conv = opt.conv
        c_growth = channels
        self.n_blocks = opt.n_blocks
        self.graph = opt.graph
        self.knn_criterion = opt.knn_criterion
        self.mlp_conn = opt.mlp_conn
        if self.knn_criterion == 'MLP' or self.graph == 'sim':
            #self.graph_mlp = torch.nn.Sequential(torch.nn.Linear(9,opt.in_channels))
            '''
            self.graph_mlp = torch.nn.Sequential(torch.nn.Linear(9,27),
                                                torch.nn.ReLU(),
                                                torch.nn.Linear(27,27),
                                                torch.nn.ReLU(),
                                                torch.nn.Linear(27,9),
                                                torch.nn.ReLU(),
                                                torch.nn.Linear(9,opt.in_channels))
            '''
            self.graph_mlp = torch.nn.Sequential(torch.nn.Linear(opt.in_channels,32),
                                                torch.nn.ReLU(),
                                                torch.nn.Dropout(self.dropout),
                                                torch.nn.Linear(32,opt.in_channels),
                                                torch.nn.Dropout(self.dropout))
            if self.mlp_conn == 'dense':
                self.head = GraphConv2d(2*opt.in_channels, channels, conv, act, norm, bias)
            else:
                self.head = GraphConv2d(opt.in_channels, channels, conv, act, norm, bias)
        else:
            self.head = GraphConv2d(opt.in_channels, channels, conv, act, norm, bias)
        self.knn = DenseKnnGraph(k)
        self.similarity_graph = SimGraph(k)

        if opt.block.lower() == 'res':
            self.backbone = Seq(*[ResBlock2d(channels, conv, act, norm, bias)
                                  for i in range(self.n_blocks-1)])
            fusion_dims = int(channels + c_growth * (self.n_blocks - 1))
        elif opt.block.lower() == 'dense':
            self.backbone = Seq(*[DenseBlock2d(channels+c_growth*i, c_growth, conv, act, norm, bias)
                                  for i in range(self.n_blocks-1)])
            fusion_dims = int(
                (channels + channels + c_growth * (self.n_blocks - 1)) * self.n_blocks // 2)
        else:

            self.backbone = Seq(*[PlainBlock2d(channels, conv, act, norm, bias)
                                  for i in range(self.n_blocks - 1)])
            fusion_dims = int(channels + c_growth * (self.n_blocks - 1))

        self.fusion_block = BasicConv([fusion_dims, 1024], act, norm, bias)
        self.prediction = Seq(*[BasicConv([fusion_dims+1024, 512], act, norm, bias),
                                BasicConv([512, 256], act, norm, bias),
                                torch.nn.Dropout(p=opt.dropout),
                                BasicConv([256, opt.n_classes], None, None, bias)])

    def forward(self, inputs):
        #remove scaled xyz
        inputs = inputs[:,:6]
        if self.graph == 'KNN':
            if self.knn_criterion == 'xyz':
                edge_index = self.knn(inputs[:, 0:3])
            elif self.knn_criterion == 'all':
                edge_index = self.knn(inputs)
            elif self.knn_criterion == 'color':
                edge_index = self.knn(inputs[:, 3:6])
            elif self.knn_criterion == 'MLP':
                #inputs shape is B,9,N_points,1
                mlp_features = self.graph_mlp(inputs.transpose(3,1)).transpose(3,1)
                edge_index = self.knn(mlp_features)
                if self.mlp_conn == 'dense':
                    inputs = torch.cat((mlp_features,inputs),dim=1)
                else:
                    inputs = mlp_features
        else:
            mlp_features = self.graph_mlp(inputs.transpose(3,1)).transpose(3,1)
            similarities, edge_index = self.similarity_graph(mlp_features)
            #print(inputs.shape, similarities.shape)
            inputs = torch.cat((mlp_features,inputs),dim=1)
        feats = [self.head(inputs, edge_index)]
        for i in range(self.n_blocks-1):
            feats.append(self.backbone[i](feats[-1], edge_index))
        feats = torch.cat(feats, dim=1)

        fusion = torch.max_pool2d(self.fusion_block(feats), kernel_size=[feats.shape[2], feats.shape[3]])
        fusion = torch.repeat_interleave(fusion, repeats=feats.shape[2], dim=2)
        return self.prediction(torch.cat((fusion, feats), dim=1)).squeeze(-1)

class MessagePassing(torch.nn.Module):
    """
    Graph convolution adapted from Kearnes et al. (2016), https://arxiv.org/abs/1603.00856
    """
    def __init__(self, in_node, in_edge, m_channels, out_channels, norm):
        super(MessagePassing, self).__init__()
        '''
        self.mlp_node = Seq(
                            torch.nn.Linear(2*in_channels,m_channels),
                            torch.nn.ReLU(),
                            torch.nn.Linear(m_channels,out_channels)
                            )
        self.mlp_edge = Seq(
                            torch.nn.Linear(3*in_channels,m_channels),
                            torch.nn.ReLU(),
                            torch.nn.Linear(m_channels,out_channels)
                            )
        '''
        self.mlp_edge = Seq(
                            BasicConv([in_edge + 2*in_node, m_channels], 'relu', norm, True),
                            BasicConv([m_channels,out_channels], 'relu', norm, True)
        )
        self.mlp_node = Seq(
                            BasicConv([in_node + out_channels, m_channels], 'relu', norm, True),
                            BasicConv([m_channels,out_channels], 'relu', norm, True)
        )

    def forward(self, node_features, e_ij, edge_index):
        h_i = batched_index_select(node_features, edge_index[1])
        h_j = batched_index_select(node_features, edge_index[0])
        e_ij_prima = self.mlp_edge(torch.cat((e_ij,h_i,h_j), dim = 1))
        #replicate last dimension so that it is K and not 1
        m = torch.sum(e_ij_prima, dim = 3, keepdim=True).expand([e_ij_prima.shape[0],e_ij_prima.shape[1],e_ij_prima.shape[2],h_i.shape[-1]])
        h_i_prima = self.mlp_node(torch.cat((h_i,m), dim=1))
        return h_i_prima[:,:,:,0].unsqueeze(-1), e_ij_prima, edge_index

class CustomDenseGCN(torch.nn.Module):
    def __init__(self, opt):
        super(CustomDenseGCN, self).__init__()
        channels = opt.n_filters
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        self.dropout = opt.dropout
        epsilon = opt.epsilon
        conv = opt.conv
        c_growth = channels
        self.n_blocks = opt.n_blocks
        self.graph = opt.graph
        self.knn_criterion = opt.knn_criterion
        self.mlp_conn = opt.mlp_conn
        if self.knn_criterion == 'MLP':
            #self.graph_mlp = torch.nn.Sequential(torch.nn.Linear(9,opt.in_channels))
            '''
            self.graph_mlp = torch.nn.Sequential(torch.nn.Linear(opt.in_channels,32),
                                                torch.nn.ReLU(),
                                                torch.nn.Dropout(self.dropout),
                                                torch.nn.Linear(32,opt.in_channels),
                                                torch.nn.Dropout(self.dropout))
            '''
            self.graph_mlp = Seq(
                                 BasicConv([opt.in_channels, 4], 'relu', norm, True),
                                 BasicConv([4,3], None, None, True)
            )
        self.head = MessagePassing(opt.in_channels, 3, int(0.5*channels), channels, norm)
        '''
        elif self.knn_criterion == 'all':
            self.head = MessagePassing(opt.in_channels, opt.in_channels, int(0.5*channels), channels, norm)
        else:
            self.head = MessagePassing(opt.in_channels, 3, int(0.5*channels), channels, norm)
        '''
        self.knn = DenseKnnGraph(k)


        # 6 -> 64 -> 128 -> 256 features per node
        self.backbone = Seq(*[MessagePassing(channels*i, channels*i, int(0.75*channels*i), channels*(i+1), norm)
                              for i in range(1,self.n_blocks)])

        fusion_dims = int(sum([channels*(i+1) for i in range(self.n_blocks)]))
        '''
        self.fusion_block = Seq(
                                torch.nn.Linear(fusion_dims,1024),
                                torch.nn.ReLU()
                                )
        self.prediction = Seq(
                              torch.nn.Linear(fusion_dims+1024, 512),
                              torch.nn.ReLU(),
                              torch.nn.Linear(512,256),
                              torch.nn.ReLU(),
                              torch.nn.Linear(256,opt.n_classes)
                              )
        '''

        self.fusion_block = BasicConv([fusion_dims, 64], act, norm, bias)
        self.prediction = Seq(*[BasicConv([fusion_dims+64, 256], act, norm, bias),
                                BasicConv([256, 128], act, norm, bias),
                                torch.nn.Dropout(p=opt.dropout),
                                BasicConv([128, opt.n_classes], None, None, bias)])
        '''
        self.prediction = Seq(*[BasicConv([self.n_blocks*channels, 512], act, norm, bias),
                                BasicConv([512, 256], act, norm, bias),
                                torch.nn.Dropout(p=opt.dropout),
                                BasicConv([256, opt.n_classes], None, None, bias)])
        '''

    def forward(self, inputs, use_mlp_graph = True):
        #remove scaled xyz
        inputs = inputs[:,:6]
        if self.knn_criterion == 'xyz':
            edge_index = self.knn(inputs[:, 0:3])
            edge_features = inputs[:,:3]
        elif self.knn_criterion == 'all':
            edge_index = self.knn(inputs)
            edge_features = inputs[:,:6]
        elif self.knn_criterion == 'color':
            edge_index = self.knn(inputs[:, 3:6])
            edge_features = inputs[:,3:6]
        elif self.knn_criterion == 'MLP':
            #inputs shape is B,9,N_points,1
            edge_features = self.graph_mlp(inputs)
            if use_mlp_graph:
                edge_index = self.knn(edge_features)
            else:
                edge_index = self.knn(inputs[:, 0:3])
        gh_i = batched_index_select(edge_features, edge_index[1])
        gh_j = batched_index_select(edge_features, edge_index[0])
        e_ij = gh_i-gh_j
        h_1, e_1, edge_index = self.head(inputs, e_ij, edge_index)
        feats = [h_1]
        feats_edges = [e_1]
        for i in range(self.n_blocks-1):
            h_i, e_i, edge_index = self.backbone[i](feats[-1],feats_edges[-1], edge_index)
            feats.append(h_i)
            feats_edges.append(e_i)

        #Fusion block
        feats = torch.cat(feats, dim=1)
        fusion = torch.max_pool2d(self.fusion_block(feats), kernel_size=[feats.shape[2], feats.shape[3]])
        fusion = torch.repeat_interleave(fusion, repeats=feats.shape[2], dim=2)
        return self.prediction(torch.cat((fusion, feats), dim=1)).squeeze(-1)
        #return self.prediction(feats[-1]).squeeze(-1)

class GraphFeatures(torch.nn.Module):
    def __init__(self, input_dims, latent_dim, output_dim):
        super(GraphFeatures, self).__init__()
        self.feat_mlp = BasicConv([input_dims, latent_dim], 'relu', None, True)

        self.fusion = BasicConv([2*latent_dim,output_dim], None, None, False)

    def forward(self,x):
        local_feat = self.feat_mlp(x)
        global_feat = torch.max_pool2d(local_feat, kernel_size=[out.shape[2], out.shape[3]])
        global_feat = torch.repeat_interleave(global_feat, repeats=feats.shape[2], dim=2)
        feat = torch.cat((local_feat, global_feat), dim=1)
        return self.fusion(feat)


class ClassificationGraphNN(torch.nn.Module):
    def __init__(self, opt):
        super(ClassificationGraphNN, self).__init__()
        channels = opt.n_filters
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        self.dropout = opt.dropout
        self.n_blocks = opt.n_blocks
        self.graph = opt.graph
        self.knn_criterion = opt.knn_criterion
        self.graph_feats = opt.graph_feats
        if self.knn_criterion == 'MLP':
            #self.graph_mlp = torch.nn.Sequential(torch.nn.Linear(9,opt.in_channels))
            '''
            self.graph_mlp = torch.nn.Sequential(torch.nn.Linear(opt.in_channels,32),
                                                torch.nn.ReLU(),
                                                torch.nn.Dropout(self.dropout),
                                                torch.nn.Linear(32,opt.in_channels),
                                                torch.nn.Dropout(self.dropout))
            '''
            self.graph_mlp = Seq(
                                 BasicConv([opt.in_channels, 16], 'relu', None, True),
                                 #BasicConv([16, 16], 'relu', norm, True),
                                 #BasicConv([16, 16], 'relu', norm, True),
                                 torch.nn.Dropout(p=opt.graph_dropout),
                                 BasicConv([16,self.graph_feats], None, None, False)
            )

            #self.head = MessagePassing(opt.in_channels, self.graph_feats, int(0.5*channels), channels, norm)
            self.head = MessagePassing(opt.in_channels, 2*self.graph_feats, int(0.5*channels), channels, norm)
        else:
            #self.head = MessagePassing(opt.in_channels, opt.in_channels, int(0.5*channels), channels, norm)
            self.head = MessagePassing(opt.in_channels, 2*opt.in_channels, int(0.5*channels), channels, norm)
        self.knn = DenseKnnGraph(k)

        # 6 -> 64 -> 128 -> 256 features per node
        self.backbone = Seq(*[MessagePassing(channels*i, channels*i, int(0.75*channels*i), channels*(i+1), norm)
                              for i in range(1,self.n_blocks)])

        fusion_dims = int(sum([channels*(i+1) for i in range(self.n_blocks)]))

        self.fusion_block = BasicConv([fusion_dims, 512], act, norm, bias)
        self.prediction = Seq(*[BasicConv([512, 256], act, norm, bias),
                                BasicConv([256, 128], act, norm, bias),
                                torch.nn.Dropout(p=opt.dropout),
                                BasicConv([128, opt.n_classes], None, None, bias)])

    def forward(self, inputs, use_mlp_graph = True):
        if self.knn_criterion == 'xyz':
            edge_index = self.knn(inputs[:, 0:3])
            edge_features = inputs
        elif self.knn_criterion == 'MLP':
            #inputs shape is B,3,N_points,1
            edge_features = self.graph_mlp(inputs)
            if use_mlp_graph:
                edge_index = self.knn(edge_features)
            else:
                edge_index = self.knn(inputs[:, 0:3])
        gh_i = batched_index_select(edge_features, edge_index[1])
        gh_j = batched_index_select(edge_features, edge_index[0])
        diff_ij = gh_i-gh_j
        e_ij = torch.cat((gh_i,diff_ij), dim=1)
        #e_ij = diff_ij
        h_1, e_1, edge_index = self.head(inputs, e_ij, edge_index)
        feats = [h_1]
        feats_edges = [e_1]
        for i in range(self.n_blocks-1):
            h_i, e_i, edge_index = self.backbone[i](feats[-1],feats_edges[-1], edge_index)
            feats.append(h_i)
            feats_edges.append(e_i)

        #Fusion block
        feats = torch.cat(feats, dim=1)
        fusion = torch.max_pool2d(self.fusion_block(feats), kernel_size=[feats.shape[2], feats.shape[3]])

        return self.prediction(fusion).squeeze(-1), edge_features
        #return self.prediction(feats[-1]).squeeze(-1)

class ClassificationGraphNN2(torch.nn.Module):
    def __init__(self, opt):
        super(ClassificationGraphNN2, self).__init__()
        channels = opt.n_filters
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        self.dropout = opt.dropout
        self.n_blocks = opt.n_blocks
        self.graph = opt.graph
        self.knn_criterion = opt.knn_criterion
        self.graph_feats = opt.graph_feats
        if self.knn_criterion == 'MLP':
            #self.graph_mlp = torch.nn.Sequential(torch.nn.Linear(9,opt.in_channels))
            '''
            self.graph_mlp = torch.nn.Sequential(torch.nn.Linear(opt.in_channels,32),
                                                torch.nn.ReLU(),
                                                torch.nn.Dropout(self.dropout),
                                                torch.nn.Linear(32,opt.in_channels),
                                                torch.nn.Dropout(self.dropout))
            '''
            self.graph_mlp = GraphFeatures(opt.in_channels, 16, opt.graph_feats)

            #self.head = MessagePassing(opt.in_channels, self.graph_feats, int(0.5*channels), channels, norm)
            self.head = MessagePassing(opt.in_channels, 2*self.graph_feats, int(0.5*channels), channels, norm)
        else:
            #self.head = MessagePassing(opt.in_channels, opt.in_channels, int(0.5*channels), channels, norm)
            self.head = MessagePassing(opt.in_channels, 2*opt.in_channels, int(0.5*channels), channels, norm)
        self.knn = DenseKnnGraph(k)

        # 6 -> 64 -> 128 -> 256 features per node
        self.backbone = Seq(*[MessagePassing(channels*i, channels*i, int(0.75*channels*i), channels*(i+1), norm)
                              for i in range(1,self.n_blocks)])

        fusion_dims = int(sum([channels*(i+1) for i in range(self.n_blocks)]))

        self.fusion_block = BasicConv([fusion_dims, 512], act, norm, bias)
        self.prediction = Seq(*[BasicConv([512, 256], act, norm, bias),
                                BasicConv([256, 128], act, norm, bias),
                                torch.nn.Dropout(p=opt.dropout),
                                BasicConv([128, opt.n_classes], None, None, bias)])

    def forward(self, inputs, use_mlp_graph = True):
        if self.knn_criterion == 'xyz':
            edge_index = self.knn(inputs[:, 0:3])
            edge_features = inputs
        elif self.knn_criterion == 'MLP':
            #inputs shape is B,3,N_points,1
            edge_features = self.graph_mlp(inputs)
            if use_mlp_graph:
                edge_index = self.knn(edge_features)
            else:
                edge_index = self.knn(inputs[:, 0:3])
        gh_i = batched_index_select(edge_features, edge_index[1])
        gh_j = batched_index_select(edge_features, edge_index[0])
        diff_ij = gh_i-gh_j
        e_ij = torch.cat((gh_i,diff_ij), dim=1)
        #e_ij = diff_ij
        h_1, e_1, edge_index = self.head(inputs, e_ij, edge_index)
        feats = [h_1]
        feats_edges = [e_1]
        for i in range(self.n_blocks-1):
            h_i, e_i, edge_index = self.backbone[i](feats[-1],feats_edges[-1], edge_index)
            feats.append(h_i)
            feats_edges.append(e_i)

        #Fusion block
        feats = torch.cat(feats, dim=1)
        fusion = torch.max_pool2d(self.fusion_block(feats), kernel_size=[feats.shape[2], feats.shape[3]])

        return self.prediction(fusion).squeeze(-1), edge_features
        #return self.prediction(feats[-1]).squeeze(-1)

if __name__ == "__main__":
    import random, numpy as np, argparse
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    batch_size = 2
    N = 1024
    device = 'cuda'

    parser = argparse.ArgumentParser(description='PyTorch implementation of Deep GCN For semantic segmentation')
    parser.add_argument('--in_channels', default=9, type=int, help='input channels (default:9)')
    parser.add_argument('--n_classes', default=13, type=int, help='num of segmentation classes (default:13)')
    parser.add_argument('--k', default=20, type=int, help='neighbor num (default:16)')
    parser.add_argument('--block', default='res', type=str, help='graph backbone block type {plain, res, dense}')
    parser.add_argument('--conv', default='edge', type=str, help='graph conv layer {edge, mr}')
    parser.add_argument('--act', default='relu', type=str, help='activation layer {relu, prelu, leakyrelu}')
    parser.add_argument('--norm', default='batch', type=str, help='{batch, instance} normalization')
    parser.add_argument('--bias', default=True, type=bool, help='bias of conv layer True or False')
    parser.add_argument('--n_filters', default=64, type=int, help='number of channels of deep features')
    parser.add_argument('--n_blocks', default=7, type=int, help='number of basic blocks')
    parser.add_argument('--dropout', default=0.5, type=float, help='ratio of dropout')
    parser.add_argument('--epsilon', default=0.2, type=float, help='stochastic epsilon for gcn')
    parser.add_argument('--stochastic', default=False, type=bool, help='stochastic for gcn, True or False')
    args = parser.parse_args()

    pos = torch.rand((batch_size, N, 3), dtype=torch.float).to(device)
    x = torch.rand((batch_size, N, 6), dtype=torch.float).to(device)

    inputs = torch.cat((pos, x), 2).transpose(1, 2).unsqueeze(-1)

    # net = DGCNNSegDense().to(device)
    net = DenseDeepGCN(args).to(device)
    print(net)
    out = net(inputs)
    print(out.shape)

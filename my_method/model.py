import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.norm import GraphNorm
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv



class GCN(torch.nn.Module):
    def __init__(self, num_in_features, num_out_features):
        super().__init__()
        self.conv1 = GCNConv(num_in_features, num_in_features)
        self.norm1 = GraphNorm(num_in_features)
        self.conv2 = GCNConv(num_in_features, num_in_features)
        self.norm2 = GraphNorm(num_in_features)
        self.conv3 = GCNConv(num_in_features,num_out_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.float()
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x,edge_index)

        return x
    
    
class GAT_(torch.nn.Module):
    def __init__(self, num_in_features, num_out_features):
        super().__init__()
        # 使用8个注意力头，concat=False保持输出维度不变
        self.conv1 = GATConv(num_in_features, num_in_features, heads=8, concat=False)
        self.norm1 = GraphNorm(num_in_features)
        # 第二层同样保持维度不变
        self.conv2 = GATConv(num_in_features, num_in_features, heads=8, concat=False)
        self.norm2 = GraphNorm(num_in_features)
        # 最后一层使用单头注意力保证输出维度
        self.conv3 = GATConv(num_in_features, num_out_features, heads=1)

    def forward(self, data):
        # 保持原有的前向传播结构不变
        x, edge_index = data.x, data.edge_index
        x = x.float()
        
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv3(x, edge_index)
        
        return x


class GCNex(torch.nn.Module):
    def __init__(self, num_in_features, num_out_features):
        super().__init__()
        self.conv1 = GCNConv(num_in_features, num_in_features)
        self.norm = GraphNorm(num_in_features)
        self.conv2 = GCNConv(num_in_features, num_out_features)

    def forward(self, data, edge_index=None):
        # x, edge_index = data.x, data.edge_index
        print(type(data))
        if type(data) == torch.Tensor:
            x = data
        else:
            x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.norm(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x

class Mapper(nn.Module):
    def __init__(self, feature_dim, out_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(feature_dim, feature_dim * 2),
                                    nn.BatchNorm1d(feature_dim * 2),
                                    nn.LeakyReLU(),
                                    nn.Linear(feature_dim * 2, out_dim),
                                    nn.LeakyReLU(),
                                    nn.Linear(out_dim, num_classes),
                                    )

    def forward(self, features):
        return self.layers(features)

def graphgen(node_embeddings):
    device = node_embeddings.device
    _, num_nodes, _ = node_embeddings.shape
    sims = torch.bmm(node_embeddings, node_embeddings.transpose(1, 2))
    sims = sims * torch.ones(num_nodes, num_nodes).fill_diagonal_(0).to(device)  # disregard self-similarities
    directed: Tensor = sims > (sims.sum(dim=2) / num_nodes - 1).unsqueeze(dim=2)  # average only over non-zero elms
    undirected = directed + directed.transpose(1, 2)
    assert torch.all(undirected == undirected.transpose(1, 2)).item()  # validate symmetrization
    edges = undirected.nonzero()

    edge_lists = []
    offset = 0
    graphs = []
    for i, sample in enumerate(undirected):
        num_edges = undirected[i].sum()
        edges_i = edges[offset: offset + num_edges]
        # Edge list in COO format
        edges_i = edges_i[:, 1:].T
        edge_lists.append(edges_i)
        offset = offset + num_edges
        graphs.append(Data(x = node_embeddings[i], edge_index=edges_i))

    return graphs

def compute_reprs(im,input_dim,feature_dim):
    device = im.device
    graphs = graphgen(im)
    graph_loader = DataLoader(graphs, batch_size=len(im))
    print(next(iter(graph_loader))._num_graphs)
    #print(graphs)
    aggregator = GCN(num_in_features=input_dim, num_out_features=feature_dim).to(device)
    ad_net=Mapper(feature_dim=feature_dim,out_dim=feature_dim,num_classes=200).to(device)
    semrel_graphs = aggregator(next(iter(graph_loader)))
    semrel_graphs = semrel_graphs.reshape(len(im), im.shape[1], -1)  # batch_size x num_local
    agg_embed = semrel_graphs.mean(dim=1)
    logits = ad_net(agg_embed)
    return logits

class DisjointRelationNet(nn.Module):
    def __init__(self, feature_dim, out_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(feature_dim, feature_dim * 2),
                                    nn.BatchNorm1d(feature_dim * 2),
                                    nn.LeakyReLU(),
                                    nn.Linear(feature_dim*2, feature_dim * 2),
                                    nn.BatchNorm1d(feature_dim * 2),
                                    nn.LeakyReLU(),
                                    nn.Linear(feature_dim * 2, out_dim),
                                    nn.LeakyReLU(),
                                    nn.Linear(out_dim, num_classes),
                                    )

    def forward(self, features_1, features_2,features_3, features_4):
        pair = torch.cat([features_1, features_2,features_3,features_4], dim=1)
        return self.layers(pair)

class DisjointRelationNet_old(nn.Module):
    def __init__(self, feature_dim, out_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(feature_dim, feature_dim * 2),
                                    nn.BatchNorm1d(feature_dim * 2),
                                    nn.LeakyReLU(),
                                    nn.Linear(feature_dim*2, feature_dim * 2),
                                    nn.BatchNorm1d(feature_dim * 2),
                                    nn.LeakyReLU(),
                                    nn.Linear(feature_dim * 2, out_dim),
                                    nn.LeakyReLU(),
                                    nn.Linear(out_dim, num_classes),
                                    )

    def forward(self, features_1, features_2,features_3):
        pair = torch.cat([features_1, features_2,features_3], dim=1)
        return self.layers(pair)
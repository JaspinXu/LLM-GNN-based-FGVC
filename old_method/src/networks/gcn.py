import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn import GATConv, GraphNorm

class GCN(torch.nn.Module):
    def __init__(self, num_in_features, num_out_features, num_heads=4):
        super().__init__()
        # 使用 GATConv 替代 GCNConv
        self.att1 = GATConv(num_in_features, num_in_features, heads=num_heads, concat=True)
        self.norm = GraphNorm(num_in_features * num_heads)  # 乘以 heads 数量，因为我们将 concat 的输出
        self.att2 = GATConv(num_in_features * num_heads, num_out_features, heads=1, concat=False)  # 第二层只保留一个输出

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.att1(x, edge_index)  # 第一层注意力机制
        x = self.norm(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.att2(x, edge_index)  # 第二层注意力机制

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

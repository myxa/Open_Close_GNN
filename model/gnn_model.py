import torch.nn as nn
import torch
from torch_geometric.nn import global_mean_pool, GCNConv, GATv2Conv, BatchNorm, GIN

# todo module list


class GCN(nn.Module):
    def __init__(self, num_features, channels, dropout=.3):
        super().__init__()
        self.conv1 = GCNConv(int(num_features), channels[0])
        self.bn1 = BatchNorm(channels[0])
        self.conv2 = GCNConv(channels[0], channels[1])
        self.dropout = nn.Dropout(dropout)
        self.relu1 = nn.ReLU()
        self.lin1 = nn.Linear(channels[1], 2)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.relu1(self.conv1(x, edge_index, edge_attr))
        x = self.bn1(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.dropout(x)
        x = global_mean_pool(x, batch)
        x = self.lin1(x)
        return x
    

class GINModel(nn.Module):
    def __init__(self, num_features, channels, layers=1, dropout=.3):
        super().__init__()
        self.conv1 = GIN(int(num_features), channels[0], 
                         num_layers=layers, out_channels=channels[1], dropout=dropout)
        self.relu1 = nn.ReLU()
        self.conv2 = GIN(int(num_features), channels[0], 
                         num_layers=layers, out_channels=channels[1], dropout=dropout)
        self.relu2 = nn.ReLU()
        self.lin1 = nn.Linear(channels[1] * 2, 2)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x1 = self.relu1(self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr))
        x1 = global_mean_pool(x1, batch)
        x2 = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x2 = global_mean_pool(x2, batch)
        x = torch.cat((x1, x2), dim=1)
        x = self.lin1(x)
        return x


class GATv2(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, heads=2, dropout=0):
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
        self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1)
        self.lin = nn.Linear(dim_out, 2)
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()
        self.bn = BatchNorm(dim_h*heads)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.gat1(x, edge_index)
        x = self.bn(self.elu(x))
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x


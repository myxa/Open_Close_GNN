import torch.nn as nn
from torch_geometric.nn import global_mean_pool, GCNConv, GATv2Conv, BatchNorm

# todo module list


class GCN(nn.Module):
    def __init__(self, num_features, channels, dropout=.3):
        super().__init__()
        self.conv1 = GCNConv(int(num_features), channels[0])
        self.bn1 = BatchNorm(channels[0])
        self.conv2 = GCNConv(channels[0], channels[1])
        self.bn2 = BatchNorm(channels[1])
        self.conv3 = GCNConv(channels[1], channels[2])
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(channels[2], 2)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch
        x = self.relu(self.conv1(x, edge_index, edge_attr))
        x = self.bn1(x)
        # x = self.dropout(x)
        x = self.relu(self.conv2(x, edge_index, edge_attr))
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.relu(self.conv3(x, edge_index, edge_attr))
        x = global_mean_pool(x, batch)
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

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = self.bn(self.elu(x))
        x = self.dropout(x)
        x = self.elu(self.gat2(x, edge_index))
        x = global_mean_pool(x, data.batch)
        x = self.lin(x)
        return x


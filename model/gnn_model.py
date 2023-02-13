import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, GCNConv, GATv2Conv, GATConv

# todo сделать module list


class GCN(nn.Module):
    def __init__(self,
                 num_features,
                 num_classes,
                 channels,
                 dropout=.3):
        super().__init__()
        self.p = dropout
        self.conv1 = GCNConv(int(num_features), channels[0])
        self.conv2 = GCNConv(channels[0], channels[1])
        self.conv3 = GCNConv(channels[1], channels[2])

        self.lin1 = nn.Linear(channels[2], int(num_classes))

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.p, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.p, training=self.training)
        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x = global_mean_pool(x, batch)
        x = self.lin1(x)
        return x

    class GAT(nn.Module):
        def __init__(self,
                     num_features,
                     num_classes,
                     channels,
                     dropout=.3):
            super().__init__()
            self.p = dropout
            self.conv1 = GATConv(int(num_features), channels[0])
            self.conv2 = GATConv(channels[0], channels[1])
            self.conv3 = GATConv(channels[1], channels[2])

            self.lin1 = nn.Linear(channels[2], int(num_classes))

        def forward(self, data):
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            batch = data.batch
            x = F.relu(self.conv1(x, edge_index, edge_attr))
            x = F.dropout(x, p=self.p, training=self.training)
            x = F.relu(self.conv2(x, edge_index, edge_attr))
            x = F.dropout(x, p=self.p, training=self.training)
            x = F.relu(self.conv3(x, edge_index, edge_attr))
            x = global_mean_pool(x, batch)
            x = self.lin1(x)
            return x


class GATv2(nn.Module):
    """Graph Attention Network"""
    def __init__(self, dim_in, dim_h, dim_out, heads=2):
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
        self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1)
        self.lin = nn.Linear(dim_out, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = F.dropout(x, p=0.6, training=self.training)
        h = self.gat1(h, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=0.6, training=self.training)
        h = F.elu(self.gat2(h, edge_index))
        h = global_mean_pool(h, data.batch)
        h = self.lin(h)
        return h


import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from scipy.sparse import coo_matrix
import numpy as np
import os
import shutil
from scipy.io import loadmat

from torch_geometric.nn import knn_graph


class OpenCloseDataset(Dataset):
    def __init__(self, datafolder, open_file, close_file, 
                 reload=False, test=False, transform=None, 
                 pre_transform=None, k_degree=10, edge_attr=None):

        self.reload = reload
        self.test = test
        self.datafolder = datafolder
        self.close = close_file
        self.open  = open_file
        self.edge_attr = edge_attr
        self.k_degree = k_degree
        

        if self.reload:
            for root, dirs, files in os.walk(f'{self.datafolder}/processed'):
                for f in files:
                    os.unlink(os.path.join(root, f))
                for d in dirs:
                    shutil.rmtree(os.path.join(root, d))

        super().__init__(root=datafolder, transform=transform, pre_transform=pre_transform)

    @property
    def raw_file_names(self):
        return ['open_84.npy', 'close_84.npy']
    
    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(len(self.close) + len(self.open))]
        
    def len(self):
        return len(self.close) + len(self.open)
    
    
    def process(self):

        for index in range(len(self.open)):
            _ = self._load_and_save(self.open[index], index, 'open')

        for index in range(len(self.close)):
            _ = self._load_and_save(self.close[index], index, 'close')


    def _load_and_save(self, matr, index, state):

        x = torch.from_numpy(matr).float()

        #adj = self.compute_KNN_graph(matr, k_degree=self.k_degree)
        
        #adj = torch.from_numpy(adj).float()
        #edge_index, edge_attr = dense_to_sparse(adj)
        #self.edge_attr = edge_attr
        edge_index = knn_graph(x, self.k_degree)


        label = torch.tensor(0 if state == 'close' else 1).long()
        data = Data(x=x, edge_index=edge_index, edge_attr=self.edge_attr, y=label)

        index = index + len(self.open) if state == 'close' else index


        torch.save(data,
                    os.path.join(self.processed_dir,
                                f'data_{index}.pt'))
        
        return data
    
    def compute_KNN_graph(self, matrix, k_degree):
        """ Calculate the adjacency matrix from the connectivity matrix.
            Adapted from: https://github.com/QKmeans0902/GCN_MDD_Classification
        """

        matrix = np.abs(matrix)
        idx = np.argsort(-matrix)[:, 0:k_degree]
        matrix.sort()
        matrix = matrix[:, ::-1]
        matrix = matrix[:, 0:k_degree]

        A = self._adjacency(matrix, idx).astype(np.float32)

        return A

    def _adjacency(self, dist, idx):

        m, k = dist.shape
        assert m, k == idx.shape
        assert dist.min() >= 0

        # Weight matrix.
        I = np.arange(0, m).repeat(k)
        J = idx.reshape(m * k)
        V = dist.reshape(m * k)
        W = coo_matrix((V, (I, J)), shape=(m, m))

        # No self-connections.
        W.setdiag(0)

        # Non-directed graph.
        bigger = W.T > W
        W = W - W.multiply(bigger) + W.T.multiply(bigger)

        return W.todense()


    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 'test_'
                                           f'data_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir,
                                           f'data_{idx}.pt'))
        return data


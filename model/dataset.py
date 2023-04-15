import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import dense_to_sparse
from scipy.sparse import coo_matrix
import numpy as np
import os
import shutil
from scipy.io import loadmat


class OpenCloseDataset(Dataset):
    def __init__(self, datafolder, open_file, close_file, 
                 reload=False, test=False, transform=None, 
                 pre_transform=None, k_degree=10, threshold=0.5, edge_attr=None,
                 noise_close=None, win_close=None, noise_open=None, win_open=None, 
                 noise_n=20, win_n=100):

        self.reload = reload
        self.test = test
        self.datafolder = datafolder
        self.close = close_file
        self.open  = open_file
        self.edge_attr = edge_attr
        self.k_degree = k_degree
        self.threshold = threshold
        self.outliers = np.array([256, 257, 258, 259]) 
        # [52, 256, 53, 257, 54, 258, 55, 259]

        if noise_close is not None:
            idx = np.random.choice(np.arange(len(noise_close)), noise_n)
            self.close = np.concatenate([
                self.close, noise_close[idx]])
            self.open = np.concatenate([
                self.open, noise_open[idx]])

        if win_close is not None:
            idx = np.random.choice(np.arange(len(win_close)), win_n)
            self.close = np.concatenate([
                self.close, win_close[idx]])
            self.open = np.concatenate([
                self.open, win_open[idx]])            

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

        matr = np.delete(matr, self.outliers, 0)
        matr = np.delete(matr, self.outliers, 1)

        x = torch.from_numpy(matr).float()

        if self.k_degree is not None:
            adj = self.compute_KNN_graph(matr, k_degree=self.k_degree)
            adj = torch.from_numpy(adj).float()
            edge_index, edge_attr = dense_to_sparse(adj)
            self.edge_attr = edge_attr
        else:
            adj = self._adjacency_threshold(x)
            edge_index, edge_attr = dense_to_sparse(adj)
            self.edge_attr = edge_attr

        label = torch.tensor(0 if state == 'close' else 1).long()
        data = Data(x=x, edge_index=edge_index, edge_attr=self.edge_attr, y=label)
        index = index + len(self.open) if state == 'close' else index

        if self.test:
            torch.save(data,
                       os.path.join(self.processed_dir, 'test_'
                                    f'data_{index}.pt'))
        else:
            torch.save(data,
                       os.path.join(self.processed_dir,
                                    f'data_{index}.pt'))
        return data
    
    def compute_KNN_graph(self, matrix, k_degree):
        """ Calculate the adjacency matrix from the connectivity matrix."""

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

    def _adjacency_threshold(self, matr):
        return ((self.threshold[0] > matr) + (matr > self.threshold[1])) * matr

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


class oldOpenCloseDataset(Dataset):
    def __init__(self, datafolder, reload=False, test=False, transform=None, pre_transform=None, k_degree=10):
        self.reload = reload
        self.test = test
        self.datafolder = datafolder
        self.close = loadmat(f'{datafolder}/raw/resultsROI_Condition001.mat')['Z']
        self.open  = loadmat(f'{datafolder}/raw/resultsROI_Condition002.mat')['Z']
        self.edge_attr = None
        self.k_degree = k_degree
        self.dl = np.array([256, 257, 258, 259]) # [52, 256, 53, 257, 54, 258, 55, 259]

        super().__init__(root=datafolder, transform=transform, pre_transform=pre_transform)

    @property
    def raw_file_names(self):
        return ['resultsROI_Condition001.mat', 'resultsROI_Condition002.mat']
    
    @property
    def processed_file_names(self):
        if self.reload:
            return [f'_data_{i}.pt' for i in range(84*2)]
        else:
            return [f'data_{i}.pt' for i in range(84*2)]
        
    def len(self):
        return 84 + 84
    
    def download(self):
        print('yo')
    
    def process(self):

        for index in range(84):
            _ = self._load_and_save(index, 'open')

        for index in range(84):
            _ = self._load_and_save(index, 'close')

    def _load_and_save(self, index, state):

        if state == 'open':
            matr = self.open[:, :, index]
        elif state == 'close':
            matr = self.close[:, :, index]

        # todo fill with 0 or 1
        np.fill_diagonal(matr, 1)
        matr = np.delete(matr, self.dl, 0)
        matr = np.delete(matr, self.dl, 1)

        x = torch.from_numpy(matr).float()

        if self.k_degree is not None:
            adj = self.compute_KNN_graph(matr, k_degree=self.k_degree)
            adj = torch.from_numpy(adj).float()
            edge_index, edge_attr = dense_to_sparse(adj)
            self.edge_attr = edge_attr
        else:
            edge_index = self._adjacency_threshold(x)

        label = torch.tensor(0 if state == 'close' else 1).long()

        data = Data(x=x, edge_index=edge_index, edge_attr=self.edge_attr, y=label)

        index = index + 84 if state == 'close' else index
        if self.test:
            torch.save(data,
                       os.path.join(self.processed_dir, 'test',
                                    f'data_{index}.pt'))
        else:
            torch.save(data,
                       os.path.join(self.processed_dir,
                                    f'data_{index}.pt'))
        return data
    
    def compute_KNN_graph(self, matrix, k_degree):
        """ Calculate the adjacency matrix from the connectivity matrix."""

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

    def _adjacency_threshold(self, matr, threshold=0.5):
        # todo optimize ???
        # todo переделать порог
        idx = []
        for i in range(len(matr)):
            for j in range(len(matr)):
                if abs(matr[i, j]) > threshold:
                    idx.append((i, j))

        return torch.tensor(idx).long().t().contiguous()

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir,
                                           f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir,
                                           f'data_{idx}.pt'))
        return data
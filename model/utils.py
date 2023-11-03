import torch
from tqdm.notebook import tqdm
import numpy as np
import pickle
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
from torchmetrics.classification import BinaryRecall, BinaryPrecision, BinaryAccuracy, F1Score


def train_epoch(train_loader, model, criterion, optimizer):
    model.train()
    for data in train_loader:
        data = data.to(device())
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def eval_epoch(loader, model, criterion):
    model.eval()
    losses = []
    acc, pr, rc, f = [], [], [], []
    accuracy = BinaryAccuracy()
    precision = BinaryPrecision()
    recall = BinaryRecall()
    f1 = F1Score(task='binary', average='macro')
    #k = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device())
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(out, data.y)
            losses.append(loss.item())
            #k +=1
            pred = out.argmax(dim=1)
            acc.append(accuracy(pred.cpu(), data.y.cpu()))
            pr.append(precision(pred.cpu(), data.y.cpu()))
            rc.append(recall(pred.cpu(), data.y.cpu()))
            f.append(f1(pred.cpu(), data.y.cpu()))

    return np.mean(losses), np.mean(acc), np.mean(pr), np.mean(rc), np.mean(f)


def train(model, epochs, train_loader, val_loader, criterion, optimizer, scheduler=None, save_best=False, path_to_save=None):

    history = []
    best_val_loss = 1000
    for epoch in tqdm(range(1, epochs+1)):
        train_epoch(train_loader, model, criterion, optimizer)
        train_loss, train_acc, train_prec, train_rec, _ = eval_epoch(train_loader, model, criterion)
        val_loss, test_acc, test_prec, test_rec, test_f1 = eval_epoch(val_loader, model, criterion)
        if scheduler is not None:
            scheduler.step()

        if save_best:
            if best_val_loss > val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), path_to_save)


        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss {val_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

        print(f'Test precision: {test_prec:.4f}, Test recall: {test_rec:.4f}, Test F1: {test_f1:.4f}')
        # f'Train precision: {train_prec:.4f}, Train recall: {train_rec:.4f}, '
        history.append((train_loss, val_loss, train_acc, test_acc))

    if save_best:
        model.load_state_dict(torch.load(path_to_save, map_location=device()))

    return history


def device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def cross_val(data, model_name, n_splits=10, epochs=20, batch_size=32,  **kwargs):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    eval_metrics = np.zeros((skf.n_splits, 3))

    labels = [data[i].y for i in range(len(data))]

    for n_fold, (train_idx, test_idx) in tqdm(enumerate(skf.split(labels, labels))):
        model = model_name(**kwargs).to(device())
        optimizer = torch.optim.Adam(model.parameters(), **kwargs)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
        train_loader = DataLoader(data[list(train_idx)], batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(data[list(test_idx)], batch_size=batch_size, shuffle=True)
        min_v_loss = np.inf
        print(n_fold)
        pr, rc, acc = [], [], []
        for epoch in range(epochs):
            train_epoch(train_loader, model, criterion, optimizer)
            train_loss, train_acc, _, _ = eval_epoch(train_loader, model, criterion)
            val_loss, test_acc, _, _ = eval_epoch(test_loader, model, criterion)
            scheduler.step()
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
                  f'Test Loss {val_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
            # print(f'Train Prec: {train_pr:.3f},
            # Train Rec: {train_rc:.3f}, Test Prec: {val_pr:.3f}, Test Rec: {val_rc:.3f}')
            # rc.append(val_rc)
            # pr.append(val_pr)
            acc.append(test_acc)
            if min_v_loss > val_loss:
                min_v_loss = val_loss
                best_test_acc = test_acc

        eval_metrics[n_fold, 0] = best_test_acc
        eval_metrics[n_fold, 1] = np.mean(acc)
        eval_metrics[n_fold, 2] = np.std(acc)

        return eval_metrics
    

def load_data_and_groups(pickle_path, filename, close_ids=None, open_ids=None):
    
    with open(pickle_path, 'rb') as handle:
        data = pickle.load(handle)
    
    opened, closed = data[filename]

    if close_ids and open_ids is not None:

        close_ids = np.loadtxt(close_ids, dtype=int)
        open_ids = np.loadtxt(open_ids, dtype=int)

        groups = np.concatenate([ # сначала открытые потом закрытые, так в датасете 
            open_ids,
            close_ids])
        
    else:
        groups = np.concatenate([
            np.arange(opened.shape[0], dtype=int), 
            np.arange(closed.shape[0], dtype=int)
            ])

    labels = [1 for _ in range(len(opened))] + [0 for _ in range(len(closed))]

    return closed, opened, groups, labels




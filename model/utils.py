import torch
from tqdm.notebook import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score


# todo precision recall


def train_epoch(train_loader, model, criterion, optimizer):
    model.train()
    for data in train_loader:
        data = data.to(device())
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def eval_epoch(loader, model, criterion):
    model.eval()
    losses = 0
    correct = 0
    pr = []
    rc = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device())
            out = model(data)
            loss = criterion(out, data.y)
            losses += loss.item()
            pred = out.argmax(dim=1)
            correct += int((pred.cpu() == data.y.cpu()).sum())
            pr.append(precision_score(data.y.cpu(), pred.cpu(), zero_division=0))
            rc.append(recall_score(data.y.cpu(), pred.cpu(), zero_division=0))

    return losses / len(loader.dataset), correct / len(loader.dataset), np.mean(pr), np.mean(rc)


def train(model, epochs, train_loader, val_loader, criterion, optimizer, scheduler=None):
    losses = []
    for epoch in tqdm(range(1, epochs+1)):
        train_epoch(train_loader, model, criterion, optimizer)
        train_loss, train_acc, _, _ = eval_epoch(train_loader, model, criterion)
        val_loss, test_acc, _, _ = eval_epoch(val_loader, model, criterion)
        if scheduler is not None:
            scheduler.step()

        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss {val_loss:.4f}, Train Acc: {train_acc:.4f},'
              f' Test Acc: {test_acc:.4f}')
        losses.append((train_loss, val_loss))
    return losses


def device(t=None):
    if t is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        return torch.device('cpu')



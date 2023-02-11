import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score


# todo precision recall


def train_epoch(train_loader, model, criterion, optimizer):
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def eval_epoch(loader, model, criterion):
    model.eval()
    losses = 0
    correct = 0
    pr = []
    rc = []
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data)
        loss = criterion(out, data.y)
        losses += loss.item()
        pred = out.argmax(dim=1)  # Use the class with the highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        pr.append(precision_score(data.y, pred, zero_division=0))
        rc.append(recall_score(data.y, pred, zero_division=0))
        # Derive ratio of correct predictions.
    return losses / len(loader.dataset), correct / len(loader.dataset), np.mean(pr), np.mean(rc)


def train(model, epochs, train_loader, val_loader, criterion, optimizer, scheduler=None):
    losses = []
    for epoch in tqdm(range(1, epochs+1)):
        train_epoch(train_loader, model, criterion, optimizer)
        train_loss, train_acc, _, _ = eval_epoch(train_loader, model, criterion)
        val_loss, test_acc, _, _ = eval_epoch(val_loader, model, criterion)
        if scheduler is not None:
            scheduler.step()

        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss {val_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        losses.append((train_loss, val_loss))
    return losses


def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')



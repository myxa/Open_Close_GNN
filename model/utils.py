import torch
from tqdm import tqdm
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F


def fit_epoch(model, loader, criterion, optimizer):
    model.train()
    loss_all = 0
    for data in loader:
        data = data.to(device())
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        loss_all += loss.item()  # data.num_graphs *
        optimizer.step()
    return np.mean(loss_all)


def eval_epoch(model, loader, criterion):
    model.eval()
    pred = []
    label = []
    loss_all = 0
    for data in loader:
        data = data.to(device())
        output = model(data)
        loss = criterion(output, data.y)
        loss_all += loss.item() # data.num_graphs *
        pred.append(F.softmax(output, dim=1).max(dim=1)[1])
        label.append(data.y)

    y_pred = torch.cat(pred, dim=0).cpu().detach().numpy()
    y_true = torch.cat(label, dim=0).cpu().detach().numpy()
    tn, fp, fn, tp = confusion_matrix(y_pred, y_true).ravel()
    # print(tn)
    epoch_sen = tp / (tp + fn + 0.000001)
    epoch_spe = tn / (tn + fp + 0.000001)
    epoch_acc = (tn + tp) / (tn + tp + fn + fp)
    return epoch_sen, epoch_spe, epoch_acc, np.mean(loss_all)


def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(train_dataset, val_dataset, model, epochs, batch_size, opt, criterion, save_best=False):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    history = []
    best_val_loss = 1000
    log_template = "train_loss {t_loss:0.4f} " \
                   "val_loss {v_loss:0.4f} val_acc {v_acc:0.4f} " \
                   "\nprecision {precision:0.3f} recall {recall:0.3f}"
    with tqdm(desc="epoch", total=epochs) as pbar_outer:

        for epoch in range(1, epochs+1):
            train_loss = fit_epoch(model, train_loader, criterion, opt)
            precision, recall, acc, val_loss = eval_epoch(model, val_loader, criterion)
            history.append((train_loss, val_loss, acc, precision, recall))

            if save_best:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), save_best)

            pbar_outer.update(1)
            tqdm.write(log_template.format(t_loss=train_loss,
                                           v_loss=val_loss, v_acc=acc,
                                           precision=precision, recall=recall))
            print('')

    if save_best:
        model.load_state_dict(torch.load(save_best, map_location=device()))

    return history

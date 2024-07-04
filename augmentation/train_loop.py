import torch
from tqdm.notebook import tqdm
import numpy as np



def train_epoch_transformer(train_loader, model, optimizer):
    model.train()
    for batch in train_loader:

        loss = model(
            past_values=batch["past_values"].to(device()),
            future_values=batch["future_values"].to(device()),
            past_time_features=batch["past_time_features"].to(device()),
            future_time_features=batch["future_time_features"].to(device()),
            past_observed_mask=batch["past_observed_mask"].to(device()),
        ).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def eval_epoch_transformer(loader, model):
    model.eval()
    losses = []

    with torch.no_grad():
        for batch in loader:

            loss = model(
                past_values=batch["past_values"].to(device()),
                future_values=batch["future_values"].to(device()),
                past_time_features=batch["past_time_features"].to(device()),
                future_time_features=batch["future_time_features"].to(device()),
                past_observed_mask=batch["past_observed_mask"].to(device()),

            ).loss
            losses.append(loss.item())

    return np.mean(losses)


def train_aug(model, epochs, train_loader, val_loader, 
              criterion, optimizer, scheduler=None, 
              save_best=False, path_to_save=None, transformer=False):

    history = []
    best_val_loss = 1000
    val_loss = None
    for epoch in tqdm(range(1, epochs+1)):
        if transformer:
            train_epoch_transformer(train_loader, model, optimizer)
            train_loss = eval_epoch_transformer(train_loader, model)


        else:
            train_epoch(train_loader, model, criterion, optimizer)
            train_loss = eval_epoch(train_loader, model, criterion)
            val_loss = eval_epoch(val_loader, model, criterion)


        if scheduler is not None:
            scheduler.step()

        if val_loss is not None:
            if save_best:
                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), path_to_save)

            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss {val_loss:.4f}')
            history.append((train_loss, val_loss))

        else:
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}')
            history.append(train_loss)

    if save_best:
        model.load_state_dict(torch.load(path_to_save, map_location=device()))

    return history


def train_epoch(train_loader, model, criterion, optimizer):
    model.train()
    for data, y in train_loader:
        data = data.to(device())
        y = y.to(device())
        out = model(data)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def eval_epoch(loader, model, criterion):
    model.eval()
    losses = []
    with torch.no_grad():
        for data, y in loader:
            data = data.to(device())
            y = y.to(device())
            out = model(data)
            loss = criterion(out, y)
            losses.append(loss.item())
    return np.mean(losses)


def device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def cv(n_epochs, lr, dp, crossval_dataset, groups, labels, test_loader):
    skf = GroupKFold(n_splits=20)
    eval_metrics = np.zeros((skf.n_splits, 3))
    #labels = torch.Tensor([i[1] for i in crossval_dataset])

    for n_fold, (train_idx, test_idx) in tqdm(enumerate(skf.split(labels, groups=groups))):
        best_val_loss = 1000
        torch.cuda.empty_cache()
        model = GCN(full_dataset.num_features, channels=[128, 32], dropout=dp).to(device())
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.02)
        criterion = torch.nn.CrossEntropyLoss()
        #criterion = nn.BCELoss()
        scheduler = lr_scheduler.StepLR(optimizer, step_size=n_epochs//2, gamma=0.1, last_epoch=-1)

        train_loader_cv = DataLoader(crossval_dataset, batch_size=32, shuffle=False, sampler=SubsetRandomSampler(train_idx))
        test_loader_cv = DataLoader(crossval_dataset, batch_size=32, shuffle=False, sampler=SubsetRandomSampler(test_idx))
        min_v_loss = np.inf
        print('==========', n_fold, '==========')
        acc = []
        for epoch in range(n_epochs):
            train_epoch(train_loader_cv, model, criterion, optimizer)
            train_loss, train_acc, _, _, _ = eval_epoch(train_loader_cv, model, criterion)
            val_loss, test_acc, _, _, _ = eval_epoch(test_loader_cv, model, criterion)
            
            #
            #scheduler.step()
            if epoch % 5 == 0:
                print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss {val_loss:.4f},',
                      f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

            acc.append(test_acc)
            
            if min_v_loss > val_loss:
                min_v_loss = val_loss
                best_test_acc = test_acc
            
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), '../best_cv_loss.pt')

        model.load_state_dict(torch.load('../best_cv_loss.pt', map_location=device()))
        real_test_loss, real_test_acc, _, _, _ = eval_epoch(test_loader, model, criterion)

        eval_metrics[n_fold, 0] = real_test_acc
        eval_metrics[n_fold, 1] = np.mean(acc)
        eval_metrics[n_fold, 2] = np.std(acc)


    return eval_metrics



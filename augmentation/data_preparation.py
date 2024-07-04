import torch
import numpy as np
from tqdm import trange


def train_target_split(data, context, predict):
    """
    returns
    -------
    tuple of torch.tensors
    
    train part of a time series (context) and test part (predict)
    """
    l = context + predict
    tr, trg = [], []
    for i in range(data.shape[0]):
        start = 0
        while start + l < data[i].shape[0]:
            tr.append(data[i][start: start+context])
            trg.append(data[i][start+context: start+l])
            start += predict
    
    return torch.stack(tr), torch.stack(trg)


def transformer_data(data, context, predict):
    """
    returns
    -------
    list of dicts with transformer input
    """
    past_values, future_values = train_target_split(data, context, predict)
    #p_enc_1d_model = PositionalEncoding1D(100)
    #pos_enc = p_enc_1d_model(data)
    
    l = context + predict
    past_time_features, future_time_features = [], []

    for i in range(data.shape[0]):
        start = 0
        while start + l < data[i].shape[0]:
            past_time_features.append(range(start, start+context))
            future_time_features.append(range(start+context, start+l))
            start += predict

    past_time_features = torch.tensor(past_time_features).unsqueeze(2) #torch.stack(past_time_features) 
    future_time_features = torch.tensor(future_time_features).unsqueeze(2) #torch.stack(future_time_features)
    past_observed_mask = torch.ones(past_values.shape)

    dataset = []
    for i in range(data.shape[0]):
        dataset.append({
            "past_values": past_values[i],
            "future_values": future_values[i],
            "past_time_features": past_time_features[i],
            "future_time_features": future_time_features[i],
            "past_observed_mask": past_observed_mask[i],
        })
    
    return dataset

def transformer_data_test(data):
    """
    function to test generation of whole ts 


    returns
    -------
    list of dicts with transformer input
    """
    past_time_features, future_time_features = [], [], []

    past_time_features = torch.arange(0, 120).repeat(data.shape[0], 1).unsqueeze(2)
    future_time_features = torch.arange(0, 120).repeat(data.shape[0], 1).unsqueeze(2)
    past_observed_mask = torch.ones(data.shape)

    dataset = []
    for i in range(data.shape[0]):
        dataset.append({
            "past_values": data[i],
            "future_values": data[i],
            "past_time_features": past_time_features[i],
            "future_time_features": future_time_features[i],
            "past_observed_mask": past_observed_mask[i],
        })
    
    return dataset


def new_aug(real, aug, aug_per_sample=1):
    print(aug.shape)
    k = 0
    new = np.zeros_like(real)
    for sub in range(len(real)):
        for net in range(14):
            m = np.argmax([
                np.corrcoef(aug[k+i, :, net], real[sub, :, net])[0, 1] for i in range(aug_per_sample)])
            new[sub, :, net] = aug[k+m, :, net]
                    
        k += aug_per_sample

    return new


def generate_data(data, model, augments_per_sample=1, context=20, predict=4):
    """
    t
    """
    augmented_data = []
    model.eval()
    model.to('cpu')
    nsub, ts, _ = data.shape
    #i = 0
    with torch.no_grad():

        for i in trange(nsub):
            for _ in range(augments_per_sample):
                augmented = data[i][0:context]
                start = 0

                while augmented.shape[0] < ts + context:
                    
                    #past_values = augmented[-context:].unsqueeze(0)
                    past_values = data[i][start: start+context].unsqueeze(0)

                    if past_values.shape[1] < context:
                        past_values = augmented[-context:].unsqueeze(0)

                    past_time_features = torch.tensor(
                        range(augmented.shape[0] - context, augmented.shape[0])).reshape(1, context, 1)
                    
                    future_time_features = torch.tensor(
                        range(augmented.shape[0], augmented.shape[0] + predict)).reshape(1, predict, 1)
                    
                    past_observed_mask = torch.ones(past_values.shape)

                    gen = model.generate(
                        past_values=past_values,
                        past_time_features=past_time_features,
                        future_time_features=future_time_features,
                        past_observed_mask=past_observed_mask
                        )["sequences"].squeeze().unsqueeze(0)
                    
                    #print(gen.shape, augmented.shape)

                    augmented = torch.cat((augmented, gen), dim=0)
                    start += predict

                augmented_data.append(augmented[context:])

    return torch.stack(augmented_data)


def generate_data_(data, model, num_samples=10, context=20, predict=4, parallel_samples=1):
    """
    t
    """
    augmented_data = []
    #p_enc_1d_model = PositionalEncoding1D(100)
    #pos_enc = p_enc_1d_model(data)
    model.eval()
    model.to('cpu')
    nsub, ts = data.shape[0], data.shape[1]
    #i = 0
    with torch.no_grad():

        for i in trange(num_samples):
            augmented = data[i][0:context].unsqueeze(0).repeat(parallel_samples, 1, 1)
            start = 0
            while augmented.shape[1] <= ts + context:

                past_values = augmented[i][:-context].unsqueeze(0)
                #print(start)
                #print(past_values.shape)

                #past_time_features = pos_enc[i, start: start+context, :].unsqueeze(0)
                #print(past_time_features.shape)
                past_time_features = torch.tensor(range(augmented.shape[1] - context, augmented.shape[1])).reshape(1, context, 1)
                
                #future_time_features = pos_enc[i, start+context: start+context+predict, :].unsqueeze(0)
                #if future_time_features.shape[1] == 0:
                    #future_time_features = pos_enc[i, -1, :].unsqueeze(0).unsqueeze(1)
                #print(future_time_features.shape)
                future_time_features = torch.tensor(range(augmented.shape[1], augmented.shape[1] + predict)).reshape(1, predict, 1)
                
                past_observed_mask = torch.ones(past_values.shape)

                gen = model.generate(
                    past_values=past_values,
                    past_time_features=past_time_features,
                    future_time_features=future_time_features,
                    past_observed_mask=past_observed_mask,
                    #static_categorical_features=static_categorical_features
                    )["sequences"].squeeze(0)#.unsqueeze(0)
                
                #print(gen.shape, augmented.shape)
                ##print(gen)

                augmented = torch.cat((augmented, gen), dim=1)
                start += predict

            augmented_data.append(augmented[:, context:, :])

    return torch.stack(augmented_data).view(-1, ts, data.shape[2])
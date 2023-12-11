from captum.attr import Saliency, IntegratedGradients
from torch import exp
import torch
import numpy as np
from collections import defaultdict


def model_forward(edge_mask, data, model):
    batch = torch.zeros(data.x.shape[0], dtype=int).to('cpu')
    out = model(data.x, data.edge_index, edge_mask, batch)
    return out


def explain(method, data, model, target=0):
    data.to('cpu')
    model.to('cpu')
    input_mask = torch.ones(data.edge_index.shape[1]).requires_grad_(True).to('cpu')
    if method == 'ig':
        ig = IntegratedGradients(model_forward)
        mask = ig.attribute(input_mask, target=target,
                            additional_forward_args=(data, model),
                            internal_batch_size=data.edge_index.shape[1])
    elif method == 'saliency':
        saliency = Saliency(model_forward)
        mask = saliency.attribute(input_mask, target=target,
                                  additional_forward_args=(data, model))
    else:
        raise Exception('Unknown explanation method')

    edge_mask = np.abs(mask.cpu().detach().numpy())
    if edge_mask.max() > 0:  # avoid division by zero
        edge_mask = edge_mask / edge_mask.max()
    return mask.cpu().detach().numpy() #edge_mask


def aggregate_edge_directions(dataset, edge_mask, data):
    edge_mask_dict = defaultdict(float)
    for val, u, v in list(zip(edge_mask, *data.edge_index)):
        u, v = u.item(), v.item()
        if u > v:
            u, v = v, u
        edge_mask_dict[(u, v)] += val

    matr = np.zeros(dataset[0].x.shape)
    for k in edge_mask_dict:
        matr[k[0], k[1]] = edge_mask_dict[k]

    return matr, edge_mask_dict


def salient_edges(dataset, model, label):
    """
    label: label of interest, int

    returns np.array
    """
    close_labels = [i for i, en in enumerate(dataset) if en.y == label]
    sal_matr = np.zeros(dataset[0].x.shape)

    for i in close_labels:
        edge_mask = explain(method='saliency',
                            data=dataset[i],
                            model=model,
                            target=dataset[i].y)

        salient_edges, edge_dict = aggregate_edge_directions(dataset,
            edge_mask, dataset[i])
        sal_matr += salient_edges > 0

    return to_symmetrical(sal_matr / len(close_labels))

def to_symmetrical(matr):
    return np.where(matr, matr, matr.T)
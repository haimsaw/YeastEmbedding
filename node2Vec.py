import os.path as osp

import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import from_networkx

import numpy as np
import networkx as nx


def train(model, optimizer, loader, device):
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def test(model, data):
    model.eval()
    z = model()
    acc = model.test(z[data.train_mask], data.y[data.train_mask],
                     z[data.test_mask], data.y[data.test_mask],
                     max_iter=150)
    return acc


@torch.no_grad()
def plot_points(model, data, device):
    model.eval()
    z = model(torch.arange(data.num_nodes, device=device))
    z = TSNE(n_components=2).fit_transform(z.cpu().numpy())
    print(z.shape)

    plt.figure(figsize=(8, 8))
    plt.scatter(*z.T, s=20)
    # plt.axis('off')
    plt.show()


def get_data():
    '''
    dataset = 'Cora'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    dataset = Planetoid(path, dataset)
    data2 = dataset[0]
    '''
    G = nx.read_edgelist('data/huri_symbol.tsv')
    G.remove_edges_from(nx.selfloop_edges(G))
    G = nx.k_core(G, 3)  # todo hyperparam
    data = from_networkx(G)

    n_train = int(len(data) * .8)
    idx = torch.randperm(len(data))
    data.train_mask = idx[:n_train]
    data.test_mask = idx[n_train:]

    return data, G


def main():
    data, G = get_data()
    nx.draw(G)
    plt.show()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=20,
                     context_size=10, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=True).to(device)
    print(model)
    # todo we want more bfs - homophily
    # p = return parameter,  likelihood of immediately revisiting a node
    # q = in-out parameter, if q > 1, the random walk is biased towards nodes close to node t. (more bfs)

    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    for epoch in range(1):
        loss = train(model, optimizer, loader, device)
        acc = 0  # todo test()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')

    plot_points(model, data, device)


if __name__ == "__main__":
    main()

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import from_networkx

import clusterUtils
import graphUtils
import networkx as nx

from sklearn.cluster import AffinityPropagation


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(model, optimizer, loader):
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
def plot_points(model, data):
    model.eval()
    z = model(torch.arange(data.num_nodes, device=device))
    z = TSNE(n_components=2).fit_transform(z.cpu().numpy())
    print(z.shape)

    plt.figure(figsize=(8, 8))
    plt.scatter(*z.T, s=20)
    # plt.axis('off')
    plt.show()


@torch.no_grad()
def plot_points_with_cluster(embbeded, labels):
    z = TSNE(n_components=2).fit_transform(embbeded)
    print(z.shape)

    plt.figure(figsize=(8, 8))
    for i in range(max(labels)+1):
        plt.scatter(z[labels == i, 0], z[labels == i, 1], s=20)
    plt.show()


def get_data():
    '''
    dataset = 'Cora'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    dataset = Planetoid(path, dataset)
    data2 = dataset[0]
    '''
    G = nx.read_edgelist('db/huri_symbol.tsv')
    G.remove_edges_from(nx.selfloop_edges(G))
    G = nx.k_core(G, 3)  # todo hyperparam
    data = from_networkx(G)

    n_train = int(len(data) * .8)
    idx = torch.randperm(len(data))
    data.train_mask = idx[:n_train]
    data.test_mask = idx[n_train:]

    gaf_data = graphUtils.readGafFile("./db/goa_human.gaf")

    return data, G, gaf_data


def cluster(embbeded):
    af = AffinityPropagation(preference=-50).fit(embbeded)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    n_clusters_ = len(np.unique(labels))
    indecies = np.array(range(len(labels)))
    print('Estimated number of clusters: %d' % n_clusters_)

    clusters = []
    for i in range(max(labels)+1):
        clusters.append(indecies[labels == i])
    return clusters, labels


def main():
    data, G, gaf_data = get_data()
    nx.draw(G)
    plt.show()

    model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=20,
                     context_size=10, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=True).to(device)
    print(model)
    # todo we want more bfs - homophily
    # p = return parameter,  likelihood of immediately revisiting a node
    # q = in-out parameter, if q > 1, the random walk is biased towards nodes close to node t. (more bfs)

    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    for epoch in range(5):
        loss = train(model, optimizer, loader)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')


    model.eval()
    embbeded, = model(torch.arange(data.num_nodes, device=device)).cpu().detach().numpy()
    clusters, labels = cluster(embbeded)

    plot_points_with_cluster(embbeded, labels)
    num_of_annotations_in_g = clusterUtils.getNumOfAnnotationsInG(G, gaf_data)
    clusters_annotation, clusters_P_Val = clusterUtils.computeClustersFuncEnrichment(G, clusters, gaf_data, num_of_annotations_in_g)

    print(f'clusters_P_Val={clusters_P_Val}')

if __name__ == "__main__":
    main()

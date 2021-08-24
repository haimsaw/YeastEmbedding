import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import from_networkx

from clusterUtils import *
from graphUtils import *
import networkx as nx

from sklearn.cluster import AffinityPropagation, MiniBatchKMeans

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# region general

def get_data():
    G = nx.read_edgelist('db/huri_symbol.tsv')
    G.remove_edges_from(nx.selfloop_edges(G))
    G = nx.k_core(G, 3)  # todo hyperparam
    data = from_networkx(G)

    # n_train = int(len(data) * .8)
    # idx = torch.randperm(len(data))
    # data.train_mask = idx[:n_train]
    # data.test_mask = idx[n_train:]

    gaf_data = readGafFile("./db/goa_human.gaf")

    return data, G, gaf_data

# endregion

# region plotting

@torch.no_grad()
def plot_2d_embeddings(model, data):
    model.eval()
    z = model(torch.arange(data.num_nodes, device=device))
    z = TSNE(n_components=2).fit_transform(z.cpu().numpy())
    print(z.shape)

    plt.figure(figsize=(8, 8))
    plt.scatter(*z.T, s=20)
    # plt.axis('off')
    plt.show()


@torch.no_grad()
def plot_2d_embeddings_with_label(embedded, labels):
    z = TSNE(n_components=2).fit_transform(embedded)
    print(z.shape)

    plt.figure(figsize=(8, 8))
    for i in range(max(labels) + 1):
        plt.scatter(z[labels == i, 0], z[labels == i, 1], s=20)
    plt.show()


@torch.no_grad()
def plot_each_cluster(G, labels):
    for i in range(max(labels) + 1):
        nodes = np.array(G.nodes)[labels == i]
        H = G.subgraph(nodes)
        print(f'\n\ncluster={i}, n_nodes={len(nodes)}:')
        nx.draw(H, node_size=70)
        plt.show()


def show_exp_results(xs, ys, scores, xlabel, ylabel, title):
    fig, ax = plt.subplots(1, 1)

    ax.set_xticklabels([0] + list(xs))
    ax.set_xlabel(xlabel)

    ax.set_yticklabels([0] + list(ys))
    ax.set_ylabel(ylabel)

    img = ax.imshow(scores, cmap='cividis')
    fig.colorbar(img)
    fig.suptitle(title)
    plt.show()


# endregion


# region embedding


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


def embed(data, epochs, p, q, embedding_dim, walk_length, walks_per_node):
    model = Node2Vec(data.edge_index, embedding_dim=embedding_dim, walk_length=walk_length,
                     context_size=10, walks_per_node=walks_per_node,
                     num_negative_samples=1, p=p, q=q, sparse=True).to(device)
    # todo we want more bfs - homophily
    # p = return parameter,  likelihood of immediately revisiting a node
    # q = in-out parameter, if q > 1, the random walk is biased towards nodes close to node t. (more bfs)

    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    losses = []
    for epoch in range(epochs):
        loss = train(model, optimizer, loader)
        losses.append(loss)
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

    plt.bar(range(len(losses)), losses)
    plt.show()

    model.eval()
    return model(torch.arange(data.num_nodes, device=device)).cpu().detach().numpy(),\
           losses[-1] if epochs > 0 else 0


# endregion


# region clustering

def cluster_embeddings(G, embedded, gaf_data, clustering_alg, **kwargs):
    if clustering_alg == "affinity_propagation":
        # preference - controls how many exemplars are used
        # damping factor - damps the responsibility and availability messages (between 0.5 and 1)
        labels = AffinityPropagation(preference=kwargs["preference"], damping=kwargs["damping"]).fit(embedded).labels_

    elif clustering_alg == "k_means":
        labels = MiniBatchKMeans(n_clusters=kwargs["n_clusters"], batch_size=kwargs["batch_size"]).fit(embedded).labels_

    else:
        assert False

    indices = np.array(range(len(labels)))
    clusters = []
    for i in range(max(labels) + 1):
        clusters.append(indices[labels == i])

    n_clusters_ = len(np.unique(labels))
    print(f'Number of clusters: {n_clusters_}')

    # plot_each_cluster(G, affinity_propagation_labels)
    # plot_2d_embbedings_with_lable(embbeded, affinity_propagation_labels)
    return get_partition_score(G, clusters, gaf_data)


# endregion


# region hyperparams

def parse_hyperparams(hyperparams):
    embedding_hyperparams = {
        "epochs": 0,
        "p": hyperparams[0],
        "q": hyperparams[1],
        "embedding_dim": 128,
        "walk_length": 20,
        "walks_per_node": 10
    }

    clustering_hyperparams = {
        "clustering_alg": "affinity_propagation",
        "preference": None,
        "damping": 0.5
    }

    clustering_hyperparams = {
        "clustering_alg": "k_means",
        "n_clusters": 1,
        "batch_size": 100
    }
    return embedding_hyperparams, clustering_hyperparams


def run_test(G, data, gaf_data, embedding_hyperparams, clustering_hyperparams):
    embedded, embedding_loss = embed(data, **embedding_hyperparams)
    score = cluster_embeddings(G, embedded, gaf_data, **clustering_hyperparams)
    print(f'embedding_hyperparams={embedding_hyperparams} clustering_hyperparams={clustering_hyperparams} embedding_loss={embedding_loss:.4f} score={score:.4f}\n\n')

    return embedding_loss, score


# endregion


def main():
    data, G, gaf_data = get_data()
    # nx.draw(G)
    # plt.show()

    ps = np.linspace(0.1, 4, 5)
    qs = np.linspace(0.1, 4, 5)
    test_matrix = np.stack(np.meshgrid(ps, qs), axis=-1)

    res = np.apply_along_axis(
        lambda hyperparams: run_test(G, data, gaf_data, *parse_hyperparams(hyperparams)),
        -1, test_matrix)

    embedding_loss, scores = res[:, :, 0], res[:, :, 1]

    show_exp_results(ps, qs, scores, "p", "q", "scores by p, q")


if __name__ == "__main__":
    main()

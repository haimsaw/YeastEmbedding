import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import from_networkx
import os

from clusterUtils import *
from graphUtils import *
import networkx as nx

from functools import lru_cache
from sklearn.cluster import AffinityPropagation, MiniBatchKMeans, DBSCAN

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# region utils

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


def plot_2d_embeddings_with_label(embedded, labels):
    z = TSNE(n_components=2).fit_transform(embedded)
    print(z.shape)

    plt.figure(figsize=(8, 8))
    for i in range(max(labels) + 1):
        plt.scatter(z[labels == i, 0], z[labels == i, 1], s=20)
    plt.show()


def plot_each_cluster(G, labels):
    for i in range(max(labels) + 1):
        nodes = np.array(G.nodes)[labels == i]
        H = G.subgraph(nodes)
        print(f'\n\ncluster={i}, n_nodes={len(nodes)}:')
        nx.draw(H, node_size=70)
        plt.show()


def show_exp_results(labels, vals, title, scores):
    # todo 1d results
    if len(labels) == 1:
        show_1d_exp_results(labels, vals, title, scores)
    elif len(labels) == 2:
        show_2d_exp_results(labels, vals, title, scores)
    else:
        show_nd_exp_results(labels, vals, title, scores)


def show_1d_exp_results(labels, vals, title, scores):
    label = labels[0]
    vals = vals[0]
    fig, ax = plt.subplots(1, 1)

    ax.bar([f'{val:.4f}' for val in vals], scores)

    ax.set_xlabel(label)

    fig.suptitle(title)
    plt.show()


def show_nd_exp_results(labels, vals, title, scores):
    labels, last_label = labels[:-1], labels[-1]
    vals, last_vals = vals[:-1], vals[-1]
    for i, val in enumerate(last_vals):
        show_exp_results(labels, vals, f'{title} {last_label}={val}', scores[..., i])


def show_2d_exp_results(labels, vals, title, scores):
    xlabel, ylabel = labels[0], labels[1]
    xs, ys = vals[0], vals[1]

    fig, ax = plt.subplots(1, 1)
    img = ax.imshow(scores, cmap='cividis', extent=[-1, 1, -1, 1], origin='lower')

    ax.set_xticks(np.linspace(-1, 1, len(xs), endpoint=False) + 1 / (len(xs)))
    ax.set_xticklabels(map(lambda x: f'{x:.3f}', xs))
    ax.set_xlabel(xlabel)

    ax.set_yticks(np.linspace(-1, 1, len(ys), endpoint=False) + 1 / (len(ys)))
    ax.set_yticklabels(map(lambda y: f'{y:.3f}', ys))
    ax.set_ylabel(ylabel)

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


@lru_cache
def embed(data, epochs, p, q, embedding_dim, walk_length, walks_per_node, verbose=False):
    random.seed(316144)
    np.random.seed(316144)
    torch.manual_seed(316144)
    torch.cuda.manual_seed(316144)

    model = Node2Vec(data.edge_index, embedding_dim=embedding_dim, walk_length=walk_length,
                     context_size=10, walks_per_node=walks_per_node,
                     num_negative_samples=1, p=p, q=q, sparse=True).to(device)
    # we want more bfs - homophily
    # p = return parameter,  likelihood of immediately revisiting a node
    # q = in-out parameter, if q > 1, the random walk is biased towards nodes close to node t. (more bfs)

    loader = model.loader(batch_size=128, shuffle=True, num_workers=os.cpu_count())
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    losses = []
    for epoch in range(epochs):
        loss = train(model, optimizer, loader)
        losses.append(loss)
        if verbose:
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

    if verbose:
        plt.bar(range(len(losses)), losses)
        plt.show()

    with torch.no_grad():
        model.eval()
        embeddings = model(torch.arange(data.num_nodes, device=device)).cpu().detach().numpy()
        loss = losses[-1] if epochs > 0 else 0
        return embeddings, loss


# endregion


# region clustering

def cluster_embeddings(G, embedded, gaf_data, clustering_alg, **kwargs):
    with torch.no_grad():
        if clustering_alg == "affinity_propagation":
            # preference - controls how many exemplars are used
            # damping factor - damps the responsibility and availability messages (between 0.5 and 1)
            labels = AffinityPropagation(random_state=316144, preference=kwargs["preference"], damping=kwargs["damping"]).fit(embedded).labels_

        elif clustering_alg == "k_means":
            labels = MiniBatchKMeans(random_state=316144, n_clusters=kwargs["n_clusters"], batch_size=kwargs["batch_size"]).fit(embedded).labels_

        elif clustering_alg == "DBSCAN":
            labels = DBSCAN(min_samples=kwargs["min_samples"], eps=kwargs["eps"]).fit(embedded).labels_

        else:
            assert False

        indices = np.array(range(len(labels)))
        clusters = []
        for i in range(max(labels) + 1):
            clusters.append(list(indices[labels == i]))

        n_clusters_ = len(np.unique(labels))
        # print(f'Number of clusters: {n_clusters_}')

        # plot_each_cluster(G, affinity_propagation_labels)
        # plot_2d_embbedings_with_lable(embbeded, affinity_propagation_labels)
        return partition_score(G, clusters, gaf_data), clusters


# endregion


# region hyperparams


def run_cycle(G, data, gaf_data, embedding_hyperparams, clustering_hyperparams, verbose=False):
    if not verbose:
        print('.', end='')
    embedded, embedding_loss = embed(data, **embedding_hyperparams)

    score, clusters = cluster_embeddings(G, embedded, gaf_data, **clustering_hyperparams)
    if verbose:
        print(f'embedding_loss={embedding_loss:.4f} score={score:.4f} embedding_hyperparams={embedding_hyperparams} clustering_hyperparams={clustering_hyperparams}')

    return np.array([embedding_loss, score, clusters], dtype=object)


def parse_hyperparams(clustering_alg, epochs, embedding_dim, walk_length, walks_per_node, p, q, n_clusters=None, batch_size=None, eps=None, min_samples=None):
    embedding_hyperparams = {
        "epochs": epochs,
        "p": p,
        "q": q,
        "embedding_dim": int(embedding_dim),
        "walk_length": int(walk_length),
        "walks_per_node": int(walks_per_node)
    }

    if clustering_alg == "affinity_propagation":
        clustering_hyperparams = {
            "preference": None,
            "damping": 0.5
        }

    elif clustering_alg == "k_means":
        clustering_hyperparams = {
            "n_clusters": int(n_clusters),
            "batch_size": int(batch_size)
        }

    elif clustering_alg == "DBSCAN":
        clustering_hyperparams = {
            "eps": eps,
            "min_samples": int(min_samples)
        }

    else:
        return None

    clustering_hyperparams["clustering_alg"] = clustering_alg
    return embedding_hyperparams, clustering_hyperparams


def test_hp(G, data, gaf_data, const_embedding_hp, const_clustering_hp, verbose=False, **hp_to_test):
    hp_names_to_test, hp_values_to_test = list(hp_to_test.keys()), list(hp_to_test.values())
    n_tests = int(np.prod([len(vals) for vals in hp_values_to_test]))

    assert n_tests <= 50
    if not verbose:
        print('n_tests' + '.' * n_tests)
        print('Running', end="")

    if n_tests == 1:
        res = run_cycle(G, data, gaf_data, *parse_hyperparams(**const_embedding_hp, **const_clustering_hp), verbose)
        embedding_loss, score, partition = res[0], res[1], res[2]
        winning_hp = {}

    else:
        test_matrix = np.stack(np.meshgrid(*hp_values_to_test), axis=-1)
        res = np.apply_along_axis(
            lambda tested_hp: run_cycle(G, data, gaf_data, *parse_hyperparams(**dict(zip(hp_names_to_test, tested_hp)), **const_embedding_hp, **const_clustering_hp), verbose),
            -1, test_matrix)

        embedding_loss, scores, partitions = res[..., 0].astype(float), res[..., 1].astype(float), res[..., 2].astype(list)

        show_exp_results(hp_names_to_test, hp_values_to_test, "scores", scores)

        idx = np.unravel_index(np.argmax(scores), scores.shape)
        winning_hp = dict(zip(hp_names_to_test, test_matrix[idx]))
        score = scores[idx]
        partition = partitions[idx]

    print(f'\nmax score={score} winning_tested_hp={winning_hp}')

    # print(f'clusters={partitions[idx]}')
    print(f'const_embedding_hp={const_embedding_hp}, const_clustering_hp={const_clustering_hp}')

    return partition


# endregion


def main():
    data, G, gaf_data = get_data()
    # nx.draw(G)
    # plt.show()

    embedding_dims = [256, 1024, 2048]

    const_hp = {
        "epochs": 0,
        "p": 1.34,
        "q": 8,
        # "embedding_dim": 1024,
        "walk_length": 20,
        "walks_per_node": 10
    }
    const_clustering_hp = {
        "clustering_alg": "DBSCAN",
        "min_samples": 3,
        "eps": 3
    }
    clusters = test_hp(G, data, gaf_data, const_hp, const_clustering_hp, embedding_dim=embedding_dims)


if __name__ == "__main__":
    main()

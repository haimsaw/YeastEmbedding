import torch
from torch import nn
#from torchvision.transforms import ToTensor, Lambda, Compose
#import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
# import scipy.sparse as sp
import networkx as nx
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler


class NetworkDataset(Dataset):  # todo use IterableDataset?
    def __init__(self, file_name, transform=None):
        self.g = nx.read_edgelist(file_name)
        self.transform = transform
        self.n_nodes = len(self.g)
        self.nodes_list = np.array(list(self.g.nodes()))

    def __len__(self):
        return self.n_nodes**2

    def __getitem__(self, idx):
        i = int(idx / self.n_nodes)
        j = idx % self.n_nodes

        u = self.nodes_list[i]
        v = self.nodes_list[j]

        x1 = np.zeros(self.n_nodes)  # F.one_hot(torch.tensor([i], dtype=torch.long), num_classes=self.n_nodes)
        x1[i] = 1
        x2 = np.zeros(self.n_nodes)  # F.one_hot(torch.tensor([j], dtype=torch.long), num_classes=self.n_nodes)
        x2[j] = 1
        propagation = 5.1

        return x1, x2, propagation


class EmbeddingNetwork(nn.Module):
    def __init__(self, n_nodes):
        super(EmbeddingNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_nodes, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 100),
            nn.ReLU()
        )

    def forward_one(self, x):
        x = self.linear_relu_stack(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        return out1, out2


class L2DistLoss(torch.nn.Module):
    def __init__(self):
        super(L2DistLoss, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, p=2)
        res = self.mse(euclidean_distance, label)
        return res


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    for batch, (x1, x2, y) in enumerate(dataloader):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        # Compute prediction error
        embed1, embed2 = model(x1, x2)
        loss = loss_fn(embed1, embed2, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x1)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def main():
    dataset = NetworkDataset('data/HuRI.tsv')
    dataloader = DataLoader(dataset, batch_size=128, sampler=RandomSampler(BatchSampler(dataset, batch_size=64, drop_last=False),
                                                                           replacement=True, num_samples=6400000))

    for X1, X2, y in dataloader:
        print("Shape of X [N, C, H, W]: ", X1.shape)
        print("Shape of y: ", y.shape, y.dtype)
        break

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    model = EmbeddingNetwork(dataset.n_nodes).to(device)
    model.double()
    print(model)

    loss_fn = L2DistLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for t in range(50):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(dataloader, model, loss_fn, optimizer, device)
    print("Done!")


if __name__ == "__main__":
    main()


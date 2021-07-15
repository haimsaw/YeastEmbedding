import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
#from torchvision.transforms import ToTensor, Lambda, Compose
#import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
# import scipy.sparse as sp

import networkx as nx

class NetworkDataset(Dataset): # todo use IterableDataset?
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


        sample = {'names': (u, v), 'xs': (x1, x2), 'propagation': 5}

        if self.transform:
            sample = self.transform(sample)

        return sample


class EmbeddingNetwork(nn.Module):
    def __init__(self, n_nodes):
        super(EmbeddingNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_nodes, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 100),
            nn.ReLU()
        )
        self.out = nn.Linear(4096, 1)

    def forward_one(self, x):
        x = self.linear_relu_stack(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        return out

def main():
    dataset = NetworkDataset('HuRI.tsv')

    dataloader = DataLoader(dataset, batch_size=64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    model = EmbeddingNetwork().to(device)

if __name__ == "__main__":
    main()



'''

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 50

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
'''

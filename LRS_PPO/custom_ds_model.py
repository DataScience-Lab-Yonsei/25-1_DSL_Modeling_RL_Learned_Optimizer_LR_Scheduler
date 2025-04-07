import numpy as np

from sklearn.random_projection import GaussianRandomProjection

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms

class ProjectedMNIST(Dataset):
    def __init__(self, train=True):
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = torchvision.datasets.MNIST(root="./data", train=train, download=True, transform=transform)

        images = dataset.data.view(len(dataset), -1).float() / 255.0
        labels = dataset.targets

        projector = GaussianRandomProjection(n_components=48)
        projected_data = projector.fit_transform(images.numpy())

        projected_data = (projected_data - np.mean(projected_data, axis=0)) / np.std(projected_data, axis=0)

        self.data = torch.tensor(projected_data, dtype=torch.float32)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'image': self.data[idx], 
            'label': self.labels[idx],
        }

class MLP(nn.Module):
    def __init__(self, input_size=48, hidden_size=48, output_size=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # No softmax (CrossEntropyLoss handles it)
        return x

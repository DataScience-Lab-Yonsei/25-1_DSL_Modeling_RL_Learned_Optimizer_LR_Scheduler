import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from sklearn.random_projection import GaussianRandomProjection
from config import BATCH_SIZE, INPUT_DIM

def get_mnist_projected_dataloader(batch_size=BATCH_SIZE):
    transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: x.view(-1).numpy()
    ])
    dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    data = np.stack([d[0] for d in dataset])
    labels = np.array([d[1] for d in dataset])
    
    projector = GaussianRandomProjection(n_components=INPUT_DIM)
    projected_data = projector.fit_transform(data)
    projected_data = (projected_data - projected_data.mean(axis=0)) / projected_data.std(axis=0)  # Normalize to unit variance
    
    dataset = TensorDataset(torch.FloatTensor(projected_data), torch.LongTensor(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
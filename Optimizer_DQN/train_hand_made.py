import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import os
import json
from models import TwoLayerNN
from dataset import get_mnist_projected_dataloader
from config import *

def train_hand_made(optimizer_type="Adam"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoLayerNN().to(device)
    train_loader = get_mnist_projected_dataloader()
    track_data_path = "track_data_hand_made"
    os.makedirs(track_data_path, exist_ok=True)
    
    if optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=LR)
    elif optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    else:
        raise ValueError("Unsupported optimizer type")
    
    total_rewards_data = []
    
    for epoch in range(10):
        epoch_start_time = time.time()
        epoch_data = []
        total_reward = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            reward = -loss.item()
            total_reward += reward
            
            epoch_data.append({
                "iteration": batch_idx + 1,
                "reward": reward,
                "loss": loss.item()
            })
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/10, Loss: {loss.item():.4f}, Reward: {total_reward:.4f}, Time: {epoch_time:.2f}s")
        
        epoch_file = os.path.join(track_data_path, f"epoch_{epoch+1}.json")
        with open(epoch_file, "w") as f:
            json.dump(epoch_data, f, indent=4)
        
        total_rewards_data.append({
            "epoch": epoch+1,
            "total_reward": total_reward
        })
    
    total_rewards_file = os.path.join(track_data_path, "total_epoch.json")
    with open(total_rewards_file, "w") as f:
        json.dump(total_rewards_data, f, indent=4)

if __name__ == "__main__":
    train_hand_made("Adam")

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import random
import numpy as np
import matplotlib.pyplot as plt
import time

# ========================
# 3. CNN Simples com Flatten Dinâmico
# ========================
class SmallCNN(nn.Module):
    def __init__(self, n_filters, n_fc, dropout):
        super().__init__()
        self.n_fc = n_fc

        # Camada 1
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=n_filters, kernel_size=5, padding=2, stride=1)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.lrn1 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)

        # Camada 2
        self.conv2 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=5, padding=2, stride=1)
        self.bn2 = nn.BatchNorm2d(n_filters * 2)
        self.lrn2 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)

        #Camada 3
        self.conv3 = nn.Conv2d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(n_filters * 4)
        self.lrn3 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)

        self.flatten_dim = None
        self.fc1 = None
        self.fc2 = None

    def build(self, device):
        with torch.no_grad():
            x = torch.zeros(1, 3, 32, 32).to(device)
            x = torch.relu(self.conv1(x))
            x = self.pool(torch.relu(self.conv2(x)))
            x = self.dropout(x)
            self.flatten_dim = x.view(1, -1).shape[1]
        self.fc1 = nn.Linear(self.flatten_dim, self.n_fc).to(device)
        self.fc2 = nn.Linear(self.n_fc, 100).to(device)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        if self.fc1 is None or self.fc2 is None:
            raise ValueError("Chame o método build(device) após instanciar o modelo!")
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

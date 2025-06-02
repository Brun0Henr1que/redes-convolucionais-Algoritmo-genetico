import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================
# 3. CNN Simples com Flatten Dinâmico
# ========================
class SmallCNN(nn.Module):
    def __init__(self, n_filters, n_fc, dropout, device):
        super().__init__()
        self.n_fc = n_fc

        # Camada 1
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=n_filters, kernel_size=3, padding=1, stride=1, device=device)
        self.bn1 = nn.BatchNorm2d(n_filters, device=device)

        # Camada 2
        self.conv2 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=3, padding=1, stride=1, device=device)
        self.bn2 = nn.BatchNorm2d(n_filters * 2, device=device)

        #Camada 3
        self.conv3 = nn.Conv2d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=3, padding=1, stride=1, device=device)
        self.bn3 = nn.BatchNorm2d(n_filters * 4, device=device)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_rate = dropout

        self.flatten_dim = None
        self.fc1 = None
        self.fc2 = None

    def build(self, device):
        with torch.no_grad():
            x = torch.zeros(1, 3, 32, 32).to(device)

            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))

            # Global Average pooling
            x = F.adaptive_avg_pool2d(x, (1, 1))

            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            self.flatten_dim = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flatten_dim, self.n_fc).to(device)
        self.fc2 = nn.Linear(self.n_fc, 100).to(device)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Global Average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))

        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc2(x)
        return x
    
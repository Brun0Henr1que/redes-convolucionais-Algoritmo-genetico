import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================
# 3. CNN Simples com Flatten Dinâmico
# ========================

# Decidi refazer a arquietetura mudando para 4 blocos convolucionais, cada um com duas convoluções
# me inspirei em modelos como o VGG
class SmallCNN(nn.Module):
    def __init__(self, n_filters, n_fc, dropout, device):
        super().__init__()
        self.n_fc = n_fc
        self.dropout_rate = dropout

        # ===== Definição dos filtros de cada bloco =====
        f1 = n_filters          # bloco 1: canais de saída
        f2 = n_filters * 2      # bloco 2: dobra de f1
        f3 = n_filters * 4      # bloco 3: dobra de f2
        f4 = n_filters * 8      # bloco 4: dobra de f3

        # -------- Bloco 1 (64, 64) --------
        self.conv1_1 = nn.Conv2d(
            in_channels=3,
            out_channels=f1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            device=device
        )
        self.bn1_1 = nn.BatchNorm2d(f1, device=device)

        self.conv1_2 = nn.Conv2d(
            in_channels=f1,
            out_channels=f1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            device=device
        )
        self.bn1_2 = nn.BatchNorm2d(f1, device=device)

        # -------- Bloco 2 (64, 128) --------
        self.conv2_1 = nn.Conv2d(
            in_channels=f1,
            out_channels=f2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            device=device
        )
        self.bn2_1 = nn.BatchNorm2d(f2, device=device)

        self.conv2_2 = nn.Conv2d(
            in_channels=f2,
            out_channels=f2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            device=device
        )
        self.bn2_2 = nn.BatchNorm2d(f2, device=device)

        # -------- Bloco 3 (128, 256) --------
        self.conv3_1 = nn.Conv2d(
            in_channels=f2,
            out_channels=f3,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            device=device
        )
        self.bn3_1 = nn.BatchNorm2d(f3, device=device)

        self.conv3_2 = nn.Conv2d(
            in_channels=f3,
            out_channels=f3,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            device=device
        )
        self.bn3_2 = nn.BatchNorm2d(f3, device=device)

        # -------- Bloco 4 (256, 512) --------
        self.conv4_1 = nn.Conv2d(
            in_channels=f3,
            out_channels=f4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            device=device
        )
        self.bn4_1 = nn.BatchNorm2d(f4, device=device)

        self.conv4_2 = nn.Conv2d(
            in_channels=f4,
            out_channels=f4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            device=device
        )
        self.bn4_2 = nn.BatchNorm2d(f4, device=device)

        # Pooling e Dropout (mesm para todos os blocos)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten_dim = None
        self.fc1 = None
        self.fc2 = None

    def build(self, device):

        with torch.no_grad():
            x = torch.zeros(1, 3, 32, 32, device=device)

            # Bloco 1
            x = F.relu(self.bn1_1(self.conv1_1(x)))
            x = F.relu(self.bn1_2(self.conv1_2(x)))
            x = self.pool(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

            # Bloco 2
            x = F.relu(self.bn2_1(self.conv2_1(x)))
            x = F.relu(self.bn2_2(self.conv2_2(x)))
            x = self.pool(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

            # Bloco 3
            x = F.relu(self.bn3_1(self.conv3_1(x)))
            x = F.relu(self.bn3_2(self.conv3_2(x)))
            x = self.pool(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

            # Bloco 4
            x = F.relu(self.bn4_1(self.conv4_1(x)))
            x = F.relu(self.bn4_2(self.conv4_2(x)))
            x = self.pool(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

            # Global Average Pooling
            x = F.adaptive_avg_pool2d(x, (1, 1))

            # flatten
            x = x.view(1, -1)
            self.flatten_dim = x.size(1)

        #fc1 e fc2 realmente definidos
        self.fc1 = nn.Linear(self.flatten_dim, self.n_fc, device=device)
        self.fc2 = nn.Linear(self.n_fc, 10, device=device)

    def forward(self, x):
        # Bloco 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Bloco 2
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Bloco 3
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Bloco 4
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = self.pool(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))

        # Flatten e camadas fully-connected (n_fc)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc2(x)

        return x
    
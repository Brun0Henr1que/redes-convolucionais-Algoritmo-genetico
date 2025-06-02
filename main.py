# ========================
# 1. Espaço de Hiperparâmetros
# ========================
space = {
    "learning_rate": [1e-3, 5e-4, 1e-4],          # Taxas de aprendizado comumente eficazes
    "batch_size": [32, 64, 128, 256],             # Tamanhos de lote variados para testar desempenho
    "n_filters": [64, 128, 256, 512],             # Número de filtros em cada camada convolucional
    "n_fc": [128],                                # Tamanhos das camadas totalmente conectadas
    "dropout": [0.25, 0.3, 0.4, 0.5],             # Taxas de dropout para regularização
    "weight_decay": [1e-4, 5e-4, 1e-3],           # Decaimento de peso para regularização
}

# ========================
# 2. Dataset CIFAR-100
# ========================
def load_data():
    # Aplicando data augmentation
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])

    full_train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)

    train_idx, val_idx = train_test_split(
        list(range(len(full_train_dataset))),
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=full_train_dataset.targets
    )

    train_subset = Subset(
        datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train),
        train_idx[:30000] # 30000 imagens de treino
    )
    val_subset = Subset(
        full_train_dataset,
        val_idx[:10000] # 10000 imagens de teste
    )

    # Para mostrar imagens depois
    full_valset = Subset(train_subset, val_idx)
    return train_subset, val_subset, full_valset

trainset, valset, full_valset = load_data()

# ========================
# 4. Funções do AG e Avaliação
# ========================
def criar_individuo():
    return {
        "learning_rate": random.choice(space["learning_rate"]),
        "batch_size": random.choice(space["batch_size"]),
        "n_filters": random.choice(space["n_filters"]),
        "n_fc": random.choice(space["n_fc"]),
        "dropout": random.choice(space["dropout"]),
        "weight_decay": random.choice(space["weight_decay"]),
    }

def crossover(pai1, pai2):
    filho = {}
    for key in pai1:
        filho[key] = random.choice([pai1[key], pai2[key]])
    return filho

def avaliar_fitness(individuo, device):
    # Dataloaders
    batch_size = individuo["batch_size"]
    pin_mem = True if device.startswith('cuda') else False # copia os valores para a memória da GPU

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8, # 8 workers para carregar as 20000 imagens
        pin_memory=pin_mem,
        persistent_workers=True
    )

    val_loader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2, # 2 workers para as 2000 imagens
        pin_memory=pin_mem,
        persistent_workers=True
    )

    # Instanciando o modelo
    model = SmallCNN(
        n_filters=individuo["n_filters"],
        n_fc=individuo["n_fc"],
        dropout=individuo["dropout"],
        device=device
    ).to(device)

    model.build(device)

    criterion = nn.CrossEntropyLoss()

    # Ainda em dúvida se mantenho o ADAM ou substituo pelo SGD
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=individuo["learning_rate"],
        weight_decay=individuo["weight_decay"]
    )

    # Scheduler que reduz learning rate se a acurácia não melhorar
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3
    )

    # Parâmetros de treinamento
    num_epochs = 120                 # 120 épocas de treinamento para cada indivíduo
    val_interval = 20                # validar a cada 20 épocas
    melhor_acc = 0.0
    val_acc = 0.0
    patience = 4                     # paciência de 4 validações sem melhora
    patience_counter = 0
    stop_training = False

    # Treinamento
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        total_train = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)
            total_train += xb.size(0)

        avg_train_loss = epoch_loss / total_train
        print(f"Época {epoch:03d}/{num_epochs} — Loss Treino: {avg_train_loss:.4f}") # Computando o loss (função de perda) por época

        # Validação do early stopping
        if epoch % val_interval == 0:
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for xb_val, yb_val in val_loader:
                    xb_val = xb_val.to(device)
                    yb_val = yb_val.to(device)
                    logits_val = model(xb_val)
                    _, preds = torch.max(logits_val, dim=1)
                    correct += (preds == yb_val).sum().item()
                    total += yb_val.size(0)

            val_acc = correct / total
            print(f"    → Validação @ Época {epoch:03d}: Acc = {val_acc:.4f}")

            # Ajusta o scheduler com base na acurácia de validação
            scheduler.step(val_acc)

            # Early Stopping & Salvamento do Melhor Modelo
            if val_acc > melhor_acc:
                melhor_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), "melhor_modelo.pt")
                print(f"    >>> Novo melhor modelo salvo! Val Acc: {melhor_acc:.4f}")
            else:
                patience_counter += 1
                print(f"    (Sem melhora: patience_counter = {patience_counter}/{patience})")

                if patience_counter >= patience:
                    print(f"\nEarly stopping acionado na época {epoch}.")
                    stop_training = True

        if stop_training:
            break

    return val_acc

# ========================
# 5. AG Modular com Coleta de Estatísticas
# ========================

def tournament_selection(populacao, fitness, tournament_size=3):
    selecionados = []
    pop_size = len(populacao)
    for _ in range(pop_size):
        participantes_idx = random.sample(range(pop_size), tournament_size)
        melhor_idx = max(participantes_idx, key=lambda i: fitness[i])
        selecionados.append(populacao[melhor_idx])
    return selecionados

def mutar_multiponto(individuo, space, num_mutacoes=2):
    chaves = list(individuo.keys())
    genes_a_mutar = random.sample(chaves, k=num_mutacoes)
    for chave in genes_a_mutar:
        individuo[chave] = random.choice(space[chave])
    return individuo

def algoritmo_genetico(pop_size=4, geracoes=3, taxa_mutacao=0.3, device='cpu'):
    historico = []
    tempo_inicio = time.time()
    populacao = [criar_individuo() for _ in range(pop_size)]

    for g in range(geracoes):
        print(f"\n--- Geração {g+1}/{geracoes} ---")
        fitness = []
        for ind in populacao:
            print(f'Indivíduo: {ind}')
            start = time.time()
            acc = avaliar_fitness(ind, device)
            elapsed = time.time() - start
            fitness.append(acc)
            print(f"Acurácia: {acc:.4f} | Tempo: {elapsed:.1f}s")

        # Salva histórico dos 4 melhores da geração
        melhores = sorted(zip(populacao, fitness), key=lambda x: x[1], reverse=True)
        historico.append([fit for _, fit in melhores[:4]])
        for i, (ind, fit) in enumerate(melhores[:4]):
            print(f"{i+1}: Acc = {fit:.4f}, {ind}")

        # Seleção por torneio
        selecionados = tournament_selection(populacao, fitness, tournament_size=3)

        # Cruzamento e mutação multiponto
        nova_populacao = selecionados[:]
        while len(nova_populacao) < pop_size:
            pai1, pai2 = random.sample(selecionados, 2)
            filho = crossover(pai1, pai2)
            if random.random() < taxa_mutacao:
                filho = mutar_multiponto(filho, space, num_mutacoes=2)
            nova_populacao.append(filho)
        populacao = nova_populacao

    # Melhor resultado final (com predição para análise)
    final_fitness = []
    for ind in populacao:
        acc = avaliar_fitness(ind, device)
        final_fitness.append(acc)
    melhor_indice = final_fitness.index(max(final_fitness))
    melhor_ind = populacao[melhor_indice]
    acc, preds, labels = avaliar_fitness(melhor_ind, device, save_preds=True)
    tempo_total = time.time() - tempo_inicio

    return melhor_ind, acc, preds, labels, historico, tempo_total

# ========================
# 6. Funções de Relatório e Visualização
# ========================
def plot_accuracies(historico):
    plt.figure(figsize=(8, 5))
    for i in range(len(historico[0])):  # 4 melhores
        plt.plot([h[i] for h in historico], label=f"Indivíduo {i+1}")
    plt.xlabel("Geração")
    plt.ylabel("Acurácia dos melhores")
    plt.title("Acurácia dos 4 melhores indivíduos por geração")
    plt.legend()
    plt.grid()
    plt.show()

def show_stats(historico, tempo_total, melhor_ind, acc):
    print("\n========== RELATÓRIO FINAL ==========")
    print(f"Tempo total de execução: {tempo_total:.1f} segundos")
    print(f"Melhor indivíduo final: {melhor_ind}")
    print(f"Acurácia do melhor: {acc:.4f}")
    acuracias = [acc for hist in historico for acc in hist]
    print(f"Acurácia média (todas): {np.mean(acuracias):.4f}")
    print(f"Acurácia máxima (histórico): {np.max(acuracias):.4f}")
    print(f"Acurácia mínima (histórico): {np.min(acuracias):.4f}")

def plot_image_examples(full_valset, preds, labels, acertos=True, n=5):
    import matplotlib.pyplot as plt
    idxs = np.where((preds == labels) if acertos else (preds != labels))[0][:n]
    if len(idxs) == 0:
        print("Nenhum exemplo encontrado.")
        return
    plt.figure(figsize=(12, 3))
    for i, idx in enumerate(idxs):
        img, label = full_valset[idx]
        img = img.permute(1,2,0) * torch.tensor([0.2675, 0.2565, 0.2761]) + torch.tensor([0.5071, 0.4867, 0.4408])
        img = img.numpy().clip(0,1)
        plt.subplot(1, n, i+1)
        plt.imshow(img)
        plt.title(f"Pred:{preds[idx]}\nTrue:{labels[idx]}")
        plt.axis('off')
    plt.suptitle("Acertos" if acertos else "Erros")
    plt.show()

# ========================
# 7. Execução Modular e Relatório
# ========================
if __name__ == '__main__':

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset
    from sklearn.model_selection import train_test_split
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    from class_CNN import SmallCNN
    import warnings
    warnings.filterwarnings("ignore", message=".*does not have many workers.*")

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")
    melhor_ind, acc, preds, labels, historico, tempo_total = algoritmo_genetico(
        pop_size=15, geracoes=30, taxa_mutacao=0.3, device=device
    )
    show_stats(historico, tempo_total, melhor_ind, acc)
    plot_accuracies(historico)
    print("\n5 exemplos que o algoritmo ACERTOU:")
    plot_image_examples(full_valset, preds, labels, acertos=True, n=5)
    print("\n5 exemplos que o algoritmo ERROU:")
    plot_image_examples(full_valset, preds, labels, acertos=False, n=5)

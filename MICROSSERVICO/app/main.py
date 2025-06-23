import os
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


# função de log
async def log(msg, ws_send=None):
    if ws_send:
        await ws_send(msg)
    else:
        print(msg)

# ========================
# 2. Dataset CIFAR-100 
# ========================

trainset, valset, full_valset = 0, 0, 0

def load_data():

    # Transforms SEM data augmentation
    transform_original = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])

    # Aplicando data augmentation
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])

    original_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_original)

    # dataset com data augmentation
    full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

    train_idx, val_idx = train_test_split(
        list(range(len(full_train_dataset))),
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=full_train_dataset.targets
    )

    train_subset = Subset(
        full_train_dataset,
        train_idx[:30000] # 30000 imagens de treino
    )
    val_subset = Subset(
        original_dataset,
        val_idx[:10000] # 10000 imagens de teste
    )

    return train_subset, val_subset, original_dataset

trainset, valset, full_valset = load_data()

# ========================
# 4. Funções do AG e Avaliação
# ========================
def criar_individuo(space):
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

async def avaliar_fitness(individuo, device, ws_send=None, save_preds=False):
    
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
        patience=2
    )

    # Parâmetros de treinamento
    num_epochs = 2                 # 120 épocas de treinamento para cada indivíduo # mudar para 30 depois
    val_interval = 1                # validar a cada 20 épocas #voltar para 3
    melhor_acc = 0.0
    val_acc = 0.0
    patience = 2                     # paciência de 4 validações sem melhora
    patience_counter = 0
    stop_training = False

    true_val_indices = valset.indices if isinstance(valset, Subset) else list(range(len(valset)))

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
        await log(f"Época {epoch:03d}/{num_epochs} — Loss Treino: {avg_train_loss:.4f}", ws_send) # Computando o loss (função de perda) por época

        # Validação do early stopping
        if epoch % val_interval == 0:
            model.eval()
            correct = 0
            total = 0
            true_val_indices = valset.indices if isinstance(valset, Subset) else list(range(len(valset)))
            all_preds, all_labels, all_true_idxs = [], [], []

            with torch.no_grad():
                idx_pointer = 0  # Ponteiro para percorrer true_val_indices de forma segura
                for xb_val, yb_val in val_loader:
                    xb_val = xb_val.to(device)
                    yb_val = yb_val.to(device)
                    logits_val = model(xb_val)
                    _, preds = torch.max(logits_val, dim=1)
                    correct += (preds == yb_val).sum().item()

                    # all_preds.extend(preds.cpu().numpy())
                    # all_labels.extend(yb_val.cpu().numpy())
                    total += yb_val.size(0)

                    batch_size_actual = len(yb_val)

                    # Mapeia os índices reais desse batch
                    batch_true_idxs = true_val_indices[idx_pointer : idx_pointer + batch_size_actual]

                    # Acumula tudo
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(yb_val.cpu().numpy())
                    all_true_idxs.extend(batch_true_idxs)

                    idx_pointer += batch_size_actual

            val_acc = correct / total
            await log(f"    Validação @ Época {epoch:03d}: Acc = {val_acc:.4f}", ws_send)

            # Ajusta o scheduler com base na acurácia de validação
            scheduler.step(val_acc)

            # Early Stopping & Salvamento do Melhor Modelo
            if val_acc > melhor_acc:
                melhor_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), "melhor_modelo.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    await log(f"\nEarly stopping acionado na época {epoch}.", ws_send)
                    stop_training = True

        if stop_training:
            break

    if save_preds:
        return val_acc, np.array(all_preds), np.array(all_labels), np.array(all_true_idxs)

    return val_acc

# ========================
# 5. AG Modular com Coleta de Estatísticas
# ========================

def tournament_selection(populacao, fitness, tournament_size=2):
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

async def algoritmo_genetico(pop_size=2, geracoes=3, taxa_mutacao=0, device='cpu', space={}, ws_send=None):
    historico = []
    tempo_inicio = time.time()
    populacao = [criar_individuo(space) for _ in range(pop_size)]

    for g in range(geracoes):
        await log(f"\n--- Geração {g+1}/{geracoes} ---", ws_send)
        fitness = []
        for ind in populacao:
            await log(f'Indivíduo: {ind}', ws_send)
            start = time.time()
            acc = await avaliar_fitness(ind, device, ws_send, save_preds=False)
            elapsed = time.time() - start
            fitness.append(acc)
            await log(f"Acurácia: {acc:.4f} | Tempo: {elapsed:.1f}s", ws_send)

        # Salva histórico dos 4 melhores da geração
        melhores = sorted(zip(populacao, fitness), key=lambda x: x[1], reverse=True)
        historico.append([fit for _, fit in melhores[:4]])
        for i, (ind, fit) in enumerate(melhores[:4]):
            await log(f"{i+1}: Acc = {fit:.4f}, {ind}", ws_send)

        # Seleção por torneio
        selecionados = tournament_selection(populacao, fitness, tournament_size=2)

        # Cruzamento e mutação multiponto
        nova_populacao = selecionados[:]
        while len(nova_populacao) < pop_size:
            pai1, pai2 = random.sample(selecionados, 2)
            filho = crossover(pai1, pai2)
            if random.random() < taxa_mutacao:
                filho = mutar_multiponto(filho, space, num_mutacoes=3)
            nova_populacao.append(filho)
        populacao = nova_populacao

    # Escolher o melhor indivíduo da última geração
    melhor_indice = fitness.index(max(fitness))
    melhor_ind = populacao[melhor_indice]

    # Reavaliar o melhor apenas para obter preds e labels
    await log(f"-====-Fazendo a avaliação final-====-", ws_send)
    acc, preds, labels, true_idx = await avaliar_fitness(melhor_ind, device, ws_send, save_preds=True)
    tempo_total = time.time() - tempo_inicio

    # Média das acurácias
    acuracias = [acc for hist in historico for acc in hist]
    media_acuracias = np.mean(acuracias)

    salvar_image_examples(full_valset, preds, labels, true_idx, acertos=True, n=5)
    salvar_image_examples(full_valset, preds, labels, true_idx, acertos=False, n=5)

    return melhor_ind, acc, historico, tempo_total, media_acuracias

# ========================
# 6. Funções de Relatório e Visualização
# ========================

def salvar_image_examples(full_valset, preds, labels, true_idxs, acertos=True, n=5, output_dir="templates/assets/img"):

    preds = np.atleast_1d(preds)
    labels = np.atleast_1d(labels)
    true_idxs = np.atleast_1d(true_idxs)

    # Define os índices de acerto ou erro
    if acertos:
        idxs = np.where(preds == labels)[0]
        prefix = "acerto"
    else:
        idxs = np.where(preds != labels)[0]
        prefix = "erro"

    if len(idxs) == 0:
        print(f"Nenhum exemplo {'acertado' if acertos else 'errado'} encontrado.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for i, idx in enumerate(idxs[:n]):
        real_idx = true_idxs[idx]
        img, label = full_valset[real_idx]

        # Desnormalizando a imagem (ajuste os valores se seus meios/std forem diferentes)
        img = img.permute(1, 2, 0) * torch.tensor([0.5071, 0.4865, 0.4409]) + torch.tensor([0.2673, 0.2564, 0.2762])
        img = img.numpy().clip(0, 1)

        plt.figure(figsize=(2, 2))
        plt.imshow(img)
        plt.title(f"Pred: {preds[idx]} | True: {labels[idx]}")
        plt.axis('off')

        file_path = os.path.join(output_dir, f"{prefix}_{i+1}.png")
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()

        print(f"Salvo: {file_path}")




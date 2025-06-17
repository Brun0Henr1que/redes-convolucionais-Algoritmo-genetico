import torch
from pathlib import Path
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from class_CNN import SmallCNN
from main import algoritmo_genetico, load_data, plot_accuracies, plot_image_examples, show_stats

# ========================
# 1. Espaço de Hiperparâmetros
# ========================
space={
        "learning_rate": [1e-3, 5e-4, 1e-4],          # Taxas de aprendizado comumente eficazes
        "batch_size": [32, 64, 128],                  # Tamanhos de lote variados para testar desempenho
        "n_filters": [64, 128, 256],                  # Número de filtros em cada camada convolucional
        "n_fc": [128],                                # Tamanhos das camadas totalmente conectadas
        "dropout": [0.25, 0.3, 0.4, 0.5],             # Taxas de dropout para regularização
        "weight_decay": [1e-4, 5e-4, 1e-3],           # Decaimento de peso para regularização
}

# Criação da aplicação FastAPI
app = FastAPI()

# Diretório base absoluto do arquivo main.py
BASE_DIR = Path(__file__).resolve().parent

# Caminho para a pasta templates dentro de app
templates_path = BASE_DIR / "templates"
templates = Jinja2Templates(directory=str(templates_path))

# Montar a pasta 'assets' para servir arquivos estáticos como CSS, JS, imagens, etc.
assets_path = BASE_DIR / "templates/assets"
app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")

def str_para_lista(s):
    return [float(x.strip()) if '.' in x or 'e' in x.lower() else int(x.strip()) for x in s.split(',') if x.strip()]

# Rota Get que exibe o formulário HTML
@app.get("/", response_class=HTMLResponse)
async def exibir_formulario(request: Request):
    return templates.TemplateResponse('Home.html', {"request": request})

@app.post("/parametros", response_class=HTMLResponse)
async def processar_parametros(request: Request,    
                               popSize: int = Form(...),
                               generations: int = Form(...),
                               mutationRate: float = Form(...),
                               nfc: str = Form(...),
                               learningRate: str = Form(...),
                               batchSize: str = Form(...),
                               filters: str = Form(...),
                               dropout: str = Form(...),
                               weightDecay: str = Form(...)
                               ):
    
    # tratando os dados 
    learningRate_list = str_para_lista(learningRate)
    batchSize_list = str_para_lista(batchSize)
    filters_list = str_para_lista(filters)
    nfc_list = str_para_lista(nfc)
    dropout_list = str_para_lista(dropout)
    weightDecay_list = str_para_lista(weightDecay)

    #concatenando as listas:
    space["learning_rate"] += learningRate_list
    space["batch_size"] += batchSize_list
    space["n_filters"] += filters_list
    space["n_fc"] += nfc_list
    space["dropout"] += dropout_list
    space["weight_decay"] += weightDecay_list

    # remover duplicatas:
    for key in space:
        space[key] = list(set(space[key]))

    # Aqui você pode fazer o que precisar com os dados recebidos
    mensagem = f"- Populacao: {popSize}\n"
    mensagem += f"- Geracões: {generations}\n"
    mensagem += f"- Taxa mutacao: {mutationRate}\n"
    mensagem += f"- nfc: {nfc}\n"

    device = 'cuda' if torch.cuda. is_available() else 'cpu'
    trainset, valset, full_valset = load_data()

    melhor_ind, acc, preds, labels, historico, tempo_total = algoritmo_genetico(pop_size=popSize, geracoes=generations, taxa_mutacao=mutationRate,device=device, space=space)

    show_stats(historico, tempo_total, melhor_ind, acc)
    # plot_accuracies(historico)
    
    # print("\n5 exemplos que o algoritmo ACERTOU:")
    # plot_image_examples(full_valset, preds, labels, acertos=True, n=5)
    # print("\n5 exemplos que o algoritmo ERROU:")
    # plot_image_examples(full_valset, preds, labels, acertos=False, n=5)

    mensagem += f"- Melhor Individuo: {melhor_ind}\n"
    mensagem += f"- Melhor Acuracia: {acc}\n"
        
    # Exemplo: renderizar uma página de resposta mostrando os dados
    return templates.TemplateResponse("dna.html", {"request": request, "mensagem": mensagem})

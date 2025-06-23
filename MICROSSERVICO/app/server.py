import os
import torch
from pathlib import Path
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import WebSocket
from class_CNN import SmallCNN
from main import algoritmo_genetico

resultado_ag = None

# ========================
# 1. Espaço de Hiperparâmetros
# ========================
# space={
#         "learning_rate": [1e-3, 5e-4, 1e-4],          # Taxas de aprendizado comumente eficazes
#         "batch_size": [4 , 8 , 16, 32, 64, 128],      # Tamanhos de lote variados para testar desempenho
#         "n_filters": [4, 8, 16, 32, 64, 128],         # Número de filtros em cada camada convolucional
#         "n_fc": [8, 16, 32, 64, 128, 256],            # Tamanhos das camadas totalmente conectadas
#         "dropout": [0.25, 0.3, 0.4, 0.5],             # Taxas de dropout para regularização
#         "weight_decay": [1e-4, 5e-4, 1e-3],           # Decaimento de peso para regularização
# }

space={
        "learning_rate": [1e-3],          # Taxas de aprendizado comumente eficazes
        "batch_size": [4],      # Tamanhos de lote variados para testar desempenho
        "n_filters": [4],         # Número de filtros em cada camada convolucional
        "n_fc": [8],            # Tamanhos das camadas totalmente conectadas
        "dropout": [0.25],             # Taxas de dropout para regularização
        "weight_decay": [1e-4],           # Decaimento de peso para regularização
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

# Servir as Imagens
app.mount("/assets", StaticFiles(directory="templates/assets"), name="assets")

# Rota Get que exibe o formulário HTML
@app.get("/", response_class=HTMLResponse)
async def exibir_formulario(request: Request):
    return templates.TemplateResponse('Home.html', {"request": request})

# Rota para receber os parametros e tratá-los
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

    # concatenando as listas:
    space["learning_rate"] += learningRate_list
    space["batch_size"] += batchSize_list
    space["n_filters"] += filters_list
    space["n_fc"] += nfc_list
    space["dropout"] += dropout_list
    space["weight_decay"] += weightDecay_list

    # remover duplicatas:
    for key in space:
        space[key] = list(set(space[key]))

    global space_final
    global pop__size
    global generations__
    global mutation__rate 

    space_final = space
    pop__size = popSize
    generations__ = generations
    mutation__rate = mutationRate

    mensagem = "Dados enviados, aguarde o processamento do seu AG (isso pode demorar um pouco)."
    return templates.TemplateResponse("dna.html", {"request": request, "mensagem": mensagem})

# Rota para iniciar um websocket + AG + CNN (Aqui que a mágica acontece)
@app.websocket("/ws/status")
async def websocket_status(websocket: WebSocket):
    await websocket.accept()

    async def ws_send(msg):
        await websocket.send_text(msg)

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(device)

        melhor_ind, acc, historico, tempo_total, media_acuracias = await algoritmo_genetico(
            pop_size=pop__size,
            geracoes=generations__,
            taxa_mutacao=mutation__rate,
            device=device,
            space=space_final,
            ws_send=ws_send
        )

        global resultado_ag 

        resultado_ag = {
            "melhor_ind": melhor_ind,
            "acc": acc,
            "tempo_total": tempo_total,
            "historico": historico,
            "num_geracoes": generations__,
            "pop_size": pop__size,
            "mutation_rate": mutation__rate,
            "media": media_acuracias
        }

        print(resultado_ag)

        await websocket.send_text("AG finalizado com sucesso!")
    except Exception as e:
        await websocket.send_text(f"Erro: {str(e)}")
    finally:
        await websocket.close()

@app.get("/resultados", response_class=HTMLResponse)
async def mostrar_resultados(request: Request):
    global resultado_ag

    if not resultado_ag:
        mensagem = "Nenhum resultado disponível ainda."
        return templates.TemplateResponse("resultados.html", {"request": request, "mensagem": mensagem})
    
    pasta_imagens = Path("templates/assets/img")
    acertos = sorted([f for f in os.listdir(pasta_imagens) if f.startswith("acerto")])
    erros = sorted([f for f in os.listdir(pasta_imagens) if f.startswith("erro")])

    return templates.TemplateResponse("resultados.html", {
        "request": request,
        "melhor_ind": resultado_ag["melhor_ind"],
        "acc": resultado_ag["acc"],
        "tempo_total": resultado_ag["tempo_total"],
        "media": resultado_ag["media"],
        "acertos": acertos,
        "erros": erros
    })
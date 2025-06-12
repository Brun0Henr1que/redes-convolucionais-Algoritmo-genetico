import torch
from pathlib import Path
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

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

# Rota Get que exibe o formulário HTML
@app.get("/", response_class=HTMLResponse)
async def exibir_formulario(request: Request):
    return templates.TemplateResponse('Home.html', {"request": request})

@app.post("/parametros", response_class=HTMLResponse)
async def processar_parametros(request: Request, nfc: str = Form(...), 
                               learningRate: str = Form(...),
                               batchSize: str = Form(...),
                               filters: str = Form(...),
                               popSize: str = Form(...),
                               generations: str = Form(...),
                               mutationRate: str = Form(...),
                               dropout: str = Form(...),
                               weightDecay: str = Form(...),):
    # Aqui você pode fazer o que precisar com os dados recebidos
    dados_recebidos = {"nfc": nfc, 
                       "learningRate": learningRate, 
                       "batchSize": batchSize,
                       "filters": filters,
                       "popSize": popSize,
                       "generations": generations,
                       "mutationRate": mutationRate,
                       "dropout": dropout,
                       "weightDecay": weightDecay
                      }
    
    # Exemplo: renderizar uma página de resposta mostrando os dados
    return templates.TemplateResponse("dna.html", {"request": request, "dados": dados_recebidos})

# Usa uma imagem base do Python
FROM python:3.12-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia os arquivos do projeto para o container
COPY ./app /app

# Instala as dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expõe a porta que o FastAPI usará
EXPOSE 8000

# Comando para rodar o servidor (ajuste se usar outro arquivo que não seja main.py)
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

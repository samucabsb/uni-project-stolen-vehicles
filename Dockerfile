FROM python:3.11-slim

WORKDIR /app

# Bibliotecas do sistema necessárias para OpenCV (libGL) e ORT (libgomp/AVX)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Instala PyTorch CPU-only antes do ultralytics (evita baixar a versão CUDA ~2 GB)
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY src/ ./src/
COPY tools/ ./tools/

# Diretórios montados via volume em tempo de execução
RUN mkdir -p models data/input data/output

CMD ["python", "main.py", "--help"]

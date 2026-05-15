# Sistema de Detecção de Placas de Veículos Roubados (Paralelismo em CPU)

# Descrição
# Este projeto implementa um sistema de detecção e reconhecimento de placas de veículos utilizando visão computacional,
# combinado com processamento paralelo em CPU.
# A aplicação recebe imagens, detecta a placa do veículo, extrai o texto via OCR e compara com uma base interna de placas roubadas.

# Arquitetura do Sistema
# Pipeline principal:
# Imagem → Detecção (YOLO) → OCR → Normalização → Comparação → Resultado

# Etapas do Processo

# 1. Entrada de Dados
# - Imagens carregadas de um diretório (/dataset)
# - Lista de placas roubadas definida manualmente no sistema

# 2. Processamento da Imagem
# Para cada imagem, o sistema executa:

# 🔹 Detecção de Placa
# Utiliza YOLO para localizar a placa na imagem

# 🔹 Recorte
# Extrai apenas a região da placa

# 🔹 OCR (Reconhecimento de Texto)
# Converte a imagem da placa em texto

# 🔹 Normalização
# Padroniza o texto (remove símbolos, converte para maiúsculo, corrige inconsistências)

# 🔹 Comparação
# Verifica se a placa está presente na base de placas roubadas

# 3. Saída
# O sistema retorna:
# Imagem X → ROUBADO
# Imagem Y → OK

# Paralelismo Implementado
# O projeto utiliza paralelismo de dados (Data Parallelism)

# Características:
# - Cada imagem é processada de forma independente
# - Distribuição entre múltiplos núcleos da CPU
# - Implementado com multiprocessing em Python

# Exemplo de divisão:
# CPU Core 1 → Imagem 1
# CPU Core 2 → Imagem 2
# CPU Core 3 → Imagem 3

# Tecnologias Utilizadas
# - Python
# - YOLOv8 (Ultralytics)
# - EasyOCR
# - Multiprocessing
# - SQLite (opcional)

# Execução Paralela (exemplo)
# from multiprocessing import Pool
# import os
# with Pool(os.cpu_count()) as p:
#     resultados = p.map(processar_imagem, lista_de_imagens)

# Avaliação de Desempenho
# O sistema pode ser avaliado através de:
# - Tempo de execução sequencial vs paralelo
# - Ganho de desempenho (speedup)
# - Uso de CPU

# Conceitos Envolvidos
# - Paralelismo de dados
# - Multiprocessing
# - Visão computacional
# - OCR
# - Processamento concorrente

# Considerações
# - Projeto adaptado para execução em CPU (sem GPU)
# - Uso de dataset controlado
# - Foco no aprendizado de paralelismo

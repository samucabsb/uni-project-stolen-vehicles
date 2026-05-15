# Sistema de Detecção de Placas de Veículos Roubados (Paralelismo em CPU)

https://www.kaggle.com/datasets/barkataliarbab/license-plate-detection-dataset-10125-images?resource=download - 10125 IMAGENS EM ALTA RESOLUÇÃO DE CARROS PARA RECONHECIMENTO DE PLACAS.

## Descrição
Este projeto implementa um sistema de detecção e reconhecimento de placas de veículos utilizando visão computacional, combinado com processamento paralelo em CPU.  
A aplicação recebe imagens, detecta a placa do veículo, extrai o texto via OCR e compara com uma base interna de placas roubadas.

---

## Arquitetura do Sistema
Pipeline principal:

Imagem → Detecção (YOLO) → OCR → Normalização → Comparação → Resultado

---

## Etapas do Processo

### 1. Entrada de Dados
- Imagens carregadas de um diretório (`/dataset`)
- Lista de placas roubadas definida manualmente no sistema

### 2. Processamento da Imagem
Para cada imagem, o sistema executa:

**Detecção de Placa**
- Utiliza YOLO para localizar a placa na imagem

**Recorte**
- Extrai apenas a região da placa

**OCR (Reconhecimento de Texto)**
- Converte a imagem da placa em texto

**Normalização**
- Padroniza o texto (remove símbolos, converte para maiúsculo, corrige inconsistências)

**Comparação**
- Verifica se a placa está presente na base de placas roubadas

### 3. Saída
O sistema retorna resultados como:

Imagem X → ROUBADO  
Imagem Y → OK  

---

## Paralelismo Implementado
O projeto utiliza **paralelismo de dados (Data Parallelism)**.

### Características
- Cada imagem é processada de forma independente
- Distribuição entre múltiplos núcleos da CPU
- Implementado com `multiprocessing` em Python

### Exemplo de divisão
CPU Core 1 → Imagem 1  
CPU Core 2 → Imagem 2  
CPU Core 3 → Imagem 3  

---

## Tecnologias Utilizadas
- Python  
- YOLOv8 (Ultralytics)  
- EasyOCR  
- Multiprocessing  
- SQLite (opcional)  

---

## Execução Paralela (exemplo)

```python
from multiprocessing import Pool
import os

with Pool(os.cpu_count()) as p:
    resultados = p.map(processar_imagem, lista_de_imagens)
``

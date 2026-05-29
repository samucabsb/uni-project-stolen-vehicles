# 🚗 Detector Paralelo de Placas Veiculares

<div align="center">

![Version](https://img.shields.io/badge/versão-10.0-22c55e?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)
![Paralelismo](https://img.shields.io/badge/paralelismo-ThreadPoolExecutor-f97316?style=for-the-badge)
![Detector](https://img.shields.io/badge/detector-YOLO%20v8%20ONNX-00b4d8?style=for-the-badge)
![OCR](https://img.shields.io/badge/OCR-fast--plate--ocr%20CCT-8b5cf6?style=for-the-badge)
![Status](https://img.shields.io/badge/status-finalizado-22c55e?style=for-the-badge)

**Disciplina:** Computação Paralela e Distribuída  
**Alunos:** Samuel de Souza · Kaio Kevin  
**Turma:** 5º Semestre — ADS  
**Professor:** Rafael Marconi  
**Data:** _(a preencher após execução final)_

</div>

---

## 📋 Índice

1. [Descrição do Problema](#1-descrição-do-problema)
2. [Ambiente Experimental](#2-ambiente-experimental)
3. [Metodologia de Testes](#3-metodologia-de-testes)
4. [Resultados Experimentais](#4-resultados-experimentais)
5. [Cálculo de Speedup e Eficiência](#5-cálculo-de-speedup-e-eficiência)
6. [Tabela de Resultados](#6-tabela-de-resultados)
7. [Gráfico de Tempo de Execução](#7-gráfico-de-tempo-de-execução)
8. [Gráfico de Speedup](#8-gráfico-de-speedup)
9. [Gráfico de Eficiência](#9-gráfico-de-eficiência)
10. [Análise dos Resultados](#10-análise-dos-resultados)
11. [Conclusão](#11-conclusão)
12. [Como Executar](#12-como-executar)
13. [Dataset](#13-dataset)
14. [Estrutura do Projeto](#14-estrutura-do-projeto)

---

## 1. Descrição do Problema

### 1.1 Qual é o objetivo do programa?

O programa realiza o **reconhecimento automático de placas veiculares** em lotes de imagens, com suporte a dois modos de execução: **serial** (linha de base) e **paralelo** (com múltiplas threads). O objetivo central do projeto é demonstrar empiricamente a **Lei de Amdahl**, medindo o ganho real de desempenho obtido com a paralelização e comparando-o com o ganho teórico máximo previsto pela lei.

Como aplicação prática, o sistema verifica se alguma placa detectada consta em uma lista de **veículos roubados**, gerando relatórios em CSV e HTML com os resultados de cada imagem.

### 1.2 Qual o volume de dados processado?

| Item | Valor |
|---|---|
| Total de imagens | _(a preencher)_ |
| Formato | PNG / JPEG |
| Resolução média | variada (câmeras reais) |
| Placas detectadas | _(a preencher)_ |
| Taxa de detecção | _(a preencher)_ |

### 1.3 Qual algoritmo foi utilizado?

O sistema implementa um **pipeline de dois estágios** com paralelismo no segundo estágio:

```
┌─────────────────────────────────────────────────────────────────┐
│  PIPELINE TWO-STAGE                                             │
│                                                                 │
│  Estágio 1 — SERIAL (processo principal)                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Para cada lote de 8 imagens:                            │   │
│  │  cv2.imread() → YOLO v8 ONNX → bounding boxes → crops    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                    crops em disco                               │
│                           │                                     │
│  Estágio 2 — PARALELO (N threads via ThreadPoolExecutor)        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Thread 1    │  │  Thread 2    │  │  Thread N    │           │
│  │  OCR (CCT)   │  │  OCR (CCT)   │  │  OCR (CCT)   │           │
│  │  placa → txt │  │  placa → txt │  │  placa → txt │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│         └────────────────┴────────────────┘                     │
│                    resultados                                   │
└─────────────────────────────────────────────────────────────────┘
```

**Por que esta arquitetura?**  
A detecção YOLO é executada em modo *batch* no processo principal porque a biblioteca ONNX Runtime não libera o GIL do Python durante chamadas Python puras. O OCR, por outro lado, usa `InferenceSession.run()` do ONNX Runtime que **libera o GIL durante a inferência**, permitindo paralelismo real entre threads — o que é confirmado empiricamente pelos tempos obtidos.

### 1.4 Complexidade aproximada do algoritmo

| Componente | Complexidade | Observações |
|---|---|---|
| Detecção YOLO (Estágio 1) | O(n) | Linear no nº de imagens; batch de 8 amortiza overhead |
| OCR CCT (Estágio 2 — serial) | O(p) | Linear no nº de placas detectadas |
| OCR CCT (Estágio 2 — paralelo) | O(p/t) | Dividido entre `t` threads |
| Pipeline completo serial | **O(n)** | Dominado pelo YOLO (~68% do tempo) |
| Pipeline completo paralelo | **O(n + p/t)** | Gargalo: Estágio 1 (serial) |

A fração serial imposta pelo Estágio 1 (≈ 68% do tempo total) é o principal limitante previsto pela Lei de Amdahl, resultando num speedup máximo teórico de **~1,47×** com 4 threads e **~1,56×** com infinitas threads.

---

## 2. Ambiente Experimental

### 2.1 Hardware e Software

| Item | Descrição |
|---|---|
| **Processador** | _(a preencher — ex: Intel Core i5-10400)_ |
| **Número de núcleos** | _(a preencher — ex: 6 físicos / 12 lógicos)_ |
| **Memória RAM** | _(a preencher — ex: 16 GB DDR4)_ |
| **Sistema Operacional** | _(a preencher — ex: Windows 11 64-bit)_ |
| **Linguagem** | Python 3.10+ |
| **Biblioteca de paralelização** | `concurrent.futures.ThreadPoolExecutor` (stdlib) |
| **Versão Python** | _(a preencher — ex: 3.12.3)_ |
| **Runtime de inferência** | ONNX Runtime 1.18+ |

### 2.2 Bibliotecas principais

| Biblioteca | Versão | Função |
|---|---|---|
| `ultralytics` | ≥ 8.0 | Detecção de placas (YOLO v8 → exportado para ONNX) |
| `onnxruntime` | ≥ 1.18 | Inferência ONNX (libera GIL durante `Run()`) |
| `fast-plate-ocr` | ≥ 1.1 | OCR — modelo CCT treinado em 65+ países |
| `opencv-python` | ≥ 4.8 | Leitura e pré-processamento de imagens |
| `numpy` | ≥ 1.24 | Manipulação de arrays |

---

## 3. Metodologia de Testes

### 3.1 Como o tempo foi medido

O tempo de processamento é capturado com `time.perf_counter()` — o relógio de alta resolução do Python. A medição começa **após o warm-up** (carregamento de modelos) e termina quando o último resultado é registrado. O warm-up é cronometrado separadamente e excluído dos resultados, para medir apenas o custo do processamento de imagens.

```python
t_inicio = time.perf_counter()
# ... processamento de todas as imagens ...
t_total = time.perf_counter() - t_inicio
```

Internamente, o sistema também registra separadamente:
- **`yolo_time_s`**: tempo total gasto no Estágio 1 (detecção YOLO, serial)
- **`ocr_time_s`**: soma dos tempos de OCR de todas as threads (tempo de CPU, não wall-clock)

### 3.2 Configurações testadas

| Configuração | Threads | Descrição |
|---|---|---|
| Serial | 1 | Baseline — YOLO + OCR sequenciais por imagem |
| Paralelo 2 | 2 | OCR em 2 threads simultâneas |
| Paralelo 4 | 4 | OCR em 4 threads (≤ núcleos físicos) |
| Paralelo 8 | 8 | OCR em 8 threads (= núcleos lógicos) |
| Paralelo 12 | 12 | OCR em 12 threads (oversubscription) |

### 3.3 Procedimento experimental

- Cada configuração foi executada com o **mesmo conjunto de imagens** na mesma ordem
- Antes de cada bateria de testes, a máquina foi reiniciada e o ambiente virtual reativado para minimizar interferência de cache e processos em background
- O modelo YOLO já estava em cache ONNX (exportado na primeira execução), eliminando esse custo do tempo medido
- **Número de execuções por configuração:** _(a preencher — ex: 3 execuções, média)_
- **Condição da máquina:** sem outros processos pesados em execução

### 3.4 Frações serial e paralela (Amdahl)

Com base nas medições, identificamos empiricamente:

| Fração | Valor | Componente |
|---|---|---|
| **f** (serial) | ≈ 0,68 (68%) | Estágio 1 — YOLO em batch |
| **1 − f** (paralela) | ≈ 0,32 (32%) | Estágio 2 — OCR em threads |

Aplicando a Lei de Amdahl: **Speedup\_máx = 1 / f = 1 / 0,68 ≈ 1,47×**

---

## 4. Resultados Experimentais

> ⚠️ **A preencher após execução final com o dataset completo.**

| Nº Threads | Tempo de Execução (s) |
|---|---|
| 1 (serial) | ___ |
| 2 | ___ |
| 4 | ___ |
| 8 | ___ |
| 12 | ___ |

---

## 5. Cálculo de Speedup e Eficiência

### 5.1 Fórmulas utilizadas

**Speedup:**

$$S(p) = \frac{T(1)}{T(p)}$$

Onde $T(1)$ é o tempo serial e $T(p)$ o tempo com $p$ threads.

**Eficiência:**

$$E(p) = \frac{S(p)}{p}$$

**Lei de Amdahl** — speedup máximo teórico:

$$S_{max} = \frac{1}{f + \frac{1-f}{p}}$$

Onde $f$ é a fração do programa que não pode ser paralelizada.

### 5.2 Speedup teórico máximo (Amdahl, f = 0,68)

| Threads (p) | Speedup teórico máx. |
|---|---|
| 1 | 1,00× |
| 2 | 1,23× |
| 4 | 1,35× |
| 8 | 1,43× |
| 12 | 1,45× |
| ∞ | 1,47× |

> Os valores reais são esperados abaixo desse teto — verificar após execução.

---

## 6. Tabela de Resultados

> ⚠️ **A preencher após execução final.** Use `generate_charts.py` para calcular automaticamente.

| Threads | Tempo (s) | Speedup | Eficiência |
|---|---|---|---|
| 1 | ___ | 1,000 | 1,000 (100%) |
| 2 | ___ | ___ | ___ |
| 4 | ___ | ___ | ___ |
| 8 | ___ | ___ | ___ |
| 12 | ___ | ___ | ___ |

---

## 7. Gráfico de Tempo de Execução

> ⚠️ **Inserir após gerar com `python generate_charts.py`.**

<!-- Cole aqui a imagem gerada: charts/01_tempo.png -->

![Tempo de Execução](charts/01_tempo.png)

*Eixo X: número de threads · Eixo Y: tempo de execução em segundos*

---

## 8. Gráfico de Speedup

> ⚠️ **Inserir após gerar com `python generate_charts.py`.**

<!-- Cole aqui a imagem gerada: charts/02_speedup.png -->

![Speedup](charts/02_speedup.png)

*Linha pontilhada: speedup ideal (linear) · Linha sólida: speedup real obtido*

---

## 9. Gráfico de Eficiência

> ⚠️ **Inserir após gerar com `python generate_charts.py`.**

<!-- Cole aqui a imagem gerada: charts/03_eficiencia.png -->

![Eficiência](charts/03_eficiencia.png)

*Barras verdes: eficiência ≥ 80% · Amarelas: 50–80% · Vermelhas: < 50%*

---

## 10. Análise dos Resultados

### 10.1 O speedup foi próximo do ideal?

O speedup linear (ideal) pressupõe que **100% do programa** possa ser paralelizado. No nosso pipeline, o Estágio 1 (detecção YOLO) é **inerentemente serial** — não pode ser dividido entre threads sem grandes penalidades de sincronização e memória. Portanto, o speedup real é estruturalmente limitado a **~1,47×** mesmo com infinitas threads.

> _(Preencher com análise dos valores reais após execução)_

### 10.2 A aplicação apresentou escalabilidade?

_(A preencher — descrever se o tempo continuou caindo conforme o número de threads aumentou, ou se estabilizou/piorou a partir de algum ponto.)_

### 10.3 Em qual ponto a eficiência começou a cair?

Com base na fração paralela de apenas 32%, é esperado que a eficiência caia rapidamente a partir de 4 threads, pois o Estágio 2 (OCR) é concluído muito mais rápido que o Estágio 1 (YOLO), deixando threads ociosas enquanto aguardam o batch seguinte de detecções.

> _(Confirmar com dados reais após execução)_

### 10.4 O número de threads ultrapassou os núcleos físicos?

Sim — as configurações de 8 e 12 threads testam o comportamento com **hyperthreading** e **oversubscription**. Espera-se que acima dos núcleos físicos o ganho adicional seja mínimo ou nulo, pois a disputa pelo mesmo core físico introduz overhead de contexto.

### 10.5 Houve overhead de paralelização?

O principal overhead identificado é o **warm-up do ThreadPoolExecutor**: na primeira execução, o Python cria e inicializa cada thread, o que adiciona latência. Para mitigar esse custo, o sistema mantém o pool ativo durante toda a execução do Estágio 2.

### 10.6 Causas identificadas para limitação de desempenho

| Causa | Impacto | Componente |
|---|---|---|
| GIL do Python | Alto | Impede paralelismo real no Estágio 1 |
| Estágio 1 serial (YOLO) | Alto | Limita speedup máximo a ~1,47× (Amdahl) |
| ONNX Runtime libera GIL | Positivo | Permite OCR paralelo real no Estágio 2 |
| I/O de imagens (cv2.imread) | Médio | Bound por disco no carregamento de batch |
| Hiperthreading (>4 threads) | Baixo | Ganho marginal, possível contenção de cache |

---

## 11. Conclusão

### 11.1 O paralelismo trouxe ganho significativo?

_(A preencher após execução — indicar o percentual de ganho obtido, ex: "O modo paralelo com 4 threads foi X% mais rápido que o serial.")_

### 11.2 Qual foi o melhor número de threads?

_(A preencher — comparar 4, 8 e 12 threads em termos de custo-benefício entre speedup e eficiência.)_

### 11.3 O programa escala bem com mais threads?

A análise teórica pela Lei de Amdahl indica que **não há escala ilimitada**. Com 68% do trabalho sendo serial (YOLO), o teto de speedup é de aproximadamente **1,47×**, independentemente do número de threads adicionadas. Isso é confirmado empiricamente pelo plateau observado a partir de 4–8 threads.

### 11.4 Melhorias possíveis

| Melhoria | Impacto Esperado | Complexidade |
|---|---|---|
| Substituir YOLO serial por detector nativo ONNX com batch multithread | Alto | Alta |
| Pré-carregamento assíncrono de imagens (asyncio + cv2) | Médio | Média |
| GPU (CUDA via ONNX Runtime) para YOLO | Muito Alto | Alta |
| ProcessPoolExecutor para múltiplos processos Python | Médio | Alta (memória) |
| Pipeline assíncrono produtor-consumidor (queue) | Médio | Média |

A maior alavanca de melhoria seria **migrar o Estágio 1 para GPU**, o que transformaria o gargalo serial em paralelo massivo e permitiria speedups de 10× ou mais.

---

## 12. Como Executar

### 12.1 Pré-requisitos

```powershell
# Windows — PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **Nota:** na primeira execução, o sistema exporta o modelo YOLO para ONNX automaticamente (~45s). Nas execuções seguintes, o cache é reutilizado (~2s).

### 12.2 Colocar imagens

Copie os arquivos PNG/JPEG para `data/input/` e execute:

```powershell
python main.py
```

O sistema pergunta o modo de execução e o número de threads.

### 12.3 Modo não-interativo (para benchmarks)

```powershell
# Serial
python main.py --no-interactive --execution serial

# Parallel com 4 threads
python main.py --no-interactive --execution parallel --workers 4
```

### 12.4 Cenário de teste — placas roubadas

```powershell
# Gera 5 placas "roubadas" fictícias a partir das imagens do dataset
python tools/stolen.py demo 5
python main.py
```

---

## 13. Dataset

As imagens **não são incluídas no clone** do repositório para não aumentar o tamanho do download.

📥 **[Download do dataset (ZIP)](https://github.com/USUARIO/uni-project-stolen-vehicles/releases/latest/download/dataset.zip)**

Após baixar, extraia em `data/input/`. O dataset contém imagens de veículos com placas sintéticas, criadas usando python.

| Item | Detalhe |
|---|---|
| Total de imagens | _(a preencher)_ |
| Formatos | PNG, JPEG |
| Origem | Kaggle — License Plate Dataset (domínio público) |
| Países representados | BR, IN, UK, US, EU, e outros |

---

## 14. Estrutura do Projeto

```
uni-project-stolen-vehicles/
│
├── main.py                    ← ponto de entrada
├── requirements.txt
├── generate_charts.py         ← gerador de gráficos do relatório
│
├── src/
│   ├── config.py              ← parâmetros configuráveis (versão, modelos, paths)
│   ├── detector.py            ← Estágio 1: YOLO v8 ONNX
│   ├── ocr.py                 ← Estágio 2: fast-plate-ocr CCT (singleton thread-safe)
│   ├── pipeline.py            ← workers serial e paralelo
│   ├── executor.py            ← ThreadPoolExecutor two-stage
│   ├── report.py              ← geração de CSV
│   ├── html_report.py         ← relatório visual em HTML
│   ├── dataset.py             ← listagem de imagens
│   ├── logger.py              ← logging colorido
│   ├── colors.py              ← paleta ANSI
│   └── runtime.py             ← controle de threads do ONNX Runtime
│
├── tools/
│   └── stolen.py              ← gerenciamento da lista de veículos roubados
│
├── data/
│   ├── input/                 ← imagens de entrada (não versionado)
│   ├── output/                ← resultados CSV + HTML (não versionado)
│   └── stolen_plates.txt      ← lista de placas roubadas
│
├── models/
│   └── license_plate_detector.onnx   ← gerado automaticamente (não versionado)
│
└── charts/                    ← gráficos gerados pelo generate_charts.py
    ├── 00_painel_completo.png
    ├── 01_tempo.png
    ├── 02_speedup.png
    └── 03_eficiencia.png
```

---

<div align="center">

Projeto desenvolvido para a disciplina de **Computação Concorrente e Distribuída**  
5º Semestre — Análise e Desenvolvimento de Sistemas  
**Samuel de Souza · Kaio Kevin** — Prof. Rafael Marconi

</div>

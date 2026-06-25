# Detector Paralelo de Placas Veiculares

![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)
![Paralelismo](https://img.shields.io/badge/Paralelismo-ProcessPoolExecutor-f97316?style=for-the-badge)
![Detector](https://img.shields.io/badge/Detector-YOLO%20v8%20ONNX-00b4d8?style=for-the-badge)
![OCR](https://img.shields.io/badge/OCR-fast--plate--ocr-8b5cf6?style=for-the-badge)
![Quantização](https://img.shields.io/badge/Quantização-INT8%20Estático-22c55e?style=for-the-badge)

**Disciplina:** Programação Concorrente e Distribuída
**Alunos:** Samuel de Souza Rodrigues · Kaio Kevin
**Turma:** 5º Semestre — ADS
**Professor:** Rafael Marconi
**Data:** 01/06/2026

---

## Dataset

As imagens utilizadas nos testes não estão armazenadas no repositório para evitar um clone pesado.

📥 [Baixar dataset](https://github.com/samucabsb/uni-project-stolen-vehicles/releases/download/dataset-v1.0/placas-20260601T192527Z-3-001.zip)

Após baixar, extraia o conteúdo para `data/input/`.

---

## Sumário

1. [Descrição do Problema](#1-descrição-do-problema)
2. [Arquitetura de Paralelismo](#2-arquitetura-de-paralelismo)
3. [Ambiente Experimental](#3-ambiente-experimental)
4. [Metodologia de Testes](#4-metodologia-de-testes)
5. [Resultados Experimentais](#5-resultados-experimentais)
6. [Análise dos Resultados](#6-análise-dos-resultados)
7. [Como Executar](#7-como-executar)
8. [Estrutura do Projeto](#8-estrutura-do-projeto)
9. [Conclusão](#9-conclusão)

---

## 1. Descrição do Problema

O projeto implementa um sistema de **reconhecimento automático de placas veiculares em lote**. O programa recebe um diretório de imagens de veículos, detecta a região da placa em cada imagem com **YOLO v8 exportado para ONNX**, recorta a placa detectada, lê o texto com **fast-plate-ocr (modelo CCT)** e compara o resultado com uma lista local de veículos roubados.

O objetivo central é demonstrar ganho de desempenho com paralelismo, comparando a execução serial (1 processo) com a execução paralela (N processos), medindo speedup e eficiência sob diferentes níveis de concorrência.

### 1.1 Pipeline de processamento

```
Entrada: diretório com imagens de veículos
   │
   ▼
Estágio 1 — Detecção YOLO/ONNX
   • Leitura da imagem (cv2.imread)
   • Inferência YOLO em mini-lotes internos
   • Seleção da detecção com maior confiança
   • Recorte da placa com expansão de bounding box (8%)
   │
   ▼
Estágio 2 — OCR com fast-plate-ocr
   • Inferência CCT (Compact Convolutional Transformer)
   • Normalização do texto (apenas A-Z e 0-9)
   • Filtros: comprimento mínimo, confiança mínima, blacklist
   • Comparação com lista de placas roubadas
   │
   ▼
Saída: CSV de resultados, log de desempenho, relatório HTML
```

### 1.2 Por que fast-plate-ocr e não um OCR genérico?

O fast-plate-ocr utiliza o modelo CCT (Compact Convolutional Transformer), treinado especificamente em placas de 65+ países. O Transformer processa a sequência completa: ao classificar o caractere na posição 2, o modelo já "viu" as posições 0 e 1 e sabe que aquela região provavelmente deve ser um dígito. Isso elimina a confusão L/D (I↔1, O↔0, S↔5) típica de OCRs genéricos que classificam cada posição de forma independente.

### 1.3 Por que ONNX e não PyTorch?

O modelo YOLO é exportado automaticamente para ONNX na primeira execução e reutilizado nas execuções seguintes. O ONNX Runtime via CPU é 2–3× mais rápido que o PyTorch em modo CPU para inferência, e permite controle preciso do número de threads internas via `SessionOptions` — essencial para evitar oversubscription quando múltiplos processos rodam em paralelo.

---

## 2. Arquitetura de Paralelismo

### 2.1 Modo parallel — pipeline completo por processo

No modo `parallel`, a lista de imagens é dividida em lotes e distribuída entre N processos via `concurrent.futures.ProcessPoolExecutor`. Cada processo:

1. Carrega **seu próprio YOLO** e **seu próprio OCR** (isolados, não compartilhados).
2. Fixa a thread interna de ambas as sessões ONNX em 1 (`intra_op_num_threads=1`).
3. Processa sua fatia de imagens do início ao fim — YOLO em mini-lotes, crop e OCR.

A separação de processos (ao invés de threads) é fundamental: cada processo tem seu próprio espaço de endereçamento e não sofre contenção do GIL (Global Interpreter Lock) do Python durante a inferência em C++.

**Por que não "YOLO serial + N processos de OCR" (arquitetura anterior)?**

Na versão anterior (v10), o YOLO rodava inteiramente no processo principal antes de qualquer paralelismo começar — representando ~70% do tempo total, completamente imune ao número de processos de OCR. A Lei de Amdahl impõe que, com 70% do trabalho em série, o speedup máximo possível é `1 / 0.3 ≈ 3.3×`, independentemente de quantos processos de OCR fossem usados. A versão atual (v11) paraleliza as duas etapas juntas: N processos = N YOLOs + N OCRs rodando ao mesmo tempo.

### 2.2 Controle de threads internas (monkeypatch ORT)

Por padrão, toda `ort.InferenceSession` criada sem `SessionOptions` explícitas usa `intra_op_num_threads=0` (automático = todos os núcleos). Com N processos × 2 sessões cada (YOLO + OCR), isso resulta em N × 2 × `cpu_count` threads disputando os mesmos núcleos simultaneamente.

Como a API pública do `ultralytics.YOLO` não expõe `SessionOptions`, o sistema aplica um monkeypatch local ao processo (`pipeline._patch_onnxruntime_threads(1)`) que substitui `ort.InferenceSession` por uma subclasse que força 1 thread interna e desativa o busy-wait (spin) em todas as sessões criadas naquele processo. Isso garante que N processos = N threads ativas, e o speedup medido reflita o paralelismo entre processos de forma limpa.

### 2.3 Warmup fora do cronômetro

Cada processo filho precisa fazer: spawn do processo, reimport de todas as bibliotecas (cv2, onnxruntime, ultralytics), carregamento e warm-up do YOLO e do OCR. Esse custo fixo de cold-start, que pode levar vários segundos por processo, **não encolhe com o aumento de workers** — se incluído no tempo medido, penalizaria desproporcionalmente configurações com poucos workers e distorceria o cálculo de speedup.

O sistema usa um `multiprocessing.Barrier(n_workers + 1)` passado via `initargs`: cada processo filho chama `barrier.wait()` ao final de sua inicialização, e o processo pai só avança (iniciando o cronômetro) quando todos os N processos estão prontos. Isso garante paridade exata com o modo serial, onde o YOLO e o OCR também são aquecidos antes do cronômetro começar.

### 2.4 INT8 estático — redução de pressão de cache L3

Com o modelo YOLO em FP32 (~11.8 MB), cada processo carrega sua própria cópia. Com 4 workers: 4 × 11.8 MB = 47.2 MB competindo por um L3 de poucos MB. O resultado é cache thrashing: os pesos são lidos da DRAM a cada inferência (~50× mais lento que L3), zerando boa parte do ganho de paralelismo.

A quantização estática INT8 (`tools/quantize_static.py`) reduz o modelo para ~3.4 MB. Com 4 workers: 4 × 3.4 MB = 13.6 MB, valor que se aproxima da capacidade do L3 do processador testado (ver §3) — os pesos ficam majoritariamente em cache, e o speedup se aproxima do ideal. A calibração usa o método Entropy (melhor que MinMax para as distribuições assimétricas do detection head sigmoid), e os nós de saída são excluídos da quantização para preservar a precisão dos scores de confiança.

---

## 3. Ambiente Experimental

Todos os benchmarks reportados neste documento foram executados na máquina abaixo, via `run_blackbox.py`, com o modelo YOLO **INT8 estático** (`license_plate_detector_static_int8.onnx`).

| Item | Descrição |
|---|---|
| Dispositivo | DESKTOP-F3Q7CGR |
| Processador | AMD Ryzen 5 PRO 4650GE with Radeon Graphics @ 3.30 GHz (boost até 4.0 GHz) |
| Núcleos físicos / Threads lógicas | 6 núcleos físicos / 12 threads (SMT habilitado) |
| Arquitetura | Zen 2 ("Renoir"), processo 7nm |
| Cache L3 | 8 MB compartilhado |
| Memória RAM | 12,0 GB instalados (11,3 GB utilizáveis) |
| Placa gráfica | AMD Radeon™ Graphics integrada (496 MB dedicados) |
| Armazenamento | 113 GB de 704 GB usados (SSD/HDD não discriminado pelo SO) |
| Sistema Operacional | Windows, 64 bits (x64) |
| Modelo YOLO | YOLO v8n exportado para ONNX — INT8 estático (~3.4 MB) |
| Modelo OCR | fast-plate-ocr CCT-xs-v2-global INT8 (~1.06 MB) |
| Runner de benchmark | `run_blackbox.py` v2 — limites externos de thread (OMP/MKL/BLIS/ORT = 1 thread antes de qualquer import) |

> **Observação sobre o hardware:** com 6 núcleos físicos / 12 threads lógicas via SMT, a configuração de **8 workers** já opera *acima* do número de núcleos físicos (apoiando-se em threads lógicas), e **12 workers** satura completamente as threads lógicas disponíveis. Isso é determinante para a leitura da queda de eficiência observada nas seções seguintes.

---

## 4. Metodologia de Testes

### 4.1 O que é medido

O tempo reportado como **"Tempo total" (interno)** corresponde ao custo de CPU puro do pipeline (YOLO + OCR), excluindo:

- Carregamento e warm-up dos modelos (medido separadamente e excluído via `multiprocessing.Barrier`, conforme §2.3)
- Gravação de crops/preprocessed em disco
- Geração do relatório HTML
- Spawn e inicialização dos processos filhos

O `run_blackbox.py` injeta variáveis de ambiente de thread (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, etc.) no subprocesso **antes de qualquer import Python** — mais confiável que patches internos que rodam depois das bibliotecas terem lido seus defaults.

### 4.2 Speedup e eficiência

```
Speedup(p)    = T(1) / T(p)
Eficiência(p) = Speedup(p) / p
```

Onde `T(1)` é o tempo serial e `T(p)` é o tempo com `p` processos. O tempo usado em todas as tabelas deste relatório é o **tempo interno** (extraído do stdout do `main.py`), que exclui overhead de spawn dos processos — é a métrica correta para avaliar o paralelismo computacional puro do pipeline.

### 4.3 Configurações testadas

| Configuração | Descrição |
|---|---|
| 1 processo (serial) | Baseline — execução sequencial |
| 2 processos | Paralelismo inicial |
| 4 processos | Configuração esperada próxima do ideal |
| 8 processos | Acima dos 6 núcleos físicos — depende de SMT |
| 12 processos | Saturação total das 12 threads lógicas |

Foram realizadas **duas baterias completas** de testes nesta máquina, ambas com o mesmo modelo INT8 estático e os mesmos parâmetros de paralelismo, diferindo apenas no tamanho do lote de imagens processado e no timeout configurado para o runner:

| Execução | Timeout/execução | Imagens processadas |
|---|---:|---:|
| Execução 1 | 360 s | 1.500 |
| Execução 2 | 400 s | 3.000 |

Todas as configurações, em cada execução, processaram o mesmo conjunto de imagens com o mesmo modelo, sem alterações de parâmetros entre as rodadas.

---

## 5. Resultados Experimentais

### 5.1 Execução 1 — 1.500 imagens (timeout = 360 s)

| Processos | Tempo (s) | Speedup | Eficiência |
|---:|---:|---:|---:|
| 1 (serial) | 89.866 | 1.000x | 100,0% |
| 2 | 48.375 | 1.858x | 92,9% |
| 4 | 26.005 | 3.456x | 86,4% |
| **8** | **20.483** | **4.387x** | **54,8%** |
| 12 | 23.286 | 3.859x | 32,2% |

🏆 **Melhor configuração: `parallel/8w`** — speedup de **4,387×**, reduzindo o tempo de 89,87 s para 20,48 s.

### 5.2 Execução 2 — 3.000 imagens (timeout = 400 s)

| Processos | Tempo (s) | Speedup | Eficiência |
|---:|---:|---:|---:|
| 1 (serial) | 202.719 | 1.000x | 100,0% |
| 2 | 90.526 | 2.239x | **112,0%** ⚡ |
| 4 | 52.261 | 3.879x | 97,0% |
| **8** | **38.593** | **5.253x** | **65,7%** |
| 12 | 43.401 | 4.671x | 38,9% |

🏆 **Melhor configuração: `parallel/8w`** — speedup de **5,253×**, reduzindo o tempo de 202,72 s para 38,59 s. Esta é a **melhor marca registrada em todo o experimento**.

⚡ Em **2 workers**, a eficiência ultrapassou 100% (**speedup superlinear**) — ponto discutido em detalhe na §6.1.

### 5.3 Resumo consolidado — melhores resultados por execução

| Execução | Imagens | Melhor config | Speedup | Eficiência |
|---|---:|---|---:|---:|
| Execução 1 | 1.500 | parallel/8w | 4,387x | 54,8% |
| **Execução 2** | **3.000** | **parallel/8w** | **5,253x** | **65,7%** |

> A Execução 2, com um lote de imagens duas vezes maior, amortiza melhor o overhead fixo de cada processo (spawn, import de bibliotecas, warm-up residual fora do cronômetro, mas ainda presente na variância) e atinge tanto speedup interno quanto eficiência superiores — evidência consistente de que workloads maiores favorecem o paralelismo neste hardware.

### 5.4 Tempos detalhados por estágio (YOLO / OCR)

> Os logs do `run_blackbox.py` capturados (`resultados1.txt`) reportam apenas os tempos agregados de wall clock e tempo interno total por configuração. A quebra por estágio (YOLO acumulado vs. OCR acumulado) exige a execução de `main.py` em modo verboso/benchmark individual por configuração, não incluída nesta coleta. Essa medição fica como trabalho futuro indicado na §9.

---

## 6. Análise dos Resultados

### 6.1 Speedup superlinear em 2 workers (Execução 2)

Na Execução 2, 2 workers atingiram eficiência de **112,0%** — speedup maior que o número de processos usados. Isso é coerente com o efeito de cache descrito em §2.4: com o modelo INT8 (~3,4 MB), **2 × 3,4 MB ≈ 6,8 MB**, valor que ainda cabe confortavelmente nos 8 MB de L3 do Ryzen 5 PRO 4650GE. Cada worker passa a ler seus pesos quase inteiramente do cache L3, enquanto o processo serial — sozinho — já sofre algum grau de poluição de cache por outras estruturas de dados do pipeline (buffers de imagem, tensores intermediários do OCR). O resultado é que o trabalho dobrado, dividido entre 2 processos com cache "limpo", roda mais que 2× mais rápido que o equivalente serial.

### 6.2 Crescimento sub-linear, mas consistente, até 4 workers

Com 4 workers, a eficiência permaneceu alta (86,4% na Execução 1; **97,0%** na Execução 2) — muito próxima do ideal teórico. Isso confirma que, com 4 × 3,4 MB ≈ 13,6 MB de pesos do YOLO disputando o L3 de 8 MB, ainda há margem suficiente (o restante do working set é pequeno o bastante) para que a maior parte das leituras ainda hits no cache, e o paralelismo entre processos domina sobre o custo de cache miss.

### 6.3 Pico em 8 workers, apesar de exceder os núcleos físicos

O melhor resultado absoluto do experimento ocorreu em **8 workers**, mesmo a máquina possuindo apenas **6 núcleos físicos**. Isso é explicado pela natureza do workload: o pipeline mistura computação intensa (inferência YOLO/OCR) com pequenas janelas de I/O (leitura de imagem, decodificação, pré-processamento) e overhead de IPC. Threads lógicas adicionais (SMT) conseguem preencher parte dessas janelas de espera com trabalho útil de outro processo, gerando ganho real mesmo acima da contagem de núcleos físicos — até o ponto em que a contenção pelas unidades de execução compartilhadas do core físico (em SMT) passa a dominar.

A eficiência, no entanto, já cai visivelmente em 8 workers (54,8% e 65,7% nas duas execuções) — sinal de que, embora o tempo absoluto continue melhorando, o retorno marginal por processo adicional já está em declínio acentuado.

### 6.4 Degradação em 12 workers — saturação total das threads lógicas

Em 12 workers (= total de threads lógicas da máquina), o tempo total **piora** em relação a 8 workers em ambas as execuções (de 20,48 s para 23,29 s na Execução 1; de 38,59 s para 43,40 s na Execução 2). Com 12 processos × 2 sessões ONNX (YOLO + OCR) competindo pelas mesmas 12 threads lógicas, não resta nenhuma capacidade ociosa para absorver picos de I/O ou overhead de IPC — todo processo adicional passa a roubar tempo de CPU de outro processo do próprio pipeline, em vez de explorar paralelismo real. A eficiência despenca para 32,2%–38,9%, confirmando saturação.

### 6.5 Por que a Execução 2 supera a Execução 1 em todas as métricas de paralelismo

Com o dobro de imagens (3.000 vs. 1.500), o custo fixo por processo (parte residual de overhead que escapa ao `Barrier` de warmup, como alocações de memória incrementais e jitter do agendador do SO) é amortizado sobre um volume maior de trabalho útil. Isso explica por que, na mesma configuração de 8 workers, a Execução 2 atinge speedup maior (5,253× vs. 4,387×) e eficiência maior (65,7% vs. 54,8%) — overheads fixos por processo (constantes, independentes do volume de imagens) pesam relativamente menos quanto maior o lote.

### 6.6 Fatores limitantes identificados

- **Cache L3 (8 MB)**: principal variável explicando o salto de eficiência entre 2 e 4 workers e a queda a partir de 8. O INT8 estático foi decisivo para manter os pesos do YOLO dentro de uma faixa administrável pelo cache até 4 workers.
- **SMT/Hyperthreading**: dois processos compartilhando o mesmo núcleo físico competem pelas mesmas unidades de execução; o ganho por worker acima de 6 processos (núcleos físicos) já é parcial, e acima de 12 (limite de threads lógicas) é negativo.
- **Overhead de IPC**: serialização/deserialização de chunks de tarefas e resultados entre processos, mitigado por `CHUNK_TARGET_IMAGES = 100`, mas presente em toda configuração paralela.
- **Memória RAM limitada (12 GB, 11,3 GB utilizáveis)**: com 12 workers, cada um carregando sua própria cópia do YOLO + OCR + buffers de imagem, a pressão de memória cresce; embora não tenha causado swapping observável nos testes, é um fator de risco em datasets maiores.

---

## 7. Como Executar

### 7.1 Instalação

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 7.2 Inserir imagens

Baixe o dataset e extraia para:

```
data/input/
```

### 7.3 Execução interativa

```powershell
python main.py
```

O programa pergunta o modo de execução e o número de workers antes de iniciar.

### 7.4 Execução para benchmark

Sem gravar crops em disco e sem HTML (isola custo de CPU):

```powershell
python main.py --no-interactive --execution serial --benchmark
python main.py --no-interactive --execution parallel --workers 2 --benchmark
python main.py --no-interactive --execution parallel --workers 4 --benchmark
python main.py --no-interactive --execution parallel --workers 8 --benchmark
python main.py --no-interactive --execution parallel --workers 12 --benchmark
```

### 7.5 Blackbox runner (recomendado para benchmarks)

O `run_blackbox.py` injeta restrições de thread **antes** de qualquer import Python e calcula speedup pelo tempo interno (excluindo spawn):

```powershell
python run_blackbox.py --workers 1 2 4 8 12
```

Com o modelo INT8 estático (configuração usada em todos os resultados deste relatório):

```powershell
python run_blackbox.py --workers 2 4 8 12 --timeout 400 --yolo-model models/license_plate_detector_static_int8.onnx
```

Opções úteis:

| Flag | Efeito |
|---|---|
| `--timeout 600` | Segundos antes de cancelar cada execução |
| `--repeat 3` | Repetições por configuração para média |
| `--pin-cpu` | Fixa cada processo em um núcleo físico distinto |
| `--exec-mode thread` | Usa modo thread em vez de processos |
| `--verbose` | Imprime o stdout completo de cada execução |

### 7.6 Gerar modelo INT8 estático (opcional, requer imagens em data/input/)

```powershell
python tools/quantize_static.py
```

Gera `models/license_plate_detector_static_int8.onnx` (~3.4 MB) com calibração Entropy sobre até 150 imagens do dataset.

Para quantização dinâmica (sem calibração, mais simples):

```powershell
python tools/quantize.py
```

### 7.7 Gerenciar lista de placas roubadas

```powershell
python tools/stolen.py
```

---

## 8. Estrutura do Projeto

```
uni-project-stolen-vehicles/
├── main.py                  # Entry point — parse de args, warmup, orquestração
├── run_blackbox.py          # Runner de benchmark com controle externo de threads
├── requirements.txt
├── README.md
├── CLAUDE.md                # Guia para ferramentas de IA (Claude Code)
│
├── src/
│   ├── config.py            # Todos os parâmetros ajustáveis do sistema
│   ├── executor.py          # Modos de execução (serial/parallel/thread) e barra de progresso
│   ├── pipeline.py          # Workers: init_full_pipeline_process, process_image_batch
│   ├── detector.py          # Export ONNX, warmup YOLO, extract_best_crop
│   ├── ocr.py               # Singleton OCR, warmup, read_plate_text, filtros pós-OCR
│   ├── yolo_infer.py        # SharedYOLO — wrapper thread-safe para modo thread
│   ├── runtime.py           # force_single_thread_env, patch_ssl_certifi
│   ├── dataset.py           # list_images, load_stolen_plates, ensure_directories
│   ├── report.py            # save_results_csv, save_performance_log, print_summary
│   ├── html_report.py       # generate_html_report
│   ├── logger.py            # setup_logger, get_logger
│   └── colors.py            # Constantes de cor ANSI para terminal
│
├── tools/
│   ├── quantize_static.py   # Quantização INT8 estática calibrada (Entropy)
│   ├── quantize.py          # Quantização INT8 dinâmica (sem calibração)
│   └── stolen.py            # Gerenciador de lista de placas roubadas
│
├── models/
│   ├── license_plate_detector.pt              # Modelo YOLO original PyTorch
│   ├── license_plate_detector.onnx            # Modelo YOLO FP32 exportado
│   └── license_plate_detector_static_int8.onnx # Modelo YOLO INT8 estático (usado nos benchmarks deste relatório)
│
└── data/
    ├── input/               # Imagens de entrada (não versionadas)
    └── output/
        ├── results.csv
        ├── performance_log.csv
        ├── report.html
        ├── crops/           # Recortes das placas detectadas
        └── preprocessed/    # Versão CLAHE+Otsu (para o relatório HTML)
```

---

## 9. Conclusão

Os experimentos confirmam, na prática, os princípios teóricos de paralelismo discutidos na disciplina:

- **A Lei de Amdahl se manifesta na escolha arquitetural**: paralelizar YOLO + OCR juntos (v11), em vez de apenas o OCR (v10), foi o que tornou speedups acima de 3,3× sequer possíveis.
- **O hardware impõe um teto físico**: com 6 núcleos físicos / 12 threads lógicas, o ganho real de paralelismo se esgota por volta de 8 workers, e qualquer configuração acima de 12 não tem mais capacidade de execução disponível para explorar.
- **Otimização de dados (INT8) pode superar o paralelismo bruto**: reduzir o modelo de ~11,8 MB para ~3,4 MB permitiu que mais cópias do modelo coexistissem no cache L3, sendo responsável pelo speedup superlinear observado em 2 workers.
- **O tamanho do workload importa**: a mesma configuração (8 workers) rendeu speedup interno de 4,387× com 1.500 imagens e 5,253× com 3.000 imagens — overheads fixos por processo se diluem melhor em lotes maiores.

A configuração recomendada para esta máquina, equilibrando tempo absoluto e eficiência de recursos, é **`parallel` com 8 workers e modelo YOLO INT8 estático**, que produziu o melhor tempo absoluto em ambas as baterias de teste (20,48 s e 38,59 s, respectivamente).

### Trabalho futuro

- Coletar a quebra de tempo por estágio (YOLO vs. OCR) por configuração, para confirmar quantitativamente o comportamento descrito em §2.1 e §6.
- Repetir os testes com `--repeat 3` para obter desvio padrão e intervalos de confiança sobre o speedup medido.
- Testar `--pin-cpu` para verificar se a fixação explícita de processos a núcleos físicos reduz a degradação observada em 12 workers.
- Gerar e comparar a bateria completa também com o modelo FP32, para quantificar isoladamente o ganho atribuível à quantização INT8 (item ainda pendente nas tabelas de §5).

---

Projeto desenvolvido para a disciplina de **Programação Concorrente e Distribuída**.

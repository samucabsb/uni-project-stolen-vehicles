# Detector Paralelo de Placas Veiculares

![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)
![Paralelismo](https://img.shields.io/badge/Paralelismo-ProcessPoolExecutor-f97316?style=for-the-badge)
![Detector](https://img.shields.io/badge/Detector-YOLO%20v8%20ONNX-00b4d8?style=for-the-badge)
![OCR](https://img.shields.io/badge/OCR-fast--plate--ocr-8b5cf6?style=for-the-badge)

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

## 1. Descrição do Problema

O projeto implementa um sistema de **reconhecimento automático de placas veiculares em lote**. O programa recebe um diretório de imagens de veículos, detecta a região da placa em cada imagem com **YOLO v8 exportado para ONNX**, recorta a placa detectada, lê o texto com **fast-plate-ocr (modelo CCT)** e compara o resultado com uma lista local de veículos roubados.

O objetivo central é demonstrar ganho de desempenho com paralelismo, comparando a execução serial (1 processo) com a execução paralela (N processos), medindo speedup e eficiência.

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

Com o modelo YOLO em FP32 (~11.8 MB), cada processo carrega sua própria cópia. Com 4 workers: 4 × 11.8 MB = 47.2 MB competindo por um L3 de ~12 MB. O resultado é cache thrashing: os pesos são lidos da DRAM a cada inferência (~50× mais lento que L3), zerando boa parte do ganho de paralelismo.

A quantização estática INT8 (`tools/quantize_static.py`) reduz o modelo para ~3.4 MB. Com 4 workers: 4 × 3.4 MB = 13.6 MB ≈ L3 → os pesos ficam em cache, e o speedup se aproxima do ideal. A calibração usa o método Entropy (melhor que MinMax para as distribuições assimétricas do detection head sigmoid), e os nós de saída são excluídos da quantização para preservar a precisão dos scores de confiança.

---

## 3. Ambiente Experimental

| Item | Descrição |
|---|---|
| Processador | _(preencher com o hardware usado)_ |
| Núcleos físicos | _(preencher)_ |
| Threads lógicas | _(preencher)_ |
| Cache L3 | _(preencher)_ |
| Memória RAM | _(preencher)_ |
| Sistema Operacional | _(preencher)_ |
| Python | _(preencher)_ |
| ONNX Runtime | _(preencher)_ |
| Modelo YOLO | YOLO v8n exportado para ONNX — FP32 (~11.8 MB) e INT8 estático (~3.4 MB) |
| Modelo OCR | fast-plate-ocr CCT-xs-v2-global INT8 (~1.06 MB) |

---

## 4. Metodologia de Testes

### 4.1 O que é medido

O tempo reportado como **"Tempo total"** corresponde ao custo de CPU puro do pipeline (YOLO + OCR), excluindo:

- Carregamento e warm-up dos modelos (medido separadamente como "Warm-up")
- Gravação de crops/preprocessed em disco (`--benchmark` desliga esse I/O)
- Geração do relatório HTML (`--benchmark` também desliga)
- Spawn e inicialização dos processos filhos (excluído via barrier, conforme §2.3)

O `run_blackbox.py` injeta variáveis de ambiente de thread (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, etc.) no subprocess **antes de qualquer import Python** — mais confiável que patches internos que rodam depois das bibliotecas terem lido seus defaults.

### 4.2 Speedup e eficiência

```
Speedup(p) = T(1) / T(p)
Eficiência(p) = Speedup(p) / p
```

Onde `T(1)` é o tempo serial e `T(p)` é o tempo com `p` processos. O `run_blackbox.py` calcula dois speedups distintos:

- **Speedup interno** (`Spd(int)`): baseado no `Tempo total` extraído do stdout do `main.py`. Exclui overhead de spawn. É a métrica correta para avaliar o paralelismo computacional.
- **Speedup wall clock** (`Spd(wall)`): baseado no tempo real de parede. Inclui spawn e toda latência visível para o usuário.

### 4.3 Configurações testadas

| Configuração | Descrição |
|---|---|
| 1 processo (serial) | Baseline — execução sequencial |
| 2 processos | Paralelismo inicial |
| 4 processos | Configuração esperada próxima do ideal |
| 8 processos | Acima dos núcleos físicos em CPUs de 6 físicos |
| 12 processos | Com hyperthreading — saturação esperada |

Todas as configurações processaram o mesmo conjunto de imagens com o mesmo modelo, sem alterações de parâmetros entre as rodadas.

---

## 5. Resultados Experimentais

### 5.1 Modelo FP32 (baseline)

| Processos | Tempo (s) | Speedup | Eficiência | Throughput (img/s) |
|---:|---:|---:|---:|---:|
| 1 (serial) | — | 1.000x | 100.0% | — |
| 2 | — | — | — | — |
| 4 | — | — | — | — |
| 8 | — | — | — | — |
| 12 | — | — | — | — |

### 5.2 Modelo INT8 estático (otimizado)

| Processos | Tempo (s) | Speedup | Eficiência | Throughput (img/s) |
|---:|---:|---:|---:|---:|
| 1 (serial) | — | 1.000x | 100.0% | — |
| 2 | — | — | — | — |
| 4 | — | — | — | — |
| 8 | — | — | — | — |
| 12 | — | — | — | — |

### 5.3 Tempos detalhados por estágio

| Processos | Warm-up (s) | YOLO acumulado (s) | OCR acumulado (s) |
|---:|---:|---:|---:|
| 1 (serial) | — | — | — |
| 2 | — | — | — |
| 4 | — | — | — |
| 8 | — | — | — |
| 12 | — | — | — |

> Os tempos acumulados de YOLO e OCR no modo parallel representam a **soma** do tempo gasto em todos os N processos — não o tempo de parede. Com N processos rodando em paralelo, a soma pode ser maior que o tempo de parede (utilização > 100%).

---

## 6. Análise dos Resultados

_(Preencher após coletar os dados experimentais.)_

### 6.1 Comportamento esperado

Com o modelo INT8 e cache L3 adequado:

- **Speedup superlinear em 2 workers**: possível quando 2 × tamanho_do_modelo < L3. Os pesos cabem inteiros no cache, e cada worker lê da DRAM menos que o serial que mal cabia.
- **Speedup próximo do ideal em 4 workers**: quando o modelo é pequeno o suficiente para que 4 cópias ainda caibam no L3.
- **Degradação acima dos núcleos físicos**: ao usar hyperthreading, dois workers compartilham as mesmas unidades de execução de um núcleo físico. Combinado com thrashing de cache, o ganho por worker cai.

### 6.2 Fatores limitantes

- **Cache L3**: principal gargalo com FP32. Resolvido parcialmente com INT8.
- **Hyperthreading (HT)**: dois processos no mesmo core físico competem pelas mesmas unidades de execução, reduzindo o ganho por worker acima do número de físicos.
- **Overhead de IPC**: serialização/deserialização dos chunks de tarefas e resultados entre processos. Mitigado por `CHUNK_TARGET_IMAGES=100`.
- **I/O de disco**: gravação de crops/preprocessed com N processos simultâneos. Eliminado em benchmarks com `--benchmark`.

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

Com modelo INT8 estático:

```powershell
python run_blackbox.py --workers 1 2 4 8 12 --yolo-model models/license_plate_detector_static_int8.onnx
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
│   ├── license_plate_detector.pt    # Modelo YOLO original PyTorch
│   └── license_plate_detector.onnx  # Modelo YOLO FP32 exportado (gerado na 1ª execução)
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

Projeto desenvolvido para a disciplina de **Programação Concorrente e Distribuída**.

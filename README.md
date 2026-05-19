# Comparador Paralelo de Placas — v7

Sistema de detecção e identificação automática de placas veiculares com
comparação contra base de placas roubadas. Pipeline **YOLO** (detecção) +
**RapidOCR** (reconhecimento) + **ProcessPoolExecutor** (paralelismo), com
relatório HTML interativo e análise comparativa serial vs paralelo.

---

## Sumário

- [Instalação](#instalação)
- [Como rodar](#como-rodar)
- [Saídas geradas](#saídas-geradas)
- [Ferramenta de placas roubadas](#ferramenta-de-placas-roubadas)
- [Coleta de dados para o relatório](#coleta-de-dados-para-o-relatório)
- [Arquitetura](#arquitetura)
- [Otimizações aplicadas](#otimizações-aplicadas)

---

## Instalação

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Coloque o modelo YOLO em `models/license_plate_detector.pt` e as imagens de
entrada em `data/input/`.

---

## Como rodar

### Modo interativo

```powershell
python main.py
```

O sistema pergunta o modo de execução (serial/parallel/pipeline) e a quantidade de workers.

### Modo direto (sem perguntas)

```powershell
# Baseline
python main.py --execution serial --no-interactive

# Data parallelism clássico
python main.py --execution parallel --workers 4 --no-interactive

# Pipeline two-stage (v8) — mais rápido em qualquer hardware
python main.py --execution pipeline --workers 4 --no-interactive
```

### Três modos disponíveis

| Modo | Como funciona | Melhor em |
|---|---|---|
| `serial` | 1 processo, YOLO → OCR sequencial | Baseline |
| `parallel` | N workers, cada um com YOLO+OCR | Hardware lento |
| `pipeline` | YOLO em batch no main + N workers só OCR | **Qualquer hardware** |

**Por que o `pipeline` é mais rápido:**
- YOLO roda em batch (8 imgs por forward pass) no processo principal, sem concorrência
- Workers carregam só OCR (~220MB) vs YOLO+OCR (~400MB)
- Menor footprint por worker → menos cache miss → OCR mais rápido
- Imagens sem placa saem imediatamente (sem custo de OCR)

---

## Saídas geradas

Toda execução grava em `data/output/`:

| Arquivo | Conteúdo |
|---|---|
| `report.html` | **Relatório visual interativo** — abra no navegador |
| `results.csv` | Uma linha por imagem com tempo, status, confiança, etc |
| `performance_log.csv` | Uma linha por execução, acumulativo (para o relatório acadêmico) |
| `crops/` | Recortes das placas detectadas |
| `preprocessed/` | Versões pré-processadas dos recortes |

### O relatório HTML (`report.html`)

Abre no navegador e mostra:

- **Cards de estatística** no topo: tempo total, throughput, imagens, workers, contagens por status
- **Alerta de roubados** com cards visuais para cada veículo identificado (imagem + placa + arquivo)
- **Tabela completa** com filtros (todos / OK / roubados / não identificadas) e busca em tempo real
- **Modal de detalhes** ao clicar em qualquer linha: foto original do carro + todas as métricas (tempo YOLO, tempo OCR, confiança, PID)

O HTML é gerado **depois** da medição de performance, portanto **zero impacto
no tempo medido**. É um arquivo único de ~50KB que funciona offline.

---

## Ferramenta de placas roubadas

`tools/stolen.py` deixa fácil gerenciar a lista de placas que devem disparar
o alerta de ROUBADO.

```powershell
# Listar placas atualmente marcadas
python tools\stolen.py list

# Adicionar placas
python tools\stolen.py add ABC1234
python tools\stolen.py add ABC1234 DEF5678 GHI9012

# Remover uma placa
python tools\stolen.py remove ABC1234

# Limpar toda a lista (com confirmação)
python tools\stolen.py clear

# DEMO: marca 5 placas aleatórias do último run como roubadas
python tools\stolen.py demo 5
```

### Fluxo de demonstração end-to-end

```powershell
# 1. Roda o sistema uma primeira vez para detectar todas as placas
python main.py --execution serial --no-interactive

# 2. Marca 5 placas detectadas como roubadas (escolhe aleatoriamente)
python tools\stolen.py demo 5

# 3. Roda de novo — agora as 5 vão aparecer com alerta visual de ROUBADO
python main.py --execution serial --no-interactive

# 4. Abre o relatório
start data\output\report.html
```

---

## Coleta de dados para o relatório

Para gerar a tabela de comparação serial vs paralelo:

```powershell
# Apaga o log antigo para começar limpo
Remove-Item data\output\performance_log.csv -ErrorAction SilentlyContinue

# Roda em diferentes configurações
python main.py --execution serial --no-interactive
python main.py --execution parallel --workers 2  --no-interactive
python main.py --execution parallel --workers 4  --no-interactive
python main.py --execution parallel --workers 8  --no-interactive
python main.py --execution parallel --workers 12 --no-interactive
```

Cada execução adiciona uma linha em `data/output/performance_log.csv` com:

`timestamp, execution_type, workers_solicitados, workers_efetivos,
total_images, warmup_time_s, total_processing_time_s, avg_time_per_image_s,
min_time_per_image_s, max_time_per_image_s, throughput_img_per_s,
images_with_plate, images_without_plate, ok_count, roubado_count,
nao_identificada_count, error_count`

---

## Arquitetura

```
project/
├── main.py                Entrada principal — CLI + orquestração
├── requirements.txt
├── README.md
│
├── src/
│   ├── colors.py          Cores ANSI para o terminal
│   ├── config.py          Constantes, paths, parâmetros
│   ├── dataset.py         Descoberta de imagens, placas roubadas
│   ├── detector.py        YOLO + crop com padding de 8%
│   ├── executor.py        Serial / paralelo + progresso colorido
│   ├── html_report.py     Geração do report.html
│   ├── ocr.py             RapidOCR + early exit + scoring genérico
│   ├── pipeline.py        Processamento por imagem (init_worker + task)
│   ├── report.py          CSVs + sumário visual no terminal
│   └── runtime.py         Limites de thread para paralelismo
│
├── tools/
│   └── stolen.py          Gerenciamento de placas roubadas
│
├── models/
│   └── license_plate_detector.pt
│
└── data/
    ├── input/             Imagens de entrada
    ├── stolen_plates.txt  Gerenciado por tools/stolen.py
    └── output/
        ├── report.html
        ├── results.csv
        ├── performance_log.csv
        ├── crops/
        └── preprocessed/
```

### Fluxo de uma imagem

```
imagem → [YOLO] → bounding box → [crop + padding 8%] →
[preprocess: CLAHE + Otsu] → [OCR com 3 variantes + early exit] →
[scoring genérico] → texto normalizado →
[comparação com stolen_plates] → status (OK/ROUBADO/NAO_IDENTIFICADA)
```

### Estratégia de paralelismo

`ProcessPoolExecutor` com `init_worker` que carrega YOLO + RapidOCR **uma
vez por processo**. Cada worker usa **1 thread interna** (env vars + APIs
de torch/cv2/onnx) para evitar contenção entre processos.

O paralelismo vem da **quantidade de processos**, não de threads dentro de
cada um. Isso é crítico: sem o limite de 1 thread por processo, N workers
× T threads brigam pelos mesmos N núcleos, causando slowdown.

---

## Otimizações aplicadas

| Otimização | Onde | Efeito |
|---|---|---|
| Limite de 1 thread por processo | `runtime.py`, `pipeline.py` | Elimina contenção entre workers paralelos |
| Padding de 8% no bbox do YOLO | `detector.py` | Evita leituras parciais por bbox apertado |
| Early exit no OCR | `ocr.py` | Pula variantes restantes quando a primeira já é boa (~30-66% menos OCR) |
| Variantes em ordem de probabilidade | `ocr.py` | Crop original primeiro, preprocessed por último |
| Scoring genérico de placa | `ocr.py` | Funciona com placas de qualquer país (não hardcoded) |
| Cap de workers por RAM disponível | `executor.py` | Evita swap em máquinas com pouca memória |
| YOLO nano (6MB) | `models/` | 8x mais rápido que YOLO large com precisão similar |
| HTML report pós-execução | `html_report.py` | Visualização rica sem impacto na medição de performance |

---

## Modos de execução: o que esperar

| Modo | Comportamento esperado em CPU 4-core |
|---|---|
| `serial` (1 worker) | Baseline — sem paralelismo |
| `parallel --workers 2` | ~1.7x speedup, underutilization de cores |
| `parallel --workers 4` | **Sweet spot** — ~1.7-2.0x speedup |
| `parallel --workers 8` | Hyperthreading, speedup similar ou levemente acima |
| `parallel --workers 12` | Oversubscription, speedup começa a cair (Lei de Amdahl) |

Esses dados são todos valiosos para o relatório acadêmico — a curva completa
demonstra o trade-off entre paralelismo e contenção.

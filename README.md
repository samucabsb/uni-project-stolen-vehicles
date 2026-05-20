# Comparador Paralelo de Placas — v9.0

Pipeline YOLO (detecção) + RapidOCR (reconhecimento) com comparação contra
lista de placas roubadas. Suporta dois modos de execução com medição de
performance para fins acadêmicos.

---

## Arquitetura

```
Processo principal:
  → YOLO em batch (8 imgs/vez)  [Estágio 1]
  → ThreadPool N threads:        [Estágio 2]
       Thread 1: OCR da imagem X
       Thread 2: OCR da imagem Y   ← compartilham processo e cache L3
       Thread 3: OCR da imagem Z
```

Threads compartilham o mesmo processo, mantendo os pesos do RapidOCR quentes
no cache L3 do CPU. O ONNX Runtime libera o GIL durante inferência, então o
paralelismo é real sem overhead de processos separados.

---

## Instalação

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Coloque o modelo YOLO em `models/license_plate_detector.pt`.

---

## Uso

```bash
python main.py
```

O sistema pergunta o modo e o número de threads interativamente.

**Flags disponíveis:**

| Flag | Descrição |
|---|---|
| `--execution serial\|parallel` | Modo sem perguntas |
| `--workers N` | Número de threads (qualquer valor ≥ 1) |
| `--no-interactive` | Roda com defaults ou flags, sem input |
| `--verbose` | Output de diagnóstico detalhado |
| `--quiet` | Suprime mensagens INFO |
| `--yolo-model PATH` | Modelo YOLO alternativo |

**Exemplo para coleta de dados de benchmark (serial + 2 + 4 + 8 + 12 threads):**

```powershell
# Limpa log anterior (opcional)
Remove-Item data\output\performance_log.csv -ErrorAction SilentlyContinue

python main.py --execution serial   --no-interactive
python main.py --execution parallel --workers 2  --no-interactive
python main.py --execution parallel --workers 4  --no-interactive
python main.py --execution parallel --workers 8  --no-interactive
python main.py --execution parallel --workers 12 --no-interactive
```

---

## Saídas geradas

| Arquivo | Descrição |
|---|---|
| `data/output/results.csv` | Um registro por imagem (sobrescreve) |
| `data/output/performance_log.csv` | Uma linha por execução (acumula) |
| `data/output/report.html` | Relatório visual interativo |
| `data/output/crops/` | Recortes de placas detectadas |
| `data/output/preprocessed/` | Versões pré-processadas dos crops |

### Campos do `performance_log.csv`

| Campo | Descrição |
|---|---|
| `execution_type` | `serial` ou `parallel` |
| `workers_efetivos` | Threads usadas no estágio OCR |
| `total_processing_time_s` | Tempo wall-clock total (sem warm-up) |
| `yolo_stage_time_s` | Wall-clock do estágio YOLO |
| `ocr_stage_time_s` | Soma dos tempos OCR individuais |
| `avg_time_per_image_s` | `total_processing_time_s / total_images` |
| `throughput_img_per_s` | Imagens processadas por segundo |

---

## Placas roubadas

```bash
python tools/stolen.py add ABC1234
python tools/stolen.py list
python tools/stolen.py demo 10    # Adiciona 10 placas aleatórias para teste
python tools/stolen.py remove ABC1234
```

---

## Estrutura do projeto

```
project/
├── main.py
├── requirements.txt
├── models/
│   └── license_plate_detector.pt
├── data/
│   ├── input/              ← coloque as imagens aqui
│   ├── stolen_plates.txt
│   └── output/
├── src/
│   ├── config.py           Constantes e caminhos
│   ├── colors.py           Cores ANSI
│   ├── logger.py           Logging estruturado
│   ├── runtime.py          Controle de threads de bibliotecas
│   ├── dataset.py          Descoberta de imagens
│   ├── detector.py         Wrapper YOLO
│   ├── ocr.py              Wrapper RapidOCR + scoring
│   ├── pipeline.py         Workers serial e threaded
│   ├── executor.py         Orquestração + barra de progresso
│   ├── report.py           CSVs + sumário terminal
│   └── html_report.py      Relatório HTML interativo
└── tools/
    └── stolen.py           Gerenciador da lista de placas roubadas
```

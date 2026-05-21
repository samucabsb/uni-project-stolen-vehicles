# Comparador Paralelo de Placas — v10.0

Pipeline YOLO (detecção) + fast-plate-ocr/CCT (reconhecimento) com comparação
contra lista de placas roubadas.

---

## O que mudou na v10

**Motor OCR substituído: RapidOCR → fast-plate-ocr (CCT)**

O fast-plate-ocr usa uma arquitetura CCT (Compact Convolutional Transformer)
treinada em 65+ países com 114k+ amostras reais. O Transformer lê a placa
inteira como sequência — quando processa a posição 2, já "viu" o estado
(posições 0-1) e sabe que deve ser dígito. A confusão L↔D (I/1, O/0, S/5)
torna-se improvável por design.

---

## Implicação para o benchmark de paralelismo (Lei de Amdahl)

Esta é a descoberta mais importante da v10:

| Configuração | v9 (RapidOCR) | v10 (fast-plate-ocr) |
|---|---|---|
| OCR por placa (CPU) | ~500–2000ms | ~20–50ms |
| YOLO (sempre serial) | ~12s | ~12s |
| Serial total | ~191s | ~13–15s |
| Parallel 4 threads | ~124s (1.55x) | ~12–13s (~1.05x) |

**Conclusão:** quando o OCR é o gargalo (v9), o threading ajuda significativamente.
Quando o YOLO domina (v10), o threading quase não ajuda. Isso demonstra a
Lei de Amdahl empiricamente: o speedup máximo é determinado pela fração
serial do pipeline, não pela velocidade da parte paralelizável.

Para o relatório acadêmico, compare os dois cenários:
- **Cenário A (v9):** OCR = 94–99% do tempo → threading útil → speedup 1.55x
- **Cenário B (v10):** YOLO = 80–90% do tempo → threading inútil → speedup ~1.05x

---

## Arquitetura

```
Processo principal:
  → YOLO em batch (8 imgs/vez)  [Estágio 1]
  → ThreadPool N threads:        [Estágio 2]
       Thread 1: fast-plate-ocr(crop_X)
       Thread 2: fast-plate-ocr(crop_Y)   ← compartilham processo
```

---

## Instalação

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/Mac

pip install -r requirements.txt
```

Na primeira execução, o modelo CCT (~poucos MB) é baixado automaticamente.

---

## Uso

```bash
python main.py
```

**Flags:**

| Flag | Descrição |
|---|---|
| `--execution serial\|parallel` | Modo sem perguntas |
| `--workers N` | Threads (qualquer valor ≥ 1) |
| `--no-interactive` | Sem input, usa flags ou defaults |
| `--verbose` | Output de diagnóstico detalhado |
| `--quiet` | Suprime INFO |

**Coleta de dados para o relatório:**

```powershell
Remove-Item data\output\performance_log.csv -ErrorAction SilentlyContinue

python main.py --execution serial   --no-interactive
python main.py --execution parallel --workers 2  --no-interactive
python main.py --execution parallel --workers 4  --no-interactive
python main.py --execution parallel --workers 8  --no-interactive
python main.py --execution parallel --workers 12 --no-interactive
```

---

## Estrutura

```
project/
├── main.py
├── requirements.txt
├── models/
│   └── license_plate_detector.pt
├── data/
│   ├── input/
│   ├── stolen_plates.txt
│   └── output/
└── src/
    ├── config.py        VERSION=10.0, FAST_OCR_MODEL
    ├── ocr.py           fast-plate-ocr (CCT global model)
    ├── pipeline.py      Two-stage serial + parallel threading
    ├── executor.py      Orquestração + barra de progresso
    ├── report.py        CSVs + sumário terminal
    └── html_report.py   Relatório HTML interativo
```

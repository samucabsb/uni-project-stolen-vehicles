## Commands

**Setup**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Run (modo interativo)**
```powershell
python main.py
```

**Run (não-interativo, benchmark)**
```powershell
python main.py --no-interactive --execution serial --benchmark
python main.py --no-interactive --execution parallel --workers 4 --benchmark
python main.py --no-interactive --execution parallel --workers 4 --benchmark --yolo-model models/license_plate_detector_static_int8.onnx
```

**Blackbox benchmark runner** (controla env vars antes de qualquer import)
```powershell
python run_blackbox.py --workers 1 2 4 8 12
python run_blackbox.py --workers 1 2 4 --yolo-model models/license_plate_detector_static_int8.onnx
python run_blackbox.py --workers 1 2 4 --exec-mode thread   # modo thread experimental
```

**Quantização dos modelos**
```powershell
# Dinâmica — rápida, sem calibração, ligeiramente mais lenta na inferência
python tools/quantize.py

# Estática (INT8 com calibração) — mais rápida na inferência, requer imagens em data/input/
python tools/quantize_static.py
```

**Gerenciar lista de placas roubadas**
```powershell
python tools/stolen.py
```

## Arquitetura

### Modos de execução

Controlados por `--execution` em `main.py`:

- **serial**: baseline — 1 processo, 1 imagem por vez: YOLO → crop → OCR.
- **parallel**: cada um dos N processos (`ProcessPoolExecutor`, contexto `spawn`) carrega seu próprio YOLO + OCR e processa uma fatia completa das imagens. Paraleliza as duas etapas juntas (ao contrário da v10, que paralelizava apenas OCR, deixando YOLO no processo principal como gargalo fixo pela Lei de Amdahl).
- **thread**: experimental — 1 `SharedYOLO` + OCR compartilhados por N threads (`ThreadPoolExecutor`). Em prática mais lento que `parallel` por contenção do GIL e thrashing de tensores de ativação.

### Invariante crítico: warmup fora do cronômetro

`executor._warmup_pool()` usa um `multiprocessing.Barrier(n_workers + 1)` passado via `initargs` para bloquear até que todos os N processos terminem spawn + carregamento dos modelos. O cronômetro do benchmark só começa depois que esse barrier é cruzado, garantindo comparação justa entre serial e parallel.

### Monkeypatch de threads ORT (pipeline.py)

`_patch_onnxruntime_threads(n)` substitui `ort.InferenceSession` no namespace do processo por uma subclasse que força `intra_op_num_threads=n` e desativa `allow_spinning`. É necessário porque `ultralytics.YOLO` não expõe `SessionOptions`. Aplicado em cada processo worker antes de carregar qualquer modelo.

### Fluxo de dados

```
main.py
  → resolve_execution() → run_warmup()
  → executor.run_tasks()
      serial: process_image_serial() por imagem
      parallel: ProcessPoolExecutor
                  init_full_pipeline_process() [1x por processo]
                    → _patch_onnxruntime_threads()
                    → YOLO(model_path) + make_ocr_engine()
                    → warmup dummy + barrier.wait()
                  process_image_batch(chunk) [N vezes por processo]
                    → prefetch I/O em ThreadPoolExecutor(1)
                    → YOLO(mini_batch, verbose=False)
                    → extract_best_crop() + read_plate_text()
```

Resultados são inseridos por índice global (`all_results[offset + i]`) — correspondência posicional, sem índice embutido em cada payload.

### Parâmetros de tuning (src/config.py)

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `YOLO_BATCH_SIZE` | 32 | Mini-lote YOLO por processo. Limitado por `MAX_TOTAL_INFLIGHT_IMAGES // n_workers`. |
| `MAX_TOTAL_INFLIGHT_IMAGES` | 384 | Cap de imagens decodificadas em RAM simultaneamente em todo o sistema. |
| `CHUNK_TARGET_IMAGES` | 100 | Tamanho do lote IPC enviado a cada future do pool. |
| `ORT_THREADS_PER_WORKER` | 1 | Threads ORT intra-op por worker. Manter em 1 para curvas de speedup limpas. |

### INT8 YOLO (tools/quantize_static.py)

Modelo FP32 (~11.8 MB) → INT8 estático calibrado (~3.4 MB). Detalhes importantes:
- Shape inference pulada (causa OOM em YOLO com shapes dinâmicos)
- Calibração Entropy (melhor que MinMax para distribuições sigmoid do detection head)
- Nós de saída e seus predecessores imediatos excluídos da quantização (scores de confiança são sensíveis a erros de range)
- Com 4 workers: 4 × 3.4 MB ≈ L3 de 12 MB → speedup próximo do ideal

### Seleção automática de modelo OCR (src/config.py)

Prioridade: `cct_xs_v2_global_int8.onnx` (1.06 MB) > `cct_s_v2_global_int8.onnx` (1.7 MB) > modelo online padrão. Os modelos INT8 do OCR são gerados com `onnxruntime.quantization.quantize_dynamic` e ficam em `models/` (ignorados pelo git — apenas localmente).

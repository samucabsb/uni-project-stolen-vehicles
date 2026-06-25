"""
config.py — Configurações centrais do projeto.

Todos os caminhos, constantes e parâmetros ajustáveis estão aqui.
Para mudar o comportamento do sistema, edite apenas este arquivo.
"""

from pathlib import Path


# ── Versão ────────────────────────────────────────────────────────────────────

VERSION = "11.3"


# ── Diretórios principais ─────────────────────────────────────────────────────

BASE_DIR         = Path(__file__).resolve().parent.parent
DATA_DIR         = BASE_DIR / "data"
INPUT_DIR        = DATA_DIR / "input"
OUTPUT_DIR       = DATA_DIR / "output"
CROPS_DIR        = OUTPUT_DIR / "crops"
PREPROCESSED_DIR = OUTPUT_DIR / "preprocessed"
MODELS_DIR       = BASE_DIR / "models"


# ── Arquivos ──────────────────────────────────────────────────────────────────

DEFAULT_YOLO_MODEL = MODELS_DIR / "license_plate_detector.pt"
STOLEN_PLATES_FILE = DATA_DIR  / "stolen_plates.txt"


# ── Extensões de imagem aceitas ───────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Parâmetros do detector YOLO ───────────────────────────────────────────────

# Expansão do bounding box antes do crop. 8% cobre casos em que o YOLO aperta
# a caixa e corta dígitos nas bordas da placa.
BBOX_PADDING_RATIO = 0.08

# Nº MÁXIMO de imagens por chamada de inferência YOLO (mini-lote interno,
# dentro de cada processo do modo parallel). Lotes maiores amortizam melhor
# o overhead fixo por chamada Python/ONNX, mas cada imagem do lote fica
# decodificada em memória simultaneamente.
#
# IMPORTANTE: este é o teto por processo, não o valor usado diretamente. O
# valor REAL usado é min(YOLO_BATCH_SIZE, MAX_TOTAL_INFLIGHT_IMAGES //
# n_workers) — ver executor._effective_yolo_batch_size. Sem esse limite,
# "imagens em memória ao mesmo tempo" cresce como YOLO_BATCH_SIZE × N
# workers: com 32 + 12 processos isso são ~384 imagens decodificadas
# simultaneamente, o tipo de pico de memória que derruba o processo (e,
# como um worker morto quebra o ProcessPoolExecutor inteiro, corrompe TODO
# o restante do run com erros em cascata). O cap mantém esse total
# limitado não importa quantos workers você peça.
YOLO_BATCH_SIZE = 32

# Teto do nº de imagens decodificadas em memória SIMULTANEAMENTE somando
# TODOS os processos do pool (ver YOLO_BATCH_SIZE acima).
#
# v11.3 — elevado de 96 para 384: com o modelo INT8 (~3 MB), a pressão de
# memória por imagem é o fator limitante, não o modelo em si. Lotes maiores
# amortizam o overhead fixo de chamada Python/ONNX (estimado em ~5ms/call):
#   batch=12 → 0.42 ms/img de overhead de chamada
#   batch=32 → 0.16 ms/img
# Com 12 workers × batch=32: 384 imagens decodificadas simultaneamente.
# Estimativa de RAM: 384 × ~1 MB/img BGR = ~380 MB — seguro com 4+ GB livres.
# Se RAM disponível for menor que 3 GB, use MAX_TOTAL_INFLIGHT_IMAGES = 128.
MAX_TOTAL_INFLIGHT_IMAGES = 384


# ── Paralelismo / particionamento de tarefas ──────────────────────────────────

# Nº alvo de imagens por lote enviado a cada processo do pool (ver
# executor._chunk_list). Lotes pequenos demais multiplicam o overhead de
# IPC (serialização de cada future); lotes grandes demais reduzem a
# granularidade do balanceamento dinâmico entre processos. 50 é um meio-termo
# para datasets de milhares de imagens — teste 100 se o overhead de IPC ainda
# for visível no perfil (muitos processos, poucos núcleos por processo).
CHUNK_TARGET_IMAGES = 100

# Quando False, os workers pulam a gravação de crop/preprocessed em disco
# (e o pré-processamento CLAHE/Otsu que só existe para gerar esse arquivo).
# Em runs de benchmark, N processos escrevendo 2 JPEGs por imagem detectada
# simultaneamente no mesmo disco é uma fonte real de contenção de I/O que
# cresce com o nº de workers — desligar isso isola o custo de CPU puro
# (YOLO + OCR) do custo de I/O. Ativado via `--no-save-images` / `--benchmark`.
SAVE_INTERMEDIATE_IMAGES = True


# ── Threading das sessões ONNX (YOLO + OCR) por worker ────────────────────────
#
# Threads ORT intra-op (e cv2/torch) usadas dentro de CADA processo worker
# do modo parallel — ver pipeline._patch_onnxruntime_threads e
# pipeline.init_full_pipeline_process.
#
# Existem duas filosofias possíveis aqui, e elas medem coisas DIFERENTES:
#
#   1. FIXO EM 1 (padrão, valor abaixo)
#      Cada worker usa exatamente 1 thread interna, sempre, não importa
#      quantos workers existam. Isolamento limpo de paralelismo por
#      PROCESSO: com 1 worker você vê 1 thread ativa; com N workers, N
#      threads ativas e — se o trabalho for CPU-bound e bem balanceado —
#      speedup próximo de N. É o que permite comparar serial vs. parallel
#      e calcular eficiência (speedup / N) de forma direta, sem outra
#      variável de confusão. Use este modo para o relatório/benchmark
#      acadêmico, gráficos de speedup e qualquer comparação 1→N workers.
#
#   2. ADAPTATIVO ("preencher núcleos ociosos")
#      Com poucos workers, cada um ganha mais threads intra-op
#      (max(1, physical_cores // n_workers)) para não deixar núcleos
#      físicos ociosos. Maximiza a taxa de processamento bruta (imagens/s)
#      quando você só se importa com o tempo total e não com a forma como
#      ele escala — mas mistura paralelismo intra-operação (várias threads
#      acelerando UMA inferência) com paralelismo entre processos (N
#      inferências distintas ao mesmo tempo). Essas duas formas de
#      paralelismo NÃO somam linearmente: o nº total de threads ativas no
#      sistema pode ficar quase constante entre 1 e 2 workers (ex.: 6
#      threads em ambos os casos num CPU de 6 físicos), fazendo o speedup
#      observado entre essas configurações parecer próximo de 1x mesmo
#      com o dobro de processos — o sintoma clássico de "rodei com 1
#      worker e vi todos os núcleos ocupados".
#
# ORT_THREADS_PER_WORKER fixa a opção (1). Para usar a estratégia
# adaptativa (filosofia 2) como experimento separado, veja o parâmetro
# fill_cores em executor.run_tasks/_run_parallel — ele é opt-in e não
# afeta o padrão.
ORT_THREADS_PER_WORKER = 1


# ── Motor OCR ─────────────────────────────────────────────────────────────────

# Modelo CCT pré-treinado em 65+ países via fast-plate-ocr (MIT License).
# O modelo é baixado automaticamente na primeira execução (~3-6 MB, uma vez).
#
# Opções disponíveis:
#   "cct-s-v2-global-model"   → mais preciso  (~0.68ms GPU / ~30-80ms CPU)  ← padrão
#   "cct-xs-v2-global-model"  → mais rápido   (~0.47ms GPU / ~20-50ms CPU)
#
# Para pipelines onde o YOLO é o gargalo dominante (este projeto), a diferença
# de velocidade entre s e xs é negligenciável. Prefira s para melhor precisão.
FAST_OCR_MODEL = "cct-s-v2-global-model"

# Caminho para modelo OCR INT8 quantizado (None = usa o modelo padrão acima).
# Gerado por: python -c "from onnxruntime.quantization import quantize_dynamic..."
# Veja run_blackbox.py --ocr-model para ativar via linha de comando.
# Prefere xs-INT8 (1.06MB) > s-INT8 (1.7MB) > fallback para o modelo online.
# xs é mais rápido E menor → menos pressão de cache L3 em paralelo.
_OCR_XS_INT8  = MODELS_DIR / "cct_xs_v2_global_int8.onnx"
_OCR_XS_CFG   = Path.home() / ".cache/fast-plate-ocr/cct-xs-v2-global-model/cct_xs_v2_global_plate_config.yaml"
_OCR_S_INT8   = MODELS_DIR / "cct_s_v2_global_int8.onnx"
_OCR_S_CFG    = Path.home() / ".cache/fast-plate-ocr/cct-s-v2-global-model/cct_s_v2_global_plate_config.yaml"

if _OCR_XS_INT8.exists() and _OCR_XS_CFG.exists():
    FAST_OCR_MODEL_PATH  = _OCR_XS_INT8
    FAST_OCR_CONFIG_PATH = _OCR_XS_CFG
elif _OCR_S_INT8.exists() and _OCR_S_CFG.exists():
    FAST_OCR_MODEL_PATH  = _OCR_S_INT8
    FAST_OCR_CONFIG_PATH = _OCR_S_CFG
else:
    FAST_OCR_MODEL_PATH  = None
    FAST_OCR_CONFIG_PATH = None

# Comprimento mínimo de placa aceito (caracteres alfanuméricos).
# Leituras com menos caracteres são descartadas como leituras parciais.
OCR_MIN_PLATE_LEN = 4

# Confiança média mínima retornada pelo OCR (0.0 a 1.0).
# Leituras com confiança abaixo deste limiar são descartadas.
OCR_MIN_CONFIDENCE = 0.20


# ── Status possíveis ──────────────────────────────────────────────────────────

STATUS_OK           = "OK"
STATUS_STOLEN       = "ROUBADO"
STATUS_UNIDENTIFIED = "NAO_IDENTIFICADA"
STATUS_ERROR        = "ERRO"


# ── Blacklist de palavras que nunca fazem parte de uma placa ──────────────────
# Pós-filtro aplicado ao resultado do OCR para descartar leituras de letreiros,
# emblemas e textos decorativos no entorno da placa.
# O filtro compara a leitura completa (normalizada, sem espaços) contra as
# entradas desta lista.

WORD_BLACKLIST: frozenset = frozenset({
    # EUA — estados e termos institucionais
    "WASHINGTON", "EVERGREEN", "STATE", "COUNTY", "DISTRICT",
    "CALIFORNIA", "TEXAS", "FLORIDA", "NEVADA", "ARIZONA",
    "OREGON", "COLORADO", "UTAH", "MONTANA", "IDAHO",
    # México
    "CDMX", "MEXICO", "JALISCO", "VERACRUZ", "OAXACA",
    # Brasil
    "BRASIL", "BRAZIL",
    # Órgãos / textos institucionais
    "POLICIA", "POLICE", "TRANSIT", "GOVERNMENT", "FEDERAL",
    "REPUBLIC", "NACIONAL", "MUNICIPAL", "ESTADUAL", "OFFICIAL",
    "UNITED", "STATES", "AMERICA",
    # Marcas e decorações comuns
    "BUS", "CITY", "HYBRID", "ELECTRIC", "GENERATION",
    # Artigos e preposições (nunca são placas)
    "THE", "AND", "FOR",
})
"""
config.py — Configurações centrais do projeto.

Todos os caminhos, constantes e parâmetros ajustáveis estão aqui.
Para mudar o comportamento do sistema, edite apenas este arquivo.
"""

from pathlib import Path


# ── Versão ────────────────────────────────────────────────────────────────────

VERSION = "10.0"


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

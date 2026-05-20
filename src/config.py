"""
config.py — Configurações centrais do projeto.

Todos os caminhos, constantes e parâmetros ajustáveis ficam aqui.
"""

from pathlib import Path


# ── Versão ────────────────────────────────────────────────────────────────────

VERSION = "9.0"


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
STOLEN_PLATES_FILE = DATA_DIR / "stolen_plates.txt"


# ── Extensões aceitas ─────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Parâmetros do detector ────────────────────────────────────────────────────

# Expansão do bounding box do YOLO antes do crop. O YOLO às vezes detecta a
# placa com a caixa apertada, cortando dígitos das bordas. 8% costuma ser o
# suficiente sem incluir muito fundo na imagem.
BBOX_PADDING_RATIO = 0.08


# ── Parâmetros do OCR ─────────────────────────────────────────────────────────

# Score mínimo para considerar uma leitura boa o suficiente e pular as
# variantes restantes (early exit). Reduzido de 10.0 → 7.5:
#   Score 7.5 = letras+dígitos (+4) + tamanho certo (+2) + conf ≥ 0.5 (+1.5)
#   É uma leitura confiável sem exigir quase-perfeição.
OCR_EARLY_EXIT_SCORE = 7.5

# Altura mínima de uma variante de crop para considerá-la viável para OCR.
OCR_MIN_VARIANT_HEIGHT = 10

# Largura máxima do crop antes de enviar ao OCR (px).
# Crops mais largos são redimensionados via INTER_AREA antes da inferência.
# O RapidOCR processa imagens largas mais devagar sem ganho de precisão
# para texto estruturado como placas.
OCR_MAX_CROP_WIDTH = 320

# Score mínimo na variante 1 para não pular a variante 2.
# Se variante 1 retornar score abaixo deste valor (quase sem texto detectado),
# pula direto para variante 3 (metade inferior), que tende a ser mais útil
# para placas de duas linhas do que a variante 2 (sem topo decorativo).
OCR_SKIP_VARIANT2_SCORE = 2.0


# ── Status possíveis ──────────────────────────────────────────────────────────

STATUS_OK              = "OK"
STATUS_STOLEN          = "ROUBADO"
STATUS_UNIDENTIFIED    = "NAO_IDENTIFICADA"
STATUS_ERROR           = "ERRO"


# ── Blacklist de palavras que nunca são parte de uma placa ────────────────────
# Nomes de estado/cidade/marca/slogans que o OCR pode capturar de letreiros
# no entorno. Genérico para múltiplos países.

WORD_BLACKLIST = {
    # EUA / América do Norte
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
    # Marcas / modelos / decorações comuns
    "BUS", "CITY", "HYBRID", "PRIUS", "GENERATION", "ELECTRIC",
    # Palavras curtas que nunca são placas
    "THE", "AND", "FOR",
}

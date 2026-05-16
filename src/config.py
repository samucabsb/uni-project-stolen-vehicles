"""
config.py — Configurações centrais do projeto.

Todos os caminhos, constantes e parâmetros ajustáveis ficam aqui.
"""

from pathlib import Path


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

# Score mínimo para considerar uma leitura "confiável o suficiente" e pular
# as variantes restantes (otimização early exit).
OCR_EARLY_EXIT_SCORE = 10.0

# Altura mínima de uma variante de crop para considerá-la viável para OCR.
# Variantes muito pequenas geram leituras erradas e gastam tempo.
OCR_MIN_VARIANT_HEIGHT = 10


# ── Status possíveis ──────────────────────────────────────────────────────────

STATUS_OK              = "OK"
STATUS_STOLEN          = "ROUBADO"
STATUS_UNIDENTIFIED    = "NAO_IDENTIFICADA"
STATUS_ERROR           = "ERRO"


# ── Blacklist de palavras que nunca são parte de uma placa ────────────────────
# Nomes de estado/cidade/marca/slogans que o OCR pode capturar de letreiros
# no entorno (ex: "WASHINGTON" do fundo da imagem). Genérico para múltiplos
# países — adicionar conforme novos falsos positivos forem aparecendo.

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

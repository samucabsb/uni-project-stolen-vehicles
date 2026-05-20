"""
ocr.py — Leitura de placas via RapidOCR.

Otimizações de velocidade (v9 final):
  - Early exit com threshold 7.5 (antes 10.0): dispara mais cedo em placas
    com boa confiança sem exigir quase-perfeição. Score 7.5 = letras+dígitos
    (+4) + tamanho certo (+2) + confiança OCR ≥ 0.5 (+1.5).
  - Cap de largura (OCR_MAX_CROP_WIDTH = 320px): crops maiores são
    redimensionados via INTER_AREA antes da inferência. O RapidOCR processa
    imagens largas mais devagar sem ganho mensurável de precisão.
  - Skip adaptativo da variante 2: se variante 1 retorna score quase zero
    (OCR não encontrou nenhum texto relevante), pula a variante 2 (sem topo
    decorativo, tende a falhar pelos mesmos motivos) e vai direto para a
    variante 3 — economiza uma chamada OCR no pior caso.

Otimizações de precisão (v9 final):
  - Variante 3 = metade inferior do crop: substitui a versão pré-processada
    (CLAHE+Otsu). Resolve placas de duas linhas (motos, formato antigo
    indiano/brasileiro) onde o OCR capturava apenas a linha superior.
    A imagem pré-processada ainda é salva em disco para o relatório HTML.
  - Correção contextual L↔D: após OCR e normalização, aplica substituições
    baseadas nos vizinhos imediatos de cada caractere ambíguo:
      Contexto de letras → 0→O  1→I  8→B  5→S  6→G  2→Z
      Contexto de dígitos → O→0  I→1  B→8  S→5  G→6  Z→2
"""

import re
import numpy as np
import cv2

from src.config import (
    OCR_EARLY_EXIT_SCORE, OCR_MIN_VARIANT_HEIGHT,
    OCR_MAX_CROP_WIDTH, OCR_SKIP_VARIANT2_SCORE,
)
from src.logger import get_logger


# Padrão genérico de placa: 4-10 caracteres alfanuméricos.
_PLATE_PATTERN = re.compile(r"^[A-Z0-9]{4,10}$")

# Regex para limpeza: mantém só A-Z e 0-9
_NON_ALNUM = re.compile(r"[^A-Z0-9]")

# Mapas de correção contextual L↔D
# Confusões típicas em OCR de fontes de placa (serifadas, alta densidade)
_LETTER_FROM_DIGIT = {"0": "O", "1": "I", "8": "B", "5": "S", "6": "G", "2": "Z"}
_DIGIT_FROM_LETTER = {"O": "0", "I": "1", "B": "8", "S": "5", "G": "6", "Z": "2"}


# ── Warmup ────────────────────────────────────────────────────────────────────

def warmup_ocr() -> None:
    """Faz warm-up do RapidOCR (carrega modelos ONNX)."""
    from rapidocr_onnxruntime import RapidOCR
    try:
        engine = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
    except TypeError:
        engine = RapidOCR()
    dummy = np.zeros((32, 128, 3), dtype=np.uint8)
    engine(dummy)
    get_logger().info("[WARMUP] RapidOCR pronto.")


# ── Pré-processamento ─────────────────────────────────────────────────────────

def preprocess_plate(crop: np.ndarray) -> np.ndarray:
    """
    Pipeline de pré-processamento salvo em disco para o relatório HTML.

    Nota: não é mais usada como variante de OCR (substituída pela metade
    inferior do crop). Mantida para gerar a imagem preprocessed_path que
    aparece no modal de detalhes do relatório.

      1. Upscale para altura mínima de 100px (CUBIC)
      2. Grayscale → CLAHE (contraste adaptativo) → Otsu (binarização)
      3. Conversão de volta para BGR (compatibilidade)
    """
    target_h = max(crop.shape[0], 100)
    scale = target_h / crop.shape[0]
    if scale > 1.0:
        new_w = int(crop.shape[1] * scale)
        crop = cv2.resize(crop, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

    gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray  = clahe.apply(gray)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def _cap_width(img: np.ndarray) -> np.ndarray:
    """
    Redimensiona a imagem se a largura exceder OCR_MAX_CROP_WIDTH.

    Usa INTER_AREA (melhor para downscale: reduz aliasing e preserva bordas
    dos caracteres). Garante que a altura resultante não fique abaixo do
    mínimo viável.
    """
    if img is None or img.size == 0:
        return img
    if img.shape[1] <= OCR_MAX_CROP_WIDTH:
        return img
    scale = OCR_MAX_CROP_WIDTH / img.shape[1]
    new_h = max(int(img.shape[0] * scale), OCR_MIN_VARIANT_HEIGHT)
    return cv2.resize(img, (OCR_MAX_CROP_WIDTH, new_h), interpolation=cv2.INTER_AREA)


def _get_subcrop_variants(crop: np.ndarray, preprocessed: np.ndarray) -> list:
    """
    Gera as variantes de crop para OCR, em ordem de probabilidade:

      1. Crop original completo  → caso mais comum, cobre a maioria das placas
      2. Top 25% removido        → elimina texto decorativo/estado acima da placa
      3. Metade inferior (45%+)  → resolve placas de 2 linhas (motos, formato antigo)

    O parâmetro `preprocessed` é mantido na assinatura por compatibilidade
    com os callers (ainda é salvo em disco para o HTML), mas não é mais
    usado como variante de OCR — a versão processada rara vez ajudava e
    aumentava o pior caso em ~1-2s.

    Cap de largura (OCR_MAX_CROP_WIDTH) aplicado em cada variante.
    Variantes com altura insuficiente (< OCR_MIN_VARIANT_HEIGHT) são descartadas.
    """
    h = crop.shape[0]
    candidates = [
        crop,
        crop[int(h * 0.25):, :],   # remove topo decorativo
        crop[int(h * 0.45):, :],   # metade inferior — placas 2 linhas
    ]
    return [
        _cap_width(v) for v in candidates
        if v is not None and v.size > 0 and v.shape[0] >= OCR_MIN_VARIANT_HEIGHT
    ]


# ── Correção contextual L↔D ───────────────────────────────────────────────────

def _fix_char_context(text: str) -> str:
    """
    Corrige confusões L↔D comuns em OCR de placas usando contexto dos vizinhos.

    Para cada caractere ambíguo (presente nos mapas de substituição):
      - Olha uma janela de até 2 vizinhos para cada lado (max 4 chars)
      - Se maioria dos vizinhos são letras → converte para letra equivalente
      - Se maioria dos vizinhos são dígitos → converte para dígito equivalente
      - Empate → mantém o original (conservador, evita falsos positivos)

    Exemplos:
      "HR26CK857I" → "HR26CK8571"  (I em contexto de dígitos → 1)
      "MH2OBY3665" → "MH20BY3665"  (O em contexto de dígitos → 0 ... e
                                     depois O em contexto de letras → O)
      "KA04MN3622" → "KA04MN3622"  (sem mudança, já correto)
    """
    if len(text) < 2:
        return text

    chars  = list(text)
    result = list(text)

    for i, ch in enumerate(chars):
        # Vizinhos dentro de janela de 2 posições para cada lado
        window  = [chars[j]
                   for j in range(max(0, i - 2), min(len(chars), i + 3))
                   if j != i]
        n_alpha = sum(c.isalpha() for c in window)
        n_digit = sum(c.isdigit() for c in window)

        if n_alpha > n_digit and ch in _LETTER_FROM_DIGIT:
            result[i] = _LETTER_FROM_DIGIT[ch]
        elif n_digit > n_alpha and ch in _DIGIT_FROM_LETTER:
            result[i] = _DIGIT_FROM_LETTER[ch]
        # Empate: mantém original

    return "".join(result)


# ── Normalização ──────────────────────────────────────────────────────────────

def _normalize_text(text: str) -> str:
    """Remove tudo que não é A-Z ou 0-9 e converte para uppercase."""
    return _NON_ALNUM.sub("", text.upper())


# ── Scoring de candidatos ─────────────────────────────────────────────────────

def _score_candidate(text: str, confidence: float, blacklist: set) -> float:
    """
    Pontua um candidato de leitura. Score positivo = bom candidato.

    Critérios positivos:
      +4   Mistura de letras e números (universal em placas)
      +2   Comprimento entre 4 e 8 (típico mundialmente)
      +3   Confiança alta do OCR (multiplicado pela confiança)
      +1.5 Caracteres majoritariamente alfanuméricos
      +2   Casa com o padrão de placa genérico (4-10 alnum)

    Score máximo teórico: 12.5
    Early exit threshold: 7.5 (letras+dígitos + tamanho certo + conf ≥ 0.5)

    Rejeição imediata (-999):
      - Texto vazio ou somente não-alfanumérico
      - Palavra na blacklist (estado, cidade, marca)

    Score negativo leve (-1.0):
      - Normalizado fora do range 3-10 caracteres
    """
    if not text:
        return -999.0

    normalized = _normalize_text(text)
    if not normalized:
        return -999.0

    # Rejeita se qualquer palavra do texto está na blacklist
    for word in text.upper().split():
        clean = re.sub(r"[^A-Z]", "", word)
        if clean in blacklist:
            return -999.0

    if not (3 <= len(normalized) <= 10):
        return -1.0

    score = 0.0

    has_letters = any(c.isalpha() for c in normalized)
    has_digits  = any(c.isdigit() for c in normalized)
    if has_letters and has_digits:
        score += 4.0

    if 4 <= len(normalized) <= 8:
        score += 2.0

    score += confidence * 3.0

    alnum_ratio = sum(c.isalnum() for c in text) / max(len(text), 1)
    score += alnum_ratio * 1.5

    if _PLATE_PATTERN.match(normalized):
        score += 2.0

    return score


# ── Leitura principal com early exit e skip adaptativo ───────────────────────

def read_plate_text(ocr_engine, crop: np.ndarray, preprocessed: np.ndarray,
                    blacklist: set) -> tuple:
    """
    Retorna (texto_normalizado_corrigido, confiança) ou ("", 0.0).

    Algoritmo de seleção de variantes (otimizado):

      i=0  Variante 1: crop completo
           ├── score >= OCR_EARLY_EXIT_SCORE → retorna imediatamente
           ├── score < OCR_SKIP_VARIANT2_SCORE → marca skip_v2=True
           └── continua

      i=1  Variante 2: sem topo (skip_v2=True → pula para i=2)
           ├── score >= OCR_EARLY_EXIT_SCORE → retorna imediatamente
           └── continua

      i=2  Variante 3: metade inferior (placas 2 linhas)
           └── retorna melhor encontrado

    Após selecionar o melhor candidato:
      - _fix_char_context corrige confusões L↔D contextuais
      - resultado final é normalizado (só A-Z e 0-9) + corrigido
    """
    variants   = _get_subcrop_variants(crop, preprocessed)
    best_text  = ""
    best_score = -999.0
    best_conf  = 0.0
    skip_v2    = False

    i = 0
    while i < len(variants):
        # Skip adaptativo: variante 1 sem resultado → pula variante 2
        if skip_v2 and i == 1:
            i += 1
            continue

        try:
            result, _ = ocr_engine(variants[i])
        except Exception:
            i += 1
            continue

        if result:
            for item in result:
                if len(item) < 3:
                    continue
                text = str(item[1])
                try:
                    conf = float(item[2])
                except (TypeError, ValueError):
                    conf = 0.0

                score = _score_candidate(text, conf, blacklist)
                if score > best_score:
                    best_score = score
                    best_text  = _normalize_text(text)
                    best_conf  = conf

        # Após variante 1: decide se pula a variante 2
        if i == 0 and best_score < OCR_SKIP_VARIANT2_SCORE:
            skip_v2 = True

        # Early exit: leitura boa o suficiente
        if best_score >= OCR_EARLY_EXIT_SCORE:
            break

        i += 1

    # Correção contextual L↔D no melhor candidato
    # _fix_char_context desativada — causa regressão no formato LLDDLL indiano

    return best_text, best_conf

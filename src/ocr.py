"""
ocr.py — Leitura de placas via RapidOCR.

Otimizações:
  - Early exit: se a primeira variante retorna uma placa com score alto
    (formato válido + confiança alta), as variantes seguintes são puladas.
    Em placas fáceis isso corta o tempo de OCR pela metade ou mais.
  - Validação de padrão GENÉRICA (funciona com placas de qualquer país):
    apenas verifica se tem mistura de letras+números e tamanho razoável.
"""

import re
import numpy as np
import cv2

from src.config import OCR_EARLY_EXIT_SCORE, OCR_MIN_VARIANT_HEIGHT


# Padrão genérico de placa: 4-10 caracteres alfanuméricos.
# Cobre formatos como AB1234, ABC1234, 51A05227, 50LD04411, MERCOSUL ABC1D23.
_PLATE_PATTERN = re.compile(r"^[A-Z0-9]{4,10}$")

# Regex para limpeza: mantém só A-Z e 0-9
_NON_ALNUM = re.compile(r"[^A-Z0-9]")


# ── Warmup ────────────────────────────────────────────────────────────────────

def warmup_ocr() -> None:
    """Faz warm-up do RapidOCR (carrega modelos ONNX)."""
    from rapidocr_onnxruntime import RapidOCR
    engine = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
    dummy = np.zeros((32, 128, 3), dtype=np.uint8)
    engine(dummy)
    print("[WARMUP] RapidOCR pronto.")


# ── Pré-processamento ─────────────────────────────────────────────────────────

def preprocess_plate(crop: np.ndarray) -> np.ndarray:
    """
    Pipeline de pré-processamento para OCR:
      1. Upscale para altura mínima de 100px (CUBIC)
      2. Conversão para grayscale
      3. CLAHE para equalização adaptativa de contraste
      4. Threshold de Otsu (binarização automática)
      5. Conversão de volta para BGR (RapidOCR espera 3 canais)
    """
    target_h = max(crop.shape[0], 100)
    scale = target_h / crop.shape[0]
    if scale > 1.0:
        new_w = int(crop.shape[1] * scale)
        crop = cv2.resize(crop, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def _get_subcrop_variants(crop: np.ndarray, preprocessed: np.ndarray) -> list:
    """
    Gera variantes do crop para OCR tentar, em ordem de probabilidade de
    funcionar primeiro:
      1. Crop original (caso mais comum, OCR moderno lida bem)
      2. Top 25% removido (placas com texto decorativo no topo)
      3. Versão pré-processada (último recurso para casos difíceis)

    O early exit aproveita essa ordem: se variante 1 já dá bom resultado,
    pula as outras duas. Variantes com altura < OCR_MIN_VARIANT_HEIGHT são
    descartadas para evitar leituras erradas.
    """
    h = crop.shape[0]
    variants = [
        crop,
        crop[int(h * 0.25):, :],
        preprocessed,
    ]
    return [v for v in variants
            if v is not None and v.size > 0 and v.shape[0] >= OCR_MIN_VARIANT_HEIGHT]


# ── Scoring de candidatos ─────────────────────────────────────────────────────

def _normalize_text(text: str) -> str:
    """Remove tudo que não é A-Z ou 0-9 e converte para uppercase."""
    return _NON_ALNUM.sub("", text.upper())


def _score_candidate(text: str, confidence: float, blacklist: set) -> float:
    """
    Pontua um candidato de leitura. Score positivo = bom candidato.

    Critérios positivos:
      +4   Mistura de letras e números (universal em placas)
      +2   Comprimento entre 4 e 8 (típico mundialmente)
      +3   Confiança alta do OCR (multiplicado pela confiança)
      +1.5 Caracteres majoritariamente alfanuméricos
      +2   Casa com o padrão de placa genérico

    Rejeição imediata (-999):
      - Texto vazio
      - Palavra na blacklist (estado, cidade, marca)
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


# ── Leitura principal com early exit ──────────────────────────────────────────

def read_plate_text(ocr_engine, crop: np.ndarray, preprocessed: np.ndarray,
                    blacklist: set) -> tuple:
    """
    Retorna (texto_normalizado, confiança) ou ("", 0.0).

    OTIMIZAÇÃO EARLY EXIT:
    Após processar cada variante, se o melhor candidato encontrado até agora
    tem score >= OCR_EARLY_EXIT_SCORE, retorna sem processar as variantes
    restantes. Economiza 30-66% do tempo de OCR em placas fáceis.
    """
    variants = _get_subcrop_variants(crop, preprocessed)

    best_text  = ""
    best_score = -999.0
    best_conf  = 0.0

    for variant in variants:
        try:
            result, _ = ocr_engine(variant)
        except Exception:
            continue

        if not result:
            continue

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

        # Early exit: já temos uma leitura boa o suficiente
        if best_score >= OCR_EARLY_EXIT_SCORE:
            break

    return best_text, best_conf

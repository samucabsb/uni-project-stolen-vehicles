"""
ocr.py — Leitura de placas via fast-plate-ocr (CCT).

ARQUITETURA
===========
Este módulo mantém um singleton thread-safe do LicensePlateRecognizer.

Por que singleton e não uma instância por thread?
  O ONNX Runtime documenta InferenceSession como thread-safe para chamadas
  concorrentes de Run(). Compartilhar uma instância elimina o overhead de
  carregar o modelo ONNX N vezes (uma por thread), que causava degradação
  visível ao rodar com 4+ threads no modo parallel.

  warmup_ocr() inicializa o singleton e aquece o modelo com uma passagem
  sintética. Todas as threads subsequentes reutilizam a mesma sessão.

POR QUE fast-plate-ocr vs RapidOCR (v9)?
  RapidOCR é um OCR genérico de texto que classifica regiões de forma
  independente. Isso gera confusões L/D (I↔1, O↔0, S↔5) porque cada
  posição é avaliada sem contexto da placa inteira.

  fast-plate-ocr usa CCT (Compact Convolutional Transformer), treinado
  especificamente em placas de 65+ países. O Transformer lê a sequência
  completa: ao processar a posição 2, já "viu" as posições 0-1 e sabe
  que aquela região deve ser dígito. A confusão L/D torna-se improvável.

preprocess_plate() é mantida para gerar a imagem processada salva em disco
e exibida no relatório HTML — não é usada pelo motor OCR.
"""

from __future__ import annotations

import re
import threading

import cv2
import numpy as np

from src.config import (
    FAST_OCR_MODEL, WORD_BLACKLIST,
    OCR_MIN_PLATE_LEN, OCR_MIN_CONFIDENCE,
)
from src.logger import get_logger


# ── Singleton thread-safe ─────────────────────────────────────────────────────

_recognizer      = None
_recognizer_lock = threading.Lock()


def _get_recognizer():
    """
    Retorna o singleton LicensePlateRecognizer, criando-o se necessário.

    Usa double-checked locking para garantir inicialização única mesmo com
    múltiplas threads chamando simultaneamente na primeira vez.
    """
    global _recognizer
    if _recognizer is None:
        with _recognizer_lock:
            if _recognizer is None:
                from fast_plate_ocr import LicensePlateRecognizer
                _recognizer = LicensePlateRecognizer(FAST_OCR_MODEL)
    return _recognizer


def make_ocr_engine():
    """
    Retorna o motor OCR compartilhado (singleton thread-safe).

    Chamado tanto pelo worker serial quanto pelas threads do modo parallel.
    """
    return _get_recognizer()


# ── Warmup ────────────────────────────────────────────────────────────────────

def warmup_ocr() -> None:
    """
    Inicializa o singleton e aquece o modelo com uma passagem sintética.

    Na primeira execução, baixa o modelo do HuggingFace (~3-6 MB) e carrega
    o arquivo ONNX na memória. Execuções subsequentes usam o cache local.

    Deve ser chamada durante o warm-up (antes do pipeline iniciar), tanto
    no modo serial quanto no parallel. Isso garante que a primeira imagem
    real não sofra latência de inicialização.
    """
    recognizer = _get_recognizer()
    dummy = np.zeros((64, 256, 3), dtype=np.uint8)
    recognizer.run(dummy)
    get_logger().info("[WARMUP] fast-plate-ocr (%s) pronto.", FAST_OCR_MODEL)


# ── Pré-processamento (para o relatório HTML) ─────────────────────────────────

def preprocess_plate(crop: np.ndarray) -> np.ndarray:
    """
    Gera versão CLAHE + Otsu do crop para salvar em disco (relatório HTML).

    Não usada pelo motor OCR: fast-plate-ocr realiza seu próprio
    pré-processamento internamente antes da inferência ONNX.
    """
    target_h = max(crop.shape[0], 100)
    scale    = target_h / crop.shape[0]
    if scale > 1.0:
        new_w = int(crop.shape[1] * scale)
        crop  = cv2.resize(crop, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

    gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray  = clahe.apply(gray)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


# ── Leitura principal ─────────────────────────────────────────────────────────

_NON_ALNUM = re.compile(r"[^A-Z0-9]")


def read_plate_text(
    ocr_engine,
    crop: np.ndarray,
    preprocessed: np.ndarray,
    blacklist: frozenset,
) -> tuple[str, float]:
    """
    Lê o texto da placa e retorna (texto, confiança).

    O fast-plate-ocr trata internamente:
      - Redimensionamento para o input do modelo
      - Normalização de pixel (÷255)
      - Remoção do padding char '_' da saída
      - Contexto global da sequência (sem confusão I/1 posição a posição)

    O parâmetro `preprocessed` é mantido na assinatura por compatibilidade
    com os callers — não é passado ao motor OCR.

    Filtros pós-OCR (em ordem):
      1. Erro de inferência → descarta
      2. Texto vazio → descarta
      3. Normalização: mantém apenas A-Z e 0-9, converte para uppercase
      4. OCR_MIN_PLATE_LEN: descarta leituras parciais (< N caracteres)
      5. OCR_MIN_CONFIDENCE: descarta leituras com baixa confiança média
      6. WORD_BLACKLIST: descarta letreiros, slogans e marcas
    """
    try:
        results = ocr_engine.run(crop, return_confidence=True)
    except Exception as exc:
        get_logger().debug("[OCR] Falha na inferência: %s", exc)
        return "", 0.0

    if not results:
        return "", 0.0

    pred = results[0]
    raw  = pred.plate if pred.plate else ""
    text = _NON_ALNUM.sub("", raw.upper())

    if not text:
        return "", 0.0

    # Filtro de comprimento mínimo
    if len(text) < OCR_MIN_PLATE_LEN:
        return "", 0.0

    # Extrai confiança média dos caracteres reconhecidos.
    # fast-plate-ocr usa `char_probs` (List[float], um valor por posição de caractere).
    # Se não disponível (campo ausente ou vazio), assume 1.0 — aceita a leitura.
    confidence = 1.0
    char_probs = getattr(pred, "char_probs", None)
    if char_probs is not None and len(char_probs) > 0:
        try:
            confidence = float(sum(char_probs)) / len(char_probs)
        except (TypeError, ValueError):
            confidence = 1.0

    if confidence < OCR_MIN_CONFIDENCE:
        return "", 0.0

    # Filtro blacklist — descarta leituras que casam com termos na lista
    if text in blacklist:
        return "", 0.0

    return text, confidence

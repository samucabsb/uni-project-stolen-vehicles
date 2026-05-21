"""
detector.py — Detecção de placas via YOLO + export ONNX + recorte com padding.

v10: O modelo YOLO é automaticamente exportado para ONNX na primeira execução.
O ONNX via onnxruntime é 2-3x mais rápido que o PyTorch em CPU, reduzindo o
gargalo do estágio 1 e melhorando o speedup do paralelismo no estágio 2.

Fluxo:
  1ª execução : .pt → exporta → .onnx  (~30-60s, feito UMA VEZ)
  Execuções   : detecta .onnx existente → carrega direto (sem re-exportar)
  Fallback    : se export falhar → usa .pt com aviso
"""

from pathlib import Path

import numpy as np
import cv2

from src.config import BBOX_PADDING_RATIO
from src.logger import get_logger


# ── Export ONNX ───────────────────────────────────────────────────────────────

def ensure_onnx_export(model_path: str) -> str:
    """
    Garante que o modelo YOLO está disponível em formato ONNX.

    Se `model_path` já for um .onnx, retorna como está.
    Se for um .pt e o .onnx correspondente ainda não existir, exporta.
    Se o .onnx já existir, reutiliza sem re-exportar.
    Se a exportação falhar, retorna o .pt original como fallback.

    Parâmetros de export:
      format='onnx'    — runtime via onnxruntime (já instalado com fast-plate-ocr)
      dynamic=True     — suporte a batch variable (1, 8, 16... imagens)
      simplify=True    — onnx-simplifier: remove ops redundantes, grafo mais enxuto
      opset=12         — compatibilidade máxima com onnxruntime >= 1.10
    """
    log  = get_logger()
    path = Path(model_path)

    # Já é ONNX — usa direto
    if path.suffix.lower() == ".onnx":
        log.info("[YOLO] Modelo já em ONNX: %s", path.name)
        return model_path

    onnx_path = path.with_suffix(".onnx")

    # ONNX já exportado em execução anterior
    if onnx_path.exists():
        log.info("[YOLO] ONNX em cache: %s", onnx_path.name)
        return str(onnx_path)

    # Primeira vez: exporta
    log.info("[YOLO] Exportando para ONNX (feito UMA VEZ, ~30-60s)...")
    try:
        from ultralytics import YOLO
        model = YOLO(str(path))
        model.export(
            format="onnx",
            dynamic=True,    # batch variável: funciona com lotes de 1, 8, etc.
            simplify=True,   # onnx-simplifier: grafo mais enxuto e rápido
            opset=12,        # compatibilidade com onnxruntime >= 1.10
        )
        log.info("[YOLO] Export concluído: %s", onnx_path.name)
        return str(onnx_path)

    except Exception as exc:
        log.warning(
            "[YOLO] Export ONNX falhou: %s — usando .pt como fallback.", exc
        )
        return model_path


# ── Warmup ────────────────────────────────────────────────────────────────────

def warmup_yolo(model_path: str) -> None:
    """Faz warm-up do YOLO com uma imagem sintética pequena."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    model(dummy, verbose=False)
    backend = "ONNX" if model_path.endswith(".onnx") else "PyTorch"
    get_logger().info("[WARMUP] YOLO (%s) pronto.", backend)


# ── BBox helpers ──────────────────────────────────────────────────────────────

def _expand_bbox(x1: int, y1: int, x2: int, y2: int,
                 img_w: int, img_h: int, ratio: float) -> tuple:
    """
    Expande o bounding box em `ratio` para cada lado, respeitando os limites
    da imagem. Útil porque o YOLO tende a apertar a caixa demais e cortar
    dígitos das bordas da placa.
    """
    bw    = x2 - x1
    bh    = y2 - y1
    pad_x = int(bw * ratio)
    pad_y = int(bh * ratio)

    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(img_w, x2 + pad_x),
        min(img_h, y2 + pad_y),
    )


def extract_best_crop(img: np.ndarray, detections) -> np.ndarray:
    """
    Extrai o recorte da melhor detecção YOLO (maior confiança), com padding.

    Retorna None se nenhuma placa for detectada.
    Limitação conhecida: considera apenas uma placa por imagem.
    """
    h, w      = img.shape[:2]
    best_crop = None
    best_conf = -1.0

    for result in detections:
        if result.boxes is None:
            continue
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf <= best_conf:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            x1, y1, x2, y2 = _expand_bbox(x1, y1, x2, y2, w, h, BBOX_PADDING_RATIO)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            best_conf = conf
            best_crop = crop

    return best_crop

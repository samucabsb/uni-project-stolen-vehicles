"""
detector.py — Detecção de placas via YOLO + recorte com padding.
"""

import numpy as np
import cv2

from src.config import BBOX_PADDING_RATIO
from src.logger import get_logger


def warmup_yolo(model_path: str) -> None:
    """Faz warm-up do YOLO com uma imagem sintética pequena."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    model(dummy, verbose=False)
    get_logger().info("[WARMUP] YOLO pronto.")


def _expand_bbox(x1: int, y1: int, x2: int, y2: int,
                 img_w: int, img_h: int, ratio: float) -> tuple:
    """
    Expande o bounding box em `ratio` para cada lado, respeitando os limites
    da imagem. Útil porque o YOLO tende a apertar a caixa demais e cortar
    dígitos das bordas da placa.
    """
    bw = x2 - x1
    bh = y2 - y1
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
    h, w = img.shape[:2]
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

"""
pipeline.py — Workers de processamento.

Dois tipos de worker:

1. init_worker / process_image_task
   Carrega YOLO + OCR. Usado pelo modo 'parallel' original (mantido para
   comparação no relatório acadêmico).

2. init_ocr_worker / process_ocr_task
   Carrega APENAS OCR. Usado pelo modo 'pipeline' (v8), onde o YOLO roda
   no processo principal em batch e só o OCR é paralelizado.
   Benefício: metade da RAM por worker, zero contenção de cache do YOLO.

ORDEM CRÍTICA: force_single_thread_env() ANTES de qualquer import pesado.
"""

import os
import time
from pathlib import Path

from src.runtime import force_single_thread_env, apply_library_thread_limits


# ── Globais por processo ──────────────────────────────────────────────────────

_yolo = None   # usado pelo modo parallel
_ocr  = None   # usado pelos dois modos


# ── Helpers comuns ────────────────────────────────────────────────────────────

def _make_ocr_engine():
    """Cria RapidOCR com try/except para compatibilidade entre versões."""
    from rapidocr_onnxruntime import RapidOCR
    try:
        return RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
    except TypeError:
        return RapidOCR()


def _new_result(image_name: str) -> dict:
    return {
        "image":             image_name,
        "plate_detected":    False,
        "plate_text":        "",
        "status":            "NAO_IDENTIFICADA",
        "yolo_time_s":       0.0,
        "ocr_time_s":        0.0,
        "total_time_s":      0.0,
        "ocr_confidence":    0.0,
        "crop_path":         "",
        "preprocessed_path": "",
        "worker_pid":        os.getpid(),
        "error":             "",
    }


# ── Modo PARALLEL — worker carrega YOLO + OCR ─────────────────────────────────

def init_worker(yolo_model_path: str) -> None:
    """Worker completo (YOLO + OCR). Usado pelo modo 'parallel'."""
    global _yolo, _ocr
    force_single_thread_env()
    apply_library_thread_limits()

    from ultralytics import YOLO
    _yolo = YOLO(yolo_model_path)
    _ocr  = _make_ocr_engine()


def process_image_task(task: dict) -> dict:
    """Processa uma imagem completa: YOLO → crop → OCR. Modo 'parallel'."""
    import cv2
    from src.config import CROPS_DIR, PREPROCESSED_DIR, WORD_BLACKLIST, STATUS_OK, STATUS_STOLEN
    from src.detector import extract_best_crop
    from src.ocr import preprocess_plate, read_plate_text

    image_path    = task["image_path"]
    stolen_plates = task["stolen_plates"]
    stem          = Path(image_path).stem
    result        = _new_result(Path(image_path).name)
    t0            = time.perf_counter()

    img = cv2.imread(image_path)
    if img is None:
        result["error"] = "Não foi possível carregar a imagem."
        result["total_time_s"] = round(time.perf_counter() - t0, 6)
        return result

    t_yolo = time.perf_counter()
    try:
        detections = _yolo(img, verbose=False)
    except Exception as exc:
        result["error"] = f"YOLO erro: {exc}"
        result["total_time_s"] = round(time.perf_counter() - t0, 6)
        return result
    result["yolo_time_s"] = round(time.perf_counter() - t_yolo, 6)

    crop = extract_best_crop(img, detections)
    if crop is None:
        result["total_time_s"] = round(time.perf_counter() - t0, 6)
        return result

    result["plate_detected"] = True
    crop_path = CROPS_DIR / f"{stem}_crop.jpg"
    cv2.imwrite(str(crop_path), crop)
    result["crop_path"] = str(crop_path)

    preprocessed = preprocess_plate(crop)
    prep_path = PREPROCESSED_DIR / f"{stem}_prep.jpg"
    cv2.imwrite(str(prep_path), preprocessed)
    result["preprocessed_path"] = str(prep_path)

    t_ocr = time.perf_counter()
    try:
        plate_text, confidence = read_plate_text(_ocr, crop, preprocessed, WORD_BLACKLIST)
    except Exception as exc:
        result["error"] = f"OCR erro: {exc}"
        result["total_time_s"] = round(time.perf_counter() - t0, 6)
        return result
    result["ocr_time_s"]     = round(time.perf_counter() - t_ocr, 6)
    result["plate_text"]     = plate_text
    result["ocr_confidence"] = round(confidence, 4)

    if plate_text:
        result["status"] = STATUS_STOLEN if plate_text in stolen_plates else STATUS_OK

    result["total_time_s"] = round(time.perf_counter() - t0, 6)
    return result


# ── Modo PIPELINE — worker carrega APENAS OCR ─────────────────────────────────

def init_ocr_worker() -> None:
    """
    Worker leve: carrega APENAS RapidOCR, sem YOLO.

    Usado pelo modo 'pipeline' (v8), onde o YOLO já rodou no processo
    principal antes de spawnar esses workers.

    Vantagens sobre o init_worker completo:
      - ~200MB de RAM a menos por worker (sem YOLO)
      - Cache L3 não precisa acomodar pesos do YOLO
      - Mais workers cabem na mesma RAM disponível
    """
    global _ocr
    force_single_thread_env()
    apply_library_thread_limits()
    _ocr = _make_ocr_engine()


def process_ocr_task(task: dict) -> dict:
    """
    Executa apenas OCR num crop já existente em disco.

    Entrada esperada (gerada pelo estágio YOLO no processo principal):
      task["crop_path"]         → path do crop salvo
      task["preprocessed_path"] → path do crop pré-processado
      task["result_base"]       → dict parcial com dados do YOLO
      task["stolen_plates"]     → set de placas roubadas
    """
    import cv2
    from src.config import WORD_BLACKLIST, STATUS_OK, STATUS_STOLEN
    from src.ocr import read_plate_text

    result        = task["result_base"].copy()
    stolen_plates = task["stolen_plates"]
    t0            = time.perf_counter()

    # Se YOLO não detectou placa, retorna imediatamente
    if not result.get("plate_detected"):
        return result

    crop_path = task.get("crop_path", "")
    prep_path = task.get("preprocessed_path", "")

    crop = cv2.imread(crop_path) if crop_path else None
    preprocessed = cv2.imread(prep_path) if prep_path else None

    if crop is None:
        result["error"]       = "Crop não encontrado para OCR."
        result["total_time_s"] = round(
            result.get("yolo_time_s", 0) + (time.perf_counter() - t0), 6
        )
        return result

    t_ocr = time.perf_counter()
    try:
        plate_text, confidence = read_plate_text(_ocr, crop, preprocessed, WORD_BLACKLIST)
    except Exception as exc:
        result["error"]       = f"OCR erro: {exc}"
        result["total_time_s"] = round(
            result.get("yolo_time_s", 0) + (time.perf_counter() - t0), 6
        )
        return result

    result["ocr_time_s"]     = round(time.perf_counter() - t_ocr, 6)
    result["plate_text"]     = plate_text
    result["ocr_confidence"] = round(confidence, 4)
    result["worker_pid"]     = os.getpid()

    if plate_text:
        result["status"] = STATUS_STOLEN if plate_text in stolen_plates else STATUS_OK

    result["total_time_s"] = round(
        result.get("yolo_time_s", 0) + result["ocr_time_s"], 6
    )
    return result

"""
pipeline.py — Workers de processamento serial e parallel.

ARQUITETURA TWO-STAGE
======================

  Estágio 1 — processo principal (serial em ambos os modos):
    YOLO/ONNX em batch de 8 imagens por vez. Detecta placas e salva crops.
    Batching melhora a utilização do ONNX Runtime vs. chamadas individuais.

  Estágio 2 — OCR (serial: 1 thread | parallel: N threads):
    fast-plate-ocr CCT sobre cada crop detectado.
    Todas as threads compartilham um único LicensePlateRecognizer via singleton
    thread-safe (ONNX Runtime documenta InferenceSession como thread-safe para
    chamadas concorrentes de Run()). Isso elimina o overhead de carregar o
    modelo ONNX N vezes — uma por thread — que causava degradação no benchmark.

NOTA SOBRE total_time_s
=======================
  Serial  : total_time_s = yolo_time_s + ocr_time_s por imagem (wall-clock).
  Parallel: total_time_s = ocr_time_s apenas, medido dentro de cada thread.
            O tempo do YOLO é medido como wall-clock do Estágio 1 em executor.py.
"""

import os
import time
from pathlib import Path

from src.runtime import force_single_thread_env, apply_library_thread_limits
from src.logger import get_logger


# ── Inicialização de runtime ──────────────────────────────────────────────────

def init_runtime() -> None:
    """Aplica limites de thread das bibliotecas no processo atual."""
    force_single_thread_env()
    apply_library_thread_limits()


# ── Motor OCR compartilhado ───────────────────────────────────────────────────

def _get_ocr_engine():
    """Retorna o singleton OCR thread-safe (inicializado no warmup)."""
    from src.ocr import make_ocr_engine
    return make_ocr_engine()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _new_result(image_name: str) -> dict:
    """Estrutura padrão de resultado com todos os campos em seus valores default."""
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
        "worker_id":         os.getpid(),
        "error":             "",
    }


# ── Worker SERIAL ─────────────────────────────────────────────────────────────

_serial_yolo = None
_serial_ocr  = None


def init_serial_worker(yolo_model_path: str) -> None:
    """
    Carrega YOLO e OCR no processo principal para execução serial.

    Deve ser chamado uma vez antes de process_image_serial().
    O motor OCR reutiliza o singleton já aquecido pelo warmup_ocr().
    """
    global _serial_yolo, _serial_ocr
    init_runtime()

    from ultralytics import YOLO
    _serial_yolo = YOLO(yolo_model_path)
    _serial_ocr  = _get_ocr_engine()


def process_image_serial(task: dict) -> dict:
    """
    Processa uma imagem em modo serial: YOLO → crop → OCR.

    Fluxo completo por imagem:
      1. cv2.imread
      2. YOLO inference (single image)
      3. extract_best_crop + padding
      4. Salva crop e preprocessed em disco
      5. fast-plate-ocr inference
      6. Consulta blacklist e lista de roubados
    """
    if _serial_yolo is None or _serial_ocr is None:
        raise RuntimeError(
            "init_serial_worker() deve ser chamado antes de process_image_serial()."
        )

    import cv2
    from src.config import (
        CROPS_DIR, PREPROCESSED_DIR, WORD_BLACKLIST,
        STATUS_OK, STATUS_STOLEN,
    )
    from src.detector import extract_best_crop
    from src.ocr import preprocess_plate, read_plate_text

    image_path    = task["image_path"]
    stolen_plates = task["stolen_plates"]
    stem          = Path(image_path).stem
    result        = _new_result(Path(image_path).name)
    t0            = time.perf_counter()

    img = cv2.imread(image_path)
    if img is None:
        result["error"]        = "Não foi possível carregar a imagem."
        result["total_time_s"] = round(time.perf_counter() - t0, 6)
        return result

    # Estágio 1 — YOLO
    t_yolo = time.perf_counter()
    try:
        detections = _serial_yolo(img, verbose=False)
    except Exception as exc:
        result["error"]        = f"YOLO: {exc}"
        result["total_time_s"] = round(time.perf_counter() - t0, 6)
        return result
    result["yolo_time_s"] = round(time.perf_counter() - t_yolo, 6)

    crop = extract_best_crop(img, detections)
    if crop is None:
        result["total_time_s"] = round(time.perf_counter() - t0, 6)
        return result

    # Salva crop e versão preprocessada em disco
    result["plate_detected"] = True
    crop_path = CROPS_DIR / f"{stem}_crop.jpg"
    cv2.imwrite(str(crop_path), crop)
    result["crop_path"] = str(crop_path)

    preprocessed = preprocess_plate(crop)
    prep_path = PREPROCESSED_DIR / f"{stem}_prep.jpg"
    cv2.imwrite(str(prep_path), preprocessed)
    result["preprocessed_path"] = str(prep_path)

    # Estágio 2 — OCR
    t_ocr = time.perf_counter()
    try:
        plate_text, confidence = read_plate_text(
            _serial_ocr, crop, preprocessed, WORD_BLACKLIST
        )
    except Exception as exc:
        result["error"]        = f"OCR: {exc}"
        result["total_time_s"] = round(time.perf_counter() - t0, 6)
        return result
    result["ocr_time_s"] = round(time.perf_counter() - t_ocr, 6)

    result["plate_text"]     = plate_text
    result["ocr_confidence"] = round(confidence, 4)
    if plate_text:
        result["status"] = STATUS_STOLEN if plate_text in stolen_plates else STATUS_OK

    result["total_time_s"] = round(time.perf_counter() - t0, 6)
    return result


# ── Worker OCR para o modo PARALLEL ──────────────────────────────────────────

def process_ocr_threaded(task: dict) -> dict:
    """
    Executa OCR sobre um crop já salvo em disco. Chamado por cada thread.

    Usa o singleton OCR compartilhado (thread-safe). Não há inicialização
    por thread — todas as threads reutilizam a mesma sessão ONNX.

    total_time_s = tempo de OCR desta imagem (não inclui YOLO do Estágio 1).
    worker_id    = thread ID truncado para 6 dígitos decimais.
    """
    import threading
    import cv2
    from src.config import WORD_BLACKLIST, STATUS_OK, STATUS_STOLEN
    from src.ocr import read_plate_text

    result        = task["result_base"].copy()
    stolen_plates = task["stolen_plates"]
    t0            = time.perf_counter()

    if not result.get("plate_detected"):
        return result

    crop_path = task.get("crop_path", "")
    prep_path = task.get("preprocessed_path", "")

    crop         = cv2.imread(crop_path) if crop_path else None
    preprocessed = cv2.imread(prep_path) if prep_path else None

    if crop is None:
        result["error"]        = "Crop não encontrado para OCR."
        result["total_time_s"] = round(time.perf_counter() - t0, 6)
        return result

    ocr = _get_ocr_engine()

    t_ocr = time.perf_counter()
    try:
        plate_text, confidence = read_plate_text(ocr, crop, preprocessed, WORD_BLACKLIST)
    except Exception as exc:
        result["error"]        = f"OCR: {exc}"
        result["total_time_s"] = round(time.perf_counter() - t0, 6)
        return result
    ocr_elapsed = round(time.perf_counter() - t_ocr, 6)

    result["ocr_time_s"]     = ocr_elapsed
    result["plate_text"]     = plate_text
    result["ocr_confidence"] = round(confidence, 4)
    result["worker_id"]      = threading.get_ident() & 0xFFFFFF
    result["total_time_s"]   = ocr_elapsed

    if plate_text:
        result["status"] = STATUS_STOLEN if plate_text in stolen_plates else STATUS_OK

    return result

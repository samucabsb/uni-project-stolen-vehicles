"""
pipeline.py — Núcleo do processamento de imagens (por worker).

ORDEM CRÍTICA PARA PERFORMANCE PARALELA:
  O init_worker DEVE configurar limites de thread ANTES de importar
  torch/ultralytics/rapidocr. Caso contrário cada worker spawn ~8 threads
  internas, gerando contenção severa entre processos (chegou a ser 2x mais
  lento que serial nos testes iniciais).

  Por isso este módulo só importa coisas leves no topo. As libs pesadas
  (torch, cv2, ultralytics, rapidocr) são importadas DENTRO de init_worker,
  depois das variáveis de ambiente terem sido configuradas.
"""

import os
import time
from pathlib import Path

from src.runtime import force_single_thread_env, apply_library_thread_limits


# ── Globais por processo (populadas pelo init_worker) ─────────────────────────

_yolo = None
_ocr  = None


# ── Inicializador do worker ───────────────────────────────────────────────────

def init_worker(yolo_model_path: str) -> None:
    """
    Chamada UMA VEZ por processo worker pelo ProcessPoolExecutor.

    Sequência:
      1. Env vars de thread (ANTES dos imports pesados)
      2. Imports pesados + ajustes finos de thread via API
      3. Carrega modelos YOLO e RapidOCR
    """
    global _yolo, _ocr

    # PASSO 1: env vars ANTES de qualquer import pesado
    force_single_thread_env()

    # PASSO 2: imports pesados, depois ajuste fino via API
    apply_library_thread_limits()

    # PASSO 3: carrega YOLO
    from ultralytics import YOLO
    _yolo = YOLO(yolo_model_path)

    # PASSO 4: cria RapidOCR com limites de thread explícitos.
    # IMPORTANTE: passar intra_op_num_threads=1 e inter_op_num_threads=1
    # senão o ONNX Runtime usa todas as cores ignorando OMP_NUM_THREADS.
    from rapidocr_onnxruntime import RapidOCR
    _ocr = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)


# ── Tarefa por imagem ─────────────────────────────────────────────────────────

def _new_result(image_name: str) -> dict:
    """Cria um dict de resultado com os campos default preenchidos."""
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


def process_image_task(task: dict) -> dict:
    """
    Processa uma única imagem usando os modelos globais do worker.

    Fluxo:
      1. Lê imagem do disco
      2. Roda YOLO para detectar a placa
      3. Recorta a placa (com padding)
      4. Pré-processa o recorte
      5. Roda OCR (com early exit)
      6. Compara com lista de placas roubadas
      7. Retorna dict com todos os tempos e status
    """
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
    t_total_start = time.perf_counter()

    # ── 1. Carrega imagem ────────────────────────────────────────────────────
    img = cv2.imread(image_path)
    if img is None:
        result["error"]        = "Não foi possível carregar a imagem."
        result["total_time_s"] = round(time.perf_counter() - t_total_start, 6)
        return result

    # ── 2. YOLO ──────────────────────────────────────────────────────────────
    t_yolo = time.perf_counter()
    try:
        detections = _yolo(img, verbose=False)
    except Exception as exc:
        result["error"]        = f"YOLO erro: {exc}"
        result["total_time_s"] = round(time.perf_counter() - t_total_start, 6)
        return result
    result["yolo_time_s"] = round(time.perf_counter() - t_yolo, 6)

    crop = extract_best_crop(img, detections)
    if crop is None:
        result["total_time_s"] = round(time.perf_counter() - t_total_start, 6)
        return result

    result["plate_detected"] = True

    # ── 3. Salva crop ────────────────────────────────────────────────────────
    crop_path = CROPS_DIR / f"{stem}_crop.jpg"
    cv2.imwrite(str(crop_path), crop)
    result["crop_path"] = str(crop_path)

    # ── 4. Pré-processa ──────────────────────────────────────────────────────
    preprocessed = preprocess_plate(crop)
    prep_path    = PREPROCESSED_DIR / f"{stem}_prep.jpg"
    cv2.imwrite(str(prep_path), preprocessed)
    result["preprocessed_path"] = str(prep_path)

    # ── 5. OCR ───────────────────────────────────────────────────────────────
    t_ocr = time.perf_counter()
    try:
        plate_text, confidence = read_plate_text(_ocr, crop, preprocessed, WORD_BLACKLIST)
    except Exception as exc:
        result["error"]        = f"OCR erro: {exc}"
        result["total_time_s"] = round(time.perf_counter() - t_total_start, 6)
        return result
    result["ocr_time_s"]     = round(time.perf_counter() - t_ocr, 6)
    result["plate_text"]     = plate_text
    result["ocr_confidence"] = round(confidence, 4)

    # ── 6. Comparação com lista de roubadas ──────────────────────────────────
    if plate_text:
        result["status"] = STATUS_STOLEN if plate_text in stolen_plates else STATUS_OK

    result["total_time_s"] = round(time.perf_counter() - t_total_start, 6)
    return result

"""
pipeline.py — Workers de processamento (v9).

ARQUITETURA v9 — Two-stage com THREADING para OCR:

  Estágio 1 (processo principal): YOLO em batch sobre todas as imagens.
    Detecta placas e salva crops em disco. Sem paralelismo aqui — o batch
    interno do YOLO já é eficiente, e múltiplos processos só fragmentariam
    o cache do modelo.

  Estágio 2 (N threads): OCR sobre os crops em paralelo.
    USA THREADS, não processos. Razão: na v8 usávamos ProcessPoolExecutor
    para o OCR e em máquinas rápidas isso era MAIS LENTO que serial. Cada
    processo carregava sua própria cópia do RapidOCR (~200MB cada), e os
    modelos ONNX competiam pelo cache L3 do CPU (~8MB). Resultado: cache
    miss massivo, cada worker rodando 3-4x mais devagar que serial.

    Com threading:
      - 1 processo só → pesos do modelo ficam quentes no cache
      - ONNX Runtime libera o GIL durante inferência → paralelismo real
      - Zero overhead de spawn de processo
      - Zero serialização de dados entre processos

    Cada thread tem sua própria instância RapidOCR via threading.local()
    para evitar contention em sessões ONNX compartilhadas.

Nota sobre total_time_s no modo parallel:
  No modo serial, total_time_s = yolo_time_s + ocr_time_s por imagem.
  No modo parallel, total_time_s = ocr_time_s da imagem apenas. O YOLO é
  medido como wall-clock do estágio inteiro em executor.py (_run_parallel)
  e não é distribuído por imagem. O avg_time_per_image_s no sumário usa
  elapsed / n (wall-clock), não a média dos total_time_s individuais.
"""

import os
import threading
import time
from pathlib import Path

from src.runtime import force_single_thread_env, apply_library_thread_limits
from src.logger import get_logger


# ── Storage thread-local ──────────────────────────────────────────────────────

# Cada thread Python tem sua própria instância de RapidOCR.
# Criada sob demanda na primeira chamada da thread, reutilizada nas seguintes.
_thread_local = threading.local()

# B5: guards — inicializados como None, protegem contra uso prematuro.
_serial_yolo = None
_serial_ocr  = None


def _make_ocr_engine():
    """Cria RapidOCR com compatibilidade entre versões da biblioteca.
    intra_op_num_threads=1 limita o ONNX Runtime a 1 thread interna,
    garantindo que o paralelismo venha das N threads de OCR, não de
    sub-threads de cada sessão ONNX.
    """
    from rapidocr_onnxruntime import RapidOCR
    try:
        return RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
    except TypeError:
        # Versões < 1.3 não aceitam esses kwargs
        return RapidOCR()


def _get_thread_ocr():
    """Retorna a instância OCR da thread atual, criando se necessário."""
    if not hasattr(_thread_local, "ocr"):
        _thread_local.ocr = _make_ocr_engine()
    return _thread_local.ocr


# ── Inicialização ─────────────────────────────────────────────────────────────

def init_runtime() -> None:
    """
    Aplica limites de thread das bibliotecas no processo atual.

    Chamada UMA VEZ no início da execução, antes de spawn de threads de OCR.
    torch e cv2 são limitados via API. ONNX Runtime é limitado via
    intra_op_num_threads=1 no construtor do RapidOCR (_make_ocr_engine).
    """
    force_single_thread_env()
    apply_library_thread_limits()


# ── Result helpers ────────────────────────────────────────────────────────────

def _new_result(image_name: str) -> dict:
    """Estrutura padrão de resultado com campos default."""
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
        "worker_id":         os.getpid(),   # B2: renomeado de worker_pid
        "error":             "",
    }


# ── Worker SERIAL — usado pelo modo 'serial' ──────────────────────────────────

def init_serial_worker(yolo_model_path: str) -> None:
    """
    Carrega YOLO e OCR no processo principal para execução serial.
    Deve ser chamado ANTES de process_image_serial().
    """
    global _serial_yolo, _serial_ocr
    init_runtime()

    from ultralytics import YOLO
    _serial_yolo = YOLO(yolo_model_path)
    _serial_ocr  = _make_ocr_engine()


def process_image_serial(task: dict) -> dict:
    """Processa uma imagem em modo serial: YOLO → crop → OCR."""
    # B5: guard contra uso prematuro dos globals
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

    t_yolo = time.perf_counter()
    try:
        detections = _serial_yolo(img, verbose=False)
    except Exception as exc:
        result["error"]        = f"YOLO erro: {exc}"
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
        plate_text, confidence = read_plate_text(
            _serial_ocr, crop, preprocessed, WORD_BLACKLIST
        )
    except Exception as exc:
        result["error"]        = f"OCR erro: {exc}"
        result["total_time_s"] = round(time.perf_counter() - t0, 6)
        return result

    result["ocr_time_s"]     = round(time.perf_counter() - t_ocr, 6)
    result["plate_text"]     = plate_text
    result["ocr_confidence"] = round(confidence, 4)
    if plate_text:
        result["status"] = STATUS_STOLEN if plate_text in stolen_plates else STATUS_OK

    result["total_time_s"] = round(time.perf_counter() - t0, 6)
    return result


# ── Worker OCR — usado pelas threads do modo 'parallel' ───────────────────────

def process_ocr_threaded(task: dict) -> dict:
    """
    Executa OCR num crop já existente em disco. Roda em uma thread.

    Cada thread tem sua própria instância RapidOCR (via threading.local),
    criada na primeira chamada. As subsequentes reusam a mesma instância.

    Entrada esperada (preparada pelo estágio YOLO no processo principal):
      task["crop_path"]         → path do crop salvo em disco
      task["preprocessed_path"] → path do crop pré-processado
      task["result_base"]       → dict parcial com dados do YOLO
      task["stolen_plates"]     → set de placas roubadas

    Nota: total_time_s = apenas o tempo de OCR desta imagem.
    O YOLO é contabilizado como wall-clock do estágio 1 em executor.py.
    """
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

    ocr = _get_thread_ocr()

    t_ocr = time.perf_counter()
    try:
        plate_text, confidence = read_plate_text(ocr, crop, preprocessed, WORD_BLACKLIST)
    except Exception as exc:
        result["error"]        = f"OCR erro: {exc}"
        result["total_time_s"] = round(time.perf_counter() - t0, 6)
        return result

    ocr_elapsed = time.perf_counter() - t_ocr

    result["ocr_time_s"]     = round(ocr_elapsed, 6)
    result["plate_text"]     = plate_text
    result["ocr_confidence"] = round(confidence, 4)
    # B2: worker_id guarda thread ID (não PID) no modo parallel
    result["worker_id"]      = threading.get_ident() & 0xFFFFFF

    if plate_text:
        result["status"] = STATUS_STOLEN if plate_text in stolen_plates else STATUS_OK

    # B1: total_time_s = apenas OCR no modo parallel
    result["total_time_s"] = round(ocr_elapsed, 6)
    return result

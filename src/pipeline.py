"""
pipeline.py — Workers de processamento serial e parallel.

ARQUITETURA (v11)
==================

  Modo serial (1 processo):
    YOLO single-image → crop → OCR, sequencial, no processo principal.
    Baseline de benchmark.

  Modo parallel (N processos) — pipeline completo por partição de imagens:
    A lista de imagens é dividida em lotes e distribuída entre N processos.
    Cada processo carrega SEU PRÓPRIO YOLO e SEU PRÓPRIO OCR
    (init_full_pipeline_process) e roda o pipeline inteiro — YOLO em
    mini-lotes internos, crop, OCR — para sua fatia (process_image_batch).

    Por que isso e não "YOLO serial + OCR paralelo" (v10.x)?
    Na v10.x, o YOLO rodava inteiramente no processo principal antes de
    qualquer paralelismo começar — sempre ~70% do tempo total, imune ao
    número de processos de OCR (Lei de Amdahl: por mais que o OCR acelere,
    o total nunca cai abaixo do tempo do YOLO sozinho). A v11 paraleliza
    AMBAS as etapas juntas: N processos = N YOLOs + N OCRs rodando ao
    mesmo tempo, cada um em uma fatia das imagens.

  THREADING DAS SESSÕES ONNX (crítico para escalar de verdade):
    Por padrão, toda `ort.InferenceSession` usa intra_op_num_threads=0
    ("auto" → todos os núcleos), e isso NÃO é controlado pelas variáveis de
    ambiente de runtime.py. Sem corrigir isso, N processos x 2 sessões cada
    (YOLO + OCR) tentariam usar todos os núcleos simultaneamente — disputa
    catastrófica, pior quanto mais processos.
      • OCR: corrigido via sess_options explícitas em ocr.py — a API do
        fast-plate-ocr aceita isso diretamente.
      • YOLO: a API pública de `ultralytics.YOLO` NÃO aceita sess_options.
        Corrigido via _patch_onnxruntime_single_thread() — um monkeypatch
        local ao processo que intercepta toda futura `InferenceSession`
        sem sess_options explícitas e força 1 thread.

NOTA SOBRE total_time_s
=======================
  Em ambos os modos: total_time_s = yolo_time_s + ocr_time_s por imagem.
  No modo parallel, yolo_time_s é o custo do lote de YOLO dividido
  igualmente entre as imagens daquele lote (custo amortizado, não medido
  imagem a imagem — mesma convenção do antigo Estágio 1).
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



# ── Monkeypatch: ONNX Runtime de thread única para sessões sem sess_options ──

def _patch_onnxruntime_single_thread() -> None:
    """
    Monkeypatch local ao processo: força toda futura `ort.InferenceSession`
    criada SEM sess_options explícitas a usar exatamente 1 thread interna.

    Por quê: a classe pública `ultralytics.YOLO` não expõe nenhum jeito de
    passar SessionOptions para a sessão ONNX que ela cria internamente —
    `ultralytics/nn/autobackend.py` chama
    `onnxruntime.InferenceSession(w, providers=providers)` sem sess_options,
    então o YOLO sempre cai no padrão intra_op_num_threads=0 ("auto" → usa
    todos os núcleos). Sem este patch, rodar N processos cada um com seu
    próprio YOLO recriaria — para o YOLO, que é bem mais pesado que o OCR —
    o mesmo bug de oversubscription que corrigimos em ocr.py.

    O patch só altera o atributo `InferenceSession` no módulo `onnxruntime`
    DESTE processo (cada processo filho do ProcessPoolExecutor importa sua
    própria cópia do módulo) — não tem efeito no processo principal nem no
    modo serial, que não chamam esta função.
    """
    import onnxruntime as ort

    if getattr(ort.InferenceSession, "_single_thread_patched", False):
        return  # já aplicado neste processo

    _Original = ort.InferenceSession

    class _SingleThreadInferenceSession(_Original):
        def __init__(self, path_or_bytes, sess_options=None, **kwargs):
            if sess_options is None:
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = 1
                sess_options.inter_op_num_threads = 1
                sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            super().__init__(path_or_bytes, sess_options=sess_options, **kwargs)

    _SingleThreadInferenceSession._single_thread_patched = True
    ort.InferenceSession = _SingleThreadInferenceSession


# ── Worker do modo PARALLEL v11: pipeline completo por processo ──────────────
#
# Em vez de "1 YOLO serial alimentando N processos de OCR" (v10.x), cada
# processo do pool carrega SEU PRÓPRIO YOLO e SEU PRÓPRIO OCR — "YOLOs
# diferentes" rodando ao mesmo tempo — e processa uma fatia completa da
# lista de imagens, do início ao fim. Isso remove o piso imposto pela Lei
# de Amdahl que limitava o speedup total na v10.x (o YOLO sempre serial,
# ~70% do tempo total, era imune ao nº de processos de OCR).

_fp_yolo          = None
_fp_ocr           = None
_fp_stolen_plates = None

_FULL_PIPELINE_YOLO_BATCH = 8  # mesmo valor usado no antigo Estágio 1


def init_full_pipeline_process(yolo_model_path: str, stolen_plates: frozenset) -> None:
    """
    Initializer do ProcessPoolExecutor no modo parallel v11 — roda UMA VEZ
    em cada processo filho, antes de qualquer tarefa.

    Carrega um YOLO e um OCR PRÓPRIOS deste processo (não compartilhados),
    ambos com threading interna fixada em 1 (YOLO via
    _patch_onnxruntime_single_thread, OCR via sess_options em ocr.py) —
    necessário para que N processos usem N núcleos de forma limpa, sem cada
    sessão tentar usar todos os núcleos sozinha.
    """
    global _fp_yolo, _fp_ocr, _fp_stolen_plates
    import numpy as np

    init_runtime()
    _patch_onnxruntime_single_thread()

    from ultralytics import YOLO
    _fp_yolo          = YOLO(yolo_model_path)
    _fp_ocr           = _get_ocr_engine()
    _fp_stolen_plates = stolen_plates

    # Aquece os dois modelos neste processo — evita que a 1ª imagem real
    # pague o custo de cold-start tanto do YOLO quanto do OCR.
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    _fp_yolo(dummy, verbose=False)
    _fp_ocr.run(np.zeros((64, 256, 3), dtype=np.uint8))


def process_image_batch(tasks: list) -> list:
    """
    Processa um LOTE de imagens do início ao fim (YOLO → crop → OCR), dentro
    de UM ÚNICO PROCESSO — worker do modo parallel v11.

    A ordem dos resultados retornados é EXATAMENTE a ordem de `tasks` — o
    chamador (executor.py) usa essa correspondência posicional para
    remontar `all_results` no índice global correto, sem precisar embutir
    índice em cada item.

    O YOLO ainda roda em mini-lotes internos (_FULL_PIPELINE_YOLO_BATCH)
    para preservar o ganho de inferência em batch do ONNX Runtime — mesmo
    com intra_op_num_threads=1, agrupar chamadas reduz overhead fixo por
    chamada Python/ONNX.

    Cada task esperado: {"image_path": str, "stolen_plates": set} (mesmo
    formato já usado em toda a base de código).
    """
    import cv2
    from src.config import (
        CROPS_DIR, PREPROCESSED_DIR, WORD_BLACKLIST,
        STATUS_OK, STATUS_STOLEN,
    )
    from src.detector import extract_best_crop
    from src.ocr import preprocess_plate, read_plate_text

    results: list = []

    for batch_start in range(0, len(tasks), _FULL_PIPELINE_YOLO_BATCH):
        batch = tasks[batch_start: batch_start + _FULL_PIPELINE_YOLO_BATCH]

        t_yolo_batch = time.perf_counter()
        images = [cv2.imread(t["image_path"]) for t in batch]
        loaded = [img is not None for img in images]
        valid_imgs = [img for img, ok in zip(images, loaded) if ok]

        valid_dets: list = []
        if valid_imgs:
            try:
                valid_dets = _fp_yolo(valid_imgs, verbose=False)
            except Exception:
                valid_dets = [None] * len(valid_imgs)
        # Tempo de YOLO do lote dividido igualmente entre as imagens do lote
        # (mesma convenção usada no antigo Estágio 1: custo de batch
        # amortizado, não medido imagem a imagem).
        yolo_per_image = (
            (time.perf_counter() - t_yolo_batch) / len(batch) if batch else 0.0
        )
        det_iter = iter(valid_dets)

        for task, img, ok in zip(batch, images, loaded):
            stem   = Path(task["image_path"]).stem
            result = _new_result(Path(task["image_path"]).name)
            result["yolo_time_s"] = round(yolo_per_image, 6)

            if not ok:
                result["error"]        = "Não foi possível carregar a imagem."
                result["total_time_s"] = result["yolo_time_s"]
                results.append(result)
                continue

            det  = next(det_iter, None)
            crop = extract_best_crop(img, [det]) if det is not None else None

            if crop is None:
                result["total_time_s"] = result["yolo_time_s"]
                results.append(result)
                continue

            result["plate_detected"] = True
            crop_path = CROPS_DIR        / f"{stem}_crop.jpg"
            prep_path = PREPROCESSED_DIR / f"{stem}_prep.jpg"
            cv2.imwrite(str(crop_path), crop)
            cv2.imwrite(str(prep_path), preprocess_plate(crop))
            result["crop_path"]         = str(crop_path)
            result["preprocessed_path"] = str(prep_path)

            t_ocr = time.perf_counter()
            try:
                plate_text, confidence = read_plate_text(
                    _fp_ocr, crop, None, WORD_BLACKLIST
                )
            except Exception as exc:
                result["error"]        = f"OCR: {exc}"
                result["total_time_s"] = result["yolo_time_s"]
                results.append(result)
                continue
            ocr_elapsed = round(time.perf_counter() - t_ocr, 6)

            result["ocr_time_s"]     = ocr_elapsed
            result["plate_text"]     = plate_text
            result["ocr_confidence"] = round(confidence, 4)
            if plate_text:
                result["status"] = (
                    STATUS_STOLEN if plate_text in _fp_stolen_plates else STATUS_OK
                )

            result["total_time_s"] = round(result["yolo_time_s"] + ocr_elapsed, 6)
            results.append(result)

    return results

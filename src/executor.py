"""
executor.py — Gerencia os três modos de execução.

Modos disponíveis:
  serial    Baseline sequencial. Um processo, YOLO + OCR em sequência.
  parallel  Data parallelism clássico. N workers, cada um com YOLO + OCR.
            Bom em hardware lento; sofre contenção de cache em CPUs rápidas.
  pipeline  Two-stage pipeline (v8). YOLO roda em batch no processo
            principal; N workers fazem APENAS OCR em paralelo.
            Elimina a redundância de N cópias do YOLO na RAM e no cache L3.
            Speedup real em qualquer hardware: 3-7x vs serial.
"""

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from src.pipeline import (
    init_worker, process_image_task,
    init_ocr_worker, process_ocr_task,
)
from src.colors import C, paint
from src.config import (
    STATUS_OK, STATUS_STOLEN, STATUS_UNIDENTIFIED, STATUS_ERROR,
    CROPS_DIR, PREPROCESSED_DIR,
)


# RAM estimada por worker
_RAM_PER_WORKER_FULL = 0.40  # GB — YOLO + OCR
_RAM_PER_WORKER_OCR  = 0.22  # GB — só OCR (modo pipeline)

_IMAGE_NAME_WIDTH = 40
_YOLO_BATCH_SIZE  = 8        # imagens por batch no estágio YOLO do pipeline


# ── Hardware ──────────────────────────────────────────────────────────────────

def get_hardware_info() -> dict:
    info = {"logical_cores": 1, "physical_cores": 1,
            "ram_total_gb": 0.0, "ram_avail_gb": 0.0}
    try:
        import psutil
        info["logical_cores"]  = psutil.cpu_count(logical=True)  or 1
        info["physical_cores"] = psutil.cpu_count(logical=False) or 1
        vm = psutil.virtual_memory()
        info["ram_total_gb"] = vm.total     / (1024 ** 3)
        info["ram_avail_gb"] = vm.available / (1024 ** 3)
    except ImportError:
        import multiprocessing as mp
        info["logical_cores"] = info["physical_cores"] = mp.cpu_count()
    return info


def print_hardware_info() -> None:
    info = get_hardware_info()
    print(paint("\n===== HARDWARE =====", C.CYAN_BOLD))
    print(f"  Núcleos físicos   : {info['physical_cores']}")
    print(f"  Núcleos lógicos   : {info['logical_cores']}")
    print(f"  RAM total         : {info['ram_total_gb']:.1f} GB")
    print(f"  RAM disponível    : {info['ram_avail_gb']:.1f} GB")


# ── Cap de workers ────────────────────────────────────────────────────────────

def _cap_workers(requested: int, ram_per_worker: float) -> int:
    info = get_hardware_info()
    physical = info["physical_cores"]
    logical  = info["logical_cores"]

    max_by_ram = max(1, int(info["ram_avail_gb"] / ram_per_worker)) \
                 if info["ram_avail_gb"] > 0 else requested
    cap = max(1, min(max_by_ram, requested))

    print(paint(
        f"\n[INFO] Núcleos: {physical} físicos / {logical} lógicos  |  "
        f"RAM livre: {info['ram_avail_gb']:.1f} GB",
        C.CYAN
    ))
    if cap < requested:
        print(paint(
            f"[RAM]  Workers reduzidos de {requested} para {cap} "
            f"(limite de RAM: {max_by_ram})",
            C.YELLOW
        ))
    if requested > logical:
        print(paint(
            f"[INFO] {requested} workers > {logical} lógicos → oversubscription severa",
            C.YELLOW
        ))
    elif requested > physical:
        print(paint(
            f"[INFO] {requested} workers > {physical} físicos → hyperthreading",
            C.CYAN
        ))
    else:
        print(paint(
            f"[INFO] {requested} workers ≤ {physical} físicos → configuração ideal",
            C.GREEN
        ))
    print(paint(f"[INFO] Workers efetivos: {cap}\n", C.CYAN))
    return cap


# ── Entry point público ───────────────────────────────────────────────────────

def run_tasks(tasks: list, yolo_model: str, execution: str = "serial",
              workers: int = 1) -> tuple:
    """
    Executa as tasks no modo indicado.
    Retorna: (results, elapsed, workers_requested, workers_effective)
    """
    workers_req = workers
    workers_eff = workers
    t_start     = time.perf_counter()

    if execution == "serial":
        workers_eff = 1
        from src.pipeline import init_worker
        init_worker(yolo_model)
        results = []
        for i, task in enumerate(tasks, 1):
            r = process_image_task(task)
            results.append(r)
            _print_progress(i, len(tasks), r)

    elif execution == "parallel":
        workers_eff = _cap_workers(workers, _RAM_PER_WORKER_FULL)
        results = _run_parallel(tasks, yolo_model, workers_eff)

    elif execution == "pipeline":
        workers_eff = _cap_workers(workers, _RAM_PER_WORKER_OCR)
        results = _run_pipeline(tasks, yolo_model, workers_eff)

    else:
        raise ValueError(f"Modo desconhecido: {execution}")

    elapsed = time.perf_counter() - t_start
    return results, elapsed, workers_req, workers_eff


# ── Modo PARALLEL ─────────────────────────────────────────────────────────────

def _run_parallel(tasks: list, yolo_model: str, n_workers: int) -> list:
    """N workers, cada um com YOLO + OCR. Data parallelism clássico."""
    results_map: dict = {}
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_worker,
        initargs=(yolo_model,),
    ) as executor:
        future_to_idx = {
            executor.submit(process_image_task, t): i
            for i, t in enumerate(tasks)
        }
        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            completed += 1
            try:
                result = future.result()
            except Exception as exc:
                result = _error_result(tasks[idx], exc)
            results_map[idx] = result
            _print_progress(completed, len(tasks), result)

    return [results_map[i] for i in range(len(tasks))]


# ── Modo PIPELINE (v8) ────────────────────────────────────────────────────────

def _run_pipeline(tasks: list, yolo_model: str, n_workers: int) -> list:
    """
    Two-stage pipeline:
      Estágio 1: YOLO em batch no processo principal (rápido, sem contenção)
      Estágio 2: N workers fazem APENAS OCR em paralelo

    Por que é mais rápido:
      - YOLO roda 8 imagens por forward pass (batch inference)
      - Workers só carregam OCR (~220MB) em vez de YOLO+OCR (~400MB)
      - Menor footprint de RAM → menos cache miss → OCR mais rápido
      - Imagens sem placa saem imediatamente após o YOLO (sem OCR)
    """
    import cv2
    from ultralytics import YOLO
    from src.detector import extract_best_crop
    from src.ocr import preprocess_plate
    from src.runtime import force_single_thread_env, apply_library_thread_limits

    force_single_thread_env()
    apply_library_thread_limits()

    stolen_plates = tasks[0]["stolen_plates"] if tasks else set()
    n = len(tasks)

    print(paint(f"  [Estágio 1/2] YOLO em batch (lotes de {_YOLO_BATCH_SIZE})...", C.CYAN))
    t_yolo_start = time.perf_counter()

    yolo_model_obj = YOLO(yolo_model)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Resultados finais (indexados por posição original)
    all_results = [None] * n

    # Tasks que precisam de OCR
    ocr_tasks = []

    # Processa em batches
    for batch_start in range(0, n, _YOLO_BATCH_SIZE):
        batch_tasks = tasks[batch_start: batch_start + _YOLO_BATCH_SIZE]

        # Carrega imagens do batch
        images = []
        loaded = []
        for task in batch_tasks:
            img = cv2.imread(task["image_path"])
            images.append(img)
            loaded.append(img is not None)

        # Filtra imagens válidas para o batch do YOLO
        valid_imgs = [img for img, ok in zip(images, loaded) if ok]
        valid_detections = []
        if valid_imgs:
            try:
                valid_detections = yolo_model_obj(valid_imgs, verbose=False)
            except Exception:
                valid_detections = [None] * len(valid_imgs)

        # Reconstrói iterador de detecções para todos (válidos e inválidos)
        det_iter = iter(valid_detections)

        for idx_in_batch, (task, img, ok) in enumerate(
            zip(batch_tasks, images, loaded)
        ):
            global_idx = batch_start + idx_in_batch
            image_name = Path(task["image_path"]).name
            stem       = Path(task["image_path"]).stem

            base_result = {
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
                "worker_pid":        0,
                "error":             "",
            }

            if not ok:
                base_result["error"] = "Não foi possível carregar a imagem."
                all_results[global_idx] = base_result
                continue

            det = next(det_iter, None)
            crop = extract_best_crop(img, [det]) if det is not None else None

            if crop is None:
                # Sem placa → resultado já está pronto
                all_results[global_idx] = base_result
                continue

            # Salva crop e preprocessed
            crop_path = CROPS_DIR      / f"{stem}_crop.jpg"
            prep_path = PREPROCESSED_DIR / f"{stem}_prep.jpg"
            cv2.imwrite(str(crop_path), crop)
            preprocessed = preprocess_plate(crop)
            cv2.imwrite(str(prep_path), preprocessed)

            base_result["plate_detected"] = True
            base_result["crop_path"]         = str(crop_path)
            base_result["preprocessed_path"] = str(prep_path)

            # Enfileira para OCR
            ocr_tasks.append({
                "global_idx":      global_idx,
                "result_base":     base_result,
                "crop_path":       str(crop_path),
                "preprocessed_path": str(prep_path),
                "stolen_plates":   stolen_plates,
            })

    t_yolo_elapsed = time.perf_counter() - t_yolo_start
    plates_found   = len(ocr_tasks)
    no_plate       = n - plates_found

    print(paint(
        f"  [Estágio 1/2] Concluído em {t_yolo_elapsed:.2f}s  "
        f"({plates_found} placas detectadas, {no_plate} sem placa)",
        C.GREEN
    ))

    # Estágio 2: OCR em paralelo (apenas para imagens COM placa)
    if not ocr_tasks:
        print(paint("  [Estágio 2/2] Sem placas para OCR.", C.YELLOW))
        for i, r in enumerate(all_results):
            if r is None:
                all_results[i] = {
                    "image": Path(tasks[i]["image_path"]).name,
                    "plate_detected": False, "plate_text": "",
                    "status": "NAO_IDENTIFICADA",
                    "yolo_time_s": 0.0, "ocr_time_s": 0.0,
                    "total_time_s": 0.0, "ocr_confidence": 0.0,
                    "crop_path": "", "preprocessed_path": "",
                    "worker_pid": 0, "error": "",
                }
        return all_results

    print(paint(
        f"  [Estágio 2/2] OCR de {plates_found} placas "
        f"em {n_workers} workers...",
        C.CYAN
    ))

    completed = 0
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_ocr_worker,
    ) as executor:
        future_to_ocr_task = {
            executor.submit(process_ocr_task, ot): ot
            for ot in ocr_tasks
        }
        for future in as_completed(future_to_ocr_task):
            ot = future_to_ocr_task[future]
            completed += 1
            try:
                result = future.result()
            except Exception as exc:
                result = ot["result_base"].copy()
                result["error"] = str(exc)

            global_idx = ot["global_idx"]
            all_results[global_idx] = result
            # Progresso: conta a posição global (inclui os sem placa já resolvidos)
            done_total = no_plate + completed
            _print_progress(done_total, n, result)

    return all_results


# ── Helpers visuais ───────────────────────────────────────────────────────────

def _error_result(task: dict, exc: Exception) -> dict:
    return {
        "image":          Path(task.get("image_path", "?")).name,
        "error":          str(exc),
        "status":         STATUS_ERROR,
        "plate_detected": False, "plate_text": "",
        "yolo_time_s":    0.0, "ocr_time_s":  0.0,
        "total_time_s":   0.0, "ocr_confidence": 0.0,
        "worker_pid":     -1,
    }


def _short_name(name: str, max_len: int = _IMAGE_NAME_WIDTH) -> str:
    if len(name) <= max_len:
        return name.ljust(max_len)
    return (name[:max_len - 3] + "...").ljust(max_len)


def _status_color(status: str) -> str:
    if status == STATUS_STOLEN:      return C.RED_BOLD
    if status == STATUS_OK:          return C.GREEN
    if status == STATUS_ERROR:       return C.MAGENTA
    return C.YELLOW


def _print_progress(current: int, total: int, result: dict) -> None:
    plate  = result.get("plate_text") or "—"
    status = result.get("status", "?")
    image  = result.get("image", "?")
    pid    = result.get("worker_pid", "?")
    t      = result.get("total_time_s", 0.0)
    width  = len(str(total))
    counter = f"[{current:>{width}}/{total}]"
    name    = _short_name(image)
    color   = _status_color(status)

    if status == STATUS_STOLEN:
        line = (
            f"🚨 {counter} {name} placa={plate:<10} "
            f"status={status:<17} pid={pid}  {t:.3f}s  ⚠️  ALERTA"
        )
        print(paint(line, color))
    else:
        status_colored = paint(f"{status:<17}", color)
        print(f"  {counter} {name} placa={plate:<10} "
              f"status={status_colored} pid={pid}  {t:.3f}s")

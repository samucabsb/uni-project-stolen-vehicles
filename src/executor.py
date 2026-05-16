"""
executor.py — Gerencia execução serial ou paralela.

ProcessPoolExecutor com initializer que carrega modelos uma vez por processo.
O progresso é renderizado com cores ANSI, destacando casos de ROUBADO.
"""

import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.pipeline import init_worker, process_image_task
from src.colors import C, paint
from src.config import STATUS_OK, STATUS_STOLEN, STATUS_UNIDENTIFIED, STATUS_ERROR


# RAM estimada por worker (YOLO nano + RapidOCR + overhead)
_RAM_PER_WORKER_GB = 0.4

# Largura máxima do nome do arquivo no progresso (linhas muito longas viram bagunça)
_IMAGE_NAME_WIDTH = 40


# ── Hardware ──────────────────────────────────────────────────────────────────

def get_hardware_info() -> dict:
    """Retorna info de hardware útil para diagnóstico."""
    info = {
        "logical_cores":  1,
        "physical_cores": 1,
        "ram_total_gb":   0.0,
        "ram_avail_gb":   0.0,
    }
    try:
        import psutil
        info["logical_cores"]  = psutil.cpu_count(logical=True)  or 1
        info["physical_cores"] = psutil.cpu_count(logical=False) or 1
        vm = psutil.virtual_memory()
        info["ram_total_gb"] = vm.total     / (1024 ** 3)
        info["ram_avail_gb"] = vm.available / (1024 ** 3)
    except ImportError:
        import multiprocessing as mp
        info["logical_cores"]  = mp.cpu_count()
        info["physical_cores"] = mp.cpu_count()
    return info


def print_hardware_info() -> None:
    """Imprime info de hardware no início da execução."""
    info = get_hardware_info()
    print(paint("\n===== HARDWARE =====", C.CYAN_BOLD))
    print(f"  Núcleos físicos   : {info['physical_cores']}")
    print(f"  Núcleos lógicos   : {info['logical_cores']}")
    print(f"  RAM total         : {info['ram_total_gb']:.1f} GB")
    print(f"  RAM disponível    : {info['ram_avail_gb']:.1f} GB")


# ── Limite de workers ─────────────────────────────────────────────────────────

def _cap_workers(requested: int) -> int:
    """
    Limita workers apenas por RAM disponível.

    O limite por núcleos físicos foi removido para permitir comparações
    acadêmicas com oversubscription. Isso gera a curva de speedup completa
    com o sweet spot e a degradação além dele (Lei de Amdahl na prática).

    Faixas de comportamento esperado:
      <  núcleos físicos    -> underutilization
      =  núcleos físicos    -> sweet spot
      >  físicos, <= lógicos -> oversubscription leve (hyperthreading)
      >  núcleos lógicos    -> oversubscription severa
    """
    info = get_hardware_info()
    physical = info["physical_cores"]
    logical  = info["logical_cores"]

    max_by_ram = (max(1, int(info["ram_avail_gb"] / _RAM_PER_WORKER_GB))
                  if info["ram_avail_gb"] > 0 else requested)
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
            f"[INFO] {requested} workers > {logical} núcleos lógicos "
            f"-> oversubscription severa",
            C.YELLOW
        ))
    elif requested > physical:
        print(paint(
            f"[INFO] {requested} workers > {physical} físicos "
            f"-> usando hyperthreading",
            C.CYAN
        ))
    else:
        print(paint(
            f"[INFO] {requested} workers <= {physical} físicos "
            f"-> configuração ideal",
            C.GREEN
        ))

    print(paint(f"[INFO] Workers efetivos: {cap}\n", C.CYAN))
    return cap


# ── Execução ──────────────────────────────────────────────────────────────────

def run_tasks(tasks: list, yolo_model: str, execution: str = "serial",
              workers: int = 1) -> tuple:
    """
    Executa as tasks em modo serial ou paralelo.
    Retorna: (results, elapsed, workers_requested, workers_effective)
    """
    workers_requested = workers
    workers_effective = workers
    t_start = time.perf_counter()

    if execution == "serial":
        workers_effective = 1
        init_worker(yolo_model)
        results = []
        for i, task in enumerate(tasks, 1):
            result = process_image_task(task)
            results.append(result)
            _print_progress(i, len(tasks), result)
    else:
        workers_effective = _cap_workers(workers)
        results = _run_parallel(tasks, yolo_model, workers_effective)

    elapsed = time.perf_counter() - t_start
    return results, elapsed, workers_requested, workers_effective


def _run_parallel(tasks: list, yolo_model: str, n_workers: int) -> list:
    """Executa em paralelo com ProcessPoolExecutor."""
    results_map: dict = {}

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_worker,
        initargs=(yolo_model,),
    ) as executor:
        future_to_idx = {
            executor.submit(process_image_task, task): idx
            for idx, task in enumerate(tasks)
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


def _error_result(task: dict, exc: Exception) -> dict:
    """Cria um resultado de erro quando o worker falha catastroficamente."""
    return {
        "image":          task.get("image_path", "?"),
        "error":          str(exc),
        "status":         STATUS_ERROR,
        "plate_detected": False,
        "plate_text":     "",
        "yolo_time_s":    0.0,
        "ocr_time_s":     0.0,
        "total_time_s":   0.0,
        "ocr_confidence": 0.0,
        "worker_pid":     -1,
    }


# ── Renderização do progresso ─────────────────────────────────────────────────

def _short_name(name: str, max_len: int = _IMAGE_NAME_WIDTH) -> str:
    """Trunca nome de arquivo longo com '...' no meio."""
    if len(name) <= max_len:
        return name.ljust(max_len)
    return (name[:max_len - 3] + "...").ljust(max_len)


def _status_color(status: str) -> str:
    """Cor ANSI apropriada para cada status."""
    if status == STATUS_STOLEN:
        return C.RED_BOLD
    if status == STATUS_OK:
        return C.GREEN
    if status == STATUS_ERROR:
        return C.MAGENTA
    return C.YELLOW  # NAO_IDENTIFICADA


def _print_progress(current: int, total: int, result: dict) -> None:
    """
    Imprime uma linha de progresso. Linhas de ROUBADO recebem destaque
    visual extra (emoji, texto em vermelho/negrito, marcador de alerta).
    """
    plate  = result.get("plate_text") or "—"
    status = result.get("status", "?")
    image  = result.get("image", "?")
    pid    = result.get("worker_pid", "?")
    t      = result.get("total_time_s", 0.0)

    width = len(str(total))
    counter = f"[{current:>{width}}/{total}]"
    name    = _short_name(image)
    color   = _status_color(status)

    if status == STATUS_STOLEN:
        # Linha destacada: prefixo de alerta + tudo em vermelho/negrito
        line = (
            f"🚨 {counter} {name} placa={plate:<10} "
            f"status={status:<17} pid={pid}  {t:.3f}s  ⚠️  ALERTA"
        )
        print(paint(line, color))
    else:
        # Linha normal: só o campo "status" recebe cor
        status_colored = paint(f"{status:<17}", color)
        line = (
            f"  {counter} {name} placa={plate:<10} "
            f"status={status_colored} pid={pid}  {t:.3f}s"
        )
        print(line)

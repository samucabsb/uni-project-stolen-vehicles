"""
executor.py — Orquestração dos modos de execução (v9).

Dois modos de execução:

  serial    Baseline sequencial — 1 thread, YOLO + OCR em sequência.
            Sem paralelismo. Usado para medir o tempo base.

  parallel  Two-stage com threading:
              Estágio 1: YOLO em batch no processo principal
              Estágio 2: N threads (não processos!) executam OCR em paralelo
            Cada thread tem sua própria instância RapidOCR via threading.local.
            ONNX Runtime libera o GIL durante inferência → paralelismo real.
"""

import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.pipeline import (
    init_runtime, init_serial_worker, process_image_serial,
    process_ocr_threaded,
)
from src.colors import C, paint
from src.config import (
    STATUS_OK, STATUS_STOLEN, STATUS_UNIDENTIFIED, STATUS_ERROR,
    CROPS_DIR, PREPROCESSED_DIR,
)
from src.logger import get_logger


_YOLO_BATCH_SIZE  = 8
_IMAGE_NAME_WIDTH = 40
_BAR_WIDTH        = 20
_LINE_WIDTH       = 95    # largura para sobrescrever a barra via \r


# ── Hardware ──────────────────────────────────────────────────────────────────

def get_hardware_info() -> dict:
    """Coleta informações de hardware para diagnóstico."""
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
    """Imprime info de hardware em formato compacto."""
    info = get_hardware_info()
    print(paint("\n===== HARDWARE =====", C.CYAN_BOLD))
    print(
        f"  {info['physical_cores']} físicos / {info['logical_cores']} lógicos  ·  "
        f"{info['ram_total_gb']:.1f} GB RAM ({info['ram_avail_gb']:.1f} GB livre)"
    )


# ── Recomendação de workers ───────────────────────────────────────────────────

def recommend_workers(hw: dict) -> int:
    """
    Calcula o número recomendado de workers baseado em hardware.

    Fórmula: min(physical_cores, floor(ram_avail_gb / 0.4))
    Cada instância RapidOCR consome ~200-400 MB; 0.4 GB é conservador.
    Sem limite máximo: o usuário pode rodar 12+ para benchmark acadêmico.
    """
    by_cpu = hw["physical_cores"]
    by_ram = max(1, int(hw["ram_avail_gb"] / 0.4))
    return min(by_cpu, by_ram)


# ── Diagnóstico de workers — chamado na seção de CONFIGURAÇÃO (main.py) ───────

def print_worker_diagnostics(requested: int) -> None:
    """
    Imprime informação sobre a configuração de workers escolhida.

    B4: chamada em resolve_execution() (main.py), na seção de CONFIGURAÇÃO,
    não durante a execução do pipeline. Assim o output não aparece
    intercalado com barras de progresso e resultados.

    Não restringe a escolha do usuário — apenas informa o contexto.
    Válido rodar com 12+ threads para coletar dados de benchmark acadêmico.
    """
    log  = get_logger()
    info = get_hardware_info()
    physical = info["physical_cores"]
    logical  = info["logical_cores"]

    if requested <= physical:
        log.info(paint(
            f"[INFO] {requested} thread(s) ≤ {physical} físicos → configuração ideal",
            C.GREEN
        ))
    elif requested <= logical:
        log.info(paint(
            f"[INFO] {requested} thread(s) > {physical} físicos → usando hyperthreading",
            C.CYAN
        ))
    else:
        log.info(paint(
            f"[INFO] {requested} thread(s) > {logical} lógicos → oversubscription "
            "(válido para benchmark)",
            C.YELLOW
        ))


# ── Barra de progresso ────────────────────────────────────────────────────────

class _ProgressBar:
    """
    Barra de progresso visual com ETA para o terminal.

    Atualiza a linha de progresso in-place via \\r (carriage return).
    Resultados são impressos acima da barra, que permanece na última linha.

    Em ambientes não-TTY (output redirecionado para arquivo), a barra é
    desativada automaticamente — apenas os resultados são impressos.
    """

    FILL  = "█"
    EMPTY = "░"

    def __init__(self, total: int, label: str = ""):
        self.total     = total
        self.label     = label
        self.completed = 0
        self.t_start   = time.perf_counter()
        self._is_tty   = sys.stdout.isatty()

    def _render(self) -> str:
        n      = self.completed
        total  = self.total
        pct    = n / total if total > 0 else 0.0
        filled = int(_BAR_WIDTH * pct)
        bar    = self.FILL * filled + self.EMPTY * (_BAR_WIDTH - filled)

        elapsed = time.perf_counter() - self.t_start
        if n > 0 and n < total:
            eta     = elapsed / n * (total - n)
            eta_str = f"~{eta:.0f}s restantes"
        elif n >= total:
            eta_str = f"{elapsed:.1f}s"
        else:
            eta_str = "calculando..."

        lbl = f"{self.label}  " if self.label else ""
        return f"  {lbl}[{bar}]  {n}/{total}  {pct*100:.0f}%  {eta_str}"

    def start(self) -> None:
        """Exibe a barra inicial sem newline (cursor fica no fim da linha)."""
        if self._is_tty:
            sys.stdout.write(self._render().ljust(_LINE_WIDTH))
            sys.stdout.flush()

    def update(self, result_line: str = "") -> None:
        """
        Incrementa contador e atualiza a barra.

        Se result_line for fornecido:
          - Sobrescreve a linha atual com o resultado (+ \\n)
          - Desenha a nova barra na linha seguinte (sem \\n)
        Sem result_line: apenas redesenha a barra no lugar.
        """
        self.completed += 1
        if not self._is_tty:
            if result_line:
                print(result_line)
            return

        bar = self._render()
        if result_line:
            sys.stdout.write(
                f"\r{result_line.ljust(_LINE_WIDTH)}\n{bar.ljust(_LINE_WIDTH)}"
            )
        else:
            sys.stdout.write(f"\r{bar.ljust(_LINE_WIDTH)}")
        sys.stdout.flush()

    def finish(self) -> None:
        """Finaliza a barra com newline."""
        if self._is_tty:
            sys.stdout.write(f"\r{self._render().ljust(_LINE_WIDTH)}\n")
            sys.stdout.flush()


# ── Entry point público ───────────────────────────────────────────────────────

def run_tasks(tasks: list, yolo_model: str, execution: str = "serial",
              workers: int = 1) -> tuple:
    """
    Executa as tasks no modo indicado.

    Retorna: (results, elapsed, workers_requested, workers_effective,
              yolo_time, ocr_time)
    """
    workers_req = workers
    t_start     = time.perf_counter()
    yolo_time   = 0.0
    ocr_time    = 0.0

    if execution == "serial":
        workers_eff = 1
        init_serial_worker(yolo_model)
        results = []
        n       = len(tasks)

        bar = _ProgressBar(n, label="")
        bar.start()

        for i, task in enumerate(tasks, 1):
            r = process_image_serial(task)
            results.append(r)
            yolo_time += r.get("yolo_time_s", 0.0)
            ocr_time  += r.get("ocr_time_s", 0.0)
            line = _format_result_line(i, n, r, mode="serial")
            bar.update(line)

        bar.finish()

    elif execution == "parallel":
        workers_eff = workers
        results, yolo_time, ocr_time = _run_parallel(tasks, yolo_model, workers_eff)
    else:
        raise ValueError(f"Modo desconhecido: {execution}")

    elapsed = time.perf_counter() - t_start
    return results, elapsed, workers_req, workers_eff, yolo_time, ocr_time


# ── Modo PARALLEL (two-stage threading) ───────────────────────────────────────

def _run_parallel(tasks: list, yolo_model: str, n_workers: int) -> tuple:
    """
    Two-stage pipeline com threading no OCR.

    Estágio 1: YOLO em batch no processo principal.
    Estágio 2: N threads consomem da fila de crops e fazem OCR.

    Retorna (all_results, yolo_time_total, ocr_time_total).
    """
    import cv2
    from ultralytics import YOLO
    from src.detector import extract_best_crop
    from src.ocr import preprocess_plate

    log = get_logger()
    init_runtime()

    stolen_plates = tasks[0]["stolen_plates"] if tasks else set()
    n = len(tasks)

    # ── Estágio 1: YOLO em batch ──────────────────────────────────────────────
    print(paint(
        f"  [Estágio 1/2] YOLO em batch (lotes de {_YOLO_BATCH_SIZE})...",
        C.CYAN
    ))
    t_yolo_start = time.perf_counter()

    yolo_model_obj = YOLO(yolo_model)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_results: list     = [None] * n
    ocr_tasks:   list     = []
    no_plate_names: list  = []   # B3: para exibição compacta

    for batch_start in range(0, n, _YOLO_BATCH_SIZE):
        batch_tasks = tasks[batch_start: batch_start + _YOLO_BATCH_SIZE]
        images      = [cv2.imread(t["image_path"]) for t in batch_tasks]
        loaded      = [img is not None for img in images]

        valid_imgs = [img for img, ok in zip(images, loaded) if ok]
        valid_detections = []
        if valid_imgs:
            try:
                valid_detections = yolo_model_obj(valid_imgs, verbose=False)
            except Exception:
                valid_detections = [None] * len(valid_imgs)
        det_iter = iter(valid_detections)

        for idx_in_batch, (task, img, ok) in enumerate(
            zip(batch_tasks, images, loaded)
        ):
            global_idx = batch_start + idx_in_batch
            stem       = Path(task["image_path"]).stem
            image_name = Path(task["image_path"]).name

            base_result = {
                "image":             image_name,
                "plate_detected":    False,
                "plate_text":        "",
                "status":            STATUS_UNIDENTIFIED,
                "yolo_time_s":       0.0,
                "ocr_time_s":        0.0,
                "total_time_s":      0.0,
                "ocr_confidence":    0.0,
                "crop_path":         "",
                "preprocessed_path": "",
                "worker_id":         0,    # B2: sem pid no parallel
                "error":             "",
            }

            if not ok:
                base_result["error"] = "Não foi possível carregar a imagem."
                all_results[global_idx] = base_result
                no_plate_names.append(image_name)
                continue

            det  = next(det_iter, None)
            crop = extract_best_crop(img, [det]) if det is not None else None

            if crop is None:
                all_results[global_idx] = base_result
                no_plate_names.append(image_name)
                continue

            crop_path = CROPS_DIR        / f"{stem}_crop.jpg"
            prep_path = PREPROCESSED_DIR / f"{stem}_prep.jpg"
            cv2.imwrite(str(crop_path), crop)
            preprocessed = preprocess_plate(crop)
            cv2.imwrite(str(prep_path), preprocessed)

            base_result["plate_detected"]    = True
            base_result["crop_path"]         = str(crop_path)
            base_result["preprocessed_path"] = str(prep_path)

            ocr_tasks.append({
                "global_idx":        global_idx,
                "result_base":       base_result,
                "crop_path":         str(crop_path),
                "preprocessed_path": str(prep_path),
                "stolen_plates":     stolen_plates,
            })

    yolo_elapsed = time.perf_counter() - t_yolo_start
    plates_found = len(ocr_tasks)
    no_plate_cnt = len(no_plate_names)

    # B3: sumário compacto do estágio 1 — sem índices globais, sem worker_id
    print(paint(
        f"  [Estágio 1/2] Concluído em {yolo_elapsed:.2f}s"
        f"  →  {plates_found} com placa  |  {no_plate_cnt} sem placa",
        C.GREEN
    ))

    if no_plate_names:
        print(paint(f"  Sem placa ({no_plate_cnt}):", C.GRAY))
        shown = no_plate_names[:6]
        for name in shown:
            short = name if len(name) <= 48 else name[:45] + "..."
            print(paint(f"    · {short}", C.GRAY))
        if no_plate_cnt > 6:
            print(paint(f"    · ...e mais {no_plate_cnt - 6} imagens", C.GRAY))
    print()

    # ── Estágio 2: OCR em threads ─────────────────────────────────────────────
    if not ocr_tasks:
        print(paint("  [Estágio 2/2] Nenhuma placa para processar.", C.YELLOW))
        return all_results, yolo_elapsed, 0.0

    print(paint(
        f"  [Estágio 2/2] OCR de {plates_found} placas em {n_workers} threads...\n",
        C.CYAN
    ))

    t_ocr_start    = time.perf_counter()
    ocr_total_time = 0.0

    bar = _ProgressBar(plates_found, label="OCR")
    bar.start()

    with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="ocr") as pool:
        future_to_task = {pool.submit(process_ocr_threaded, ot): ot
                          for ot in ocr_tasks}
        for future in as_completed(future_to_task):
            ot         = future_to_task[future]
            global_idx = ot["global_idx"]
            try:
                result = future.result()
            except Exception as exc:
                result           = ot["result_base"].copy()
                result["error"]  = str(exc)
                result["status"] = STATUS_ERROR

            all_results[global_idx]  = result
            ocr_total_time          += result.get("ocr_time_s", 0.0)

            line = _format_result_line(global_idx + 1, n, result, mode="parallel")
            bar.update(line)

    bar.finish()

    ocr_wall = time.perf_counter() - t_ocr_start
    print(paint(f"\n  [Estágio 2/2] Concluído em {ocr_wall:.2f}s", C.GREEN))

    return all_results, yolo_elapsed, ocr_total_time


# ── Formatação de resultado ───────────────────────────────────────────────────

def _short_name(name: str, max_len: int = _IMAGE_NAME_WIDTH) -> str:
    """Trunca nome de arquivo longo com '...' no fim."""
    if len(name) <= max_len:
        return name.ljust(max_len)
    return (name[:max_len - 3] + "...").ljust(max_len)


def _status_color(status: str) -> str:
    """Mapeamento de status para cor ANSI."""
    if status == STATUS_STOLEN:    return C.RED_BOLD
    if status == STATUS_OK:        return C.GREEN
    if status == STATUS_ERROR:     return C.MAGENTA
    return C.YELLOW   # NAO_IDENTIFICADA


def _format_result_line(current: int, total: int, result: dict,
                         mode: str = "serial") -> str:
    """
    Formata uma linha de resultado completa como string.

    B2: serial usa 'pid=', parallel usa 'tid=' para deixar claro o tipo
    de identificador. worker_id guarda o PID real no serial e o thread ID
    truncado no parallel.
    """
    plate  = result.get("plate_text") or "—"
    status = result.get("status", "?")
    image  = result.get("image", "?")
    wid    = result.get("worker_id", "?")
    t      = result.get("total_time_s", 0.0)
    width  = len(str(total))
    counter = f"[{current:>{width}}/{total}]"
    name    = _short_name(image)
    color   = _status_color(status)
    wlabel  = "pid" if mode == "serial" else "tid"

    if status == STATUS_STOLEN:
        return paint(
            f"🚨 {counter} {name} placa={plate:<10} "
            f"status={status:<17} {wlabel}={wid}  {t:.3f}s  ⚠️  ALERTA",
            color
        )
    status_colored = paint(f"{status:<17}", color)
    return (
        f"  {counter} {name} placa={plate:<10} "
        f"status={status_colored} {wlabel}={wid}  {t:.3f}s"
    )

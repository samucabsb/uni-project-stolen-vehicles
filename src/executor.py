"""
executor.py — Orquestração dos modos de execução.

Dois modos disponíveis:

  serial    Baseline sequencial. 1 processo, YOLO → OCR por imagem.
            Usado para medir o tempo base sem qualquer paralelismo.

  parallel  v11 — pipeline completo (YOLO + OCR) distribuído entre N
            processos. Cada processo carrega SEU PRÓPRIO YOLO e SEU
            PRÓPRIO OCR e processa uma fatia da lista de imagens do
            início ao fim — "YOLOs diferentes" rodando ao mesmo tempo.

  Por que pipeline completo e não "YOLO serial + OCR paralelo" (v10.x)?
    Na v10.x, o YOLO rodava inteiramente no processo principal antes de
    qualquer paralelismo começar — sempre ~70% do tempo total, imune ao
    número de processos de OCR. Por mais que o OCR escalasse bem, o tempo
    TOTAL nunca caía proporcionalmente (Lei de Amdahl: a fração serial
    domina o piso). A v11 paraleliza as duas etapas juntas: N processos =
    N YOLOs + N OCRs simultâneos, cada um numa fatia das imagens.

  Threading das sessões ONNX (ver pipeline.py para detalhes):
    Tanto o YOLO quanto o OCR têm sua sessão ONNX fixada em 1 thread
    interna em cada processo — sem isso, N processos x 2 sessões cada
    tentariam usar todos os núcleos simultaneamente, gerando oversubscription
    severa (pior que a já observada com o OCR isolado, pois o YOLO é mais
    pesado). OCR: via sess_options diretas (ocr.py). YOLO: via monkeypatch
    de processo, já que a API pública do ultralytics.YOLO não aceita
    SessionOptions (ver _patch_onnxruntime_single_thread em pipeline.py).

  Diagnósticos:
    Tarefas são agrupadas em lotes (ver _chunk_list) para reduzir o nº de
    mensagens IPC com os processos filhos. Ao final, imprime: variação de
    clock de CPU (detecta throttling térmico) e distribuição de
    imagens/tempo por processo (detecta desequilíbrio de carga).
"""

from __future__ import annotations

import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from src.pipeline import (
    init_runtime, init_serial_worker, process_image_serial,
    init_full_pipeline_process, process_image_batch,
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
_LINE_WIDTH       = 95   # largura para sobrescrever a barra via \r


# ── Hardware ──────────────────────────────────────────────────────────────────

def get_hardware_info() -> dict:
    """Coleta informações de hardware para diagnóstico e recomendação."""
    info = {
        "logical_cores":  1,
        "physical_cores": 1,
        "ram_total_gb":   0.0,
        "ram_avail_gb":   0.0,
        "cpu_freq_mhz":   None,
        "cpu_freq_max_mhz": None,
    }
    try:
        import psutil
        info["logical_cores"]  = psutil.cpu_count(logical=True)  or 1
        info["physical_cores"] = psutil.cpu_count(logical=False) or 1
        vm = psutil.virtual_memory()
        info["ram_total_gb"] = vm.total     / (1024 ** 3)
        info["ram_avail_gb"] = vm.available / (1024 ** 3)
        try:
            freq = psutil.cpu_freq()
            if freq:
                info["cpu_freq_mhz"]     = freq.current
                info["cpu_freq_max_mhz"] = freq.max or None
        except Exception:
            pass  # alguns ambientes (containers, certas VMs) não expõem cpu_freq
    except ImportError:
        import multiprocessing as mp
        info["logical_cores"] = info["physical_cores"] = mp.cpu_count()
    return info


def print_hardware_info() -> None:
    """Imprime informações de hardware em formato compacto."""
    info = get_hardware_info()
    print(paint("\n===== HARDWARE =====", C.CYAN_BOLD))
    print(
        f"  {info['physical_cores']} físicos / {info['logical_cores']} lógicos  ·  "
        f"{info['ram_total_gb']:.1f} GB RAM ({info['ram_avail_gb']:.1f} GB livre)"
    )
    if info["cpu_freq_mhz"]:
        max_part = f" / máx {info['cpu_freq_max_mhz']:.0f} MHz" if info["cpu_freq_max_mhz"] else ""
        print(f"  CPU clock atual: {info['cpu_freq_mhz']:.0f} MHz{max_part}")


# ── Recomendação de workers ───────────────────────────────────────────────────

def recommend_workers(hw: dict) -> int:
    """
    Calcula o número recomendado de workers com base em hardware.

    Fórmula: min(physical_cores, floor(ram_avail_gb / 0.35))

    A partir da v11, cada processo carrega SEU PRÓPRIO YOLO + OCR (não mais
    um único OCR compartilhado) — footprint de RAM por processo maior que
    nas versões anteriores. 0.35 GB/processo é uma estimativa conservadora
    para os dois modelos + buffers de imagem.

    O usuário pode exceder este valor para benchmarks acadêmicos.
    """
    by_cpu = hw["physical_cores"]
    by_ram = max(1, int(hw["ram_avail_gb"] / 0.35))
    return min(by_cpu, by_ram)


# ── Diagnóstico de workers ────────────────────────────────────────────────────

def print_worker_diagnostics(requested: int) -> None:
    """
    Informa sobre a configuração de workers escolhida.

    Chamada em resolve_execution() (main.py), na seção CONFIGURAÇÃO,
    antes do pipeline iniciar — evita output intercalado com progresso.
    Não restringe a escolha do usuário.
    """
    log      = get_logger()
    info     = get_hardware_info()
    physical = info["physical_cores"]
    logical  = info["logical_cores"]

    if requested <= physical:
        log.info(paint(
            f"[INFO] {requested} processo(s) ≤ {physical} físicos → configuração ideal",
            C.GREEN
        ))
    elif requested <= logical:
        log.info(paint(
            f"[INFO] {requested} processo(s) > {physical} físicos → usando hyperthreading",
            C.CYAN
        ))
    else:
        log.info(paint(
            f"[INFO] {requested} processo(s) > {logical} lógicos → "
            "oversubscription (válido para benchmark)",
            C.YELLOW
        ))


# ── Barra de progresso ────────────────────────────────────────────────────────

class _ProgressBar:
    """
    Barra de progresso visual com ETA para o terminal.

    Atualiza a linha de progresso in-place via \\r (carriage return).
    Resultados são impressos acima da barra, que permanece na última linha.
    Em ambientes não-TTY (output redirecionado), imprime apenas os resultados.
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
        if 0 < n < total:
            eta     = elapsed / n * (total - n)
            eta_str = f"~{eta:.0f}s restantes"
        elif n >= total:
            eta_str = f"{elapsed:.1f}s"
        else:
            eta_str = "calculando..."

        lbl = f"{self.label}  " if self.label else ""
        return f"  {lbl}[{bar}]  {n}/{total}  {pct*100:.0f}%  {eta_str}"

    def start(self) -> None:
        """Exibe a barra inicial sem newline."""
        if self._is_tty:
            sys.stdout.write(self._render().ljust(_LINE_WIDTH))
            sys.stdout.flush()

    def update(self, result_line: str = "") -> None:
        """
        Incrementa contador e atualiza a barra.

        Se result_line for fornecido, imprime-o acima da barra (com newline)
        e redesenha a barra na mesma linha (sem newline).
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

def run_tasks(
    tasks: list, yolo_model: str,
    execution: str = "serial", workers: int = 1,
) -> tuple:
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
        bar     = _ProgressBar(n)
        bar.start()

        for i, task in enumerate(tasks, 1):
            r = process_image_serial(task)
            results.append(r)
            yolo_time += r.get("yolo_time_s", 0.0)
            ocr_time  += r.get("ocr_time_s",  0.0)
            bar.update(_format_result_line(i, n, r, mode="serial"))

        bar.finish()

    elif execution == "parallel":
        workers_eff = workers
        results, yolo_time, ocr_time = _run_parallel(tasks, yolo_model, workers_eff)
    else:
        raise ValueError(f"Modo desconhecido: {execution!r}")

    elapsed = time.perf_counter() - t_start
    return results, elapsed, workers_req, workers_eff, yolo_time, ocr_time



# ── Modo PARALLEL (v11 — pipeline completo: YOLO + OCR juntos por processo) ──

def _run_parallel(tasks: list, yolo_model: str, n_workers: int) -> tuple:
    """
    v11 — divide a lista de imagens em lotes e distribui entre N processos.
    Cada processo carrega SEU PRÓPRIO YOLO e SEU PRÓPRIO OCR
    (init_full_pipeline_process, em pipeline.py) e roda o pipeline completo
    (YOLO → crop → OCR) para sua fatia — "YOLOs diferentes" rodando ao
    mesmo tempo, em vez de 1 YOLO serial alimentando N processos de OCR.

    Por que essa mudança em relação à v10.x?
      Na v10.x, o YOLO rodava inteiramente no processo principal ANTES de
      qualquer paralelismo começar (~70% do tempo total, fixo, imune ao nº
      de processos de OCR) — Lei de Amdahl limitando o speedup TOTAL a bem
      menos do que o speedup do OCR isoladamente, por mais que esse
      escalasse bem. Paralelizar as duas etapas juntas remove esse piso.

    Retorna (all_results, yolo_time_sum, ocr_time_sum) — mesma assinatura
    de antes (compatibilidade com main.py/report.py/html_report.py). Os
    dois valores são SOMAS acumuladas do tempo gasto em cada etapa, dentro
    de cada processo, somadas entre os N processos — não wall-clock. Os
    números de wall-clock real (o que de fato importa para o speedup do
    benchmark) são impressos no terminal ao final desta função.
    """
    from src.config import CROPS_DIR, PREPROCESSED_DIR

    init_runtime()

    stolen_plates = tasks[0]["stolen_plates"] if tasks else set()
    n             = len(tasks)

    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(paint(
        f"  [Pipeline paralelo] {n} imagens em {n_workers} processos"
        f" (cada um com seu próprio YOLO + OCR)...\n",
        C.CYAN
    ))

    freq_before = _read_cpu_freq_mhz()
    t_start     = time.perf_counter()

    all_results:    list = [None] * n
    no_plate_names: list = []
    worker_stats:   dict = {}   # pid -> [count, yolo_sum, ocr_sum]
    yolo_total_time = 0.0
    ocr_total_time  = 0.0

    bar = _ProgressBar(n, label="Pipeline")
    bar.start()

    ctx    = mp.get_context("spawn")
    chunks = _chunk_list(tasks, n_workers)

    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=init_full_pipeline_process,
        initargs=(yolo_model, stolen_plates),
    ) as pool:
        future_to_chunk: dict = {}
        chunk_offset:    dict = {}
        offset = 0
        for chunk in chunks:
            fut = pool.submit(process_image_batch, chunk)
            future_to_chunk[fut] = chunk
            chunk_offset[fut]    = offset
            offset += len(chunk)

        for future in as_completed(future_to_chunk):
            chunk  = future_to_chunk[future]
            offset = chunk_offset[future]
            try:
                chunk_results = future.result()
            except Exception as exc:
                chunk_results = [_error_result(t, str(exc)) for t in chunk]

            for i, result in enumerate(chunk_results):
                global_idx               = offset + i
                all_results[global_idx]  = result
                yolo_total_time         += result.get("yolo_time_s", 0.0)
                ocr_total_time          += result.get("ocr_time_s", 0.0)

                if not result.get("plate_detected"):
                    no_plate_names.append(result.get("image", "?"))

                pid  = result.get("worker_id", 0)
                stat = worker_stats.setdefault(pid, [0, 0.0, 0.0])
                stat[0] += 1
                stat[1] += result.get("yolo_time_s", 0.0)
                stat[2] += result.get("ocr_time_s", 0.0)

                bar.update(_format_result_line(global_idx + 1, n, result, mode="parallel"))

    bar.finish()

    wall        = time.perf_counter() - t_start
    freq_after  = _read_cpu_freq_mhz()
    no_plate_cnt = len(no_plate_names)
    plates_found = n - no_plate_cnt
    accumulated  = yolo_total_time + ocr_total_time
    utilization  = (accumulated / (wall * n_workers) * 100.0) if wall > 0 else 0.0

    print(paint(
        f"\n  [Pipeline paralelo] Concluído em {wall:.2f}s"
        f"  →  {plates_found} com placa  |  {no_plate_cnt} sem placa",
        C.GREEN
    ))
    if no_plate_names:
        print(paint(f"  Sem placa ({no_plate_cnt}):", C.GRAY))
        for name in no_plate_names[:6]:
            short = name if len(name) <= 48 else name[:45] + "..."
            print(paint(f"    · {short}", C.GRAY))
        if no_plate_cnt > 6:
            print(paint(f"    · ...e mais {no_plate_cnt - 6} imagem(ns)", C.GRAY))

    print(paint(
        f"  YOLO+OCR acumulado (soma dos {n_workers} processos): {accumulated:.2f}s"
        f"  ·  utilização média: {utilization:.0f}%",
        C.GRAY
    ))
    if freq_before and freq_after:
        drop_pct = (freq_before - freq_after) / freq_before * 100.0 if freq_before else 0.0
        flag = C.YELLOW if drop_pct > 10 else C.GRAY
        print(paint(
            f"  CPU clock: {freq_before:.0f} MHz → {freq_after:.0f} MHz"
            f"  ({drop_pct:+.0f}% — {'possível throttling' if drop_pct > 10 else 'estável'})",
            flag
        ))
    _print_worker_distribution(worker_stats)

    return all_results, yolo_total_time, ocr_total_time


# ── Helpers de diagnóstico e particionamento ──────────────────────────────────

def _read_cpu_freq_mhz() -> float | None:
    """Lê a frequência atual da CPU (MHz), se disponível na plataforma."""
    try:
        import psutil
        freq = psutil.cpu_freq()
        return freq.current if freq else None
    except Exception:
        return None


_CHUNK_TARGET = 25  # imagens por lote — equilíbrio entre overhead de IPC e fluidez da barra


def _chunk_list(tasks: list, n_workers: int) -> list:
    """
    Agrupa `tasks` em lotes pequenos, um lote = um future enviado ao pool.

    Reduz o nº de mensagens IPC trocadas com os processos filhos (1 por
    lote, não 1 por imagem) — relevante com milhares de imagens. Tamanho do
    lote: pequeno o bastante para manter vários lotes por worker (a barra
    de progresso continua atualizando com frequência), grande o bastante
    para amortizar o overhead de serialização por chamada.
    """
    n = len(tasks)
    if n == 0:
        return []
    target = max(1, min(_CHUNK_TARGET, n // max(1, n_workers * 4) or 1))
    return [tasks[i:i + target] for i in range(0, n, target)]


def _error_result(task: dict, msg: str) -> dict:
    """Resultado de erro mínimo para quando um lote inteiro falha no pool."""
    return {
        "image":             Path(task["image_path"]).name,
        "plate_detected":    False,
        "plate_text":        "",
        "status":            STATUS_ERROR,
        "yolo_time_s":       0.0,
        "ocr_time_s":        0.0,
        "total_time_s":      0.0,
        "ocr_confidence":    0.0,
        "crop_path":         "",
        "preprocessed_path": "",
        "worker_id":         0,
        "error":             msg,
    }


def _print_worker_distribution(worker_stats: dict) -> None:
    """
    Imprime quantas imagens cada processo (PID) tratou e o tempo médio de
    YOLO/OCR por imagem em cada um. Distribuição equilibrada + tempos
    uniformes entre PIDs → comportamento esperado de paralelismo real.
    Um PID isolado muito mais lento → algo específico daquele processo
    (afinidade de núcleo, processo em segundo plano competindo, etc.), não
    um problema sistêmico de todos os workers.
    """
    if not worker_stats:
        return
    print(paint("\n  Distribuição por processo:", C.CYAN))
    for pid in sorted(worker_stats):
        count, yolo_sum, ocr_sum = worker_stats[pid]
        if count == 0:
            continue
        yolo_ms = yolo_sum / count * 1000.0
        ocr_ms  = ocr_sum  / count * 1000.0
        print(paint(
            f"    pid={pid}  {count:>5} imagens  ·  "
            f"YOLO {yolo_ms:.1f} ms/img  ·  OCR {ocr_ms:.1f} ms/img",
            C.GRAY
        ))


def _short_name(name: str, max_len: int = _IMAGE_NAME_WIDTH) -> str:
    """Trunca nome de arquivo longo com '...' mantendo comprimento fixo."""
    if len(name) <= max_len:
        return name.ljust(max_len)
    return (name[:max_len - 3] + "...").ljust(max_len)


def _status_color(status: str) -> str:
    """Mapeia status para código de cor ANSI."""
    return {
        STATUS_STOLEN:      C.RED_BOLD,
        STATUS_OK:          C.GREEN,
        STATUS_ERROR:       C.MAGENTA,
        STATUS_UNIDENTIFIED: C.YELLOW,
    }.get(status, C.YELLOW)


def _format_result_line(
    current: int, total: int, result: dict, mode: str = "serial"
) -> str:
    """
    Formata linha de resultado completa para exibição no terminal.

    worker_id agora é sempre um PID de processo (os.getpid()):
      serial   → PID do processo principal (sempre o mesmo).
      parallel → PID do processo do pool que de fato processou a imagem
                 (varia entre os N processos do ProcessPoolExecutor).
    """
    plate   = result.get("plate_text") or "—"
    status  = result.get("status", "?")
    image   = result.get("image", "?")
    wid     = result.get("worker_id", "?")
    t       = result.get("total_time_s", 0.0)
    width   = len(str(total))
    counter = f"[{current:>{width}}/{total}]"
    name    = _short_name(image)
    color   = _status_color(status)
    wlabel  = "pid"

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

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
    Tanto o YOLO quanto o OCR têm sua sessão ONNX fixada em
    config.ORT_THREADS_PER_WORKER threads internas em cada processo —
    sem isso, N processos x 2 sessões cada tentariam usar todos os
    núcleos simultaneamente, gerando oversubscription severa (pior do
    que a já observada com o OCR isolado, pois o YOLO é mais pesado).
    OCR e YOLO compartilham o mesmo monkeypatch de processo, já que a
    API pública do ultralytics.YOLO não aceita SessionOptions (ver
    _patch_onnxruntime_threads em pipeline.py).

  v11.1 — threads ORT fixas por padrão (não mais adaptativas):
    Antes, o nº de threads ORT por worker escalava como
    max(1, physical_cores // n_workers) — com poucos workers, cada um
    ganhava MAIS threads internas para "preencher" núcleos ociosos. Isso
    maximiza throughput bruto, mas mistura paralelismo intra-operação
    (várias threads acelerando UMA inferência) com paralelismo entre
    processos (N inferências distintas ao mesmo tempo): o nº total de
    threads ativas no sistema podia ficar quase constante entre
    1 e 2 workers, fazendo o speedup observado parecer ~1x mesmo com o
    dobro de processos, e o "1 worker" sozinho aparecia usando todos os
    núcleos lógicos da máquina — o paralelismo real (entre processos)
    fica invisível.
    Agora o padrão é fixo (config.ORT_THREADS_PER_WORKER, =1): 1 worker
    = 1 thread ORT ativa, sempre. N workers = N threads ativas. Speedup
    observado passa a refletir paralelismo de processo de forma direta —
    o que se quer demonstrar/medir num benchmark de concorrência. A
    estratégia adaptativa antiga continua disponível como experimento
    opt-in via fill_cores=True em run_tasks/_run_parallel, para quem
    quiser comparar throughput bruto separadamente.

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
    CROPS_DIR, PREPROCESSED_DIR, CHUNK_TARGET_IMAGES,
    YOLO_BATCH_SIZE, MAX_TOTAL_INFLIGHT_IMAGES, ORT_THREADS_PER_WORKER,
)
from src.logger import get_logger


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

_RAM_PER_WORKER_GB = 0.6  # ver docstring de recommend_workers


def recommend_workers(hw: dict) -> int:
    """
    Calcula o número recomendado de workers com base em hardware.

    Fórmula: min(physical_cores, floor(ram_avail_gb / 0.6))

    Cada processo carrega SEU PRÓPRIO YOLO + OCR. RSS medido empiricamente
    de um processo já carregado e aquecido: ~425 MB. 0.6 GB/processo dá
    margem para o crescimento de RSS durante o processamento real (buffers
    de imagem, lotes de inferência) sem subestimar o risco de OOM — ver
    config.MAX_TOTAL_INFLIGHT_IMAGES para o outro lado dessa mesma
    proteção (limita quanto cada processo pode crescer via o tamanho do
    lote YOLO).

    O usuário pode exceder este valor manualmente para benchmarks
    acadêmicos — recommend_workers só define o PADRÃO sugerido.
    """
    by_cpu = hw["physical_cores"]
    by_ram = max(1, int(hw["ram_avail_gb"] / _RAM_PER_WORKER_GB))
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
    save_images: bool = True, show_progress_lines: bool = True,
    pin_cpu: bool = False, fill_cores: bool = False,
) -> tuple:
    """
    Executa as tasks no modo indicado.

    save_images=False pula a gravação de crop/preprocessed em disco (modo
    benchmark — isola o custo de CPU do custo de I/O).
    show_progress_lines=False suprime a linha de resultado por imagem,
    mantendo só a barra de progresso — reduz overhead de console em runs
    com milhares de imagens.

    fill_cores (padrão False — ver _ort_threads_per_worker):
      False → cada worker do modo parallel usa exatamente
              config.ORT_THREADS_PER_WORKER (1) thread interna na sessão
              ONNX, sempre. 1 worker = 1 thread visível; N workers = N
              threads visíveis. É o que permite uma curva de speedup
              limpa (1→N workers) e comparável ao baseline serial.
      True  → modo adaptativo antigo: cada worker recebe
              max(1, physical_cores // n_workers) threads internas, para
              preencher núcleos que ficariam ociosos com poucos workers.
              Maximiza throughput bruto, mas não é apropriado para medir
              speedup por Nº DE PROCESSOS, já que mistura paralelismo
              intra-operação com paralelismo entre processos.

    IMPORTANTE sobre o que entra no `elapsed` medido:
      Em ambos os modos, qualquer carregamento/aquecimento de modelo
      acontece ANTES do cronômetro começar — paridade entre serial e
      parallel. No modo parallel isso é crítico: cada um dos N processos
      precisa fazer spawn + reimportar cv2/onnxruntime/ultralytics do zero
      + carregar e aquecer seu próprio YOLO/OCR (ver _warmup_pool), um
      custo fixo que NÃO encolhe com mais workers. Medi-lo dentro do
      `elapsed` penaliza desproporcionalmente configurações com poucos
      workers e é a causa mais provável de "parallel com 1 worker" aparecer
      mais lento que o serial puro.

    Retorna: (results, elapsed, workers_requested, workers_effective,
              yolo_time, ocr_time)
    """
    workers_req = workers
    yolo_time   = 0.0
    ocr_time    = 0.0

    if execution == "serial":
        workers_eff = 1
        init_serial_worker(yolo_model, save_images)   # fora do tempo medido
        t_start = time.perf_counter()
        results = []
        n       = len(tasks)
        bar     = _ProgressBar(n)
        bar.start()

        for i, task in enumerate(tasks, 1):
            r = process_image_serial(task)
            results.append(r)
            yolo_time += r.get("yolo_time_s", 0.0)
            ocr_time  += r.get("ocr_time_s",  0.0)
            line = _format_result_line(i, n, r, mode="serial") if show_progress_lines else ""
            bar.update(line)

        bar.finish()
        elapsed = time.perf_counter() - t_start

    elif execution == "parallel":
        workers_eff = workers
        results, yolo_time, ocr_time, elapsed = _run_parallel(
            tasks, yolo_model, workers_eff, save_images, show_progress_lines,
            pin_cpu, fill_cores,
        )
    else:
        raise ValueError(f"Modo desconhecido: {execution!r}")

    return results, elapsed, workers_req, workers_eff, yolo_time, ocr_time



# ── Modo PARALLEL (v11 — pipeline completo: YOLO + OCR juntos por processo) ──

def _run_parallel(
    tasks: list, yolo_model: str, n_workers: int,
    save_images: bool = True, show_progress_lines: bool = True,
    pin_cpu: bool = False, fill_cores: bool = False,
) -> tuple:
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

    fill_cores: ver docstring de run_tasks. Repassado direto para
    _ort_threads_per_worker — não afeta nada além do nº de threads ORT
    internas por worker.

    Retorna (all_results, yolo_time_sum, ocr_time_sum, wall). Os dois
    valores de tempo de estágio são SOMAS acumuladas do tempo gasto em cada
    etapa, dentro de cada processo, somadas entre os N processos — não
    wall-clock. `wall` é o tempo de parede real medido SOMENTE durante o
    processamento das imagens (pool já aquecido — ver _warmup_pool) e é o
    que de fato importa para o cálculo de speedup.
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

    all_results:    list = [None] * n
    no_plate_names: list = []
    worker_stats:   dict = {}   # pid -> [count, yolo_sum, ocr_sum]
    yolo_total_time = 0.0
    ocr_total_time  = 0.0

    ctx         = mp.get_context("spawn")
    chunks      = _chunk_list(tasks, n_workers)
    barrier     = ctx.Barrier(n_workers + 1)   # +1 = processo pai (ver _warmup_pool)
    yolo_batch  = _effective_yolo_batch_size(n_workers)
    ort_threads = _ort_threads_per_worker(n_workers, fill_cores)

    affinity_list, worker_index_counter = None, None
    if pin_cpu:
        affinity_list = _compute_cpu_affinity(n_workers)
        if affinity_list:
            worker_index_counter = ctx.Value("i", 0)

    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=init_full_pipeline_process,
        initargs=(
            yolo_model, stolen_plates, save_images, barrier, yolo_batch,
            affinity_list, worker_index_counter, ort_threads,
        ),
    ) as pool:
        t_warmup = time.perf_counter()
        _warmup_pool(pool, n_workers, barrier)
        warmup_s = time.perf_counter() - t_warmup
        affinity_note = (
            f"  ·  CPU pinning: núcleos {affinity_list}" if affinity_list
            else "  ·  CPU pinning: desligado" if pin_cpu
            else ""
        )
        thread_mode_note = "adaptativo (fill-cores)" if fill_cores else "fixo"
        print(paint(
            f"  [Pipeline paralelo] {n_workers} processo(s) prontos em "
            f"{warmup_s:.2f}s (spawn + import + YOLO/OCR — fora do tempo medido)"
            f"  ·  lote YOLO/processo: {yolo_batch}"
            + (f" (reduzido de {YOLO_BATCH_SIZE} p/ limitar pico de memória)"
               if yolo_batch < YOLO_BATCH_SIZE else "")
            + f"  ·  ORT threads/processo: {ort_threads} ({thread_mode_note})"
            + affinity_note,
            C.GRAY
        ))

        freq_before = _read_cpu_freq_mhz()
        t_start     = time.perf_counter()

        bar = _ProgressBar(n, label="Pipeline")
        bar.start()

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

                line = (
                    _format_result_line(global_idx + 1, n, result, mode="parallel")
                    if show_progress_lines else ""
                )
                bar.update(line)

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

    return all_results, yolo_total_time, ocr_total_time, wall


# ── Helpers de diagnóstico e particionamento ──────────────────────────────────

def _noop() -> None:
    """Task vazia — só serve para disparar o spawn dos processos do pool
    (ProcessPoolExecutor cria processos sob demanda, não eagerly na
    construção). A sincronização real de prontidão é feita pelo barrier
    em _warmup_pool, não pelo resultado desta task."""


def _warmup_pool(pool: ProcessPoolExecutor, n_workers: int, barrier) -> None:
    """
    Bloqueia até que os `n_workers` processos do pool tenham terminado
    spawn + import de cv2/onnxruntime/ultralytics + carregamento e
    warm-up de YOLO/OCR.

    `barrier` tem n_workers+1 partes e foi passado a cada processo via
    initargs (ver pipeline.init_full_pipeline_process) — cada um chama
    barrier.wait() ao FINAL do seu initializer, depois dos modelos
    carregados. Aqui o processo pai ocupa a parte +1: quando esta chamada
    retorna, é GARANTIDO que todos os n_workers processos terminaram a
    inicialização — diferente de esperar n_workers futures de tasks no-op,
    onde um processo rápido poderia concluir 2 tasks enquanto outro, ainda
    carregando o modelo, não pegou nenhuma (não provaria nada sobre esse
    processo lento).

    Chamado ANTES de iniciar o cronômetro do benchmark em _run_parallel.
    Sem isso, o custo fixo de cold-start de N processos (vários segundos
    por processo, em runs reais) entra no wall-clock medido — penaliza
    desproporcionalmente configurações com poucos workers e não some com
    o aumento de workers, distorcendo o speedup observado.
    """
    noop_futures = [pool.submit(_noop) for _ in range(n_workers)]
    barrier.wait()
    for fut in noop_futures:
        fut.result()


def _read_cpu_freq_mhz() -> float | None:
    """Lê a frequência atual da CPU (MHz), se disponível na plataforma."""
    try:
        import psutil
        freq = psutil.cpu_freq()
        return freq.current if freq else None
    except Exception:
        return None


def _compute_cpu_affinity(n_workers: int) -> list | None:
    """
    Escolhe a quais núcleos lógicos fixar cada um dos n_workers processos.
    Retorna None quando pinning não ajuda ou ativamente piora (ver abaixo).

    Pinning só é aplicado quando n_workers < physical_cores:
      Coloca cada worker no primeiro thread lógico de um core físico distinto.
      No layout Intel (phys_k = logical k e logical k+physical), [0..n-1]
      são todos em cores físicos diferentes — evita que 2 workers compartilhem
      L1/L2 via HT.
      Ex.: 4 workers, 6 físicos/12 lógicos → [0, 1, 2, 3] (cores 0-3 ✓)
      Ex.: 2 workers, 4 físicos/8 lógicos  → [0, 1]       (cores 0-1 ✓)

    Sem pinning (None) quando n_workers >= physical_cores:
      n_workers == physical: cada worker ficaria travado em 1 único lógico,
        mas o processo agora tem 3 threads nativas (Python main, ORT intra-op,
        io_pool prefetch) disputando esse 1 lógico — overhead de context-switch
        degrada ~12 % vs deixar o SO distribuir livremente.
      n_workers > physical: já em hyperthreading; sem mapa perfeito de topology
        portátil no Windows, o SO faz melhor que qualquer heurística estática.
    """
    try:
        import psutil
        logical  = psutil.cpu_count(logical=True)  or 0
        physical = psutil.cpu_count(logical=False) or 0
    except Exception:
        return None

    if logical <= 0 or physical <= 0 or n_workers <= 0 or n_workers > logical:
        return None

    if n_workers >= physical:
        # Caso n_workers == physical: cada worker ficaria preso em 1 único
        # núcleo lógico, mas o worker agora tem 3 threads nativas (Python main,
        # ORT intra-op, io_pool prefetch) — todas disputando esse 1 lógico.
        # O overhead de context-switch degrada ~12% vs deixar o SO decidir
        # livremente entre os N lógicos disponíveis.
        #
        # Caso n_workers > physical: já estamos em hyperthreading; espalhar
        # pelos lógicos é melhor que concentrar, mas o SO costuma fazer isso
        # bem por conta própria.
        #
        # Em ambos os casos: sem pinning = melhor.
        return None

    # n_workers < physical: cada worker vai para um core físico distinto.
    # No layout Intel (sibling = logical_idx + physical), [0..n_workers-1]
    # são os primeiros threads de n_workers cores físicos distintos.
    # Evita que 2 workers compartilhem L1/L2 via HT.
    return list(range(n_workers))


def _ort_threads_per_worker(n_workers: int, fill_cores: bool = False) -> int:
    """
    Threads ORT intra-op por processo worker.

    fill_cores=False (PADRÃO):
      Retorna sempre config.ORT_THREADS_PER_WORKER (1), não importa
      n_workers. 1 worker = 1 thread ativa, N workers = N threads ativas.
      Isso é o que torna o paralelismo entre PROCESSOS visível e medível
      de forma limpa — task manager mostra exatamente N threads em uso
      com N workers, e o speedup (tempo_serial / tempo_parallel) reflete
      diretamente o nº de processos, sem outra fonte de paralelismo
      interferindo. Use este modo para benchmark/relatório acadêmico.

    fill_cores=True (modo adaptativo, opt-in):
      Fórmula: max(1, physical_cores // n_workers)
      Com poucos workers, cada um recebe mais threads ORT para preencher
      os núcleos que ficariam ociosos com o esquema "1 thread por
      processo":
        2 workers em 6 físicos → 3 threads cada  (6 núcleos ativos)
        4 workers em 6 físicos → 1 thread cada   (4 núcleos ativos)
        6 workers em 6 físicos → 1 thread cada   (6 núcleos ativos)
        8+ workers             → 1 thread cada   (sem oversubscription)
      Maximiza throughput bruto (imagens/s), mas mistura paralelismo
      intra-operação com paralelismo entre processos — o nº total de
      threads ativas no sistema pode ficar quase constante entre 1 e 2
      workers (3 threads × 2 workers = 6, igual a 1 worker × 6 threads),
      fazendo o speedup entre essas duas configurações parecer ~1x. Não
      é apropriado para demonstrar/medir speedup por Nº DE PROCESSOS —
      só para comparar throughput bruto como experimento separado.
    """
    if not fill_cores:
        return ORT_THREADS_PER_WORKER

    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or 1
    except Exception:
        physical = mp.cpu_count()
    return max(1, physical // max(1, n_workers))


def _effective_yolo_batch_size(n_workers: int) -> int:
    """
    Lote YOLO REAL por processo, possivelmente menor que config.YOLO_BATCH_SIZE.

    O nº de imagens decodificadas em memória ao mesmo tempo em TODO o
    sistema é (lote por processo) × n_workers — sem limitar isso, pedir
    mais workers multiplica o pico de memória, podendo derrubar um
    processo por falta de RAM. E como UM processo morto quebra o
    ProcessPoolExecutor inteiro (BrokenProcessPool propaga para todos os
    futures pendentes), isso não aparece como "um pouco mais lento": vira
    uma cascata de erros no restante do run inteiro.

    Mantém o total de imagens em memória limitado a
    config.MAX_TOTAL_INFLIGHT_IMAGES não importa quantos workers sejam
    pedidos.
    """
    return max(1, min(YOLO_BATCH_SIZE, MAX_TOTAL_INFLIGHT_IMAGES // max(1, n_workers)))


def _chunk_list(tasks: list, n_workers: int) -> list:
    """
    Agrupa `tasks` em lotes pequenos, um lote = um future enviado ao pool.

    Reduz o nº de mensagens IPC trocadas com os processos filhos (1 por
    lote, não 1 por imagem) — relevante com milhares de imagens. Tamanho do
    lote: pequeno o bastante para manter vários lotes por worker (a barra
    de progresso continua atualizando com frequência), grande o bastante
    para amortizar o overhead de serialização por chamada. Alvo definido em
    config.CHUNK_TARGET_IMAGES.
    """
    n = len(tasks)
    if n == 0:
        return []
    target = max(1, min(CHUNK_TARGET_IMAGES, n // max(1, n_workers * 4) or 1))
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
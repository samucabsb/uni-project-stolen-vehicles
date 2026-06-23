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
from src.config import YOLO_BATCH_SIZE as _FULL_PIPELINE_YOLO_BATCH
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

_serial_yolo        = None
_serial_ocr         = None
_serial_save_images = True


def init_serial_worker(yolo_model_path: str, save_images: bool = True) -> None:
    """
    Carrega YOLO e OCR no processo principal para execução serial.

    Deve ser chamado uma vez antes de process_image_serial().
    O motor OCR reutiliza o singleton já aquecido pelo warmup_ocr().

    save_images=False pula a gravação de crop/preprocessed em disco (e o
    pré-processamento que só existe para gerar esse arquivo) — útil para
    isolar o custo de CPU (YOLO+OCR) do custo de I/O em benchmarks.
    """
    global _serial_yolo, _serial_ocr, _serial_save_images
    init_runtime()

    from ultralytics import YOLO
    _serial_yolo        = YOLO(yolo_model_path)
    _serial_ocr         = _get_ocr_engine()
    _serial_save_images = save_images


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

    result["plate_detected"] = True

    # Salva crop e versão preprocessada em disco (pulado em benchmark puro —
    # save_images=False evita tanto o I/O quanto o custo de CPU do CLAHE/Otsu,
    # já que `preprocessed` não é usado pelo motor OCR, só para exibição).
    preprocessed = None
    if _serial_save_images:
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



# ── Monkeypatch: ONNX Runtime com orçamento de threads por worker ────────────

def _patch_onnxruntime_threads(n_threads: int = 1) -> None:
    """
    Monkeypatch local ao processo: força toda futura `ort.InferenceSession`
    criada SEM sess_options explícitas a usar exatamente `n_threads` threads
    internas de intra-op (paralelismo dentro de cada operação ONNX).

    Por quê: a classe pública `ultralytics.YOLO` não expõe nenhum jeito de
    passar SessionOptions para a sessão ONNX que ela cria internamente —
    `ultralytics/nn/autobackend.py` chama
    `onnxruntime.InferenceSession(w, providers=providers)` sem sess_options,
    então o YOLO sempre cai no padrão intra_op_num_threads=0 ("auto" → usa
    todos os núcleos). Sem este patch, rodar N processos cada um com seu
    próprio YOLO recriaria o bug de oversubscription.

    n_threads é calculado adaptativamente em executor._ort_threads_per_worker:
      max(1, physical_cores // n_workers)
    → com 2 workers em 6 físicos: 3 threads cada (6 núcleos ativos)
    → com 4+ workers: 1 thread cada (sem oversubscription)

    O patch só altera o atributo `InferenceSession` no módulo `onnxruntime`
    DESTE processo (cada processo filho importa sua própria cópia do módulo)
    — não tem efeito no processo principal nem no modo serial.
    """
    import onnxruntime as ort

    if getattr(ort.InferenceSession, "_ort_thread_patch_n", None) == n_threads:
        return  # já aplicado com este mesmo valor neste processo

    _Original = ort.InferenceSession

    class _LimitedThreadInferenceSession(_Original):
        def __init__(self, path_or_bytes, sess_options=None, **kwargs):
            if sess_options is None:
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = n_threads
                sess_options.inter_op_num_threads = 1
                sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            super().__init__(path_or_bytes, sess_options=sess_options, **kwargs)

    _LimitedThreadInferenceSession._ort_thread_patch_n = n_threads
    ort.InferenceSession = _LimitedThreadInferenceSession


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
_fp_save_images   = True
_fp_yolo_batch    = _FULL_PIPELINE_YOLO_BATCH


def init_full_pipeline_process(
    yolo_model_path: str, stolen_plates: frozenset,
    save_images: bool = True, ready_barrier=None,
    yolo_batch_size: int | None = None,
    affinity_list: list | None = None, worker_index_counter=None,
    ort_threads: int = 1,
) -> None:
    """
    Initializer do ProcessPoolExecutor no modo parallel v11 — roda UMA VEZ
    em cada processo filho, antes de qualquer tarefa.

    Carrega um YOLO e um OCR PRÓPRIOS deste processo (não compartilhados),
    ambos com threading interna fixada em 1 (YOLO via
    _patch_onnxruntime_single_thread, OCR via sess_options em ocr.py) —
    necessário para que N processos usem N núcleos de forma limpa, sem cada
    sessão tentar usar todos os núcleos sozinha.

    save_images=False pula a gravação de crop/preprocessed em disco em
    process_image_batch — com N processos escrevendo 2 JPEGs por imagem
    simultaneamente, isso é uma fonte real de contenção de I/O que cresce
    com o nº de workers (ver config.SAVE_INTERMEDIATE_IMAGES).

    yolo_batch_size, se fornecido, sobrescreve config.YOLO_BATCH_SIZE para
    ESTE processo — computado em executor._effective_yolo_batch_size em
    função do nº de workers, para que o nº de imagens decodificadas em
    memória simultaneamente em TODO o sistema (lote × nº de processos)
    fique limitado independente de quantos workers forem pedidos (ver
    config.MAX_TOTAL_INFLIGHT_IMAGES). Sem isso, pedir muitos workers com
    um lote grande multiplica o pico de memória e pode derrubar processos
    por falta de RAM — e um processo morto quebra o ProcessPoolExecutor
    inteiro, corrompendo o restante do run com erros em cascata.

    affinity_list + worker_index_counter, se fornecidos, fixam ESTE
    processo em UM núcleo lógico via psutil — sem isso, o agendador do
    Windows é livre para migrar o processo entre núcleos durante a
    execução (invalidando cache a cada migração) ou para colocar 2
    processos no mesmo núcleo físico (2 threads de hyperthreading do
    mesmo core competem pelas MESMAS unidades de execução — bem pior que
    núcleos físicos distintos). Ver executor._compute_cpu_affinity para
    como os núcleos da lista são escolhidos.

    Por que um contador compartilhado e não passar o núcleo já escolhido
    diretamente? `initargs` do ProcessPoolExecutor é IDÊNTICO para todos
    os processos — não há como passar "este é o processo nº 2" via
    initargs sozinho. O contador (multiprocessing.Value com lock) permite
    que cada processo, ao inicializar, reivindique atomicamente o PRÓXIMO
    índice disponível em affinity_list — não importa a ordem real de
    spawn/boot de cada processo.

    Best-effort: se falhar (plataforma sem suporte, permissão), é
    ignorado silenciosamente — não deve derrubar o processo.

    ort_threads controla quantas threads intra-op o ORT usa em CADA sessão
    ONNX criada neste processo (YOLO e OCR). Calculado pelo executor como
    max(1, physical_cores // n_workers): com poucos workers, cada um recebe
    mais threads ORT para usar os núcleos que ficariam ociosos; com muitos
    workers, permanece em 1 para evitar oversubscription. Propaga também para
    cv2 (decode JPEG) e torch.

    ready_barrier, se fornecido, é um `ctx.Barrier(n_workers + 1)` —
    chamado ao FINAL desta função, depois que os modelos já foram
    carregados e aquecidos, para sinalizar ao processo pai que este
    processo está 100% pronto (ver executor._warmup_pool). Só pode chegar
    aqui via `initargs` (pickling de "inheritance", igual a passar um Lock
    para `Process(args=...)`) — um Barrier NÃO pode ser enviado depois,
    via `pool.submit()`, pois o processo já não está mais em bootstrap.
    """
    global _fp_yolo, _fp_ocr, _fp_stolen_plates, _fp_save_images, _fp_yolo_batch
    import numpy as np

    if affinity_list and worker_index_counter is not None:
        try:
            with worker_index_counter.get_lock():
                idx = worker_index_counter.value
                worker_index_counter.value += 1
            if idx < len(affinity_list):
                import psutil
                psutil.Process().cpu_affinity([affinity_list[idx]])
        except Exception:
            pass

    init_runtime()
    _patch_onnxruntime_threads(ort_threads)

    # Propaga o orçamento de threads para cv2 (decode JPEG) e torch (ops CPU),
    # sobrescrevendo o limite fixo de 1 que init_runtime() / force_single_thread_env()
    # aplicaram — agora usamos exatamente ort_threads por worker.
    try:
        import cv2
        cv2.setNumThreads(ort_threads)
    except Exception:
        pass
    try:
        import torch
        torch.set_num_threads(ort_threads)
    except (ImportError, RuntimeError):
        pass

    from ultralytics import YOLO
    _fp_yolo          = YOLO(yolo_model_path)
    _fp_ocr           = _get_ocr_engine()
    _fp_stolen_plates = stolen_plates
    _fp_save_images   = save_images
    if yolo_batch_size is not None:
        _fp_yolo_batch = max(1, yolo_batch_size)

    # Aquece os dois modelos neste processo — evita que a 1ª imagem real
    # pague o custo de cold-start tanto do YOLO quanto do OCR.
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    _fp_yolo(dummy, verbose=False)
    _fp_ocr.run(np.zeros((64, 256, 3), dtype=np.uint8))

    if ready_barrier is not None:
        ready_barrier.wait()


def _imread_batch(paths: list) -> list:
    """Lê uma lista de imagens via cv2 — chamado em thread de I/O (prefetch)."""
    import cv2
    return [cv2.imread(p) for p in paths]


def process_image_batch(tasks: list) -> list:
    """
    Processa um LOTE de imagens do início ao fim (YOLO → crop → OCR), dentro
    de UM ÚNICO PROCESSO — worker do modo parallel v11.

    A ordem dos resultados retornados é EXATAMENTE a ordem de `tasks` — o
    chamador (executor.py) usa essa correspondência posicional para
    remontar `all_results` no índice global correto, sem precisar embutir
    índice em cada item.

    YOLO roda em mini-lotes internos (tamanho _fp_yolo_batch) para amortizar
    o overhead fixo de chamada Python/ONNX. O carregamento das imagens do
    próximo mini-lote é feito em uma thread de I/O em background enquanto o
    YOLO processa o mini-lote atual (prefetch) — elimina a espera de disco do
    caminho crítico para todos os mini-lotes exceto o último.

    Cada task esperado: {"image_path": str, "stolen_plates": set}.
    """
    import cv2
    import concurrent.futures
    from src.config import (
        CROPS_DIR, PREPROCESSED_DIR, WORD_BLACKLIST,
        STATUS_OK, STATUS_STOLEN,
    )
    from src.detector import extract_best_crop
    from src.ocr import preprocess_plate, read_plate_text

    results: list = []

    # Divide as tasks em mini-lotes fixos para controle de memória
    mini_batches = [
        tasks[s: s + _fp_yolo_batch]
        for s in range(0, len(tasks), _fp_yolo_batch)
    ]
    if not mini_batches:
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as io_pool:
        # Pré-carrega o primeiro mini-lote enquanto ainda não há nada para processar
        next_images_fut = io_pool.submit(
            _imread_batch, [t["image_path"] for t in mini_batches[0]]
        )

        for batch_idx, batch in enumerate(mini_batches):
            # Obtém as imagens do mini-lote atual (já carregadas em background)
            images = next_images_fut.result()

            # Inicia imediatamente o carregamento do próximo mini-lote
            # (roda em paralelo ao YOLO + OCR do lote atual)
            if batch_idx + 1 < len(mini_batches):
                next_images_fut = io_pool.submit(
                    _imread_batch, [t["image_path"] for t in mini_batches[batch_idx + 1]]
                )

            t_yolo_batch = time.perf_counter()
            loaded    = [img is not None for img in images]
            valid_imgs = [img for img, ok in zip(images, loaded) if ok]

            valid_dets: list = []
            if valid_imgs:
                try:
                    valid_dets = _fp_yolo(valid_imgs, verbose=False)
                except Exception:
                    valid_dets = [None] * len(valid_imgs)
            # Tempo de YOLO do lote dividido igualmente entre as imagens
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

                # Pulado em benchmark puro — evita I/O e custo de CPU do
                # CLAHE/Otsu que só serve para gerar o arquivo no relatório HTML.
                if _fp_save_images:
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

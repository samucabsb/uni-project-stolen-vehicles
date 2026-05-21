"""
runtime.py — Configuração de threading para evitar contenção entre processos.

CONTEXTO
========
PyTorch, ONNX Runtime e OpenCV criam pools internos de threads. Por padrão,
cada um tenta usar todos os núcleos da CPU.

Com N workers (threads) de OCR, cada biblioteca spawnando T threads internas,
temos N×T threads brigando pelos mesmos núcleos. Isso gera contenção massiva
e pode tornar o sistema mais lento que a execução serial.

SOLUÇÃO
=======
Forçar 1 thread interna por biblioteca via:
  1. Variáveis de ambiente — lidas pelas libs na inicialização (antes do import)
  2. APIs diretas — aplicadas após o import, para libs que ignoram env vars

O paralelismo real vem das N threads de OCR gerenciadas pelo ThreadPoolExecutor,
não dos sub-threads de cada biblioteca.

ORDEM DE CHAMADA
================
  force_single_thread_env()   ← ANTES de qualquer import de torch/cv2/onnx
  apply_library_thread_limits() ← DEPOIS dos imports
"""

import os


# Variáveis de ambiente que controlam pools de threads em diversas libs.
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def force_single_thread_env() -> None:
    """
    Define todas as variáveis de ambiente que controlam pools de threads.

    IMPORTANTE: Deve ser chamada ANTES de qualquer `import torch`, `import cv2`,
    `import onnxruntime` ou `import ultralytics`. Caso contrário, as libs já
    terão lido os valores padrão e ignorarão as variáveis.
    """
    for var in _THREAD_ENV_VARS:
        os.environ[var] = "1"

    # Silencia logs de progresso e versão do Ultralytics
    os.environ["YOLO_VERBOSE"] = "False"


def apply_library_thread_limits() -> None:
    """
    Limita threads diretamente via APIs de torch e cv2.

    Deve ser chamada DEPOIS que as libs foram importadas. Complementa
    force_single_thread_env() para libs que ignoram variáveis de ambiente
    após já terem sido carregadas.

    ONNX Runtime (usado pelo YOLO e pelo fast-plate-ocr) gerencia seus
    próprios threads internos e é configurado pelo runtime_options do
    InferenceSession. Para o fast-plate-ocr, o comportamento padrão do
    onnxruntime é adequado: a sessão é thread-safe para inferência e o
    número de threads internos é controlado automaticamente.
    """
    try:
        import torch
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # set_num_interop_threads só pode ser chamado uma vez por processo.
            pass
    except ImportError:
        pass

    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass

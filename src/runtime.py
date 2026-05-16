"""
runtime.py — Configuração de threading para evitar contenção entre processos.

CONTEXTO:
  PyTorch, ONNX Runtime e OpenCV criam pools internos de threads. Por padrão,
  cada um tenta usar todas as cores da CPU.

  Com N workers (processos), cada um spawnando T threads, temos N*T threads
  brigando pelos mesmos N núcleos. Resultado: contenção massiva e slowdown
  (chegou a ser 2x mais lento que serial nos testes).

  Solução: cada processo worker usa 1 thread interna. O paralelismo vem da
  quantidade de processos, não de threads internas de cada um.
"""

import os


# Variáveis de ambiente que diferentes libs leem para configurar thread pools.
# Cada uma controla uma camada diferente: OMP é OpenMP (base do BLAS), MKL é
# Intel MKL, OPENBLAS é o backend padrão do NumPy em muitas distros, etc.
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
    terão lido os valores padrão.
    """
    for var in _THREAD_ENV_VARS:
        os.environ[var] = "1"

    # Silencia logs verbosos do Ultralytics (não tem haver com threading mas
    # é o lugar natural pra setar antes do import)
    os.environ["YOLO_VERBOSE"] = "False"


def apply_library_thread_limits() -> None:
    """
    Limita threads diretamente via APIs das bibliotecas.
    Chamar DEPOIS que as libs foram importadas, complementa o force_single_thread_env.
    """
    try:
        import torch
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # set_num_interop_threads só pode ser chamado uma vez por processo.
            # Se já foi chamado, ignora silenciosamente.
            pass
    except ImportError:
        pass

    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass

"""
runtime.py — Configuração de threading para evitar contenção entre processos.

CONTEXTO:
  PyTorch, ONNX Runtime e OpenCV criam pools internos de threads. Por padrão,
  cada um tenta usar todas as cores da CPU.

  Com N workers (threads) de OCR, cada biblioteca spawnando T threads internas,
  temos N*T threads brigando pelos mesmos núcleos. Resultado: contenção massiva
  e slowdown (chegou a ser 2x mais lento que serial nos testes).

  Solução: forçar 1 thread interna por biblioteca. O paralelismo vem das N
  threads de OCR, não de sub-threads de cada biblioteca.

  Nota sobre ONNX Runtime: não é limitado via API aqui. O ONNX é limitado
  via intra_op_num_threads=1 passado ao construtor do RapidOCR em pipeline.py
  (_make_ocr_engine). torch e cv2 são limitados via API neste módulo.
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
    terão lido os valores padrão e ignorarão as variáveis.
    """
    for var in _THREAD_ENV_VARS:
        os.environ[var] = "1"

    # Silencia logs verbosos do Ultralytics
    os.environ["YOLO_VERBOSE"] = "False"


def apply_library_thread_limits() -> None:
    """
    Limita threads diretamente via APIs de torch e cv2.

    Deve ser chamada DEPOIS que as libs foram importadas, complementando
    force_single_thread_env para as libs que ignoram variáveis de ambiente
    após já terem sido carregadas.

    ONNX Runtime é tratado separadamente via intra_op_num_threads=1 no
    construtor do RapidOCR (pipeline.py::_make_ocr_engine).
    """
    try:
        import torch
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # set_num_interop_threads só pode ser chamado uma vez por processo.
            # Se já foi chamado (ex: em warmup), ignora silenciosamente.
            pass
    except ImportError:
        pass

    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass

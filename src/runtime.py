"""
runtime.py — Configuração de threading e SSL para evitar problemas de ambiente.

THREADING
=========
PyTorch, ONNX Runtime e OpenCV criam pools internos de threads. Por padrão,
cada um tenta usar todos os núcleos da CPU.

Com N workers (threads) de OCR, cada biblioteca spawnando T threads internas,
temos N×T threads brigando pelos mesmos núcleos. Isso gera contenção massiva
e pode tornar o sistema mais lento que a execução serial.

SOLUÇÃO: forçar 1 thread interna por biblioteca via variáveis de ambiente (antes
do import) e via APIs diretas (depois do import). O paralelismo real vem das N
threads de OCR gerenciadas pelo ThreadPoolExecutor.

SSL
===
Em ambientes institucionais (universidades, empresas) e Windows sem OpenSSL
configurado, o Python pode falhar em verificar certificados HTTPS porque:
  1. O arquivo de cert do sistema não existe (ex: Windows sem OpenSSL instalado)
  2. Um proxy corporativo substitui certificados SSL por um cert próprio

patch_ssl_certifi() aponta o SSL para o bundle do certifi (já instalado como
dependência), resolvendo ambos os casos sem comprometer a segurança.

ORDEM DE CHAMADA
================
  patch_ssl_certifi()           ← ANTES de qualquer import de requests/urllib3
  force_single_thread_env()     ← ANTES de qualquer import de torch/cv2/onnx
  apply_library_thread_limits() ← DEPOIS dos imports
"""

import os


# ── SSL ───────────────────────────────────────────────────────────────────────

def patch_ssl_certifi() -> None:
    """
    Aponta o SSL do Python para o bundle de certificados do certifi.

    Resolve dois problemas comuns em ambientes institucionais e Windows:
      1. Arquivo de cert do sistema ausente — Python falha em todo HTTPS
      2. Proxy corporativo com interceptação SSL

    Usa setdefault() para não sobrescrever variáveis já definidas pelo usuário.
    Deve ser chamada ANTES de qualquer import de requests / urllib3 /
    huggingface_hub (que faz o download dos modelos OCR e YOLO).
    """
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE",      certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except ImportError:
        pass  # certifi não instalado — ignora silenciosamente


# ── Threading ─────────────────────────────────────────────────────────────────

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
    """
    try:
        import torch
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except ImportError:
        pass

    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass

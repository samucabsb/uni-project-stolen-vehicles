"""
logger.py — Logging estruturado centralizado.

Separa mensagens de diagnóstico (INFO, WARNING, DEBUG) do output visual
(barras de progresso, resultados por imagem), que continuam via print().

Uso típico:
    from src.logger import get_logger
    log = get_logger()
    log.info("[WARMUP] YOLO pronto.")
    log.warning("[AVISO] RAM disponível baixa.")

setup_logger() deve ser chamado UMA VEZ em main(), antes de qualquer
get_logger() nos submódulos.
"""

import logging
import sys

_LOGGER_NAME = "comparador"


def setup_logger(verbose: bool = False, quiet: bool = False) -> logging.Logger:
    """
    Configura e retorna o logger principal.

    verbose=True  → DEBUG e acima  (máximo detalhe)
    quiet=True    → WARNING e acima (suprime INFO)
    padrão        → INFO e acima

    Idempotente: chamadas subsequentes retornam o logger já configurado.
    """
    logger = logging.getLogger(_LOGGER_NAME)

    if logger.handlers:
        return logger  # Já configurado — evita handlers duplicados

    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logger.setLevel(level)
    logger.propagate = False  # Impede que mensagens subam ao root logger (evita duplicatas)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    # Formato minimalista: sem timestamp nem nível — lê como output normal do programa
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    return logger


def get_logger() -> logging.Logger:
    """
    Retorna o logger principal.
    setup_logger() deve ter sido chamado antes; caso contrário retorna um
    logger sem handlers (saída suprimida).
    """
    return logging.getLogger(_LOGGER_NAME)

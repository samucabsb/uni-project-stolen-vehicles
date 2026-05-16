"""
colors.py — Cores ANSI para o terminal.

Implementa um helper minimalista para aplicar cores ANSI ao output do console,
com fallback gracioso quando o terminal não suporta (output redirecionado para
arquivo, terminal antigo, etc).

Uso típico:
    from src.colors import C, paint
    print(paint("ROUBADO", C.RED_BOLD))
    print(f"{C.GREEN}OK{C.RESET}")
"""

import os
import sys


# ── Constantes ANSI ───────────────────────────────────────────────────────────

class C:
    """Códigos ANSI organizados por categoria."""

    RESET = "\033[0m"
    BOLD  = "\033[1m"
    DIM   = "\033[2m"

    # Cores básicas
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"
    GRAY    = "\033[90m"

    # Cores brilhantes (mais saturadas)
    RED_BRIGHT    = "\033[91m"
    GREEN_BRIGHT  = "\033[92m"
    YELLOW_BRIGHT = "\033[93m"
    CYAN_BRIGHT   = "\033[96m"

    # Combinações comuns
    RED_BOLD   = "\033[1;91m"
    GREEN_BOLD = "\033[1;92m"
    CYAN_BOLD  = "\033[1;96m"

    # Background (para alertas críticos)
    BG_RED = "\033[41;97m"  # fundo vermelho, texto branco


# ── Detecção de suporte ───────────────────────────────────────────────────────

_colors_enabled = True


def enable_ansi_colors() -> bool:
    """
    Ativa o processamento de códigos ANSI no terminal.

    No Windows 10+, o terminal suporta ANSI mas precisa ser "acordado".
    O truque `os.system("")` faz isso sem efeitos colaterais visíveis.

    Se a saída não for um terminal (foi redirecionada para arquivo, pipe, etc),
    desativa as cores para não poluir o arquivo com escape codes.

    Retorna True se cores estão ativas, False caso contrário.
    """
    global _colors_enabled

    if not sys.stdout.isatty():
        _colors_enabled = False
        return False

    if sys.platform == "win32":
        # Acorda o processamento ANSI no console do Windows
        os.system("")

    _colors_enabled = True
    return True


def paint(text: str, color: str) -> str:
    """Envolve texto com cor ANSI + reset. Vira no-op se cores desativadas."""
    if not _colors_enabled:
        return text
    return f"{color}{text}{C.RESET}"


def colors_enabled() -> bool:
    """Indica se cores estão ativas no momento."""
    return _colors_enabled

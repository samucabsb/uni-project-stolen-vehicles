"""
stolen.py — Ferramenta de gerenciamento de placas roubadas.

USO:
    python tools/stolen.py list           Mostra placas atualmente marcadas
    python tools/stolen.py add ABC1234    Adiciona uma placa à lista
    python tools/stolen.py add ABC1 DEF2  Adiciona várias placas de uma vez
    python tools/stolen.py remove ABC1234 Remove uma placa da lista
    python tools/stolen.py clear          Limpa toda a lista (com confirmação)
    python tools/stolen.py demo [N]       Marca N placas aleatórias do último
                                          run como roubadas (padrão: 5)

FLUXO DE DEMONSTRAÇÃO:
    1. python main.py --execution serial --no-interactive    (gera results.csv)
    2. python tools/stolen.py demo 5                         (marca 5 como roubadas)
    3. python main.py --execution serial --no-interactive    (agora detecta)
"""

import csv
import random
import sys
from pathlib import Path

# Permite importar src.* a partir de tools/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.colors import enable_ansi_colors, C, paint


# Paths resolvidos a partir da raiz do projeto
ROOT_DIR     = Path(__file__).resolve().parent.parent
STOLEN_FILE  = ROOT_DIR / "data" / "stolen_plates.txt"
RESULTS_FILE = ROOT_DIR / "data" / "output" / "results.csv"


# ── Persistência ──────────────────────────────────────────────────────────────

def _load_stolen() -> set:
    if not STOLEN_FILE.exists():
        return set()
    plates = set()
    with open(STOLEN_FILE, encoding="utf-8") as fh:
        for line in fh:
            plate = line.strip().upper()
            if plate:
                plates.add(plate)
    return plates


def _save_stolen(plates: set) -> None:
    STOLEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STOLEN_FILE, "w", encoding="utf-8") as fh:
        for plate in sorted(plates):
            fh.write(plate + "\n")


def _normalize(plate: str) -> str:
    """Normaliza uma placa: uppercase, sem espaços, sem hífens."""
    return plate.strip().upper().replace("-", "").replace(" ", "")


def _read_detected_plates() -> list:
    """Lê as placas detectadas no último run a partir do results.csv."""
    if not RESULTS_FILE.exists():
        return []
    plates = []
    with open(RESULTS_FILE, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            plate = row.get("plate_text", "").strip().upper()
            if plate and row.get("status") in ("OK", "ROUBADO"):
                plates.append(plate)
    return list(set(plates))


# ── Comandos ──────────────────────────────────────────────────────────────────

def cmd_list() -> None:
    plates = _load_stolen()
    if not plates:
        print(paint("Nenhuma placa marcada como roubada.", C.YELLOW))
        print(f"Arquivo: {STOLEN_FILE}")
        return
    print(paint(f"Placas roubadas ({len(plates)}):", C.RED_BOLD))
    for plate in sorted(plates):
        print(f"  {paint('•', C.RED)} {plate}")
    print(f"\nArquivo: {STOLEN_FILE}")


def cmd_add(args: list) -> None:
    if not args:
        print(paint("ERRO: informe ao menos uma placa.", C.RED_BOLD))
        print("Uso: python tools/stolen.py add PLACA1 [PLACA2 ...]")
        return
    plates = _load_stolen()
    added = []
    for plate in args:
        normalized = _normalize(plate)
        if normalized and normalized not in plates:
            plates.add(normalized)
            added.append(normalized)
    _save_stolen(plates)
    if added:
        print(paint(f"Adicionada(s) {len(added)} placa(s):", C.GREEN_BOLD))
        for p in added:
            print(f"  {paint('+', C.GREEN)} {p}")
    else:
        print(paint("Nenhuma placa nova adicionada (já estavam na lista).", C.YELLOW))
    print(f"Total agora: {len(plates)}")


def cmd_remove(args: list) -> None:
    if not args:
        print(paint("ERRO: informe ao menos uma placa.", C.RED_BOLD))
        return
    plates = _load_stolen()
    removed = []
    for plate in args:
        normalized = _normalize(plate)
        if normalized in plates:
            plates.discard(normalized)
            removed.append(normalized)
    _save_stolen(plates)
    if removed:
        print(paint(f"Removida(s) {len(removed)} placa(s):", C.YELLOW))
        for p in removed:
            print(f"  {paint('-', C.YELLOW)} {p}")
    else:
        print(paint("Nenhuma das placas informadas estava na lista.", C.YELLOW))
    print(f"Total agora: {len(plates)}")


def cmd_clear() -> None:
    plates = _load_stolen()
    if not plates:
        print(paint("A lista já está vazia.", C.GRAY))
        return
    answer = input(paint(
        f"Confirma apagar {len(plates)} placa(s)? [s/N]: ", C.YELLOW_BRIGHT
    )).strip().lower()
    if answer not in ("s", "sim", "y", "yes"):
        print("Cancelado.")
        return
    _save_stolen(set())
    print(paint(f"Lista limpa. {len(plates)} placa(s) removida(s).", C.GREEN))


def cmd_demo(args: list) -> None:
    """Marca N placas aleatórias do último run como roubadas."""
    n = 5
    if args:
        try:
            n = int(args[0])
        except ValueError:
            print(paint(f"Aviso: '{args[0]}' não é um número, usando padrão 5.", C.YELLOW))

    detected = _read_detected_plates()
    if not detected:
        print(paint(f"ERRO: nenhuma placa encontrada em {RESULTS_FILE}", C.RED_BOLD))
        print("Rode o sistema primeiro:")
        print("  python main.py --execution serial --no-interactive")
        return

    n = min(n, len(detected))
    sampled = random.sample(detected, n)

    plates = _load_stolen()
    for plate in sampled:
        plates.add(plate)
    _save_stolen(plates)

    print(paint(f"Marcadas {n} placa(s) aleatória(s) como roubadas:", C.RED_BOLD))
    for p in sampled:
        print(f"  {paint('+', C.RED)} {p}")
    print(paint("\nAgora rode o sistema para ver o status ROUBADO:", C.CYAN))
    print("  python main.py --execution serial --no-interactive")


# ── Main ──────────────────────────────────────────────────────────────────────

USAGE = """\
Uso: python tools/stolen.py <comando> [argumentos]

Comandos:
  list                  Lista as placas marcadas como roubadas
  add PLACA1 [PLACA2..] Adiciona placas à lista
  remove PLACA          Remove uma placa da lista
  clear                 Limpa toda a lista
  demo [N]              Marca N placas aleatórias do último run (padrão: 5)
"""


def main() -> None:
    enable_ansi_colors()

    if len(sys.argv) < 2:
        print(USAGE)
        return

    command = sys.argv[1].lower()
    args = sys.argv[2:]

    handlers = {
        "list":   lambda: cmd_list(),
        "add":    lambda: cmd_add(args),
        "remove": lambda: cmd_remove(args),
        "clear":  lambda: cmd_clear(),
        "demo":   lambda: cmd_demo(args),
    }

    handler = handlers.get(command)
    if handler is None:
        print(paint(f"Comando desconhecido: {command}\n", C.RED_BOLD))
        print(USAGE)
        return

    handler()


if __name__ == "__main__":
    main()

"""
dataset.py — Descoberta de imagens e carregamento da lista de placas roubadas.
"""

from pathlib import Path

from src.config import (
    INPUT_DIR, OUTPUT_DIR, CROPS_DIR, PREPROCESSED_DIR, MODELS_DIR,
    IMAGE_EXTENSIONS,
)


def ensure_directories() -> None:
    """Cria os diretórios necessários caso não existam."""
    for directory in (INPUT_DIR, OUTPUT_DIR, CROPS_DIR, PREPROCESSED_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def list_images(directory: Path) -> list:
    """Retorna lista ordenada de imagens válidas no diretório."""
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_stolen_plates(filepath: Path) -> set:
    """
    Carrega placas roubadas de um arquivo texto (uma por linha).

    Normalização aplicada: uppercase, remove espaços e hífens. Isso garante
    que `ABC-1234` e `abc 1234` casem com o resultado do OCR `ABC1234`.

    Cria o arquivo vazio se não existir.
    """
    if not filepath.exists():
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.touch()
        print(f"[INFO] Arquivo de placas roubadas criado vazio: {filepath}")
        print("[INFO] Use 'python tools/stolen.py add PLACA' para adicionar placas.")
        return set()

    plates: set = set()
    with open(filepath, encoding="utf-8") as fh:
        for line in fh:
            plate = line.strip().upper().replace(" ", "").replace("-", "")
            if plate:
                plates.add(plate)

    if plates:
        print(f"[INFO] {len(plates)} placa(s) roubada(s) carregada(s).")
    else:
        print("[INFO] Lista de placas roubadas vazia.")
        print("[INFO] Use 'python tools/stolen.py demo 5' para gerar um cenário de teste.")

    return plates

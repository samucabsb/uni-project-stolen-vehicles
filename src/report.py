"""
report.py — Geração de relatórios CSV e sumário visual no terminal.

CSVs gerados:
  results.csv          Um registro por imagem.
  performance_log.csv  Um registro por execução (acumula sem sobrescrever).
"""

import csv
from datetime import datetime
from pathlib import Path

from src.colors import C, paint
from src.config import STATUS_OK, STATUS_STOLEN, STATUS_UNIDENTIFIED, STATUS_ERROR


# ── Colunas dos CSVs ──────────────────────────────────────────────────────────

RESULTS_FIELDS = [
    "image", "plate_detected", "plate_text", "status",
    "yolo_time_s", "ocr_time_s", "total_time_s", "ocr_confidence",
    "worker_pid", "crop_path", "preprocessed_path", "error",
]

PERFORMANCE_FIELDS = [
    "timestamp", "execution_type",
    "workers_solicitados", "workers_efetivos",
    "total_images", "warmup_time_s",
    "total_processing_time_s", "avg_time_per_image_s",
    "min_time_per_image_s", "max_time_per_image_s",
    "throughput_img_per_s",
    "images_with_plate", "images_without_plate",
    "ok_count", "roubado_count", "nao_identificada_count", "error_count",
]


# ── CSVs ──────────────────────────────────────────────────────────────────────

def save_results_csv(results: list, filepath: Path) -> None:
    """Salva results.csv com um registro por imagem."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULTS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(paint(f"[CSV] Resultados salvos em: {filepath}", C.CYAN))


def save_performance_log(
    filepath: Path, results: list, elapsed: float, execution: str,
    workers_requested: int, workers_effective: int, warmup_time: float,
) -> None:
    """
    Acumula métricas de desempenho em performance_log.csv.
    Não sobrescreve registros anteriores — cada execução adiciona uma linha.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    times = [r.get("total_time_s", 0.0) for r in results]
    counts, plate_detected = _count_statuses(results)
    n = len(results)

    row = {
        "timestamp":                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "execution_type":           execution,
        "workers_solicitados":      workers_requested,
        "workers_efetivos":         workers_effective,
        "total_images":             n,
        "warmup_time_s":            round(warmup_time, 4),
        "total_processing_time_s":  round(elapsed, 4),
        "avg_time_per_image_s":     round(sum(times) / n, 4) if n else 0.0,
        "min_time_per_image_s":     round(min(times), 4) if times else 0.0,
        "max_time_per_image_s":     round(max(times), 4) if times else 0.0,
        "throughput_img_per_s":     round(n / elapsed, 4) if elapsed > 0 else 0.0,
        "images_with_plate":        plate_detected,
        "images_without_plate":     n - plate_detected,
        "ok_count":                 counts[STATUS_OK],
        "roubado_count":            counts[STATUS_STOLEN],
        "nao_identificada_count":   counts[STATUS_UNIDENTIFIED],
        "error_count":              counts[STATUS_ERROR],
    }

    file_exists = filepath.exists()
    with open(filepath, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PERFORMANCE_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(paint(f"[CSV] Performance salva em: {filepath}", C.CYAN))


def _count_statuses(results: list) -> tuple:
    """Conta status e quantas têm placa detectada."""
    counts = {STATUS_OK: 0, STATUS_STOLEN: 0,
              STATUS_UNIDENTIFIED: 0, STATUS_ERROR: 0}
    plate_detected = 0
    for r in results:
        s = r.get("status", STATUS_UNIDENTIFIED)
        counts[s] = counts.get(s, 0) + 1
        if r.get("plate_detected"):
            plate_detected += 1
    return counts, plate_detected


# ── Sumário visual no terminal ────────────────────────────────────────────────

_LINE_WIDTH = 64
_THICK_LINE = "=" * _LINE_WIDTH
_THIN_LINE  = "-" * _LINE_WIDTH


def print_summary(
    results: list, elapsed: float, execution: str,
    workers_requested: int, workers_effective: int, warmup_time: float,
    results_file: Path, performance_file: Path,
) -> None:
    """Imprime sumário visual completo com destaque para placas roubadas."""
    n = len(results)
    counts, plate_detected = _count_statuses(results)
    times = [r.get("total_time_s", 0.0) for r in results]
    avg   = sum(times) / n if n else 0.0
    thr   = n / elapsed if elapsed > 0 else 0.0

    workers_str = (str(workers_effective)
                   if workers_requested == workers_effective
                   else f"{workers_effective} (solicitado: {workers_requested})")

    # ── Cabeçalho ─────────────────────────────────────────────────────────────
    print()
    print(paint(_THICK_LINE, C.CYAN_BOLD))
    print(paint("  SUMÁRIO DE EXECUÇÃO", C.CYAN_BOLD))
    print(paint(_THICK_LINE, C.CYAN_BOLD))

    # ── Configuração ──────────────────────────────────────────────────────────
    print(f"  Modo              : {paint(execution.upper(), C.CYAN_BOLD)}")
    print(f"  Workers           : {workers_str}")
    print(f"  Imagens           : {n}")
    print(f"  Warm-up           : {warmup_time:.4f} s")
    print(f"  Tempo total       : {paint(f'{elapsed:.4f} s', C.YELLOW_BRIGHT)}")
    print(f"  Média por imagem  : {avg:.4f} s")
    print(f"  Throughput        : {paint(f'{thr:.2f} img/s', C.GREEN_BOLD)}")

    print(paint(_THIN_LINE, C.GRAY))

    # ── Contagens ─────────────────────────────────────────────────────────────
    print(f"  Com placa         : {plate_detected}")
    print(f"  Sem placa         : {n - plate_detected}")
    print(f"  Status OK         : {paint(str(counts[STATUS_OK]), C.GREEN)}")
    print(f"  Status ROUBADO    : {paint(str(counts[STATUS_STOLEN]),
                                          C.RED_BOLD if counts[STATUS_STOLEN] else C.GRAY)}")
    print(f"  Não identificado  : {paint(str(counts[STATUS_UNIDENTIFIED]), C.YELLOW)}")
    if counts[STATUS_ERROR]:
        print(f"  Erros             : {paint(str(counts[STATUS_ERROR]), C.MAGENTA)}")

    # ── Alerta de carros roubados (se houver) ─────────────────────────────────
    if counts[STATUS_STOLEN]:
        _print_stolen_alert(results)

    # ── Arquivos gerados ──────────────────────────────────────────────────────
    print(paint(_THIN_LINE, C.GRAY))
    print(f"  results.csv       : {results_file}")
    print(f"  performance_log   : {performance_file}")
    print(paint(_THICK_LINE, C.CYAN_BOLD))


def _print_stolen_alert(results: list) -> None:
    """Imprime a caixa de alerta destacada com a lista de placas roubadas."""
    stolen = [r for r in results if r.get("status") == STATUS_STOLEN]
    count  = len(stolen)

    print()
    print(paint(_THICK_LINE, C.RED_BOLD))
    title = f"  🚨  ALERTA: {count} VEÍCULO(S) ROUBADO(S) IDENTIFICADO(S)  🚨"
    print(paint(title, C.RED_BOLD))
    print(paint(_THICK_LINE, C.RED_BOLD))
    print()

    for r in stolen:
        plate = r.get("plate_text", "?")
        image = r.get("image", "?")
        print(paint(f"  • Placa: {plate}", C.RED_BOLD))
        print(paint(f"    Arquivo: {image}", C.RED))
        print()

    print(paint(_THICK_LINE, C.RED_BOLD))

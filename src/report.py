"""
report.py — Geração de relatórios CSV e sumário visual no terminal (v9).

CSVs gerados:
  results.csv          Um registro por imagem.
  performance_log.csv  Um registro por execução (acumula sem sobrescrever).
                       Inclui breakdown de tempo YOLO vs OCR para análise.

Nota sobre avg_time_per_image_s:
  Este campo usa elapsed / n (wall-clock / total de imagens), não a média dos
  total_time_s individuais. No modo parallel, total_time_s por imagem reflete
  apenas o OCR da thread; usar a média desses valores subestimaria o tempo
  real por imagem. O wall-clock captura o tempo real do pipeline completo.
"""

import csv
from datetime import datetime
from pathlib import Path

from src.colors import C, paint
from src.config import (
    STATUS_OK, STATUS_STOLEN, STATUS_UNIDENTIFIED, STATUS_ERROR, VERSION,
)


# ── Colunas dos CSVs ──────────────────────────────────────────────────────────

RESULTS_FIELDS = [
    "image", "plate_detected", "plate_text", "status",
    "yolo_time_s", "ocr_time_s", "total_time_s", "ocr_confidence",
    "worker_id",           # B2: renomeado de worker_pid
    "crop_path", "preprocessed_path", "error",
]

# v9: campos adicionais para timing breakdown.
PERFORMANCE_FIELDS = [
    "timestamp", "version", "execution_type",
    "workers_solicitados", "workers_efetivos",
    "total_images",
    "warmup_time_s",
    "total_processing_time_s",
    "yolo_stage_time_s",
    "ocr_stage_time_s",
    "avg_time_per_image_s",    # = elapsed / n (wall-clock, não média de total_time_s)
    "min_time_per_image_s",
    "max_time_per_image_s",
    "throughput_img_per_s",
    "images_with_plate", "images_without_plate",
    "ok_count", "roubado_count", "nao_identificada_count", "error_count",
]


# ── CSVs ──────────────────────────────────────────────────────────────────────

def save_results_csv(results: list, filepath: Path) -> None:
    """Salva results.csv (sobrescreve a cada execução)."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULTS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(paint(f"  [CSV] {filepath.name} ({len(results)} linhas)", C.GRAY))


def save_performance_log(
    filepath: Path, results: list, elapsed: float, execution: str,
    workers_requested: int, workers_effective: int, warmup_time: float,
    yolo_time: float = 0.0, ocr_time: float = 0.0,
) -> None:
    """
    Acumula métricas em performance_log.csv. Cada execução = 1 nova linha.

    yolo_stage_time_s: wall-clock do estágio YOLO (ambos os modos).
    ocr_stage_time_s:  soma dos tempos OCR individuais (não wall-clock do estágio,
                       pois no parallel rodou em paralelo).
    avg_time_per_image_s: elapsed / n — wall-clock real por imagem.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    times  = [r.get("total_time_s", 0.0) for r in results]
    counts, plate_detected = _count_statuses(results)
    n = len(results)

    # B1: avg usa wall-clock (elapsed/n), não média dos total_time_s
    row = {
        "timestamp":                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version":                  VERSION,
        "execution_type":           execution,
        "workers_solicitados":      workers_requested,
        "workers_efetivos":         workers_effective,
        "total_images":             n,
        "warmup_time_s":            round(warmup_time, 4),
        "total_processing_time_s":  round(elapsed, 4),
        "yolo_stage_time_s":        round(yolo_time, 4),
        "ocr_stage_time_s":         round(ocr_time, 4),
        "avg_time_per_image_s":     round(elapsed / n, 4) if n else 0.0,
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
        writer = csv.DictWriter(fh, fieldnames=PERFORMANCE_FIELDS,
                                extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(paint(f"  [CSV] {filepath.name} (linha adicionada)", C.GRAY))


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
    yolo_time: float = 0.0, ocr_time: float = 0.0,
) -> None:
    """Sumário visual com breakdown de timing por estágio (v9)."""
    n      = len(results)
    counts, plate_detected = _count_statuses(results)
    # B1: avg usa wall-clock (elapsed/n)
    avg    = elapsed / n if n else 0.0
    thr    = n / elapsed if elapsed > 0 else 0.0

    workers_str = (str(workers_effective)
                   if workers_requested == workers_effective
                   else f"{workers_effective} (solicitado: {workers_requested})")

    print()
    print(paint(_THICK_LINE, C.CYAN_BOLD))
    print(paint("  SUMÁRIO DE EXECUÇÃO", C.CYAN_BOLD))
    print(paint(_THICK_LINE, C.CYAN_BOLD))

    print(f"  Modo              : {paint(execution.upper(), C.CYAN_BOLD)}")
    print(f"  Workers           : {workers_str}")
    print(f"  Imagens           : {n}")
    print(f"  Warm-up           : {warmup_time:.4f} s")
    print(f"  Tempo total       : {paint(f'{elapsed:.4f} s', C.YELLOW_BRIGHT)}")
    print(f"  Média por imagem  : {avg:.4f} s")
    print(f"  Throughput        : {paint(f'{thr:.2f} img/s', C.GREEN_BOLD)}")

    # Breakdown de tempo
    if yolo_time > 0 or ocr_time > 0:
        total_stage = yolo_time + ocr_time
        if total_stage > 0:
            yolo_pct = yolo_time / total_stage * 100
            ocr_pct  = ocr_time  / total_stage * 100
            print(paint(_THIN_LINE, C.GRAY))
            print(f"  YOLO (detecção)   : {yolo_time:6.2f} s  ({yolo_pct:4.1f}%)")
            print(f"  OCR  (leitura)    : {ocr_time:6.2f} s  ({ocr_pct:4.1f}%)")

    print(paint(_THIN_LINE, C.GRAY))

    print(f"  Com placa         : {plate_detected}")
    print(f"  Sem placa         : {n - plate_detected}")
    print(f"  Status OK         : {paint(str(counts[STATUS_OK]), C.GREEN)}")
    stolen_color = C.RED_BOLD if counts[STATUS_STOLEN] else C.GRAY
    print(f"  Status ROUBADO    : {paint(str(counts[STATUS_STOLEN]), stolen_color)}")
    print(f"  Não identificado  : {paint(str(counts[STATUS_UNIDENTIFIED]), C.YELLOW)}")
    if counts[STATUS_ERROR]:
        print(f"  Erros             : {paint(str(counts[STATUS_ERROR]), C.MAGENTA)}")

    if counts[STATUS_STOLEN]:
        _print_stolen_alert(results)

    print(paint(_THIN_LINE, C.GRAY))
    print(f"  results.csv       : {results_file}")
    print(f"  performance_log   : {performance_file}")
    print(paint(_THICK_LINE, C.CYAN_BOLD))


def _print_stolen_alert(results: list) -> None:
    """Caixa de alerta destacada listando todas as placas roubadas."""
    stolen = [r for r in results if r.get("status") == STATUS_STOLEN]
    count  = len(stolen)

    print()
    print(paint(_THICK_LINE, C.RED_BOLD))
    print(paint(
        f"  🚨  ALERTA: {count} VEÍCULO(S) ROUBADO(S) IDENTIFICADO(S)  🚨",
        C.RED_BOLD
    ))
    print(paint(_THICK_LINE, C.RED_BOLD))
    print()

    for r in stolen:
        plate = r.get("plate_text", "?")
        image = r.get("image", "?")
        print(paint(f"  • Placa: {plate}", C.RED_BOLD))
        print(paint(f"    Arquivo: {image}", C.RED))
        print()

    print(paint(_THICK_LINE, C.RED_BOLD))

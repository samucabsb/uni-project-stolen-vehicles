"""
Comparador Paralelo de Placas — v7.0
====================================

Pipeline com YOLO (detecção) + RapidOCR (reconhecimento) +
ProcessPoolExecutor (paralelismo), com comparação contra base de placas
roubadas e métricas detalhadas de desempenho.

Mudanças v7:
  - Output visual com cores ANSI e destaque para placas roubadas
  - Revisão completa de código (deduplicação, type hints, magic numbers movidos)
  - Tratamento de erros robusto no fluxo principal
"""

# Configura env vars antes de QUALQUER import que use torch/cv2/onnx.
from src.runtime import force_single_thread_env
force_single_thread_env()

import argparse
import sys
import time
from pathlib import Path

from src.colors import enable_ansi_colors, C, paint
from src.config import (
    INPUT_DIR, OUTPUT_DIR, STOLEN_PLATES_FILE, DEFAULT_YOLO_MODEL,
)
from src.dataset import ensure_directories, list_images, load_stolen_plates
from src.detector import warmup_yolo
from src.ocr import warmup_ocr
from src.executor import run_tasks, print_hardware_info, get_hardware_info
from src.report import save_results_csv, save_performance_log, print_summary
from src.html_report import generate_html_report


# ── CLI parsing ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comparador Paralelo de Placas v7.0"
    )
    parser.add_argument("--execution", choices=["serial", "parallel"],
                        help="Modo de execução")
    parser.add_argument("--workers", type=int,
                        help="Quantidade de workers (modo parallel)")
    parser.add_argument("--yolo-model", default=str(DEFAULT_YOLO_MODEL),
                        help="Caminho do modelo YOLO (.pt)")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Não pergunta nada, usa defaults ou --execution/--workers")
    return parser.parse_args()


def _ask_choice(question: str, valid: list, default: str) -> str:
    """Pergunta uma escolha entre opções válidas, com default se Enter."""
    opts = "/".join(valid)
    while True:
        answer = input(f"{question} ({opts}) [padrão: {default}]: ").strip().lower()
        if not answer:
            return default
        if answer in valid:
            return answer
        print(paint(f"  Opção inválida. Use: {opts}", C.YELLOW))


def _ask_workers(default: int) -> int:
    """Pergunta um inteiro >= 1, com default se Enter."""
    while True:
        answer = input(f"Quantidade de workers [padrão: {default}]: ").strip()
        if not answer:
            return default
        try:
            value = int(answer)
            if value >= 1:
                return value
        except ValueError:
            pass
        print(paint("  Digite um inteiro >= 1", C.YELLOW))


def resolve_execution(args: argparse.Namespace) -> tuple:
    """Determina modo de execução e número de workers."""
    hw = get_hardware_info()
    physical = hw["physical_cores"]

    if args.no_interactive:
        execution = args.execution or "serial"
        if execution == "serial":
            return "serial", 1
        return "parallel", args.workers or physical

    print(paint("\n===== CONFIGURAÇÃO =====", C.CYAN_BOLD))
    execution = args.execution or _ask_choice("Execução", ["serial", "parallel"], "serial")
    if execution == "serial":
        return "serial", 1

    workers = args.workers or _ask_workers(physical)
    return "parallel", workers


# ── Warmup ────────────────────────────────────────────────────────────────────

def run_warmup(execution: str, yolo_model: str) -> float:
    """
    Executa o warm-up apropriado para o modo de execução.

    Em serial, carrega YOLO e OCR no processo principal (eles ficarão lá
    durante o processamento). Em paralelo, apenas valida que o modelo YOLO
    pode ser carregado, já que os workers vão criar suas próprias instâncias.
    """
    print(paint("\n===== WARM-UP =====", C.CYAN_BOLD))
    t0 = time.perf_counter()

    if execution == "serial":
        warmup_yolo(yolo_model)
        warmup_ocr()
    else:
        from ultralytics import YOLO
        YOLO(yolo_model)
        print("[VALIDAÇÃO] YOLO OK. OCR será carregado em cada worker.")

    elapsed = time.perf_counter() - t0
    print(f"  Concluído em {elapsed:.4f}s")
    return elapsed


# ── Fluxo principal ───────────────────────────────────────────────────────────

def main() -> int:
    enable_ansi_colors()

    args = parse_args()

    print_hardware_info()

    execution, workers = resolve_execution(args)
    yolo_model = args.yolo_model

    ensure_directories()

    if not Path(yolo_model).exists():
        print(paint(f"\n[ERRO] Modelo YOLO não encontrado: {yolo_model}", C.RED_BOLD))
        print("  Coloque o arquivo .pt em models/license_plate_detector.pt")
        return 1

    images = list_images(INPUT_DIR)
    if not images:
        print(paint(f"\n[ERRO] Nenhuma imagem encontrada em {INPUT_DIR}", C.RED_BOLD))
        return 1

    try:
        warmup_time = run_warmup(execution, yolo_model)
    except Exception as exc:
        print(paint(
            f"\n[ERRO] Falha no warm-up: {type(exc).__name__}: {exc}",
            C.RED_BOLD
        ))
        return 1

    stolen_plates = load_stolen_plates(STOLEN_PLATES_FILE)

    tasks = [
        {"image_path": str(p), "stolen_plates": stolen_plates, "yolo_model": yolo_model}
        for p in images
    ]

    print(paint("\n===== EXECUTANDO =====", C.CYAN_BOLD))
    print(f"  Execução  : {execution}")
    print(f"  Workers   : {workers}")
    print(f"  Imagens   : {len(images)}")
    print(f"  Modelo    : {yolo_model}\n")

    results, elapsed, workers_req, workers_eff = run_tasks(
        tasks, yolo_model, execution=execution, workers=workers
    )

    results_file     = OUTPUT_DIR / "results.csv"
    performance_file = OUTPUT_DIR / "performance_log.csv"
    html_report_file = OUTPUT_DIR / "report.html"

    save_results_csv(results, results_file)
    save_performance_log(performance_file, results, elapsed,
                         execution, workers_req, workers_eff, warmup_time)
    print_summary(results, elapsed, execution, workers_req, workers_eff,
                  warmup_time, results_file, performance_file)

    # Geração do HTML — DEPOIS do summary, fora da medição de tempo
    generate_html_report(results, elapsed, execution,
                         workers_req, workers_eff, warmup_time,
                         html_report_file)
    print(paint(f"\n[HTML] Relatório visual gerado: {html_report_file}", C.CYAN_BOLD))
    print(paint(f"       Abra no navegador para visualizar.", C.GRAY))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(paint("\n\n[INTERROMPIDO] Execução cancelada pelo usuário.", C.YELLOW))
        sys.exit(130)

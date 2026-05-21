"""
Comparador Paralelo de Placas v10.0
=====================================

Pipeline YOLO/ONNX (detecção) + fast-plate-ocr/CCT (reconhecimento) com
comparação contra lista de placas roubadas. Dois modos de execução:

  serial    1 thread, YOLO → OCR sequencial. Baseline para benchmark.

  parallel  Two-stage com threading:
              Estágio 1: YOLO/ONNX em batch (processo principal)
              Estágio 2: N threads executam fast-plate-ocr em paralelo
            Todas as threads compartilham um único modelo ONNX carregado no
            warmup — sem overhead de inicialização por thread.

Destaques da v10:
  • OCR: RapidOCR → CCT/fast-plate-ocr  (16x mais rápido, sem confusão L↔D)
  • YOLO: PyTorch  → ONNX (exportado automaticamente na 1ª execução, ~2-3x)
  • Speedup de paralelismo limitado pela Lei de Amdahl: com YOLO dominando
    ~95% do tempo serial, o benefício de N threads é próximo de 1.0x.
    O ONNX do YOLO reduz essa fração, recuperando parte do speedup.
"""

# ⚠️  force_single_thread_env() DEVE ser chamada antes de qualquer import de
# torch / cv2 / onnxruntime / ultralytics — ou as libs spawnam sub-threads
# que competem com os workers de OCR e geram contenção.
from src.runtime import force_single_thread_env, patch_ssl_certifi
force_single_thread_env()
patch_ssl_certifi()   # resolve SSL em Windows sem cert.pem e proxies corporativos

import argparse
import sys
import time
from pathlib import Path

from src.colors  import enable_ansi_colors, C, paint
from src.config  import (
    INPUT_DIR, OUTPUT_DIR, STOLEN_PLATES_FILE, DEFAULT_YOLO_MODEL, VERSION,
)
from src.dataset  import ensure_directories, list_images, load_stolen_plates
from src.detector import warmup_yolo, ensure_onnx_export
from src.ocr      import warmup_ocr
from src.executor import (
    run_tasks, print_hardware_info, get_hardware_info,
    recommend_workers, print_worker_diagnostics,
)
from src.report    import save_results_csv, save_performance_log, print_summary
from src.html_report import generate_html_report
from src.logger    import setup_logger, get_logger


VALID_EXECUTIONS = ["serial", "parallel"]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Comparador Paralelo de Placas v{VERSION}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--execution", choices=VALID_EXECUTIONS,
        help="Modo de execução (serial ou parallel)",
    )
    parser.add_argument(
        "--workers", type=int,
        help="Número de threads para o estágio OCR (modo parallel)",
    )
    parser.add_argument(
        "--yolo-model", default=str(DEFAULT_YOLO_MODEL),
        help="Caminho do modelo YOLO (.pt ou .onnx)",
    )
    parser.add_argument(
        "--no-interactive", action="store_true",
        help="Executa sem perguntas interativas, usando flags ou defaults",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Output de diagnóstico detalhado (nível DEBUG)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suprime mensagens INFO; mantém apenas WARNING+",
    )
    return parser.parse_args()


def _ask_choice(question: str, valid: list, default: str) -> str:
    """Solicita uma escolha entre opções válidas com fallback ao default."""
    opts = "/".join(valid)
    while True:
        answer = input(f"{question} ({opts}) [padrão: {default}]: ").strip().lower()
        if not answer:
            return default
        if answer in valid:
            return answer
        print(paint(f"  Opção inválida. Use: {opts}", C.YELLOW))


def _ask_workers(hw: dict) -> int:
    """
    Solicita o número de threads com recomendação baseada em hardware.

    Aceita qualquer inteiro >= 1. Valores acima do número de núcleos físicos
    são válidos para coletar dados de benchmark com oversubscription.
    """
    recommended = recommend_workers(hw)
    while True:
        answer = input(
            f"Quantidade de threads [recomendado: {recommended}, qualquer valor aceito]: "
        ).strip()
        if not answer:
            return recommended
        try:
            v = int(answer)
            if v >= 1:
                return v
        except ValueError:
            pass
        print(paint("  Digite um inteiro >= 1.", C.YELLOW))


def _check_ram_warning(hw: dict, workers: int) -> None:
    """
    Emite aviso se a RAM disponível pode ser insuficiente.

    fast-plate-ocr usa um singleton compartilhado (~100 MB), mas cada
    thread ainda consome RAM para buffers de imagem e overhead (~50-150 MB).
    Não bloqueia a execução — apenas informa o usuário.
    """
    log        = get_logger()
    avail      = hw["ram_avail_gb"]
    ram_needed = workers * 0.15   # estimativa conservadora por thread

    if avail < 0.5:
        log.warning(paint(
            f"[AVISO] RAM disponível muito baixa ({avail:.1f} GB). "
            "Considere usar modo serial.",
            C.YELLOW
        ))
    elif avail < ram_needed:
        log.warning(paint(
            f"[AVISO] {workers} threads podem precisar de ~{ram_needed:.1f} GB, "
            f"mas apenas {avail:.1f} GB disponível.",
            C.YELLOW
        ))


def resolve_execution(args: argparse.Namespace) -> tuple:
    """
    Determina modo e número de workers a partir dos args ou input interativo.

    print_worker_diagnostics é chamada aqui (seção CONFIGURAÇÃO), não durante
    o pipeline — evita output intercalado com barras de progresso.
    """
    hw = get_hardware_info()

    if args.no_interactive:
        execution = args.execution or "serial"
        if execution == "serial":
            return "serial", 1
        workers = args.workers or recommend_workers(hw)
        _check_ram_warning(hw, workers)
        print_worker_diagnostics(workers)
        return "parallel", workers

    print(paint("\n===== CONFIGURAÇÃO =====", C.CYAN_BOLD))
    execution = args.execution or _ask_choice("Execução", VALID_EXECUTIONS, "serial")

    if execution == "serial":
        return "serial", 1

    workers = args.workers or _ask_workers(hw)
    _check_ram_warning(hw, workers)
    print_worker_diagnostics(workers)
    return execution, workers


# ── Warmup ────────────────────────────────────────────────────────────────────

def run_warmup(execution: str, yolo_model: str) -> float:
    """
    Carrega e aquece YOLO e OCR. Retorna o tempo gasto.

    Ambos os modos (serial e parallel) pré-aquecem o OCR aqui.
    Isso elimina a latência de cold-start na primeira imagem real.
    """
    print(paint("\n===== WARM-UP =====", C.CYAN_BOLD))
    t0 = time.perf_counter()

    warmup_yolo(yolo_model)
    warmup_ocr()

    elapsed = time.perf_counter() - t0
    print(f"  Concluído em {elapsed:.4f}s")
    return elapsed


# ── Fluxo principal ───────────────────────────────────────────────────────────

def main() -> int:
    enable_ansi_colors()
    args = parse_args()
    setup_logger(verbose=args.verbose, quiet=args.quiet)

    print(paint(f"\n  Comparador de Placas v{VERSION}", C.CYAN_BOLD))
    print_hardware_info()

    execution, workers = resolve_execution(args)
    yolo_model         = args.yolo_model

    ensure_directories()

    if not Path(yolo_model).exists():
        print(paint(f"\n[ERRO] Modelo YOLO não encontrado: {yolo_model}", C.RED_BOLD))
        print("  Coloque o .pt em models/license_plate_detector.pt")
        return 1

    images = list_images(INPUT_DIR)
    if not images:
        print(paint(f"\n[ERRO] Nenhuma imagem em {INPUT_DIR}", C.RED_BOLD))
        return 1

    # ── Export ONNX (feito uma vez, reutilizado nas execuções seguintes) ───────
    onnx_path = Path(yolo_model).with_suffix(".onnx")
    needs_export = not onnx_path.exists() and not yolo_model.endswith(".onnx")

    if needs_export:
        print(paint("\n===== ONNX EXPORT =====", C.CYAN_BOLD))
        print("  Exportando YOLO para ONNX — feito UMA VEZ, ~30-60s...")
        t_export = time.perf_counter()
        yolo_model = ensure_onnx_export(yolo_model)
        elapsed_export = time.perf_counter() - t_export
        print(paint(
            f"  Concluído em {elapsed_export:.1f}s  →  {Path(yolo_model).name}",
            C.GREEN
        ))
    else:
        yolo_model = ensure_onnx_export(yolo_model)

    try:
        warmup_time = run_warmup(execution, yolo_model)
    except Exception as exc:
        print(paint(f"\n[ERRO] Warm-up falhou: {exc}", C.RED_BOLD))
        return 1

    stolen_plates = load_stolen_plates(STOLEN_PLATES_FILE)
    tasks = [
        {"image_path": str(p), "stolen_plates": stolen_plates}
        for p in images
    ]

    mode_label = "SERIAL" if execution == "serial" else f"PARALLEL · {workers} threads"
    print(paint("\n===== EXECUTANDO =====", C.CYAN_BOLD))
    print(f"  Modo      : {paint(mode_label, C.CYAN_BOLD)}")
    print(f"  Imagens   : {len(images)}")
    print(f"  Modelo    : {yolo_model}\n")

    results, elapsed, w_req, w_eff, yolo_time, ocr_time = run_tasks(
        tasks, yolo_model, execution=execution, workers=workers,
    )

    results_file     = OUTPUT_DIR / "results.csv"
    performance_file = OUTPUT_DIR / "performance_log.csv"
    html_report_file = OUTPUT_DIR / "report.html"

    print()
    save_results_csv(results, results_file)
    save_performance_log(
        performance_file, results, elapsed, execution, w_req, w_eff,
        warmup_time, yolo_time=yolo_time, ocr_time=ocr_time,
    )

    print_summary(
        results, elapsed, execution, w_req, w_eff, warmup_time,
        results_file, performance_file,
        yolo_time=yolo_time, ocr_time=ocr_time,
    )

    generate_html_report(
        results, elapsed, execution, w_req, w_eff, warmup_time,
        html_report_file,
    )
    print(paint(f"\n[HTML] Relatório: {html_report_file}", C.CYAN_BOLD))
    print(paint("       Abra no navegador para visualizar.", C.GRAY))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(paint("\n\n[INTERROMPIDO] Cancelado pelo usuário.", C.YELLOW))
        sys.exit(130)

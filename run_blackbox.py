"""
run_blackbox.py - Executor caixa preta com limites externos de thread.

TECNICA DA CAIXA PRETA v2
==========================
Trata main.py como opaco. Nao toca nenhum codigo interno.

Aplica externamente:
  1. Restricoes de thread via env vars NO SUBPROCESS, antes de qualquer
     import Python. Mais forte que patches internos que rodam depois das
     libs terem lido os defaults.

  2. --pin-cpu repassado para o main.py via CLI: pina cada worker num
     nucleo fisico diferente (via psutil), reduzindo migracao de cache.

  3. Speedup calculado pelo TEMPO INTERNO (Tempo total reportado pelo
     main.py), nao pelo wall clock. Isso exclui o overhead de spawn de
     processos (~5-11s que cresce com workers) do denominador, dando uma
     medicao justa do speedup computacional real.

     Wall clock ainda e mostrado na tabela para transparencia.

Uso:
    python run_blackbox.py
    python run_blackbox.py --workers 1 2 4 6 --pin-cpu
    python run_blackbox.py --timeout 240 --repeat 3
    python run_blackbox.py --yolo-model models/license_plate_detector_int8.onnx --workers 1 2 4 6
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PYTHON = sys.executable
MAIN   = str(Path(__file__).parent / "main.py")

# Env vars injetadas no subprocess ANTES de qualquer import Python.
# Mais confiavel que patches internos: o processo filho ja nasce com 1 thread.
_BASE_THREAD_ENV = {
    "OMP_NUM_THREADS":        "1",
    "OPENBLAS_NUM_THREADS":   "1",
    "MKL_NUM_THREADS":        "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS":    "1",
    "BLIS_NUM_THREADS":       "1",
    "KMP_BLOCKTIME":          "0",       # Intel OMP: sem busy-wait
    "OMP_WAIT_POLICY":        "passive", # OpenMP generico: sem busy-wait
    "GOMP_SPINCOUNT":         "0",       # GCC OpenMP: sem spin
    "KMP_SETTINGS":           "0",       # silencia log Intel OMP
    "YOLO_VERBOSE":           "False",
    "PYTHONIOENCODING":       "utf-8",
}


def _make_env() -> dict:
    env = os.environ.copy()
    env.update(_BASE_THREAD_ENV)
    return env


# ── Extracao de metricas do stdout ────────────────────────────────────────────

def _find(pattern: str, text: str, default: str = "") -> str:
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1).strip() if m else default


def parse_metrics(stdout: str) -> dict:
    """Extrai metricas do stdout do main.py via regex."""
    return {
        "n_images":    _find(r"Imagens\s*[:\|]\s*(\d+)",        stdout, "?"),
        "tempo_total": _find(r"Tempo total\s*[:\|]\s*([\d.]+)", stdout, ""),
        "throughput":  _find(r"Throughput\s*[:\|]\s*([\d.]+)",  stdout, ""),
        "detected":    _find(r"Com placa\s*[:\|]\s*(\d+)",      stdout, "?"),
        "stolen":      _find(r"ROUBAD[OA]\s*[:\|]\s*(\d+)",     stdout, "?"),
        "yolo_s":      _find(r"YOLO.*?:\s*([\d.]+)\s*s",        stdout, ""),
        "ocr_s":       _find(r"OCR.*?:\s*([\d.]+)\s*s",         stdout, ""),
    }


# ── Runner ────────────────────────────────────────────────────────────────────

def run_one(
    execution: str,
    workers: int = 1,
    timeout: int = 300,
    yolo_model: str | None = None,
    pin_cpu: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Executa main.py como subprocess isolado com env controlado.

    O speedup sera calculado com base em 'Tempo total' extraido do stdout
    (tempo interno de computacao, excluindo spawn/warmup de processos),
    nao pelo wall clock externo.
    """
    cmd = [
        PYTHON, MAIN,
        "--no-interactive",
        "--execution", execution,
        "--benchmark",
        "--no-html",
    ]
    if execution in ("parallel", "thread"):
        cmd += ["--workers", str(workers)]
    if yolo_model:
        cmd += ["--yolo-model", yolo_model]
    if pin_cpu and execution == "parallel":
        cmd += ["--pin-cpu"]   # repassado externamente - caixa preta nao altera o codigo

    label = "serial" if execution == "serial" else f"{execution}/{workers}w"
    pin_tag = "+pin" if (pin_cpu and execution == "parallel") else ""
    print(f"\n[{label}{pin_tag}] iniciando... (timeout={timeout}s)", flush=True)

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,          # bytes: evita UnicodeDecodeError do cp1252
            timeout=timeout,
            env=_make_env(),
            cwd=str(Path(__file__).parent),
        )
        wall_s     = round(time.perf_counter() - t0, 3)
        timed_out  = False
        returncode = proc.returncode
        stdout     = (proc.stdout or b"").decode("utf-8", errors="replace")
        stderr     = (proc.stderr or b"").decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired as exc:
        wall_s     = round(time.perf_counter() - t0, 3)
        timed_out  = True
        returncode = -9
        stdout     = (exc.stdout or b"").decode("utf-8", errors="replace") if exc.stdout else ""
        stderr     = (exc.stderr or b"").decode("utf-8", errors="replace") if exc.stderr else ""

    if verbose and stdout:
        for line in stdout.splitlines():
            print(f"    {line}")
    if stderr:
        for line in stderr.splitlines()[-8:]:
            print(f"  STDERR: {line}")

    ok     = not timed_out and returncode == 0
    status = "TIMEOUT" if timed_out else ("OK" if ok else f"ERRO({returncode})")
    m      = parse_metrics(stdout) if ok else {}

    t_int = float(m.get("tempo_total") or 0)
    print(f"  -> {status}  wall={wall_s}s  interno={t_int:.3f}s", flush=True)

    return {
        "label":      label + pin_tag,
        "execution":  execution,
        "workers":    workers,
        "status":     status,
        "wall_s":     wall_s,
        **m,
    }


# ── Calculo de speedup ────────────────────────────────────────────────────────

def compute_speedup(rows: list[dict], serial_internal: float, serial_wall: float) -> None:
    """
    Calcula dois speedups para cada linha:

      speedup_int  - baseado no Tempo total interno (exclui spawn/warmup).
                     Esta e a metrica correta para avaliar paralelismo
                     computacional: compara so o tempo de CPU, sem penalizar
                     o parallel pelo overhead fixo de subir processos.

      speedup_wall - baseado no wall clock externo (inclui tudo).
                     Mostra o tempo real do ponto de vista do usuario.

    Eficiencia = speedup_int / n_workers  (quanto do ideal foi aproveitado)
    """
    for r in rows:
        w = r["workers"] if r["execution"] == "parallel" else 1

        if r["status"] != "OK":
            r["speedup_int"] = r["speedup_wall"] = r["eficiencia"] = None
            r["speedup_ideal"] = float(w)
            continue

        t_int = float(r.get("tempo_total") or 0)

        r["speedup_int"]  = round(serial_internal / t_int, 3)  if (t_int  > 0 and serial_internal > 0) else None
        r["speedup_wall"] = round(serial_wall / r["wall_s"], 3) if serial_wall > 0 else None
        r["speedup_ideal"] = float(w)

        sp = r["speedup_int"] or r["speedup_wall"]
        r["eficiencia"] = round(sp / w, 3) if sp else None


# ── Tabela ────────────────────────────────────────────────────────────────────

def print_table(rows: list[dict]) -> None:
    print(f"\n{'='*80}")
    print("  RESUMO - CAIXA PRETA (speedup interno vs wall clock)")
    print(f"{'='*80}")
    header = (
        f"  {'Config':<16} {'Status':<8} {'Wall(s)':<9} {'Int(s)':<8}"
        f" {'Spd(int)':>9} {'Spd(wall)':>10} {'Ideal':>6} {'Efic%':>7} {'Imgs':>5}"
    )
    print(header)
    print(f"  {'-'*76}")

    for r in rows:
        si   = f"{r['speedup_int']:.3f}"   if r.get("speedup_int")  is not None else "  -- "
        sw   = f"{r['speedup_wall']:.3f}"  if r.get("speedup_wall") is not None else "  -- "
        eff  = f"{r['eficiencia']*100:.1f}%" if r.get("eficiencia") is not None else "  -- "
        idc  = f"{r.get('speedup_ideal',1):.0f}x"
        tint = r.get("tempo_total", "")
        tint_str = f"{float(tint):.3f}" if tint else "  --  "

        print(
            f"  {r['label']:<16} {r['status']:<8} {r['wall_s']:<9.3f} {tint_str:<8}"
            f" {si:>9} {sw:>10} {idc:>6} {eff:>7} {r.get('n_images','?'):>5}"
        )

    print()
    print("  Spd(int)  = serial_interno / parallel_interno  [sem overhead de spawn]")
    print("  Spd(wall) = serial_wall / parallel_wall        [tempo real do usuario]")
    print("  Efic%     = Spd(int) / n_workers * 100")

    # Destaca a melhor configuracao
    ok_rows = [r for r in rows if r.get("speedup_int") and r["execution"] == "parallel"]
    if ok_rows:
        best = max(ok_rows, key=lambda r: r["speedup_int"])
        print(f"\n  Melhor configuracao: {best['label']}"
              f"  (speedup interno = {best['speedup_int']:.3f}x,"
              f" eficiencia = {best['eficiencia']*100:.1f}%)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Caixa preta: limites externos + speedup por tempo interno"
    )
    p.add_argument("--timeout",    type=int, default=300,
                   help="Segundos antes de matar cada execucao (padrao: 300)")
    p.add_argument("--workers",    type=int, nargs="+", default=[1, 2, 4],
                   help="Workers parallel a testar (padrao: 1 2 4)")
    p.add_argument("--repeat",     type=int, default=1,
                   help="Repeticoes por configuracao (padrao: 1)")
    p.add_argument("--yolo-model", type=str,
                   help="Caminho do modelo YOLO (.pt ou .onnx)")
    p.add_argument("--pin-cpu",    action="store_true",
                   help="Repassa --pin-cpu ao main.py: fixa cada worker num nucleo fisico")
    p.add_argument("--exec-mode", dest="exec_mode", default="parallel",
                   choices=["parallel", "thread"],
                   help="Modo de execucao parallel: processos ou threads (padrao: parallel)")
    p.add_argument("--serial-only",   action="store_true")
    p.add_argument("--parallel-only", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Imprime stdout completo de cada execucao")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pin_note = "sim (--pin-cpu)" if args.pin_cpu else "nao"
    print(f"\n{'='*60}")
    print("  BLACKBOX RUNNER v2 - LIMITES EXTERNOS DE THREAD")
    print(f"  Timeout     : {args.timeout}s por execucao")
    print(f"  Repeticoes  : {args.repeat}")
    print(f"  Workers     : {args.workers}")
    print(f"  Env externo : OMP/MKL/BLIS/ORT = 1 thread antes de qualquer import")
    print(f"  Pin-CPU     : {pin_note}")
    print(f"  Speedup     : pelo Tempo total interno (exclui spawn)")
    print(f"{'='*60}")

    all_rows: list[dict] = []

    for rep in range(1, args.repeat + 1):
        if args.repeat > 1:
            print(f"\n{'#'*40}  repeticao {rep}/{args.repeat}")

        rep_rows: list[dict] = []
        serial_internal = serial_wall = None

        if not args.parallel_only:
            r = run_one("serial", timeout=args.timeout,
                        yolo_model=args.yolo_model, verbose=args.verbose)
            if args.repeat > 1:
                r["label"] = f"serial r{rep}"
            rep_rows.append(r)
            if r["status"] == "OK":
                serial_wall     = r["wall_s"]
                serial_internal = float(r.get("tempo_total") or 0) or serial_wall

        if not args.serial_only:
            exec_mode = getattr(args, "exec_mode", "parallel")
            for w in args.workers:
                r = run_one(exec_mode, workers=w, timeout=args.timeout,
                            yolo_model=args.yolo_model, pin_cpu=args.pin_cpu,
                            verbose=args.verbose)
                if args.repeat > 1:
                    r["label"] = f"p/{w}w r{rep}"
                rep_rows.append(r)

        if serial_internal and serial_wall:
            compute_speedup(rep_rows, serial_internal, serial_wall)
        elif args.parallel_only and rep_rows:
            ok = [r for r in rep_rows if r["status"] == "OK"]
            if ok:
                base_int  = min(float(r.get("tempo_total") or r["wall_s"]) for r in ok)
                base_wall = min(r["wall_s"] for r in ok)
                compute_speedup(rep_rows, base_int, base_wall)

        all_rows.extend(rep_rows)

    print_table(all_rows)

    if any(r["status"] != "OK" for r in all_rows):
        sys.exit(1)


if __name__ == "__main__":
    main()

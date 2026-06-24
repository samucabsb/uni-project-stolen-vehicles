<#
.SYNOPSIS
    Benchmark de speedup com isolamento de CPU via Docker.

.DESCRIPTION
    Roda o pipeline em serial (1 CPU) e em parallel (2, 4 CPUs) usando
    --cpuset-cpus do Docker para limitar exatamente quantos núcleos o
    container pode usar. Isso garante que:
      - serial (1 CPU)   → baseline honesto de 1 núcleo
      - 2 workers/2 CPUs → cada worker ocupa 1 núcleo → speedup ~2x
      - 4 workers/4 CPUs → cada worker ocupa 1 núcleo → speedup ~4x

.EXAMPLE
    .\docker-bench.ps1
    .\docker-bench.ps1 -YoloModel "models/license_plate_detector_int8.onnx"
    .\docker-bench.ps1 -MaxWorkers 8
#>

param(
    [string]$YoloModel  = "models/license_plate_detector.onnx",
    [int]$MaxWorkers    = 4,
    [string]$ImageName  = "plates-bench"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Caminhos (Docker Desktop no Windows aceita barras para frente) ─────────────

$ProjectDir = $PSScriptRoot
$ModelsDir  = "$ProjectDir\models"  -replace '\\', '/'
$DataDir    = "$ProjectDir\data"    -replace '\\', '/'
$HFCache    = "$env:USERPROFILE\.cache\huggingface" -replace '\\', '/'

# Volume flags reutilizados em todos os runs
$Volumes = @(
    "-v", "${ModelsDir}:/app/models:ro",
    "-v", "${DataDir}:/app/data",
    "-v", "${HFCache}:/root/.cache/huggingface"
)

# ── Verificações iniciais ──────────────────────────────────────────────────────

Write-Host ""
Write-Host "=== Benchmark Docker — Speedup por isolamento de CPU ===" -ForegroundColor Cyan
Write-Host ""

# Verifica se o Docker está rodando
$null = docker info 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERRO] Docker não está rodando. Abra o Docker Desktop e tente novamente." -ForegroundColor Red
    exit 1
}

# Verifica quantas CPUs o Docker enxerga
$dockerCpus = [int](docker run --rm --entrypoint="" python:3.11-slim python -c "import os; print(os.cpu_count())" 2>$null)
if ($dockerCpus -lt 1) { $dockerCpus = 4 }
Write-Host "  CPUs disponíveis no Docker: $dockerCpus" -ForegroundColor Gray

if ($MaxWorkers -gt $dockerCpus) {
    Write-Host "  [AVISO] MaxWorkers=$MaxWorkers > CPUs Docker=$dockerCpus — reduzindo para $dockerCpus" -ForegroundColor Yellow
    $MaxWorkers = $dockerCpus
}

# Garante que o modelo ONNX exista
$onnxPath = Join-Path $ProjectDir ($YoloModel -replace '/', '\')
if (-not (Test-Path $onnxPath)) {
    Write-Host "[ERRO] Modelo não encontrado: $onnxPath" -ForegroundColor Red
    Write-Host "       Rode primeiro: python main.py --execution serial --no-interactive" -ForegroundColor Yellow
    Write-Host "       (isso exporta o .pt para .onnx automaticamente)" -ForegroundColor Yellow
    exit 1
}

# ── Build da imagem ────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "--- BUILD: $ImageName ---" -ForegroundColor Cyan
docker build -t $ImageName $ProjectDir
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERRO] Build falhou." -ForegroundColor Red
    exit 1
}

# ── Função auxiliar ────────────────────────────────────────────────────────────

function Run-Bench {
    param([string]$Label, [string]$Cpuset, [string[]]$ExtraArgs)

    Write-Host ""
    Write-Host "--- $Label (--cpuset-cpus=$Cpuset) ---" -ForegroundColor Yellow

    $args = @("run", "--rm", "--cpuset-cpus=$Cpuset") + $Volumes + @(
        $ImageName,
        "python", "main.py",
        "--yolo-model", $YoloModel,
        "--benchmark",
        "--no-interactive"
    ) + $ExtraArgs

    & docker @args
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[AVISO] Saiu com código $LASTEXITCODE" -ForegroundColor Yellow
    }
}

# ── Runs ───────────────────────────────────────────────────────────────────────

# Serial: 1 CPU, 1 processo
Run-Bench -Label "SERIAL · 1 worker · 1 CPU" -Cpuset "0" -ExtraArgs @(
    "--execution", "serial"
)

# Parallel: 2 workers, 2 CPUs
if ($MaxWorkers -ge 2) {
    Run-Bench -Label "PARALLEL · 2 workers · 2 CPUs" -Cpuset "0-1" -ExtraArgs @(
        "--execution", "parallel", "--workers", "2"
    )
}

# Parallel: 4 workers, 4 CPUs
if ($MaxWorkers -ge 4) {
    Run-Bench -Label "PARALLEL · 4 workers · 4 CPUs" -Cpuset "0-3" -ExtraArgs @(
        "--execution", "parallel", "--workers", "4"
    )
}

# Parallel: 8 workers, 8 CPUs (se disponível)
if ($MaxWorkers -ge 8) {
    Run-Bench -Label "PARALLEL · 8 workers · 8 CPUs" -Cpuset "0-7" -ExtraArgs @(
        "--execution", "parallel", "--workers", "8"
    )
}

Write-Host ""
Write-Host "=== Benchmark concluído ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Para calcular o speedup:" -ForegroundColor Gray
Write-Host "  speedup(N) = tempo_serial / tempo_N_workers" -ForegroundColor Gray
Write-Host "  eficiência(N) = speedup(N) / N  (ideal = 1.0 = 100%)" -ForegroundColor Gray

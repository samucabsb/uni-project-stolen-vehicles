"""
html_report.py — Gerador de relatório HTML pós-execução.

Gera um arquivo HTML standalone em data/output/report.html com:
  - Cabeçalho com métricas grandes (tempo, throughput, workers, etc)
  - Alerta destacado para veículos roubados (cards com imagem)
  - Tabela completa com filtro/busca/ordenação
  - Modal para ampliar imagens

Princípios de design:
  - Zero impacto em performance: chamado DEPOIS do `elapsed` ser capturado
  - Standalone: um único arquivo HTML, sem dependências de CDN
  - Imagens via path relativo (crops/ e ../input/) — HTML não fica gigante
  - Funciona offline em qualquer navegador moderno
"""

import json
from datetime import datetime
from pathlib import Path

from src.config import (
    STATUS_OK, STATUS_STOLEN, STATUS_UNIDENTIFIED, STATUS_ERROR,
)


# ── CSS embutido ──────────────────────────────────────────────────────────────

_CSS = """
:root {
  --bg: #0a0a0b;
  --bg-card: #131316;
  --bg-elevated: #1a1a1f;
  --bg-hover: #22222a;
  --border: #2a2a33;
  --border-strong: #3a3a45;
  --text: #ededf0;
  --text-dim: #8a8a95;
  --text-faint: #5a5a65;
  --accent: #6366f1;
  --green: #10b981;
  --red: #ef4444;
  --red-bg: rgba(239, 68, 68, 0.1);
  --yellow: #f59e0b;
  --magenta: #a855f7;
  --shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
  --shadow-lg: 0 12px 32px rgba(0, 0, 0, 0.6);
}

* { box-sizing: border-box; margin: 0; padding: 0; }

html, body {
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  font-size: 14px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}

.container {
  max-width: 1280px;
  margin: 0 auto;
  padding: 32px 24px;
}

/* ── Hero ──────────────────────────────────────────────────── */

.hero {
  border-bottom: 1px solid var(--border);
  padding-bottom: 32px;
  margin-bottom: 32px;
}

.hero-title {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 28px;
  font-weight: 700;
  letter-spacing: -0.02em;
  margin-bottom: 8px;
}

.hero-badge {
  display: inline-block;
  padding: 2px 8px;
  background: var(--accent);
  color: white;
  font-size: 11px;
  font-weight: 600;
  border-radius: 4px;
  letter-spacing: 0.05em;
}

.hero-meta {
  color: var(--text-dim);
  font-size: 13px;
  margin-bottom: 24px;
}

.hero-meta strong { color: var(--text); font-weight: 600; }

/* ── Stats grid ────────────────────────────────────────────── */

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 16px;
}

.stat-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 20px;
  transition: border-color 0.15s, transform 0.15s;
}

.stat-card:hover { border-color: var(--border-strong); transform: translateY(-1px); }

.stat-value {
  font-size: 26px;
  font-weight: 700;
  letter-spacing: -0.02em;
  margin-bottom: 4px;
  font-variant-numeric: tabular-nums;
}

.stat-label {
  color: var(--text-dim);
  font-size: 12px;
  font-weight: 500;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.stat-value.green  { color: var(--green); }
.stat-value.red    { color: var(--red); }
.stat-value.yellow { color: var(--yellow); }

/* ── Stolen alert ──────────────────────────────────────────── */

.stolen-section {
  background: var(--red-bg);
  border: 1px solid var(--red);
  border-radius: 12px;
  padding: 24px;
  margin-bottom: 32px;
}

.stolen-header {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 18px;
  font-weight: 700;
  color: var(--red);
  margin-bottom: 20px;
}

.stolen-pulse {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  background: var(--red);
  border-radius: 50%;
  color: white;
  font-size: 16px;
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.6); }
  50%      { box-shadow: 0 0 0 8px rgba(239, 68, 68, 0); }
}

.stolen-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 16px;
}

.stolen-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
  cursor: pointer;
  transition: transform 0.15s, border-color 0.15s;
}

.stolen-card:hover {
  transform: translateY(-2px);
  border-color: var(--red);
}

.stolen-card-img {
  width: 100%;
  height: 140px;
  background: var(--bg-elevated);
  background-size: cover;
  background-position: center;
  border-bottom: 1px solid var(--border);
}

.stolen-card-body { padding: 14px; }

.stolen-card-plate {
  font-family: "SF Mono", Menlo, Consolas, monospace;
  font-size: 20px;
  font-weight: 700;
  letter-spacing: 0.05em;
  color: var(--red);
  margin-bottom: 4px;
}

.stolen-card-file {
  font-size: 11px;
  color: var(--text-dim);
  word-break: break-all;
  line-height: 1.3;
}

/* ── Filters ──────────────────────────────────────────────── */

.filters {
  display: flex;
  gap: 12px;
  align-items: center;
  margin-bottom: 16px;
  flex-wrap: wrap;
}

.search-input {
  flex: 1;
  min-width: 240px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 9px 14px;
  color: var(--text);
  font-family: inherit;
  font-size: 13px;
  outline: none;
  transition: border-color 0.15s;
}

.search-input:focus { border-color: var(--accent); }
.search-input::placeholder { color: var(--text-faint); }

.filter-buttons { display: flex; gap: 6px; }

.filter-btn {
  background: var(--bg-card);
  border: 1px solid var(--border);
  color: var(--text-dim);
  padding: 8px 14px;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 500;
  font-family: inherit;
  cursor: pointer;
  transition: all 0.15s;
}

.filter-btn:hover { color: var(--text); border-color: var(--border-strong); }

.filter-btn.active {
  background: var(--accent);
  border-color: var(--accent);
  color: white;
}

.filter-btn.active.stolen { background: var(--red); border-color: var(--red); }
.filter-btn.active.ok     { background: var(--green); border-color: var(--green); }

/* ── Results table ────────────────────────────────────────── */

.results-wrapper {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

thead {
  background: var(--bg-elevated);
  border-bottom: 1px solid var(--border);
}

th {
  text-align: left;
  padding: 12px 16px;
  font-weight: 600;
  color: var(--text-dim);
  font-size: 11px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

td {
  padding: 10px 16px;
  border-bottom: 1px solid var(--border);
  vertical-align: middle;
}

tbody tr { transition: background 0.1s; cursor: pointer; }
tbody tr:hover { background: var(--bg-hover); }
tbody tr.stolen { background: rgba(239, 68, 68, 0.05); }
tbody tr.stolen:hover { background: rgba(239, 68, 68, 0.1); }
tbody tr:last-child td { border-bottom: none; }

.thumb {
  width: 56px;
  height: 32px;
  background: var(--bg-elevated);
  background-size: cover;
  background-position: center;
  border-radius: 4px;
  border: 1px solid var(--border);
}

.thumb.empty {
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-faint);
  font-size: 14px;
}

.filename {
  color: var(--text-dim);
  font-size: 12px;
  font-family: "SF Mono", Menlo, Consolas, monospace;
  max-width: 280px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.plate {
  font-family: "SF Mono", Menlo, Consolas, monospace;
  font-weight: 600;
  letter-spacing: 0.04em;
}

.plate.empty { color: var(--text-faint); }

.badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.03em;
}

.badge.ok       { background: rgba(16, 185, 129, 0.15); color: var(--green); }
.badge.stolen   { background: rgba(239, 68, 68, 0.15);  color: var(--red); }
.badge.unident  { background: rgba(245, 158, 11, 0.15); color: var(--yellow); }
.badge.error    { background: rgba(168, 85, 247, 0.15); color: var(--magenta); }

.num {
  font-variant-numeric: tabular-nums;
  color: var(--text-dim);
  font-size: 12px;
}

.conf-bar {
  display: inline-block;
  width: 50px;
  height: 4px;
  background: var(--bg-elevated);
  border-radius: 2px;
  overflow: hidden;
  vertical-align: middle;
  margin-right: 6px;
}

.conf-bar-fill {
  height: 100%;
  background: var(--green);
  transition: width 0.3s;
}

.results-empty {
  padding: 60px 20px;
  text-align: center;
  color: var(--text-dim);
}

/* ── Modal ────────────────────────────────────────────────── */

.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.85);
  display: none;
  align-items: center;
  justify-content: center;
  z-index: 100;
  padding: 24px;
}

.modal-overlay.active { display: flex; }

.modal {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  max-width: 800px;
  width: 100%;
  max-height: 90vh;
  overflow-y: auto;
  box-shadow: var(--shadow-lg);
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  padding: 20px 24px;
  border-bottom: 1px solid var(--border);
}

.modal-title {
  font-size: 22px;
  font-weight: 700;
  font-family: "SF Mono", Menlo, Consolas, monospace;
  letter-spacing: 0.04em;
}

.modal-close {
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  color: var(--text-dim);
  width: 32px;
  height: 32px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 18px;
  line-height: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.15s;
}

.modal-close:hover { background: var(--bg-hover); color: var(--text); }

.modal-body { padding: 24px; }

.modal-image {
  width: 100%;
  border-radius: 8px;
  background: var(--bg-elevated);
  margin-bottom: 20px;
}

.modal-meta {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px;
  margin-top: 16px;
}

.modal-meta-item {
  background: var(--bg-elevated);
  padding: 12px;
  border-radius: 6px;
}

.modal-meta-label {
  font-size: 11px;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 4px;
}

.modal-meta-value {
  font-size: 14px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}

/* ── Footer ───────────────────────────────────────────────── */

.footer {
  margin-top: 40px;
  padding-top: 24px;
  border-top: 1px solid var(--border);
  color: var(--text-faint);
  font-size: 12px;
  text-align: center;
}

/* ── Section title ────────────────────────────────────────── */

.section-title {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 16px;
  color: var(--text);
}

/* ── Responsive ───────────────────────────────────────────── */

@media (max-width: 768px) {
  .container { padding: 20px 12px; }
  .hero-title { font-size: 22px; }
  .stat-value { font-size: 22px; }
  .filename { max-width: 140px; }
  th, td { padding: 8px 10px; }
}
"""


# ── JavaScript embutido ───────────────────────────────────────────────────────

_JS = """
const STATUS_LABELS = {
  OK: { class: 'ok', label: 'OK' },
  ROUBADO: { class: 'stolen', label: 'ROUBADO' },
  NAO_IDENTIFICADA: { class: 'unident', label: 'NÃO IDENTIFICADA' },
  ERRO: { class: 'error', label: 'ERRO' },
};

let currentFilter = 'all';
let currentSearch = '';

function renderTable() {
  const tbody = document.getElementById('results-body');
  tbody.innerHTML = '';

  const filtered = RESULTS.filter(r => {
    if (currentFilter !== 'all' && r.status !== currentFilter) return false;
    if (currentSearch) {
      const q = currentSearch.toLowerCase();
      return (r.plate_text || '').toLowerCase().includes(q) ||
             (r.image || '').toLowerCase().includes(q);
    }
    return true;
  });

  if (filtered.length === 0) {
    const tr = document.createElement('tr');
    tr.innerHTML = '<td colspan="7" class="results-empty">Nenhum resultado encontrado.</td>';
    tbody.appendChild(tr);
    return;
  }

  filtered.forEach((r, idx) => {
    const status = STATUS_LABELS[r.status] || { class: 'unident', label: r.status };
    const isStolen = r.status === 'ROUBADO';
    const conf = Math.round((r.ocr_confidence || 0) * 100);
    const thumb = r.crop_rel
      ? `<div class="thumb" style="background-image: url('${r.crop_rel}')"></div>`
      : `<div class="thumb empty">—</div>`;
    const plate = r.plate_text
      ? `<span class="plate">${r.plate_text}</span>`
      : `<span class="plate empty">—</span>`;

    const tr = document.createElement('tr');
    if (isStolen) tr.classList.add('stolen');
    tr.dataset.idx = RESULTS.indexOf(r);
    tr.innerHTML = `
      <td>${thumb}</td>
      <td><span class="filename" title="${r.image}">${r.image}</span></td>
      <td>${plate}</td>
      <td><span class="badge ${status.class}">${status.label}</span></td>
      <td>
        <span class="conf-bar"><span class="conf-bar-fill" style="width:${conf}%"></span></span>
        <span class="num">${conf}%</span>
      </td>
      <td><span class="num">${(r.total_time_s || 0).toFixed(3)}s</span></td>
      <td><span class="num">${r.worker_pid || '—'}</span></td>
    `;
    tr.addEventListener('click', () => openModal(parseInt(tr.dataset.idx)));
    tbody.appendChild(tr);
  });
}

function openModal(idx) {
  const r = RESULTS[idx];
  if (!r) return;

  document.getElementById('modal-title').textContent = r.plate_text || '— Sem placa —';

  const img = document.getElementById('modal-image');
  if (r.original_rel) {
    img.src = r.original_rel;
    img.style.display = 'block';
  } else {
    img.style.display = 'none';
  }

  const status = STATUS_LABELS[r.status] || { class: 'unident', label: r.status };
  const conf = Math.round((r.ocr_confidence || 0) * 100);

  document.getElementById('modal-meta').innerHTML = `
    <div class="modal-meta-item">
      <div class="modal-meta-label">Status</div>
      <div class="modal-meta-value"><span class="badge ${status.class}">${status.label}</span></div>
    </div>
    <div class="modal-meta-item">
      <div class="modal-meta-label">Confiança OCR</div>
      <div class="modal-meta-value">${conf}%</div>
    </div>
    <div class="modal-meta-item">
      <div class="modal-meta-label">Tempo Total</div>
      <div class="modal-meta-value">${(r.total_time_s || 0).toFixed(3)}s</div>
    </div>
    <div class="modal-meta-item">
      <div class="modal-meta-label">YOLO</div>
      <div class="modal-meta-value">${(r.yolo_time_s || 0).toFixed(3)}s</div>
    </div>
    <div class="modal-meta-item">
      <div class="modal-meta-label">OCR</div>
      <div class="modal-meta-value">${(r.ocr_time_s || 0).toFixed(3)}s</div>
    </div>
    <div class="modal-meta-item">
      <div class="modal-meta-label">Worker PID</div>
      <div class="modal-meta-value">${r.worker_pid || '—'}</div>
    </div>
  `;

  document.getElementById('modal-overlay').classList.add('active');
}

function closeModal() {
  document.getElementById('modal-overlay').classList.remove('active');
}

document.addEventListener('DOMContentLoaded', () => {
  renderTable();

  document.getElementById('search-input').addEventListener('input', (e) => {
    currentSearch = e.target.value;
    renderTable();
  });

  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentFilter = btn.dataset.filter;
      renderTable();
    });
  });

  document.getElementById('modal-overlay').addEventListener('click', (e) => {
    if (e.target.id === 'modal-overlay') closeModal();
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModal();
  });
});
"""


# ── Geração do HTML ───────────────────────────────────────────────────────────

def _make_relative_paths(result: dict) -> dict:
    """Converte paths absolutos em paths relativos ao report.html."""
    crop_path = result.get("crop_path", "")
    image     = result.get("image", "")

    # HTML é gravado em data/output/report.html
    # Crops ficam em data/output/crops/
    # Imagens originais em data/input/
    crop_rel = f"crops/{Path(crop_path).name}" if crop_path else ""
    # Tenta usar imagem original para o modal (cara em destaque, não só placa)
    original_rel = f"../input/{image}" if image else ""

    return {**result, "crop_rel": crop_rel, "original_rel": original_rel}


def _count_statuses(results: list) -> dict:
    counts = {STATUS_OK: 0, STATUS_STOLEN: 0,
              STATUS_UNIDENTIFIED: 0, STATUS_ERROR: 0}
    for r in results:
        s = r.get("status", STATUS_UNIDENTIFIED)
        counts[s] = counts.get(s, 0) + 1
    return counts


def _build_stats_cards(results: list, elapsed: float, throughput: float,
                       workers_effective: int) -> str:
    """HTML dos cards de estatística do hero."""
    counts = _count_statuses(results)

    def card(value: str, label: str, color_class: str = "") -> str:
        return (
            f'<div class="stat-card">'
            f'<div class="stat-value {color_class}">{value}</div>'
            f'<div class="stat-label">{label}</div>'
            f'</div>'
        )

    cards = [
        card(f"{elapsed:.2f}s", "Tempo Total"),
        card(f"{throughput:.2f}", "Imagens / s", "green"),
        card(str(len(results)), "Imagens Processadas"),
        card(str(workers_effective), "Workers Efetivos"),
        card(str(counts[STATUS_OK]), "Status OK", "green"),
        card(str(counts[STATUS_STOLEN]),
             "Roubados Detectados",
             "red" if counts[STATUS_STOLEN] else ""),
    ]
    return '<div class="stats-grid">' + "".join(cards) + '</div>'


def _build_stolen_section(stolen_results: list) -> str:
    """HTML da seção de alerta com cards de veículos roubados."""
    if not stolen_results:
        return ""

    cards = []
    for r in stolen_results:
        crop = r.get("crop_rel", "")
        bg_style = f'style="background-image: url(\'{crop}\')"' if crop else ""
        cards.append(
            f'<div class="stolen-card" data-idx="{r.get("_idx", 0)}">'
            f'<div class="stolen-card-img" {bg_style}></div>'
            f'<div class="stolen-card-body">'
            f'<div class="stolen-card-plate">{r.get("plate_text", "?")}</div>'
            f'<div class="stolen-card-file">{r.get("image", "?")}</div>'
            f'</div>'
            f'</div>'
        )

    return (
        f'<section class="stolen-section">'
        f'<div class="stolen-header">'
        f'<span class="stolen-pulse">🚨</span>'
        f'<span>ALERTA: {len(stolen_results)} VEÍCULO(S) ROUBADO(S) IDENTIFICADO(S)</span>'
        f'</div>'
        f'<div class="stolen-grid">' + "".join(cards) + '</div>'
        f'</section>'
    )


def _build_html(
    results: list, elapsed: float, execution: str,
    workers_requested: int, workers_effective: int, warmup_time: float,
) -> str:
    """Monta o HTML completo."""
    # Adiciona paths relativos e índice em cada resultado
    enriched = []
    for i, r in enumerate(results):
        rel = _make_relative_paths(r)
        rel["_idx"] = i
        enriched.append(rel)

    stolen = [r for r in enriched if r.get("status") == STATUS_STOLEN]
    counts = _count_statuses(results)
    throughput = len(results) / elapsed if elapsed > 0 else 0.0
    timestamp = datetime.now().strftime("%d/%m/%Y às %H:%M:%S")

    workers_str = (str(workers_effective)
                   if workers_requested == workers_effective
                   else f"{workers_effective} (solicitado: {workers_requested})")

    # Resultados como JSON para o JS renderizar
    # default=str para serializar Path, etc; ensure_ascii=False para acentos
    results_json = json.dumps(enriched, ensure_ascii=False, default=str)

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Relatório · Comparador de Placas v7</title>
<style>{_CSS}</style>
</head>
<body>
<div class="container">

<header class="hero">
  <h1 class="hero-title">
    <span>Comparador de Placas</span>
    <span class="hero-badge">v7</span>
  </h1>
  <div class="hero-meta">
    Executado em <strong>{timestamp}</strong> ·
    Modo <strong>{execution.upper()}</strong> ·
    <strong>{workers_str}</strong> worker(s) ·
    Warm-up <strong>{warmup_time:.2f}s</strong>
  </div>
  {_build_stats_cards(results, elapsed, throughput, workers_effective)}
</header>

{_build_stolen_section(stolen)}

<section>
  <div class="section-title">Resultados Detalhados</div>
  <div class="filters">
    <input id="search-input" type="search" class="search-input"
           placeholder="🔍  Buscar por placa ou nome de arquivo…">
    <div class="filter-buttons">
      <button class="filter-btn active" data-filter="all">Todos ({len(results)})</button>
      <button class="filter-btn ok" data-filter="OK">OK ({counts[STATUS_OK]})</button>
      <button class="filter-btn stolen" data-filter="ROUBADO">Roubados ({counts[STATUS_STOLEN]})</button>
      <button class="filter-btn" data-filter="NAO_IDENTIFICADA">N/I ({counts[STATUS_UNIDENTIFIED]})</button>
    </div>
  </div>

  <div class="results-wrapper">
    <table>
      <thead>
        <tr>
          <th>Placa Detectada</th>
          <th>Arquivo</th>
          <th>Texto</th>
          <th>Status</th>
          <th>Confiança</th>
          <th>Tempo</th>
          <th>PID</th>
        </tr>
      </thead>
      <tbody id="results-body"></tbody>
    </table>
  </div>
</section>

<div class="footer">
  Relatório gerado automaticamente pelo Comparador de Placas v7 · {timestamp}<br>
  Clique em qualquer linha para ver detalhes
</div>

</div>

<div class="modal-overlay" id="modal-overlay">
  <div class="modal">
    <div class="modal-header">
      <h2 class="modal-title" id="modal-title">—</h2>
      <button class="modal-close" onclick="closeModal()">×</button>
    </div>
    <div class="modal-body">
      <img class="modal-image" id="modal-image" alt="">
      <div class="modal-meta" id="modal-meta"></div>
    </div>
  </div>
</div>

<script>
const RESULTS = {results_json};
{_JS}
</script>

</body>
</html>"""


# ── Função pública ────────────────────────────────────────────────────────────

def generate_html_report(
    results: list, elapsed: float, execution: str,
    workers_requested: int, workers_effective: int, warmup_time: float,
    output_path: Path,
) -> None:
    """
    Gera o relatório HTML no caminho indicado.

    Esta função é chamada DEPOIS que o tempo de execução foi capturado,
    então não impacta as medições de performance.
    """
    html = _build_html(
        results, elapsed, execution,
        workers_requested, workers_effective, warmup_time,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

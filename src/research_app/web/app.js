const STAGES = ["Research", "Promotion", "Freeze", "Robustness", "Paper / Live"];

const HYPOTHESES = [
  {
    id: "H1",
    name: "Risk-off short base",
    target: "QQQ 15m",
    status: "replaced",
    tone: "warn",
    stageIndex: 2,
    progress: 48,
    summary: "The first rule found that horizon 6 had edge, but validation folds were too uneven for production.",
    outcome: "Kept as the parent branch. It showed enough signal to justify repair work, but not enough stability to trade directly.",
    next: "Use as lineage only. H1B and H1C are the actionable branches.",
    evidence: ["results/strategy/risk_off_short/QQQ/15min/report.md"],
    metrics: [
      ["Validation net", "+2.14%", "h=6, 2 bps"],
      ["Test net", "+6.67%", "h=6, 2 bps"],
      ["Validation folds", "3 / 5", "positive"],
      ["Test folds", "4 / 5", "positive"],
    ],
    backtest: [
      { label: "Validation", value: 0.0214 },
      { label: "Test", value: 0.0667 },
    ],
  },
  {
    id: "H1B",
    name: "Credit weak repair",
    target: "QQQ 15m",
    status: "needs research",
    tone: "warn",
    stageIndex: 3,
    progress: 68,
    summary: "Credit confirmation repaired concentration and passed promotion, but robustness showed dependence on an exact credit quantile.",
    outcome: "Frozen and reviewed, but not promoted to paper. The economics remain interesting and the robustness warning is explicit.",
    next: "Resolve the credit quantile dependency or leave H1B behind H1C.",
    evidence: [
      "results/strategy/risk_off_short/QQQ/15min/h1b_concentration_sweep/report.md",
      "results/strategy/risk_off_short/QQQ/15min/robustness/qqq_15min_risk_off_short_h1b_v1/report.md",
    ],
    metrics: [
      ["Validation net", "+7.23%", "142 trades"],
      ["Test net", "+7.08%", "135 trades"],
      ["Stress 5 bps", "positive", "validation and test"],
      ["Robustness", "6 / 27", "local passes"],
    ],
    backtest: [
      { label: "Validation", value: 0.0723 },
      { label: "Test", value: 0.0708 },
    ],
  },
  {
    id: "H1C",
    name: "Credit-spread repair",
    target: "QQQ 15m",
    status: "paper candidate",
    tone: "good",
    stageIndex: 4,
    progress: 92,
    summary: "The promoted H1 branch. It replaces the exact credit quantile with spread_credit_12 <= 0 and is now monitored in paper/live flow.",
    outcome: "Promoted to paper candidate after freeze review and pre-paper robustness. The active warning is cost sensitivity at 10 bps.",
    next: "Close paper observability: fills, slippage, ex-post PnL and deterioration against the frozen backtest expectations.",
    evidence: [
      "configs/strategy/qqq_15min_risk_off_short_h1c_v1.yaml",
      "results/strategy/risk_off_short/QQQ/15min/freeze_review/qqq_15min_risk_off_short_h1c_v1/manifest.yaml",
      "results/strategy/risk_off_short/QQQ/15min/robustness/qqq_15min_risk_off_short_h1c_v1/report.md",
    ],
    metrics: [
      ["Validation net", "+7.90%", "139 trades"],
      ["Test net", "+7.60%", "133 trades"],
      ["Avg trade", "5.7 bps", "validation/test"],
      ["Local passes", "6 / 9", "threshold sweep"],
    ],
    backtest: [
      { label: "Validation", value: 0.0790 },
      { label: "Test", value: 0.0760 },
    ],
    cost: [
      { label: "2 bps", validation: 0.0790, test: 0.0760 },
      { label: "5 bps", validation: 0.0373, test: 0.0361 },
      { label: "7.5 bps", validation: 0.0026, test: 0.0029 },
      { label: "10 bps", validation: -0.0322, test: -0.0304 },
    ],
  },
  {
    id: "H2.2",
    name: "Equity ORB pairs",
    target: "Relative spreads",
    status: "rejected",
    tone: "danger",
    stageIndex: 1,
    progress: 18,
    summary: "Continuation ORB on dollar-neutral equity spreads failed at the current cost level.",
    outcome: "Rejected. Every pair/window/horizon was negative in validation and test.",
    next: "Do not build options on top of this branch until the underlying timing edge is fixed.",
    evidence: ["results/strategy/equity_orb_pairs/5min/report.md"],
    metrics: [
      ["Best validation", "-8.12%", "2 bps"],
      ["Best test", "-7.88%", "2 bps"],
      ["Data coverage", "100%", "8 symbols"],
      ["Decision", "reject", "screening"],
    ],
    backtest: [
      { label: "Validation", value: -0.0812 },
      { label: "Test", value: -0.0788 },
    ],
  },
  {
    id: "H2.4",
    name: "Range-quality ORB",
    target: "Relative spreads",
    status: "not promoted",
    tone: "warn",
    stageIndex: 1,
    progress: 32,
    summary: "Opening-range quality created small positive pockets, but they failed control and concentration checks.",
    outcome: "Not promoted. Positive returns were too concentrated and weaker than controls.",
    next: "Only revisit if the control gap is fixed before adding more filters.",
    evidence: ["results/strategy/equity_orb_range_quality/5min/report.md"],
    metrics: [
      ["Best validation", "+1.07%", "XLY/XLP pocket"],
      ["Best test", "+1.17%", "screening"],
      ["Control check", "failed", "positive pocket"],
      ["Top5 share", "high", "concentrated"],
    ],
    backtest: [
      { label: "Validation", value: 0.0107 },
      { label: "Test", value: 0.0117 },
    ],
  },
  {
    id: "H2.5",
    name: "Failed ORB pairs",
    target: "Relative spreads",
    status: "rejected",
    tone: "danger",
    stageIndex: 1,
    progress: 18,
    summary: "Failed breakout / reversion on relative spreads also failed after costs.",
    outcome: "Rejected. Best validation and test candidates stayed negative.",
    next: "Keep as a closed branch unless a new market microstructure filter is introduced.",
    evidence: ["results/strategy/equity_orb_failed_pairs/5min/report.md"],
    metrics: [
      ["Best validation", "-7.11%", "2 bps"],
      ["Best test", "-5.69%", "2 bps"],
      ["Positive folds", "weak", "top candidates"],
      ["Decision", "reject", "screening"],
    ],
    backtest: [
      { label: "Validation", value: -0.0711 },
      { label: "Test", value: -0.0569 },
    ],
  },
  {
    id: "H3",
    name: "Options ORB",
    target: "1-4 DTE",
    status: "queued",
    tone: "info",
    stageIndex: 0,
    progress: 8,
    summary: "Options remain separated from H1C. No historical option backtest is promoted yet.",
    outcome: "Documented but not implemented as a paper strategy. It needs quote capture and explicit bid/ask assumptions first.",
    next: "Data probe, quote capture, and underlying control before buying historical option data.",
    evidence: ["docs/options_orb_hypothesis.md"],
    metrics: [
      ["Backtest", "none", "not promoted"],
      ["Data", "needed", "OPRA / IBKR"],
      ["Paper", "blocked", "until probe"],
      ["Branch", "separate", "not H1C"],
    ],
    backtest: [],
  },
];

const state = {
  snapshot: null,
  operations: null,
  h8Targets: [],
  h8Result: null,
  h8cResult: null,
  view: "operations",
  selectedHypothesisId: "H1C",
};

const views = {
  operations: "Strategy Map",
  h8lab: "H8 Regime Lab",
  candidates: "Evidence",
  runs: "Registry",
  reports: "Reports",
  decisions: "Decisions",
};

function byId(id) {
  return document.getElementById(id);
}

function text(value) {
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }
  return String(value);
}

function money(value) {
  const number = Number(value || 0);
  return number.toLocaleString("en-US", { style: "currency", currency: "USD" });
}

function pct(value) {
  if (!Number.isFinite(Number(value))) {
    return "-";
  }
  const number = Number(value) * 100;
  const sign = number > 0 ? "+" : "";
  return `${sign}${number.toFixed(2)}%`;
}

function bps(value) {
  if (!Number.isFinite(Number(value))) {
    return "-";
  }
  const number = Number(value) * 10000;
  const sign = number > 0 ? "+" : "";
  return `${sign}${number.toFixed(2)} bps`;
}

function bpsValue(value) {
  if (!Number.isFinite(Number(value))) {
    return "-";
  }
  const number = Number(value);
  const sign = number > 0 ? "+" : "";
  return `${sign}${number.toFixed(2)} bps`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function selectedHypothesis() {
  return HYPOTHESES.find((item) => item.id === state.selectedHypothesisId) || HYPOTHESES[0];
}

function toneForValue(value) {
  const raw = String(value || "").toLowerCase();
  if (!raw || raw === "-") return "";
  if (["pass", "passed", "ok", "open", "flat", "positive", "paper_candidate", "filled"].some((word) => raw.includes(word))) return "good";
  if (["warn", "warning", "missing", "closed", "pending", "unavailable", "skip", "hold", "queued", "not promoted", "needs"].some((word) => raw.includes(word))) return "warn";
  if (["reject", "rejected", "failed", "error", "blocked", "negative", "cancel"].some((word) => raw.includes(word))) return "danger";
  if (["note", "freeze", "research", "draft", "no_signal", "none"].some((word) => raw.includes(word))) return "info";
  return "";
}

function isNumericColumn(key) {
  return /(^|_)(count|events|orders|trades|quantity|qty|rows|artifacts|reports)$/.test(key)
    || /(pnl|net|sharpe|factor|return|price|entry|exit|threshold|margin)$/.test(key);
}

function renderCell(column, row) {
  const raw = row[column.key];
  const value = column.format ? column.format(raw, row) : text(raw);
  const safe = escapeHtml(value);
  const key = column.key.toLowerCase();
  const tone = toneForValue(value);
  const shouldBadge = tone && /(status|decision|validation|reconciliation|signal|event|type|source)/.test(key);
  if (shouldBadge) {
    return { html: `<span class="badge ${tone}">${safe}</span>`, className: "" };
  }
  return { html: safe, className: isNumericColumn(key) ? "numeric" : "" };
}

function setStatus(message, ok = true) {
  byId("status-line").textContent = message;
  const apiState = byId("api-state");
  apiState.textContent = ok ? "Connected" : "Error";
  apiState.className = `state-pill ${ok ? "" : "warn"}`;
}

async function requestJson(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || response.statusText);
  }
  return response.json();
}

function activateView(view) {
  state.view = view;
  byId("view-title").textContent = views[view];
  document.querySelectorAll(".view").forEach((node) => node.classList.toggle("is-active", node.id === view));
  document.querySelectorAll(".nav-button").forEach((node) => node.classList.toggle("is-active", node.dataset.view === view));
}

function renderTable(containerId, columns, rows, actions = null) {
  const container = byId(containerId);
  if (!container) return;
  if (!rows.length) {
    container.innerHTML = '<div class="empty-row">No rows</div>';
    return;
  }
  const actionHead = actions ? "<th>Action</th>" : "";
  const head = columns.map((column) => `<th>${escapeHtml(column.label)}</th>`).join("") + actionHead;
  const body = rows
    .map((row, index) => {
      const cells = columns
        .map((column) => {
          const cell = renderCell(column, row);
          return `<td class="${cell.className}">${cell.html}</td>`;
        })
        .join("");
      const action = actions ? `<td>${actions(row, index)}</td>` : "";
      return `<tr>${cells}${action}</tr>`;
    })
    .join("");
  container.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function chartSize(canvas) {
  const ctx = canvas.getContext("2d");
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, canvas.clientWidth || canvas.width);
  const height = Math.max(1, canvas.clientHeight || canvas.height);
  canvas.width = Math.floor(width * ratio);
  canvas.height = Math.floor(height * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return { ctx, width, height };
}

function chartBounds(values) {
  const finite = values.filter((value) => Number.isFinite(value));
  if (!finite.length) return { min: 0, max: 1 };
  let min = Math.min(...finite);
  let max = Math.max(...finite);
  if (min === max) {
    min -= 1;
    max += 1;
  }
  const pad = (max - min) * 0.12;
  return { min: min - pad, max: max + pad };
}

function drawLineChart(canvasId, rows, valueKey, options = {}) {
  const canvas = byId(canvasId);
  if (!canvas) return;
  const { ctx, width, height } = chartSize(canvas);
  const padding = { top: 20, right: 22, bottom: 34, left: 58 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  ctx.strokeStyle = "#d9dfd6";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, height - padding.bottom);
  ctx.lineTo(width - padding.right, height - padding.bottom);
  ctx.stroke();
  if (!rows.length) {
    ctx.fillStyle = "#667066";
    ctx.font = "13px sans-serif";
    ctx.fillText(options.emptyLabel || "No data", padding.left + 10, padding.top + 24);
    return;
  }
  const values = rows.map((row) => Number(row[valueKey])).filter((value) => Number.isFinite(value));
  const bounds = chartBounds(values);
  const xFor = (index) => padding.left + (rows.length <= 1 ? 0 : (index / (rows.length - 1)) * plotWidth);
  const yFor = (value) => padding.top + (1 - ((value - bounds.min) / (bounds.max - bounds.min))) * plotHeight;
  ctx.fillStyle = "#667066";
  ctx.font = "12px sans-serif";
  ctx.fillText(text(bounds.max), 8, padding.top + 4);
  ctx.fillText(text(bounds.min), 8, height - padding.bottom);
  ctx.strokeStyle = options.lineColor || "#00866f";
  ctx.lineWidth = 2;
  ctx.beginPath();
  rows.forEach((row, index) => {
    const value = Number(row[valueKey]);
    if (!Number.isFinite(value)) return;
    const x = xFor(index);
    const y = yFor(value);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  if (options.zeroLine && bounds.min < 0 && bounds.max > 0) {
    const y = yFor(0);
    ctx.strokeStyle = "#aab5ac";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

function drawH8RegimeChart() {
  const canvas = byId("h8-regime-chart");
  if (!canvas) return;
  const { ctx, width, height } = chartSize(canvas);
  const payload = state.h8Result?.chart || {};
  const rows = payload.rows || [];
  byId("h8-chart-label").textContent = rows.length ? `${payload.variant} fold ${payload.fold} ${payload.split}` : "-";
  byId("h8-chart-title").textContent = state.h8Result?.target_symbol ? `${state.h8Result.target_symbol} regime probability` : "Probability + price";
  const padding = { top: 26, right: 62, bottom: 42, left: 62 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  ctx.strokeStyle = "#d9dfd6";
  ctx.lineWidth = 1;
  [0, 0.25, 0.5, 0.75, 1].forEach((step) => {
    const y = padding.top + step * plotHeight;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
  });
  if (!rows.length) {
    ctx.fillStyle = "#667066";
    ctx.font = "14px sans-serif";
    ctx.fillText("No H8 run loaded", padding.left, padding.top + 26);
    return;
  }
  const xFor = (index) => padding.left + (rows.length <= 1 ? 0 : (index / (rows.length - 1)) * plotWidth);
  const pY = (value) => padding.top + (1 - Math.max(0, Math.min(1, Number(value) || 0))) * plotHeight;
  const prices = rows.map((row) => Number(row.close)).filter((value) => Number.isFinite(value));
  const priceBounds = chartBounds(prices);
  const priceY = (value) => padding.top + (1 - ((Number(value) - priceBounds.min) / (priceBounds.max - priceBounds.min))) * plotHeight;
  const regimeColors = {
    bull_trend: "rgba(0,134,111,0.10)",
    bear_stress: "rgba(177,61,45,0.10)",
    chop_compression: "rgba(47,111,150,0.08)",
    volatile_noise: "rgba(169,103,18,0.10)",
  };
  rows.forEach((row, index) => {
    const color = regimeColors[row.regime];
    if (!color) return;
    const x = xFor(index);
    const nextX = index < rows.length - 1 ? xFor(index + 1) : width - padding.right;
    ctx.fillStyle = color;
    ctx.fillRect(x, padding.top, Math.max(1, nextX - x), plotHeight);
  });
  const probabilityLines = [
    ["p_bull_trend", "#00866f", "bull"],
    ["p_bear_stress", "#b13d2d", "bear"],
    ["p_chop_compression", "#2f6f96", "chop"],
    ["p_volatile_noise", "#a96712", "noise"],
  ];
  probabilityLines.forEach(([key, color]) => {
    if (!rows.some((row) => Number.isFinite(Number(row[key])))) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    rows.forEach((row, index) => {
      const y = pY(row[key]);
      const x = xFor(index);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
  if (prices.length) {
    ctx.strokeStyle = "#151815";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    rows.forEach((row, index) => {
      const value = Number(row.close);
      if (!Number.isFinite(value)) return;
      const x = xFor(index);
      const y = priceY(value);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  ctx.fillStyle = "#667066";
  ctx.font = "12px sans-serif";
  ctx.textAlign = "right";
  ctx.fillText("100%", padding.left - 10, padding.top + 4);
  ctx.fillText("0%", padding.left - 10, height - padding.bottom);
  ctx.textAlign = "left";
  let lx = padding.left;
  probabilityLines.forEach(([, color, label]) => {
    ctx.fillStyle = color;
    ctx.fillRect(lx, height - 24, 10, 10);
    ctx.fillStyle = "#151815";
    ctx.fillText(label, lx + 15, height - 15);
    lx += 72;
  });
  ctx.fillStyle = "#151815";
  ctx.fillRect(lx, height - 24, 18, 3);
  ctx.fillText("price", lx + 24, height - 15);
  ctx.textAlign = "right";
  ctx.fillStyle = "#667066";
  ctx.fillText(text(priceBounds.max), width - 8, padding.top + 4);
  ctx.fillText(text(priceBounds.min), width - 8, height - padding.bottom);
  ctx.textAlign = "left";
}

function drawPortfolioMap() {
  const canvas = byId("portfolio-map");
  if (!canvas) return;
  const { ctx, width, height } = chartSize(canvas);
  const left = 46;
  const right = width - 28;
  const top = 44;
  const bottom = height - 44;
  const colWidth = (right - left) / (STAGES.length - 1);
  ctx.strokeStyle = "#d9dfd6";
  ctx.lineWidth = 1;
  STAGES.forEach((stage, index) => {
    const x = left + index * colWidth;
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, bottom);
    ctx.stroke();
    ctx.fillStyle = "#667066";
    ctx.font = "11px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(stage, x, 22);
  });
  ctx.textAlign = "left";
  HYPOTHESES.forEach((hypothesis, index) => {
    const x = left + hypothesis.stageIndex * colWidth;
    const y = top + 22 + index * ((bottom - top - 22) / Math.max(1, HYPOTHESES.length - 1));
    const color = hypothesis.tone === "good" ? "#00866f" : hypothesis.tone === "danger" ? "#b13d2d" : hypothesis.tone === "warn" ? "#a96712" : "#2f6f96";
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = hypothesis.id === state.selectedHypothesisId ? 4 : 2;
    ctx.beginPath();
    ctx.arc(x, y, hypothesis.id === state.selectedHypothesisId ? 11 : 8, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 0.12;
    ctx.beginPath();
    ctx.arc(x, y, 24, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#151815";
    ctx.font = hypothesis.id === state.selectedHypothesisId ? "700 13px sans-serif" : "12px sans-serif";
    ctx.fillText(hypothesis.id, Math.min(x + 16, width - 70), y + 4);
  });
}

function drawBacktestChart(hypothesis) {
  const canvas = byId("backtest-chart");
  if (!canvas) return;
  const { ctx, width, height } = chartSize(canvas);
  const data = hypothesis.backtest || [];
  byId("backtest-chart-label").textContent = hypothesis.id;
  byId("backtest-chart-title").textContent = `${hypothesis.id} validation vs test`;
  if (!data.length) {
    ctx.fillStyle = "#667066";
    ctx.font = "14px sans-serif";
    ctx.fillText("No promoted backtest yet", 24, 42);
    return;
  }
  const padding = { top: 28, right: 24, bottom: 42, left: 64 };
  const values = data.map((item) => item.value);
  const bounds = chartBounds([...values, 0]);
  const plotHeight = height - padding.top - padding.bottom;
  const zeroY = padding.top + (1 - ((0 - bounds.min) / (bounds.max - bounds.min))) * plotHeight;
  ctx.strokeStyle = "#d9dfd6";
  ctx.lineWidth = 1;
  ctx.font = "11px sans-serif";
  ctx.fillStyle = "#667066";
  ctx.textAlign = "right";
  [0, 0.25, 0.5, 0.75, 1].forEach((step) => {
    const value = bounds.max - step * (bounds.max - bounds.min);
    const y = padding.top + step * plotHeight;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
    ctx.fillText(pct(value), padding.left - 10, y + 4);
  });
  ctx.strokeStyle = "#8f9b91";
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, height - padding.bottom);
  ctx.lineTo(width - padding.right, height - padding.bottom);
  ctx.stroke();
  ctx.strokeStyle = "#aab5ac";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(padding.left, zeroY);
  ctx.lineTo(width - padding.right, zeroY);
  ctx.stroke();
  ctx.setLineDash([]);
  const barWidth = Math.min(110, (width - padding.left - padding.right) / (data.length * 2.2));
  data.forEach((item, index) => {
    const center = padding.left + (index + 0.65) * ((width - padding.left - padding.right) / data.length);
    const y = padding.top + (1 - ((item.value - bounds.min) / (bounds.max - bounds.min))) * plotHeight;
    const topY = Math.min(y, zeroY);
    const barHeight = Math.max(3, Math.abs(zeroY - y));
    ctx.fillStyle = item.value >= 0 ? "#00866f" : "#b13d2d";
    ctx.fillRect(center - barWidth / 2, topY, barWidth, barHeight);
    ctx.fillStyle = "#151815";
    ctx.font = "700 13px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(pct(item.value), center, topY - 8);
    ctx.fillStyle = "#667066";
    ctx.font = "12px sans-serif";
    ctx.fillText(item.label, center, height - 16);
  });
  ctx.textAlign = "left";
}

function drawCostChart(hypothesis) {
  const canvas = byId("cost-chart");
  if (!canvas) return;
  const { ctx, width, height } = chartSize(canvas);
  const data = hypothesis.cost || [];
  if (!data.length) {
    byId("cost-chart-label").textContent = hypothesis.id;
    ctx.fillStyle = "#667066";
    ctx.font = "14px sans-serif";
    ctx.fillText("No cost stress curve for this module", 24, 42);
    return;
  }
  const padding = { top: 26, right: 32, bottom: 42, left: 62 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const values = data.flatMap((item) => [item.validation, item.test]);
  const bounds = chartBounds([...values, 0]);
  const xFor = (index) => padding.left + (data.length <= 1 ? 0 : (index / (data.length - 1)) * plotWidth);
  const yFor = (value) => padding.top + (1 - ((value - bounds.min) / (bounds.max - bounds.min))) * plotHeight;
  const zeroY = yFor(0);
  byId("cost-chart-label").textContent = `${data[0].label} - ${data[data.length - 1].label}`;
  ctx.strokeStyle = "#d9dfd6";
  ctx.lineWidth = 1;
  ctx.font = "11px sans-serif";
  ctx.fillStyle = "#667066";
  ctx.textAlign = "right";
  [0, 0.25, 0.5, 0.75, 1].forEach((step) => {
    const value = bounds.max - step * (bounds.max - bounds.min);
    const y = padding.top + step * plotHeight;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
    ctx.fillText(pct(value), padding.left - 10, y + 4);
  });
  ctx.strokeStyle = "#8f9b91";
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, height - padding.bottom);
  ctx.lineTo(width - padding.right, height - padding.bottom);
  ctx.stroke();
  ctx.strokeStyle = "#aab5ac";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(padding.left, zeroY);
  ctx.lineTo(width - padding.right, zeroY);
  ctx.stroke();
  ctx.setLineDash([]);
  [
    ["validation", "#00866f"],
    ["test", "#2f6f96"],
  ].forEach(([key, color]) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    data.forEach((item, index) => {
      const x = xFor(index);
      const y = yFor(item[key]);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = color;
    data.forEach((item, index) => {
      ctx.beginPath();
      ctx.arc(xFor(index), yFor(item[key]), 4, 0, Math.PI * 2);
      ctx.fill();
    });
  });
  ctx.fillStyle = "#667066";
  ctx.font = "12px sans-serif";
  ctx.textAlign = "center";
  data.forEach((item, index) => ctx.fillText(item.label, xFor(index), height - 15));
  ctx.textAlign = "left";
  ctx.fillStyle = "#00866f";
  ctx.fillText("validation", padding.left, 16);
  ctx.fillStyle = "#2f6f96";
  ctx.fillText("test", padding.left + 86, 16);
}

function drawSignalReadiness() {
  const canvas = byId("signal-readiness-chart");
  if (!canvas) return;
  const { ctx, width, height } = chartSize(canvas);
  const summary = state.operations?.signal_diagnostics?.summary || {};
  const passed = Number(summary.passed_conditions || 0);
  const required = Number(summary.required_conditions || 0);
  const unavailable = Number(summary.unavailable_conditions || 0);
  const rate = required ? passed / required : 0;
  const cx = width * 0.36;
  const cy = height * 0.52;
  const radius = Math.min(width, height) * 0.30;
  ctx.lineWidth = 18;
  ctx.strokeStyle = "#edf1ec";
  ctx.beginPath();
  ctx.arc(cx, cy, radius, Math.PI * 0.75, Math.PI * 2.25);
  ctx.stroke();
  ctx.strokeStyle = rate >= 0.8 ? "#00866f" : rate >= 0.5 ? "#a96712" : "#b13d2d";
  ctx.beginPath();
  ctx.arc(cx, cy, radius, Math.PI * 0.75, Math.PI * (0.75 + 1.5 * rate));
  ctx.stroke();
  ctx.fillStyle = "#151815";
  ctx.font = "700 34px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(`${passed}/${required || 0}`, cx, cy + 9);
  ctx.fillStyle = "#667066";
  ctx.font = "12px sans-serif";
  ctx.fillText("conditions", cx, cy + 32);
  ctx.textAlign = "left";
  const lines = [
    ["passed", passed, "#00866f"],
    ["missing", Math.max(0, required - passed - unavailable), "#b13d2d"],
    ["unavailable", unavailable, "#a96712"],
  ];
  lines.forEach((line, index) => {
    const y = 58 + index * 34;
    ctx.fillStyle = line[2];
    ctx.fillRect(width * 0.66, y - 11, 12, 12);
    ctx.fillStyle = "#151815";
    ctx.font = "700 14px sans-serif";
    ctx.fillText(`${line[1]} ${line[0]}`, width * 0.66 + 20, y);
  });
}

function renderPortfolioStats() {
  const promoted = HYPOTHESES.filter((item) => item.stageIndex >= 4).length;
  const rejected = HYPOTHESES.filter((item) => item.tone === "danger").length;
  const liveStatus = state.operations?.summary?.current_status || "unknown";
  const rows = [
    ["Modules", HYPOTHESES.length],
    ["Promoted", promoted],
    ["Rejected", rejected],
    ["Live state", liveStatus],
  ];
  byId("portfolio-stats").innerHTML = rows
    .map(([label, value]) => `<div class="portfolio-stat"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`)
    .join("");
}

function renderHypothesisCards() {
  byId("hypothesis-cards").innerHTML = HYPOTHESES.map((hypothesis) => {
    const active = hypothesis.id === state.selectedHypothesisId ? " is-active" : "";
    return `
      <button class="hypothesis-card${active}" type="button" data-hypothesis-id="${escapeHtml(hypothesis.id)}">
        <div class="code">
          <strong>${escapeHtml(hypothesis.id)}</strong>
          <span class="badge ${hypothesis.tone}">${escapeHtml(hypothesis.status)}</span>
        </div>
        <h4>${escapeHtml(hypothesis.name)}</h4>
        <p>${escapeHtml(hypothesis.target)}</p>
        <div class="mini-meter ${hypothesis.tone}"><span style="width:${hypothesis.progress}%"></span></div>
      </button>`;
  }).join("");
}

function renderSelectedHypothesis() {
  const hypothesis = selectedHypothesis();
  byId("selected-title").textContent = `${hypothesis.id}: ${hypothesis.name}`;
  byId("selected-status").textContent = hypothesis.status;
  byId("selected-status").className = `badge ${hypothesis.tone}`;
  byId("selected-summary").textContent = hypothesis.summary;
  byId("stage-rail").innerHTML = STAGES.map((stage, index) => {
    let klass = "stage-node";
    if (index < hypothesis.stageIndex) klass += " is-done";
    if (index === hypothesis.stageIndex) klass += hypothesis.tone === "danger" ? " is-blocked" : " is-current";
    return `<div class="${klass}"><span>${escapeHtml(`Step ${index + 1}`)}</span><strong>${escapeHtml(stage)}</strong></div>`;
  }).join("");
  byId("metric-gallery").innerHTML = hypothesis.metrics
    .map(([label, value, note]) => `
      <div class="metric-tile">
        <span>${escapeHtml(label)}</span>
        <strong>${escapeHtml(value)}</strong>
        <small>${escapeHtml(note)}</small>
      </div>`)
    .join("");
  byId("story-outcome").textContent = hypothesis.outcome;
  byId("story-next").textContent = hypothesis.next;
  byId("story-evidence").innerHTML = hypothesis.evidence
    .map((path) => `<span>${escapeHtml(path)}</span>`)
    .join("");
  drawBacktestChart(hypothesis);
  drawCostChart(hypothesis);
  drawPortfolioMap();
}

function renderLiveTrading() {
  const summary = state.operations?.summary || {};
  const diagnostics = state.operations?.signal_diagnostics || {};
  const signalSummary = diagnostics.summary || {};
  const daemon = state.operations?.daemon_status || {};
  const scheduler = daemon.scheduler || {};
  const runner = daemon.runner_summary || {};
  const reconciliation = runner.pre_trade_reconciliation || {};
  const marketState = scheduler.market_open ? "market open" : "market closed";
  const liveTone = scheduler.market_open ? "good" : "warn";
  byId("sidebar-live-state").textContent = "H1C";
  byId("sidebar-live-detail").textContent = summary.current_status || "paper monitor";
  byId("metric-state").textContent = text(summary.current_status);
  byId("live-signal-action").textContent = signalSummary.action || summary.last_auto_decision || "-";
  byId("live-market-pill").textContent = marketState;
  byId("live-market-pill").className = `badge ${liveTone}`;
  byId("daemon-pill").textContent = `daemon ${runner.decision || "unknown"}`;
  byId("daemon-pill").className = `daemon-pill ${reconciliation.severity === "ok" ? "ok" : "warn"}`;
  const conditions = diagnostics.conditions || [];
  byId("live-checks").innerHTML = conditions.slice(0, 7).map((row) => {
    const klass = row.unavailable ? "warn" : row.passed ? "good" : "danger";
    return `<span class="check-chip ${klass}">${escapeHtml(row.label)}</span>`;
  }).join("");
  drawSignalReadiness();
}

function renderDaemon() {
  const daemon = state.operations?.daemon_status || {};
  const runner = daemon.runner_summary || {};
  const scheduler = daemon.scheduler || {};
  const reconciliation = runner.pre_trade_reconciliation || {};
  byId("daemon-updated").textContent = daemon.mtime_utc || daemon.created_at_utc || "-";
  const fields = [
    ["Market", scheduler.market_open ? "open" : "closed"],
    ["Decision", runner.decision],
    ["Reason", runner.reason],
    ["Reconciliation", reconciliation.decision],
    ["State", reconciliation.state_status],
    ["Sleep", scheduler.sleep_seconds],
  ];
  byId("daemon-status").innerHTML = fields
    .map(([key, value]) => {
      const displayValue = text(value);
      const tone = ["Market", "Decision", "Reconciliation", "State"].includes(key) ? toneForValue(displayValue) : "";
      const renderedValue = tone
        ? `<span class="badge ${tone}">${escapeHtml(displayValue)}</span>`
        : escapeHtml(displayValue);
      return `<dt>${escapeHtml(key)}</dt><dd>${renderedValue}</dd>`;
    })
    .join("");
}

function renderOperationsDetail() {
  const operations = state.operations || {};
  const charts = operations.charts || {};
  const diagnostics = operations.signal_diagnostics || {};
  const signalSummary = diagnostics.summary || {};
  const signalConditions = diagnostics.conditions || [];
  const autoRuns = operations.auto_runs || [];
  const submitted = operations.submitted_orders || [];
  const pnl = operations.pnl_events || [];
  const stateEvents = operations.state_events || [];
  const priceSeries = charts.price || [];
  const pnlSeries = charts.pnl || [];
  const summary = operations.summary || {};
  byId("metric-submitted").textContent = text(summary.submitted_orders || 0);
  byId("metric-pnl").textContent = money(summary.realized_pnl || 0);
  byId("metric-pnl-events").textContent = text(summary.pnl_events || 0);
  byId("metric-auto-runs").textContent = text(summary.auto_runs || autoRuns.length || 0);
  byId("auto-run-count-label").textContent = `${autoRuns.length} rows`;
  byId("submitted-count-label").textContent = `${submitted.length} rows`;
  byId("state-event-count-label").textContent = `${stateEvents.length} rows`;
  byId("price-chart-label").textContent = `${priceSeries.length} bars`;
  byId("pnl-chart-label").textContent = `${pnlSeries.length} events`;
  byId("signal-summary-label").textContent = `${signalSummary.passed_conditions || 0} / ${signalSummary.required_conditions || 0}`;
  drawLineChart("price-chart", priceSeries, "close", { lineColor: "#00866f", emptyLabel: "No QQQ price data" });
  drawLineChart("pnl-chart", pnlSeries, "cumulative_realized_pnl", {
    lineColor: "#a96712",
    zeroLine: true,
    emptyLabel: "No realized PnL events yet",
  });
  renderTable("signal-condition-table", [
    { key: "label", label: "Condition" },
    { key: "value", label: "Value" },
    { key: "operator", label: "Rule" },
    { key: "threshold", label: "Threshold" },
    { key: "margin", label: "Margin" },
    { key: "status", label: "Status" },
  ], signalConditions.map((row) => ({
    ...row,
    status: row.unavailable ? "unavailable" : row.passed ? "pass" : "missing",
  })));
  renderTable("auto-run-table", [
    { key: "created_at_utc", label: "Created" },
    { key: "decision", label: "Decision" },
    { key: "reason", label: "Reason" },
    { key: "signal_action", label: "Signal" },
    { key: "ticket_quantity", label: "Qty" },
    { key: "pre_trade_reconciliation", label: "Pre-recon" },
  ], autoRuns.slice(0, 80));
  renderTable("submitted-order-table", [
    { key: "created_at_utc", label: "Created" },
    { key: "symbol", label: "Symbol" },
    { key: "action", label: "Action" },
    { key: "quantity", label: "Qty" },
    { key: "order_type", label: "Type" },
    { key: "execution_status", label: "Status" },
  ], submitted);
  renderTable("state-event-table", [
    { key: "created_at_utc", label: "Created" },
    { key: "event_type", label: "Event" },
    { key: "previous_status", label: "Previous" },
    { key: "new_status", label: "New" },
    { key: "ticket_action", label: "Action" },
    { key: "signal_timestamp", label: "Signal timestamp" },
  ], stateEvents.slice(0, 80));
}

function h8GateColumns() {
  return [
    { key: "variant", label: "Variant" },
    { key: "horizon_bars", label: "H" },
    { key: "probability_threshold", label: "P min" },
    { key: "cost_bps", label: "Cost" },
    { key: "positive_folds", label: "+ folds" },
    { key: "trades", label: "Trades" },
    { key: "net_return", label: "Net", format: pct },
    { key: "avg_trade_net_pooled", label: "Avg", format: bps },
    { key: "avg_trade_net_min", label: "Worst fold", format: bps },
    { key: "max_drawdown_max", label: "Max DD", format: pct },
  ];
}

function h8cSelectedColumns() {
  return [
    { key: "split", label: "Split" },
    { key: "fold", label: "Fold" },
    { key: "gate_mode", label: "Mode" },
    { key: "threshold", label: "P" },
    { key: "min_probability_gap", label: "Gap" },
    { key: "regime_threshold", label: "Regime P" },
    { key: "trades", label: "Trades" },
    { key: "long_trades", label: "Long" },
    { key: "short_trades", label: "Short" },
    { key: "net_return", label: "Net", format: pct },
    { key: "avg_trade_net", label: "Avg", format: bps },
    { key: "max_drawdown", label: "DD", format: pct },
  ];
}

function h8cCostColumns() {
  return [
    { key: "split", label: "Split" },
    { key: "cost_scenario", label: "Scenario" },
    { key: "effective_cost_bps", label: "Eff cost", format: bpsValue },
    { key: "notional_usd", label: "Notional", format: money },
    { key: "positive_folds", label: "+ folds" },
    { key: "trades", label: "Trades" },
    { key: "gross_return", label: "Gross", format: pct },
    { key: "total_cost", label: "Cost", format: pct },
    { key: "net_return", label: "Net", format: pct },
    { key: "avg_trade_net_pooled", label: "Avg", format: bps },
    { key: "daily_sharpe_mean", label: "Sharpe" },
    { key: "max_drawdown_max", label: "DD", format: pct },
  ];
}

function renderH8Targets() {
  const rows = state.h8Targets || [];
  if (!rows.length) {
    byId("h8-targets").innerHTML = '<span class="check-chip warn">no local feature sets</span>';
    return;
  }
  byId("h8-targets").innerHTML = rows.slice(0, 8).map((row) => {
    const label = `${row.target_symbol} ${row.timeframe} ${row.sessions || 0}s`;
    return `<button class="target-chip" type="button" data-h8-target="${escapeHtml(row.target_symbol)}">${escapeHtml(label)}</button>`;
  }).join("");
}

function renderH8cResult() {
  const result = state.h8cResult;
  if (!result) {
    byId("h8c-result-title").textContent = "Position Probability";
    byId("h8c-run-state").textContent = "idle";
    byId("h8c-run-state").className = "badge info";
    byId("h8c-metrics").innerHTML = "";
    renderTable("h8c-selected-table", h8cSelectedColumns(), []);
    renderTable("h8c-cost-table", h8cCostColumns(), []);
    return;
  }
  const ok = result.available;
  byId("h8c-run-state").textContent = ok ? "ready" : "blocked";
  byId("h8c-run-state").className = `badge ${ok ? "good" : "warn"}`;
  byId("h8c-result-title").textContent = ok ? `${result.target_symbol} H8c position` : `${result.target_symbol || "H8c"} unavailable`;
  const selected = result.summary?.selected_gate || {};
  const validation = result.summary?.validation || {};
  const test = result.summary?.test || {};
  byId("h8c-metrics").innerHTML = ok ? [
    ["Selected", text(selected.gate_mode), `P ${text(selected.threshold)} / H8 ${text(selected.regime_threshold)}`],
    ["Val avg", bps(validation.avg_trade_net_pooled), `${validation.positive_folds || 0}/${validation.folds || 0} folds`],
    ["Test avg", bps(test.avg_trade_net_pooled), `${test.positive_folds || 0}/${test.folds || 0} folds`],
    ["Test net", pct(test.net_return), `${test.trades || 0} trades`],
  ].map(([label, value, note]) => `
    <div class="metric-tile">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
      <small>${escapeHtml(note)}</small>
    </div>`).join("") : `
    <div class="metric-tile wide">
      <span>Status</span>
      <strong>${escapeHtml(result.status || "missing")}</strong>
      <small>${escapeHtml(result.reason || "")}</small>
    </div>`;
  renderTable("h8c-selected-table", h8cSelectedColumns(), result.selected_metrics || []);
  renderTable("h8c-cost-table", h8cCostColumns(), result.cost_sensitivity || []);
}

function renderH8Lab() {
  renderH8Targets();
  const result = state.h8Result;
  if (!result) {
    byId("h8-result-title").textContent = "No run";
    byId("h8-dataset-label").textContent = "-";
    byId("h8-run-state").textContent = "idle";
    byId("h8-run-state").className = "badge info";
    byId("h8-metrics").innerHTML = "";
    renderTable("h8-validation-table", h8GateColumns(), []);
    renderTable("h8-test-table", h8GateColumns(), []);
    renderTable("h8-profile-table", [
      { key: "variant", label: "Variant" },
      { key: "fold", label: "Fold" },
      { key: "regime", label: "Regime" },
      { key: "frequency", label: "Freq", format: pct },
      { key: "mean_duration", label: "Duration" },
      { key: "mean_max_prob", label: "P max", format: pct },
      { key: "mean_entropy", label: "Entropy", format: pct },
      { key: "mean_mom_z", label: "Mom" },
      { key: "mean_vol_z", label: "Vol" },
      { key: "mean_eff_z", label: "Eff" },
    ], []);
    renderH8cResult();
    drawH8RegimeChart();
    return;
  }
  const ok = result.available;
  byId("h8-run-state").textContent = ok ? "ready" : "blocked";
  byId("h8-run-state").className = `badge ${ok ? "good" : "warn"}`;
  byId("h8-result-title").textContent = ok ? `${result.target_symbol} H8` : `${result.target_symbol || "H8"} unavailable`;
  byId("h8-dataset-label").textContent = ok && result.dataset ? `${result.dataset.sessions} sessions` : result.status || "-";
  const best = result.summary?.best_validation || {};
  const holdout = result.summary?.matching_test || {};
  byId("h8-metrics").innerHTML = ok ? [
    ["Best val avg", bps(best.avg_trade_net_pooled), `${best.positive_folds || 0}/${best.folds || 0} folds`],
    ["Best val net", pct(best.net_return), `${best.trades || 0} trades`],
    ["Matched test avg", bps(holdout.avg_trade_net_pooled), `${holdout.positive_folds || 0}/${holdout.folds || 0} folds`],
    ["Dataset", text(result.dataset?.sessions), result.dataset?.feature_path || ""],
  ].map(([label, value, note]) => `
    <div class="metric-tile">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
      <small>${escapeHtml(note)}</small>
    </div>`).join("") : `
    <div class="metric-tile wide">
      <span>Status</span>
      <strong>${escapeHtml(result.status || "missing")}</strong>
      <small>${escapeHtml(result.reason || "")}</small>
    </div>`;
  renderTable("h8-validation-table", h8GateColumns(), result.aggregate_validation || []);
  renderTable("h8-test-table", h8GateColumns(), result.aggregate_test || []);
  renderTable("h8-profile-table", [
    { key: "variant", label: "Variant" },
    { key: "fold", label: "Fold" },
    { key: "regime", label: "Regime" },
    { key: "frequency", label: "Freq", format: pct },
    { key: "mean_duration", label: "Duration" },
    { key: "mean_max_prob", label: "P max", format: pct },
    { key: "mean_entropy", label: "Entropy", format: pct },
    { key: "mean_mom_z", label: "Mom" },
    { key: "mean_vol_z", label: "Vol" },
    { key: "mean_eff_z", label: "Eff" },
  ], result.profiles || []);
  renderH8cResult();
  drawH8RegimeChart();
}

function renderStrategyMap() {
  renderPortfolioStats();
  renderHypothesisCards();
  renderSelectedHypothesis();
  renderLiveTrading();
  renderDaemon();
  renderOperationsDetail();
}

function candidateColumns() {
  return [
    { key: "target_symbol", label: "Target" },
    { key: "timeframe", label: "Timeframe" },
    { key: "candidate_id", label: "Candidate" },
    { key: "source_file", label: "Source" },
    { key: "decision", label: "Decision" },
    { key: "validation_status", label: "Validation" },
    { key: "test_net_primary", label: "Test net" },
    { key: "test_sharpe_primary", label: "Sharpe" },
    { key: "run_id", label: "Run" },
  ];
}

function renderCandidates() {
  const rows = state.snapshot?.candidates || [];
  const filter = byId("candidate-filter").value.trim().toLowerCase();
  const filtered = filter
    ? rows.filter((row) => JSON.stringify(row).toLowerCase().includes(filter))
    : rows;
  byId("candidate-count-label").textContent = `${filtered.length} rows`;
  renderTable("candidate-table", candidateColumns(), filtered);
}

function renderRuns() {
  const rows = state.snapshot?.runs || [];
  byId("run-count-label").textContent = `${rows.length} rows`;
  renderTable("run-table", [
    { key: "instrument", label: "Instrument" },
    { key: "timeframe", label: "Timeframe" },
    { key: "run_type", label: "Type" },
    { key: "source_kind", label: "Source" },
    { key: "artifact_count", label: "Artifacts" },
    { key: "report_count", label: "Reports" },
    { key: "warning", label: "Warning" },
    { key: "run_id", label: "Run" },
  ], rows);
}

function renderReports() {
  const rows = state.snapshot?.reports || [];
  byId("report-count-label").textContent = `${rows.length} rows`;
  renderTable("report-table", [
    { key: "run_id", label: "Run" },
    { key: "report_type", label: "Type" },
    { key: "path", label: "Path" },
  ], rows, (row, index) => `<button class="table-button" data-report-index="${index}">Open</button>`);
}

function renderDecisions() {
  const rows = state.snapshot?.decisions || [];
  byId("decision-count-label").textContent = `${rows.length} rows`;
  renderTable("decision-table", [
    { key: "created_at_utc", label: "Created" },
    { key: "decision_type", label: "Type" },
    { key: "candidate_id", label: "Candidate" },
    { key: "decision", label: "Decision" },
    { key: "next_action", label: "Next" },
  ], rows);
}

function renderAll() {
  renderStrategyMap();
  renderH8Lab();
  renderCandidates();
  renderRuns();
  renderReports();
  renderDecisions();
}

async function loadAll() {
  setStatus("Loading strategy map");
  const [snapshot, operations, h8Targets] = await Promise.all([
    requestJson("/registry/snapshot?limit=500"),
    requestJson("/operations/h1c?limit=500"),
    requestJson("/hypotheses/h8/targets?limit=200"),
  ]);
  state.snapshot = snapshot;
  state.operations = operations;
  state.h8Targets = h8Targets;
  renderAll();
  setStatus("Strategy map loaded");
}

async function openReport(index) {
  const row = state.snapshot.reports[index];
  if (!row) return;
  byId("report-title").textContent = row.path;
  byId("report-markdown").textContent = "Loading";
  try {
    const payload = await requestJson(`/reports/markdown?path=${encodeURIComponent(row.path)}`);
    byId("report-markdown").textContent = payload.markdown;
  } catch (error) {
    byId("report-markdown").textContent = error.message;
  }
}

async function indexRegistry() {
  setStatus("Indexing registry");
  await requestJson("/registry/index", {
    method: "POST",
    body: JSON.stringify({ reset: true }),
  });
  await loadAll();
}

async function runH8(event) {
  event.preventDefault();
  const target = byId("h8-target").value.trim().toUpperCase();
  const config = byId("h8-config").value.trim();
  if (!target) return;
  byId("h8-run-state").textContent = "running";
  byId("h8-run-state").className = "badge warn";
  byId("h8-run-button").disabled = true;
  setStatus(`Running H8 for ${target}`);
  try {
    state.h8Result = await requestJson("/hypotheses/h8/run", {
      method: "POST",
      body: JSON.stringify({ target_symbol: target, config_path: config }),
    });
    renderH8Lab();
    activateView("h8lab");
    setStatus(state.h8Result.available ? `H8 loaded for ${target}` : `H8 blocked for ${target}`, state.h8Result.available);
  } finally {
    byId("h8-run-button").disabled = false;
  }
}

async function runH8c() {
  const target = byId("h8-target").value.trim().toUpperCase();
  const config = byId("h8c-config").value.trim();
  if (!target) return;
  byId("h8c-run-state").textContent = "running";
  byId("h8c-run-state").className = "badge warn";
  byId("h8c-run-button").disabled = true;
  setStatus(`Running H8c for ${target}`);
  try {
    state.h8cResult = await requestJson("/hypotheses/h8c/run", {
      method: "POST",
      body: JSON.stringify({ target_symbol: target, config_path: config }),
    });
    renderH8Lab();
    activateView("h8lab");
    setStatus(state.h8cResult.available ? `H8c loaded for ${target}` : `H8c blocked for ${target}`, state.h8cResult.available);
  } finally {
    byId("h8c-run-button").disabled = false;
  }
}

async function saveDecision(event) {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const evidencePath = String(form.get("evidence_path") || "");
  await requestJson("/decisions", {
    method: "POST",
    body: JSON.stringify({
      decision_type: form.get("decision_type"),
      decision: form.get("decision"),
      candidate_id: form.get("candidate_id") || null,
      run_id: form.get("run_id") || null,
      rationale: form.get("rationale") || null,
      next_action: form.get("next_action") || null,
      evidence: [{ path: evidencePath }],
    }),
  });
  event.currentTarget.reset();
  await loadAll();
  activateView("decisions");
}

document.querySelectorAll(".nav-button").forEach((button) => {
  button.addEventListener("click", () => activateView(button.dataset.view));
});

byId("refresh-button").addEventListener("click", () => loadAll().catch((error) => setStatus(error.message, false)));
byId("index-button").addEventListener("click", () => indexRegistry().catch((error) => setStatus(error.message, false)));
byId("candidate-filter").addEventListener("input", renderCandidates);
byId("decision-form").addEventListener("submit", (event) => saveDecision(event).catch((error) => setStatus(error.message, false)));
byId("h8-form").addEventListener("submit", (event) => runH8(event).catch((error) => setStatus(error.message, false)));
byId("h8c-run-button").addEventListener("click", () => runH8c().catch((error) => setStatus(error.message, false)));

document.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;
  const hypothesisButton = target.closest("[data-hypothesis-id]");
  if (hypothesisButton) {
    state.selectedHypothesisId = hypothesisButton.dataset.hypothesisId;
    renderStrategyMap();
    return;
  }
  if (target.dataset.reportIndex) {
    openReport(Number(target.dataset.reportIndex));
    return;
  }
  if (target.dataset.h8Target) {
    byId("h8-target").value = target.dataset.h8Target;
  }
});

window.addEventListener("resize", () => {
  renderSelectedHypothesis();
  renderLiveTrading();
  drawH8RegimeChart();
});

loadAll().catch((error) => setStatus(error.message, false));

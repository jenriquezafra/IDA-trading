const state = {
  snapshot: null,
  selectedId: null,
  mode: "paper",
};

function byId(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function text(value) {
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  return String(value);
}

function number(value, digits = 2) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "-";
  }
  return numeric.toFixed(digits);
}

function money(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "-";
  }
  const sign = numeric > 0 ? "+" : numeric < 0 ? "-" : "";
  return `${sign}$${Math.abs(numeric).toFixed(2)}`;
}

function pct(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "-";
  }
  return `${(numeric * 100).toFixed(1)}%`;
}

function compactMoney(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "-";
  }
  const sign = numeric > 0 ? "+" : numeric < 0 ? "-" : "";
  const absolute = Math.abs(numeric);
  if (absolute >= 1000) {
    return `${sign}$${(absolute / 1000).toFixed(1)}k`;
  }
  return `${sign}$${absolute.toFixed(2)}`;
}

function tooltipAttr(lines) {
  return escapeHtml(lines.filter((line) => line !== null && line !== undefined && line !== "").join("\n"));
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || response.statusText);
  }
  return response.json();
}

function activeCandidates() {
  if (state.mode === "paper") {
    return state.snapshot?.sections?.paper || [];
  }
  if (state.mode === "live") {
    return state.snapshot?.sections?.live || [];
  }
  return [];
}

function selectedCandidate() {
  return activeCandidates().find((candidate) => candidate.candidate_id === state.selectedId) || activeCandidates()[0] || null;
}

function setSystemBadge(overallState) {
  const badge = byId("system-state");
  badge.textContent = overallState || "ready";
  badge.className = `state-pill ${overallState || ""}`;
}

function renderNav() {
  const candidates = activeCandidates();
  if (!state.selectedId && candidates.length) {
    state.selectedId = candidates.find((candidate) => candidate.candidate_id === "ko-defensive-paper-demo")?.candidate_id || candidates[0].candidate_id;
  }
  byId("candidate-nav").innerHTML = candidates
    .map(
      (candidate) => `
        <button class="candidate-button ${candidate.candidate_id === state.selectedId ? "is-active" : ""}" data-id="${escapeHtml(candidate.candidate_id)}">
          <strong>${escapeHtml(candidate.name)}</strong>
          <span>${escapeHtml(candidate.mode)} · ${escapeHtml(candidate.symbol)} · ${escapeHtml(candidate.overall_state)}</span>
        </button>`
    )
    .join("") || `<div class="candidate-button"><strong>Sin estrategias</strong><span>${escapeHtml(state.mode)}</span></div>`;
}

function renderModeButtons() {
  document.querySelectorAll(".mode-button").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.mode === state.mode);
  });
}

function renderSummary() {
  const summary = state.snapshot?.summary || {};
  byId("active-count").textContent = summary.active_count || 0;
  byId("paused-count").textContent = summary.paused_count || 0;
  byId("critical-count").textContent = summary.critical_alerts || 0;
  byId("warning-count").textContent = summary.warning_alerts || 0;
}

function renderConnections() {
  const connection = state.snapshot?.connection || { checks: [] };
  byId("candidate-title").textContent = "Conexión / VPN";
  byId("candidate-subtitle").textContent = `${connection.ok_count || 0}/${connection.check_count || 0} checks OK · ${connection.status || "unknown"}`;
  setSystemBadge(connection.status === "ok" ? "running" : "attention");
  byId("alerts-panel").innerHTML = "";
  byId("strategy-view").style.display = "none";
  byId("connection-view").classList.add("is-active");
  byId("connection-view").innerHTML = (connection.checks || [])
    .map(
      (check) => `
        <article class="connection-card">
          <span class="badge ${check.ok ? "running" : "blocked"}">${check.ok ? "OK" : "FAIL"}</span>
          <strong>${escapeHtml(check.name)}</strong>
          <span class="muted">${escapeHtml(check.host ? `${check.host}:${check.port}` : check.path || "-")}</span>
          <span>${escapeHtml(check.error || `${number(check.latency_ms, 2)} ms`)}</span>
          <span class="muted">${escapeHtml(check.config_path || check.mtime_utc || "")}</span>
        </article>`
    )
    .join("") || `<article class="connection-card muted">Sin checks configurados</article>`;
}

function renderAlerts(candidate) {
  byId("alerts-panel").innerHTML = (candidate.alerts || [])
    .map(
      (alert) => `
        <article class="alert alert-${escapeHtml(alert.severity)}">
          <strong>${escapeHtml(alert.title)}</strong>
          <p>${escapeHtml(alert.message)}</p>
        </article>`
    )
    .join("");
}

function lineChart(points, key = "cumulative_realized_pnl") {
  if (!points || !points.length) {
    return `<svg class="chart" viewBox="0 0 300 250" role="img"><text x="18" y="125" fill="#68726d">Sin PnL registrado todavía</text></svg>`;
  }
  const coords = points
    .map((point, index) => {
      const value = Number(point[key]);
      if (!Number.isFinite(value)) {
        return null;
      }
      return { point, value, sourceIndex: index };
    })
    .filter(Boolean);
  if (!coords.length) {
    return `<svg class="chart" viewBox="0 0 300 250" role="img"><text x="18" y="125" fill="#68726d">Sin serie numérica</text></svg>`;
  }
  const rawMin = Math.min(0, ...coords.map((coord) => coord.value));
  const rawMax = Math.max(0, ...coords.map((coord) => coord.value));
  const pad = Math.max((rawMax - rawMin) * 0.12, 10);
  const min = rawMin - pad;
  const max = rawMax + pad;
  const spread = max - min || 1;
  const mapped = coords.map((coord, index) => {
    const x = 34 + (index * 238) / Math.max(1, coords.length - 1);
    const y = 205 - ((coord.value - min) / spread) * 160;
    return { ...coord, x, y };
  });
  const polyline = mapped.map((coord) => `${coord.x.toFixed(1)},${coord.y.toFixed(1)}`).join(" ");
  const zeroY = 205 - ((0 - min) / spread) * 160;
  const pointLabels = mapped
    .filter((coord, index) => mapped.length <= 12 || index === mapped.length - 1)
    .map((coord) => {
      const y = Math.max(18, coord.y - 10);
      return `<text x="${coord.x.toFixed(1)}" y="${y.toFixed(1)}" text-anchor="middle" fill="#007c68" font-size="9" font-weight="800">${compactMoney(coord.value)}</text>`;
    })
    .join(" ");
  const hotspots = mapped
    .map((coord) => {
      const point = coord.point;
      const tooltip = tooltipAttr([
        text(point.timestamp || point.event_at).slice(0, 19),
        `Evento: ${text(point.event_type)}`,
        `PnL evento: ${money(point.realized_pnl)}`,
        `PnL acumulado: ${money(point[key])}`,
        `Qty: ${number(point.quantity, 0)}`,
      ]);
      return `
        <g class="chart-hotspot" tabindex="0" data-tooltip="${tooltip}">
          <title>${tooltip}</title>
          <circle cx="${coord.x.toFixed(1)}" cy="${coord.y.toFixed(1)}" r="4.8" fill="#007c68" stroke="#ffffff" stroke-width="2" vector-effect="non-scaling-stroke" />
        </g>`;
    })
    .join("");
  return `
    <svg class="chart" viewBox="0 0 300 250" role="img">
      <line x1="34" y1="45" x2="34" y2="205" stroke="#d9dfd8" />
      <line x1="34" y1="205" x2="272" y2="205" stroke="#d9dfd8" />
      <line x1="34" y1="${zeroY.toFixed(1)}" x2="272" y2="${zeroY.toFixed(1)}" stroke="#d9dfd8" stroke-dasharray="4 4" />
      <text x="8" y="49" fill="#68726d" font-size="10">${compactMoney(max)}</text>
      <text x="8" y="${zeroY.toFixed(1)}" fill="#68726d" font-size="10">${compactMoney(0)}</text>
      <text x="8" y="208" fill="#68726d" font-size="10">${compactMoney(min)}</text>
      <polyline points="${polyline}" fill="none" stroke="#007c68" stroke-width="2.8" vector-effect="non-scaling-stroke" />
      ${pointLabels}
      ${hotspots}
    </svg>`;
}

function markerColor(action) {
  if (action === "buy") {
    return "#007c68";
  }
  if (action === "sell") {
    return "#b42318";
  }
  if (action === "hold") {
    return "#9a641c";
  }
  return "#4f5a54";
}

function priceChart(points) {
  if (!points || !points.length) {
    return `<svg class="chart" viewBox="0 0 300 250" role="img"><text x="18" y="125" fill="#68726d">Sin serie de precio</text></svg>`;
  }
  const values = points.map((point) => Number(point.close)).filter((value) => Number.isFinite(value));
  if (!values.length) {
    return `<svg class="chart" viewBox="0 0 300 250" role="img"><text x="18" y="125" fill="#68726d">Sin precios numéricos</text></svg>`;
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const pad = Math.max((max - min) * 0.12, 0.35);
  const lower = min - pad;
  const upper = max + pad;
  const spread = upper - lower || 1;
  const coords = points
    .map((point, index) => {
      const value = Number(point.close);
      const x = 18 + (index * 264) / Math.max(1, points.length - 1);
      const y = 212 - ((value - lower) / spread) * 176;
      return { point, x, y };
    });
  const polyline = coords.map((coord) => `${coord.x.toFixed(1)},${coord.y.toFixed(1)}`).join(" ");
  const markers = coords
    .map((coord) => {
      const marker = coord.point.marker || {};
      const action = marker.action || "";
      const fill = marker.action ? markerColor(action) : "#2d647f";
      const labelY = action === "sell" ? coord.y - 12 : coord.y + 20;
      const tooltip = tooltipAttr([
        text(coord.point.timestamp).slice(0, 19),
        `Close: $${number(coord.point.close, 2)}`,
        marker.action ? `Señal: ${text(marker.label || marker.action)}` : "Señal: -",
        marker.quantity !== undefined ? `Qty: ${number(marker.quantity, 0)}` : "",
      ]);
      const visibleLabel = marker.action
        ? `<text x="${coord.x.toFixed(1)}" y="${labelY.toFixed(1)}" text-anchor="middle" fill="${fill}" font-size="9" font-weight="800">${escapeHtml(marker.label || action.toUpperCase())}</text>`
        : "";
      const radius = marker.action ? 5.8 : 3.2;
      return `
        <g class="chart-hotspot" tabindex="0" data-tooltip="${tooltip}">
          <title>${tooltip}</title>
          <circle cx="${coord.x.toFixed(1)}" cy="${coord.y.toFixed(1)}" r="${radius}" fill="${fill}" stroke="#fff" stroke-width="2" vector-effect="non-scaling-stroke" />
          ${visibleLabel}
        </g>`;
    })
    .join("");
  const firstDate = text(points[0]?.date || points[0]?.timestamp).slice(0, 10);
  const lastDate = text(points[points.length - 1]?.date || points[points.length - 1]?.timestamp).slice(0, 10);
  return `
    <svg class="chart" viewBox="0 0 300 250" role="img">
      <line x1="18" y1="36" x2="18" y2="212" stroke="#d9dfd8" />
      <line x1="18" y1="212" x2="282" y2="212" stroke="#d9dfd8" />
      <text x="20" y="28" fill="#68726d" font-size="10">$${upper.toFixed(2)}</text>
      <text x="20" y="228" fill="#68726d" font-size="10">$${lower.toFixed(2)}</text>
      <text x="18" y="242" fill="#68726d" font-size="10">${escapeHtml(firstDate)}</text>
      <text x="282" y="242" fill="#68726d" font-size="10" text-anchor="end">${escapeHtml(lastDate)}</text>
      <polyline points="${polyline}" fill="none" stroke="#2d647f" stroke-width="2.6" vector-effect="non-scaling-stroke" />
      ${markers}
    </svg>`;
}

function renderFacts(candidate) {
  const daemon = candidate.daemon || {};
  const scheduler = daemon.scheduler || {};
  const facts = [
    ["Market", scheduler.market_open === true ? "open" : scheduler.market_open === false ? "closed" : "-"],
    ["Scheduler", scheduler.reason || "-"],
    ["Next open", text(scheduler.next_open_utc).slice(0, 19)],
    ["Errors", daemon.error_streak ?? 0],
    ["Status file", daemon.available ? "available" : "missing"],
    ["Updated", text(daemon.mtime_utc).slice(0, 19)],
  ];
  byId("daemon-facts").innerHTML = facts
    .map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`)
    .join("");
  byId("daemon-updated").textContent = text(daemon.mtime_utc).slice(0, 19);
}

function renderRuns(candidate) {
  const rows = (candidate.recent_runs || []).slice(0, 30).map((run) => {
    const recon = run.pre_trade_reconciliation || run.post_execution_reconciliation || "-";
    const orders = `${run.planned_orders || 0}/${run.submitted_orders || 0}`;
    return `
      <tr>
        <td>${escapeHtml(text(run.created_at_utc).slice(0, 19))}</td>
        <td><span class="badge ${run.decision?.startsWith?.("blocked") ? "blocked" : ""}">${escapeHtml(text(run.decision))}</span></td>
        <td>${escapeHtml(text(run.reason))}</td>
        <td>${escapeHtml(text(run.signal_action))}</td>
        <td>${number(run.ticket_quantity, 0)}</td>
        <td>${escapeHtml(text(recon))}</td>
        <td>${escapeHtml(orders)}</td>
        <td>${number(run.latency_seconds, 1)}s</td>
      </tr>`;
  });
  byId("run-rows").innerHTML = rows.join("") || `<tr><td colspan="8" class="muted">Sin runs registrados</td></tr>`;
}

function renderStateEvents(candidate) {
  const rows = (candidate.state_events || []).slice(0, 30).map(
    (event) => `
      <tr>
        <td>${escapeHtml(text(event.created_at_utc).slice(0, 19))}</td>
        <td>${escapeHtml(text(event.event_type))}</td>
        <td>${escapeHtml(text(event.previous_status))} → ${escapeHtml(text(event.new_status))}</td>
        <td>${escapeHtml(text(event.ticket_action))}</td>
        <td>${number(event.ticket_quantity, 0)}</td>
        <td>${escapeHtml(text(event.state_updated))}</td>
      </tr>`
  );
  byId("state-event-rows").innerHTML = rows.join("") || `<tr><td colspan="6" class="muted">Sin eventos de estado</td></tr>`;
}

function renderLedger(candidate) {
  const rows = (candidate.ledger?.events || []).slice(0, 80).map(
    (entry) => `
      <tr>
        <td>${escapeHtml(text(entry.event_at).slice(0, 19))}</td>
        <td>${escapeHtml(text(entry.source))}</td>
        <td>${escapeHtml(text(entry.event_type))}</td>
        <td>${escapeHtml(text(entry.symbol))}</td>
        <td>${escapeHtml(text(entry.side))}</td>
        <td>${number(entry.quantity, 0)}</td>
        <td>${number(entry.price, 2)}</td>
        <td>${money(entry.net_pnl)}</td>
        <td>${money(entry.exposure)}</td>
        <td>${escapeHtml(text(entry.notes))}</td>
      </tr>`
  );
  byId("ledger-rows").innerHTML = rows.join("") || `<tr><td colspan="10" class="muted">Sin eventos de ledger para esta estrategia</td></tr>`;
}

function syncCapitalBasis() {
  const mode = byId("capital-mode").value;
  const basis = byId("capital-basis");
  if (mode === "absolute_usd") {
    basis.value = "max_order_notional_usd";
    basis.disabled = true;
  } else {
    basis.disabled = false;
    if (basis.value === "max_order_notional_usd") {
      basis.value = "buying_power_fraction";
    }
  }
}

function renderCandidate() {
  renderModeButtons();
  if (state.mode === "connections") {
    renderNav();
    renderConnections();
    return;
  }
  byId("strategy-view").style.display = "";
  byId("connection-view").classList.remove("is-active");
  const candidate = selectedCandidate();
  if (!candidate) {
    byId("candidate-title").textContent = `Sin estrategias ${state.mode}`;
    byId("candidate-subtitle").textContent = "No hay fuentes operativas configuradas para este modo.";
    byId("alerts-panel").innerHTML = "";
    byId("strategy-view").style.display = "none";
    return;
  }
  state.selectedId = candidate.candidate_id;
  byId("candidate-title").textContent = candidate.name;
  byId("candidate-subtitle").textContent = `${candidate.strategy_id} · ${candidate.mode} · ${candidate.symbol}`;
  setSystemBadge(candidate.overall_state);

  const control = candidate.control || {};
  const daemon = candidate.daemon || {};
  const scheduler = daemon.scheduler || {};
  const paperState = candidate.state || {};
  const latestRun = candidate.latest_run || {};
  const pnl = candidate.pnl || {};
  const market = candidate.market || {};
  const runtime = control.runtime || {};

  byId("automation-state").textContent = control.kill_switch_exists ? "PAUSED" : text(candidate.overall_state).toUpperCase();
  byId("automation-reason").textContent = control.kill_switch_exists
    ? `Kill switch: ${control.kill_switch_path}`
    : scheduler.reason || "automation enabled";
  byId("pause-button").disabled = !control.pause_enabled;
  byId("resume-button").disabled = !control.resume_enabled;

  byId("state-card-label").textContent = `${candidate.mode || "strategy"} state`;
  byId("paper-state").textContent = text(paperState.status).toUpperCase();
  byId("paper-position").textContent = `${text(paperState.symbol)} qty ${number(paperState.quantity, 0)} · desired ${number(paperState.desired_position_unit, 1)}`;
  byId("last-decision").textContent = text(latestRun.decision).toUpperCase();
  byId("last-run-time").textContent = `${text(latestRun.created_at_utc).slice(0, 19)} · ${text(latestRun.reason)}`;
  byId("pnl-realized").textContent = money(pnl.realized_pnl);
  byId("pnl-detail").textContent = `${number(pnl.event_count, 0)} events · win ${pct(pnl.win_rate)} · DD ${money(pnl.max_drawdown)}`;
  byId("pnl-source").textContent = pnl.source_available ? `${text(pnl.source_type)} · ${text(pnl.source_path)}` : "PnL log missing";
  byId("pnl-chart").innerHTML = lineChart(pnl.curve || []);
  byId("market-source").textContent = `${text(market.symbol)} · ${text(market.source)}`;
  byId("market-chart").innerHTML = priceChart(market.series || []);
  byId("control-enabled").value = String(Boolean(runtime.enabled));
  byId("capital-mode").value = runtime.capital_mode || "net_fraction";
  byId("capital-value").value = runtime.capital_value ?? 1.0;
  byId("capital-basis").value = runtime.capital_basis || "buying_power_fraction";
  syncCapitalBasis();
  byId("capital-status").textContent = `${runtime.updated_by || "config"} · ${runtime.updated_at || "current config"} · effective ${control.effective_enabled ? "on" : "off"}`;

  renderAlerts(candidate);
  renderFacts(candidate);
  renderRuns(candidate);
  renderStateEvents(candidate);
  renderLedger(candidate);
}

async function refresh() {
  state.snapshot = await api("/control-center");
  renderSummary();
  renderNav();
  renderCandidate();
}

async function saveRuntimeControl(event) {
  event.preventDefault();
  const candidate = selectedCandidate();
  if (!candidate) {
    return;
  }
  const capitalMode = byId("capital-mode").value;
  await api(`/control-center/${encodeURIComponent(candidate.candidate_id)}/runtime`, {
    method: "PATCH",
    body: JSON.stringify({
      enabled: byId("control-enabled").value === "true",
      capital_mode: capitalMode,
      capital_value: Number(byId("capital-value").value),
      capital_basis: capitalMode === "absolute_usd" ? "max_order_notional_usd" : byId("capital-basis").value,
      actor: "dashboard",
      notes: "manual runtime control update",
      apply_to_config: byId("apply-config").checked,
    }),
  });
  byId("apply-config").checked = false;
  await refresh();
}

async function controlAction(action) {
  const candidate = selectedCandidate();
  if (!candidate) {
    return;
  }
  const label = action === "pause" ? "pausar la automatización paper" : "reanudar la automatización paper";
  const accepted = window.confirm(`Confirmar: ${label} para ${candidate.name}.`);
  if (!accepted) {
    return;
  }
  await api(`/control-center/${encodeURIComponent(candidate.candidate_id)}/control`, {
    method: "POST",
    body: JSON.stringify({ action, actor: "dashboard", reason: `manual ${action} from control center` }),
  });
  await refresh();
}

function chartTooltip() {
  return byId("chart-tooltip");
}

function showChartTooltip(event, target) {
  const tooltip = chartTooltip();
  if (!tooltip) {
    return;
  }
  if (!target) {
    tooltip.hidden = true;
    return;
  }
  const content = target.getAttribute("data-tooltip");
  if (!content) {
    tooltip.hidden = true;
    return;
  }
  tooltip.textContent = content;
  tooltip.hidden = false;
  const rect = target.getBoundingClientRect();
  const x = event.clientX || rect.left + rect.width / 2;
  const y = event.clientY || rect.top + rect.height / 2;
  tooltip.style.left = `${Math.min(window.innerWidth - 260, x + 14)}px`;
  tooltip.style.top = `${Math.max(12, y - 18)}px`;
}

function hideChartTooltip() {
  const tooltip = chartTooltip();
  if (tooltip) {
    tooltip.hidden = true;
  }
}

function bindChartTooltip(containerId) {
  const container = byId(containerId);
  if (!container) {
    return;
  }
  container.addEventListener("mousemove", (event) => {
    showChartTooltip(event, event.target.closest(".chart-hotspot"));
  });
  container.addEventListener("mouseleave", hideChartTooltip);
  container.addEventListener("focusin", (event) => {
    showChartTooltip(event, event.target.closest(".chart-hotspot"));
  });
  container.addEventListener("focusout", hideChartTooltip);
}

function bindEvents() {
  byId("candidate-nav").addEventListener("click", (event) => {
    const button = event.target.closest(".candidate-button");
    if (!button) {
      return;
    }
    state.selectedId = button.dataset.id;
    renderNav();
    renderCandidate();
  });
  document.querySelectorAll(".mode-button").forEach((button) => {
    button.addEventListener("click", () => {
      state.mode = button.dataset.mode;
      state.selectedId = null;
      renderNav();
      renderCandidate();
    });
  });
  byId("pause-button").addEventListener("click", () => controlAction("pause"));
  byId("resume-button").addEventListener("click", () => controlAction("resume"));
  byId("capital-mode").addEventListener("change", syncCapitalBasis);
  byId("capital-form").addEventListener("submit", saveRuntimeControl);
  bindChartTooltip("market-chart");
  bindChartTooltip("pnl-chart");
}

async function init() {
  bindEvents();
  try {
    await refresh();
    window.setInterval(refresh, 30000);
  } catch (error) {
    byId("system-state").textContent = "error";
    byId("alerts-panel").innerHTML = `<article class="alert alert-critical"><strong>Error</strong><p>${escapeHtml(error.message)}</p></article>`;
  }
}

init();

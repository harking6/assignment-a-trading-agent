const els = {
  dynamicUniverseInput: document.querySelector("#dynamicUniverseInput"),
  dynamicUniverseLimitInput: document.querySelector("#dynamicUniverseLimitInput"),
  // Core config
  universeInput: document.querySelector("#universeInput"),
  daysInput: document.querySelector("#daysInput"),
  deepModelInput: document.querySelector("#deepModelInput"),
  llmBaseUrlInput: document.querySelector("#llmBaseUrlInput"),

  // Advanced config
  providerSelect: document.querySelector("#providerSelect"),
  symbolInput: document.querySelector("#symbolInput"),
  benchmarkInput: document.querySelector("#benchmarkInput"),
  priceAdjustSelect: document.querySelector("#priceAdjustSelect"),
  startInput: document.querySelector("#startInput"),
  endInput: document.querySelector("#endInput"),
  datasetSelect: document.querySelector("#datasetSelect"),
  strategySelect: document.querySelector("#strategySelect"),
  agentModeSelect: document.querySelector("#agentModeSelect"),
  llmModelInput: document.querySelector("#llmModelInput"),
  debateRoundsInput: document.querySelector("#debateRoundsInput"),
  riskInput: document.querySelector("#riskInput"),
  feeInput: document.querySelector("#feeInput"),
  maxPositionsInput: document.querySelector("#maxPositionsInput"),
  topNSectorsInput: document.querySelector("#topNSectorsInput"),
  stocksPerSectorInput: document.querySelector("#stocksPerSectorInput"),
  rebalanceFrequencyInput: document.querySelector("#rebalanceFrequencyInput"),
  cashReserveInput: document.querySelector("#cashReserveInput"),
  maxPositionRatioInput: document.querySelector("#maxPositionRatioInput"),
  initialCashInput: document.querySelector("#initialCashInput"),
  realtimeOrderInput: document.querySelector("#realtimeOrderInput"),
  applyAdvancedBtn: document.querySelector("#applyAdvancedBtn"),

  // Actions
  resetBtn: document.querySelector("#resetBtn"),
  stepBtn: document.querySelector("#stepBtn"),
  runBtn: document.querySelector("#runBtn"),
  exportBtn: document.querySelector("#exportBtn"),

  // Manual trading
  orderSharesInput: document.querySelector("#orderSharesInput"),
  buyBtn: document.querySelector("#buyBtn"),
  sellBtn: document.querySelector("#sellBtn"),
  liveQuoteBtn: document.querySelector("#liveQuoteBtn"),
  targetRatioInput: document.querySelector("#targetRatioInput"),
  targetRatioLabel: document.querySelector("#targetRatioLabel"),
  rebalanceBtn: document.querySelector("#rebalanceBtn"),
  addBtn: document.querySelector("#addBtn"),
  reduceBtn: document.querySelector("#reduceBtn"),

  // Display
  modeBadge: document.querySelector("#modeBadge"),
  llmBadge: document.querySelector("#llmBadge"),
  dayMetric: document.querySelector("#dayMetric"),
  equityMetric: document.querySelector("#equityMetric"),
  returnMetric: document.querySelector("#returnMetric"),
  activeMetric: document.querySelector("#activeMetric"),
  drawdownMetric: document.querySelector("#drawdownMetric"),
  tradeMetric: document.querySelector("#tradeMetric"),
  chart: document.querySelector("#chart"),
  sectorViewBox: document.querySelector("#sectorViewBox"),
  positionsBody: document.querySelector("#positionsBody"),
  logBody: document.querySelector("#logBody"),
  traceList: document.querySelector("#traceList"),
  researchBox: document.querySelector("#researchBox"),
};

let state = null;
let timer = null;
let liveQuote = null;
let resetPromise = null;
let resetDebounceTimer = null;

async function api(path, payload) {
  const options = payload
    ? {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }
    : {};
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const errorPayload = await response.json();
      detail = errorPayload.error || detail;
    } catch (_error) {
      // Keep HTTP status text.
    }
    throw new Error(detail);
  }
  return response.json();
}

function configPayload() {
  const universeRaw = els.universeInput.value.trim();
  const universe = universeRaw
    ? universeRaw.split(",").map((s) => s.trim()).filter(Boolean)
    : [];
  const days = (() => {
    const n = Number(els.daysInput.value);
    return n > 0 ? n : null;
  })();
  return {
    dynamic_universe: Boolean(els.dynamicUniverseInput.checked),
    dynamic_universe_limit: Number(els.dynamicUniverseLimitInput.value) || 30,
    provider: els.providerSelect.value,
    symbol: els.symbolInput.value.trim() || "DEMO",
    universe,
    benchmark_symbol: els.benchmarkInput.value.trim() || defaultBenchmark(els.providerSelect.value),
    price_adjust: els.priceAdjustSelect.value,
    start: els.startInput.value.trim() || undefined,
    end: els.endInput.value.trim() || undefined,
    days,
    dataset: els.datasetSelect.value,
    strategy: els.strategySelect.value,
    agent_mode: els.agentModeSelect.value,
    llm_model: els.llmModelInput.value.trim() || "glm-5.2",
    deep_model: els.deepModelInput.value.trim() || "glm-5.2",
    llm_base_url: els.llmBaseUrlInput.value.trim() || undefined,
    max_debate_rounds: Number(els.debateRoundsInput.value) || 2,
    max_risk_debate_rounds: 1,
    risk_budget: Number(els.riskInput.value) / 100,
    fee_rate: Number(els.feeInput.value) / 100,
    max_positions: Number(els.maxPositionsInput.value) || 5,
    top_n_sectors: Number(els.topNSectorsInput.value) || 2,
    stocks_per_sector: Number(els.stocksPerSectorInput.value) || 3,
    rebalance_frequency: Number(els.rebalanceFrequencyInput.value) || 5,
    cash_reserve_ratio: Number(els.cashReserveInput.value) || 0.2,
    max_position_ratio: Number(els.maxPositionRatioInput.value) || 0.4,
    initial_cash: Number(els.initialCashInput.value) || 100000,
  };
}

function defaultBenchmark(provider) {
  if (provider === "akshare" || provider === "tushare") return "000001.SH";
  if (provider === "yfinance") return "^GSPC";
  return "BENCH";
}

function money(value, digits = 0) {
  return new Intl.NumberFormat("zh-CN", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(Number(value || 0));
}

function pct(value) {
  return `${((Number(value) || 0) * 100).toFixed(2)}%`;
}

function signedPct(value) {
  const numeric = Number(value) || 0;
  return `${numeric >= 0 ? "+" : ""}${(numeric * 100).toFixed(2)}%`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function isMultiSymbol() {
  return Boolean(state?.config?.dynamic_universe || (state?.config?.universe && state.config.universe.length > 0));
}

async function loadState() {
  try {
    state = await api("/api/state");
    syncControlsFromState();
    render();
  } catch (error) {
    els.sectorViewBox.innerHTML = `<div class="empty-hint">
      <strong>Python 后端未启动</strong>
      <p>请在 assignment-a-trading-agent 目录运行：python app.py，然后刷新页面。</p>
    </div>`;
  }
}

function syncControlsFromState() {
  if (!state) return;
  const cfg = state.config || {};
  els.providerSelect.value = cfg.provider || "synthetic";
  els.symbolInput.value = cfg.symbol || "DEMO";
  els.universeInput.value = (cfg.universe || []).join(",");
  els.dynamicUniverseInput.checked = Boolean(cfg.dynamic_universe);
  els.dynamicUniverseLimitInput.value = cfg.dynamic_universe_limit || 30;
  els.benchmarkInput.value = cfg.benchmark_symbol || defaultBenchmark(cfg.provider);
  els.priceAdjustSelect.value = cfg.price_adjust || "qfq";
  els.startInput.value = cfg.start || "";
  els.endInput.value = cfg.end || "";
  els.daysInput.value = cfg.days || "";
  els.datasetSelect.value = cfg.dataset || "trend";
  els.strategySelect.value = cfg.strategy || "multi_agent";
  els.agentModeSelect.value = cfg.agent_mode || "llm";
  els.llmModelInput.value = cfg.llm_model || "glm-5.2";
  els.deepModelInput.value = cfg.deep_model || "glm-5.2";
  els.llmBaseUrlInput.value = cfg.llm_base_url || "";
  els.debateRoundsInput.value = cfg.max_debate_rounds || 2;
  els.riskInput.value = Math.round((cfg.risk_budget || 0.3) * 100);
  els.feeInput.value = ((cfg.fee_rate || 0.0012) * 100).toFixed(2);
  els.maxPositionsInput.value = cfg.max_positions || 5;
  els.topNSectorsInput.value = cfg.top_n_sectors || 2;
  els.stocksPerSectorInput.value = cfg.stocks_per_sector || 3;
  els.rebalanceFrequencyInput.value = cfg.rebalance_frequency || 5;
  els.cashReserveInput.value = cfg.cash_reserve_ratio || 0.2;
  els.maxPositionRatioInput.value = cfg.max_position_ratio || 0.4;
  els.initialCashInput.value = cfg.initial_cash || 100000;
  updateTargetLabel();
}

async function reset() {
  if (resetPromise) return resetPromise;
  stopRun();
  liveQuote = null;
  const originalLabel = els.resetBtn.textContent;
  els.resetBtn.disabled = true;
  els.stepBtn.disabled = true;
  els.runBtn.disabled = true;
  els.applyAdvancedBtn.disabled = true;
  els.resetBtn.textContent = "重置中…";
  resetPromise = (async () => {
    try {
      state = await api("/api/reset", configPayload());
      render();
      return state;
    } finally {
      resetPromise = null;
      els.resetBtn.disabled = false;
      els.stepBtn.disabled = false;
      els.runBtn.disabled = false;
      els.applyAdvancedBtn.disabled = false;
      els.resetBtn.textContent = originalLabel;
    }
  })();
  return resetPromise;
}

function scheduleReset() {
  window.clearTimeout(resetDebounceTimer);
  resetDebounceTimer = window.setTimeout(
    () => reset().catch((error) => renderError(error.message)),
    400
  );
}

function resetNow() {
  window.clearTimeout(resetDebounceTimer);
  return reset();
}

async function step() {
  state = await api("/api/step", {});
  render();
}

async function runAll() {
  stopRun();
  state = await api("/api/run", {});
  render();
}

function orderShares() {
  return Math.max(1, Math.floor(Number(els.orderSharesInput.value) || 0));
}

async function paperOrder(side) {
  state = await api("/api/paper/order", {
    side,
    shares: orderShares(),
    realtime: els.realtimeOrderInput.checked,
    reason: `manual paper ${side} ${orderShares()} shares`,
  });
  render();
}

async function refreshLiveQuote() {
  const payload = await api("/api/paper/account?realtime=true");
  if (!state) state = await api("/api/state");
  state.account = payload.account;
  state.report = payload.report;
  liveQuote = payload.account.quote;
  render();
}

async function rebalanceToTarget() {
  state = await api("/api/paper/rebalance", {
    target_ratio: Number(els.targetRatioInput.value) / 100,
    realtime: els.realtimeOrderInput.checked,
    reason: "manual target exposure rebalance",
  });
  render();
}

async function adjustPosition(direction) {
  state = await api("/api/paper/adjust", {
    direction,
    ratio_delta: 0.1,
    realtime: els.realtimeOrderInput.checked,
  });
  render();
}

function startRun() {
  if (timer) {
    stopRun();
    return;
  }
  els.runBtn.textContent = "暂停";
  timer = window.setInterval(async () => {
    try {
      await step();
      if (state.report.finished) stopRun();
    } catch (error) {
      stopRun();
      renderError(error.message);
    }
  }, 280);
}

function stopRun() {
  if (timer) {
    window.clearInterval(timer);
    timer = null;
  }
  els.runBtn.textContent = "Agent 自动运行";
}

function renderError(message) {
  els.sectorViewBox.innerHTML = `<div class="empty-hint"><strong>运行失败</strong><p>${escapeHtml(message)}</p></div>`;
}

function render() {
  if (!state) return;
  const report = state.report || {};
  const portfolio = state.portfolio || {};
  const account = state.account || {};

  els.dayMetric.textContent = `${state.day} / ${state.total_days}`;
  els.equityMetric.textContent = money(portfolio.equity);
  els.returnMetric.textContent = pct(report.total_return);
  els.activeMetric.textContent = signedPct(report.active_return);
  els.drawdownMetric.textContent = pct(Math.abs(report.max_drawdown));
  els.tradeMetric.textContent = String(report.trade_count || 0);

  els.activeMetric.className = Number(report.active_return || 0) >= 0 ? "good" : "risk";
  els.drawdownMetric.className = "risk";

  renderModeBadge();
  renderSectorView();
  renderPositions();
  renderLog();
  renderTrace();
  renderResearch();
  drawChart();
}

function renderModeBadge() {
  const cfg = state?.config || {};
  const mode = cfg.agent_mode === "offline" ? "离线规则" : "LLM";
  const scope = cfg.universe && cfg.universe.length > 0 ? "多标 TopDown" : "单标";
  els.modeBadge.textContent = `${scope} · ${mode}`;
}

function renderSectorView() {
  const view = state.sector_view;
  const llmTrace = view?.llm_trace;

  // Update LLM badge in panel title
  if (llmTrace) {
    if (llmTrace.enabled) {
      els.llmBadge.className = "tag " + (llmTrace.error ? "sell" : "buy");
      els.llmBadge.textContent = llmTrace.error
        ? "LLM 失败"
        : `${llmTrace.model_used || "LLM"} · ${llmTrace.runtime || "-"}`;
    } else {
      els.llmBadge.className = "tag hold";
      els.llmBadge.textContent = "离线规则";
    }
  } else {
    els.llmBadge.className = "tag hold";
    els.llmBadge.textContent = "等待运行";
  }

  if (!view || Object.keys(view).length === 0) {
    els.sectorViewBox.innerHTML = `<div class="empty-hint">
      <strong>等待生成板块观点</strong>
      <p>点击上方“Agent 决策一步”让 LLM 选择板块和个股。</p>
    </div>`;
    return;
  }

  const marketView = view.market_view || "neutral";
  const marketClass = marketView === "bullish" ? "buy" : marketView === "bearish" ? "sell" : "hold";

  const selected = (view.selected_sectors || [])
    .map((s) => `<span class="tag buy">${escapeHtml(s)}</span>`)
    .join(" ");

  const ranking = (view.sector_ranking || [])
    .slice(0, 8)
    .map(
      (s) =>
        `<li>${escapeHtml(s.sector)} · 平均 ${(Number(s.avg_change) * 100).toFixed(2)}% · 因子 ${Number(
          s.avg_factor_score || 0
        ).toFixed(2)}</li>`
    )
    .join("");

  const stocks = (view.selected_stocks || [])
    .map(
      (s) =>
        `<li>
          <div class="stock-head">
            <strong>${escapeHtml(s.symbol)}</strong>
            <span class="tag buy">${escapeHtml(s.sector)}</span>
            <span class="stock-weight">${(Number(s.weight) * 100).toFixed(1)}%</span>
          </div>
          <div class="stock-meta">因子得分 ${Number(s.factor_score || 0).toFixed(2)}</div>
          ${s.rationale ? `<p class="stock-rationale">${escapeHtml(s.rationale)}</p>` : ""}
        </li>`
    )
    .join("");

  els.sectorViewBox.innerHTML = `
    <div class="sector-header">
      <div>
        <span class="sector-label">市场观点</span>
        <span class="tag ${marketClass}">${escapeHtml(marketView)}</span>
      </div>
      ${view.benchmark_return !== undefined ? `<div class="sector-label">基准收益 ${signedPct(view.benchmark_return)}</div>` : ""}
    </div>

    <div class="sector-section">
      <div class="sector-label">选中板块</div>
      <div class="sector-tags">${selected || "<span class='muted'>无</span>"}</div>
      ${view.reasoning ? `<p class="sector-reasoning">${escapeHtml(view.reasoning)}</p>` : ""}
    </div>

    ${view.aggregate_weight_reasoning ? `
      <div class="sector-section">
        <div class="sector-label">权重逻辑</div>
        <p class="sector-reasoning">${escapeHtml(view.aggregate_weight_reasoning)}</p>
      </div>` : ""}

    <div class="sector-section">
      <div class="sector-label">板块排名</div>
      <ol class="sector-list">${ranking || "<li>暂无板块数据</li>"}</ol>
    </div>

    ${stocks ? `
      <div class="sector-section">
        <div class="sector-label">选中个股</div>
        <ul class="stock-list">${stocks}</ul>
      </div>` : ""}
  `;
}

function renderPositions() {
  const positions = state.portfolio?.positions || {};
  const avgEntries = state.portfolio?.avg_entries || {};
  const symbols = Object.keys(positions).filter((s) => positions[s] > 0);
  if (symbols.length === 0) {
    els.positionsBody.innerHTML = `<tr><td colspan="3" class="muted">暂无持仓</td></tr>`;
    return;
  }
  els.positionsBody.innerHTML = symbols
    .map(
      (symbol) => `
        <tr>
          <td>${escapeHtml(symbol)}</td>
          <td>${money(positions[symbol])}</td>
          <td>${money(avgEntries[symbol] || 0, 2)}</td>
        </tr>`
    )
    .join("");
}

function renderLog() {
  const rows = state.trades
    .filter((trade) => trade.side !== "hold" || trade.blocked)
    .slice(-50)
    .reverse()
    .map(
      (trade) => `
        <tr>
          <td>${trade.day}</td>
          <td>${escapeHtml(trade.symbol || "-")}</td>
          <td>${Number(trade.price).toFixed(2)}</td>
          <td>${escapeHtml(trade.signal)}</td>
          <td><span class="tag ${trade.side}">${trade.side}</span> ${trade.shares}</td>
          <td>${escapeHtml(trade.reason)}</td>
          <td>${money(trade.fee, 2)}</td>
          <td>${money(trade.realized_pnl, 2)}</td>
          <td>${money(trade.cash)}</td>
          <td>${trade.position}</td>
          <td>${money(trade.equity)}</td>
        </tr>`
    );
  els.logBody.innerHTML = rows.join("");
}

function renderTrace() {
  const trace = state.trace || [];
  if (trace.length === 0) {
    els.traceList.innerHTML = `<li class="muted">等待 Agent 运行…</li>`;
    return;
  }
  els.traceList.innerHTML = trace
    .map((t) => {
      const detail = typeof t.detail === "string" ? t.detail : JSON.stringify(t.detail);
      return `<li><strong>${escapeHtml(t.tool)}</strong> ${escapeHtml(detail)}</li>`;
    })
    .join("");
}

function renderResearch() {
  if (isMultiSymbol()) {
    els.researchBox.innerHTML = "当前为多标 TopDown 模式，研究辩论/单标决策不适用。";
    return;
  }
  const latest = state.research_log?.[state.research_log.length - 1];
  if (!latest) {
    els.researchBox.innerHTML = "等待 analyst team 和 bull/bear researchers 生成结论。";
    return;
  }
  const reports = (latest.reports || [])
    .map(
      (report) => `
        <article>
          <strong>${escapeHtml(report.agent)} · ${escapeHtml(report.stance)}</strong>
          ${escapeHtml(report.summary)}<br />
          score=${Number(report.score).toFixed(2)}, confidence=${Number(report.confidence).toFixed(2)}
        </article>`
    )
    .join("");
  els.researchBox.innerHTML = `
    ${reports}
    <article>
      <strong>bull / bear debate · ${escapeHtml(latest.debate?.verdict || "-")}</strong>
      多头：${escapeHtml(latest.debate?.bull_thesis || "")}<br />
      空头：${escapeHtml(latest.debate?.bear_thesis || "")}<br />
      ${escapeHtml(latest.debate?.manager_notes || "")}
    </article>
  `;
}

function drawChart() {
  if (!state) return;
  const canvas = els.chart;
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * scale);
  canvas.height = Math.round(rect.height * scale);
  const ctx = canvas.getContext("2d");
  ctx.scale(scale, scale);

  const width = rect.width;
  const height = rect.height;
  const pad = { left: 48, right: 20, top: 16, bottom: 28 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const prices = state.series.map((item) => item.price);
  const equity = state.equity_history;
  const benchmark = state.benchmark_equity || [];
  const activePriceLength = Math.max(1, Math.min(prices.length, state.day + 1));
  const activeBenchmarkLength = Math.max(1, Math.min(benchmark.length, state.day + 1));

  const activePrices = prices.slice(0, activePriceLength);
  const equityDomain = [...equity, ...benchmark.slice(0, activeBenchmarkLength), state.config.initial_cash];
  const minPrice = Math.min(...activePrices) * 0.98;
  const maxPrice = Math.max(...activePrices) * 1.02;
  const minEquity = Math.min(...equityDomain) * 0.998;
  const maxEquity = Math.max(...equityDomain) * 1.002;

  const x = (idx, length = state.series.length) => pad.left + (idx / Math.max(1, length - 1)) * plotW;
  const yPrice = (value) => pad.top + (1 - (value - minPrice) / Math.max(0.0001, maxPrice - minPrice)) * plotH;
  const yEquity = (value) => pad.top + (1 - (value - minEquity) / Math.max(0.0001, maxEquity - minEquity)) * plotH;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fbfdfc";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#e3eaee";
  ctx.lineWidth = 1;
  ctx.font = "12px Inter, sans-serif";
  ctx.fillStyle = "#6b7780";
  for (let i = 0; i <= 4; i += 1) {
    const yy = pad.top + (plotH / 4) * i;
    ctx.beginPath();
    ctx.moveTo(pad.left, yy);
    ctx.lineTo(width - pad.right, yy);
    ctx.stroke();
    ctx.fillText((maxPrice - ((maxPrice - minPrice) / 4) * i).toFixed(0), 8, yy + 4);
  }

  function line(values, yFn, color, activeLength, lengthForX = state.series.length) {
    if (!values.length || activeLength <= 0) return;
    ctx.beginPath();
    values.slice(0, activeLength).forEach((value, idx) => {
      const xx = x(idx, lengthForX);
      const yy = yFn(value);
      if (idx === 0) ctx.moveTo(xx, yy);
      else ctx.lineTo(xx, yy);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.stroke();
  }

  line(prices, yPrice, "#2563eb", activePriceLength);
  line(equity, yEquity, "#b45309", equity.length);
  line(benchmark, yEquity, "#0f766e", activeBenchmarkLength, benchmark.length);

  state.trades.forEach((trade) => {
    if (trade.side === "hold" || trade.shares <= 0) return;
    ctx.beginPath();
    ctx.arc(x(Math.max(0, trade.day - 1)), yPrice(trade.price), 4, 0, Math.PI * 2);
    ctx.fillStyle = trade.side === "buy" ? "#0f766e" : "#b91c1c";
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.stroke();
  });
}

async function exportReport() {
  const payload = await api("/api/report");
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "trading-agent-report.json";
  link.click();
  URL.revokeObjectURL(url);
}

function updateTargetLabel() {
  els.targetRatioLabel.textContent = `${els.targetRatioInput.value}%`;
}

function bindEvents() {
  els.resetBtn.addEventListener("click", () => resetNow().catch((error) => renderError(error.message)));
  els.stepBtn.addEventListener("click", () => step().catch((error) => renderError(error.message)));
  els.runBtn.addEventListener("click", startRun);
  els.exportBtn.addEventListener("click", () => exportReport().catch((error) => renderError(error.message)));

  els.buyBtn.addEventListener("click", () => paperOrder("buy").catch((error) => renderError(error.message)));
  els.sellBtn.addEventListener("click", () => paperOrder("sell").catch((error) => renderError(error.message)));
  els.liveQuoteBtn.addEventListener("click", () => refreshLiveQuote().catch((error) => renderError(error.message)));
  els.rebalanceBtn.addEventListener("click", () => rebalanceToTarget().catch((error) => renderError(error.message)));
  els.addBtn.addEventListener("click", () => adjustPosition("increase").catch((error) => renderError(error.message)));
  els.reduceBtn.addEventListener("click", () => adjustPosition("decrease").catch((error) => renderError(error.message)));
  els.targetRatioInput.addEventListener("input", updateTargetLabel);

  // Core config auto-resets
  [els.dynamicUniverseInput, els.dynamicUniverseLimitInput, els.universeInput, els.daysInput, els.deepModelInput, els.llmBaseUrlInput].forEach((el) => {
    el.addEventListener("change", scheduleReset);
  });

  // Advanced config applies via button only
  els.applyAdvancedBtn.addEventListener("click", () => resetNow().catch((error) => renderError(error.message)));

  window.addEventListener("resize", drawChart);
}

bindEvents();
loadState();

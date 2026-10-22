// ShipCarbon website: reads the FastAPI endpoints on the same origin and draws the page.
"use strict";

const ACCENT = "#ffb547";
const ACCENT_DIM = "rgba(255, 181, 71, 0.32)";
const GRID = "rgba(255, 255, 255, 0.06)";
const TICK = "#7c87a6";
const TIMEOUT_MS = 20000;

const $ = (sel) => document.querySelector(sel);
const state = { year: "", shipType: "" };
const charts = {};
let requestId = 0;

// ---------- formatting ----------

const nf0 = new Intl.NumberFormat("en-GB", { maximumFractionDigits: 0 });
const nf1 = new Intl.NumberFormat("en-GB", { maximumFractionDigits: 1, minimumFractionDigits: 1 });

/** Tonnes as a short label: 1,116 Mt, 51.7 Mt, 10.3 kt, 640 t. */
function tonnes(t) {
  const units = [[1e6, "Mt"], [1e3, "kt"]];
  for (const [size, unit] of units) {
    if (Math.abs(t) >= size) {
      const v = t / size;
      return `${v >= 100 ? nf0.format(v) : nf1.format(v)} ${unit}`;
    }
  }
  return `${nf0.format(t)} t`;
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---------- API ----------

class ApiError extends Error {
  constructor(message, status, body) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

async function api(path, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  let res;
  try {
    res = await fetch(path, { ...options, signal: controller.signal, headers: { Accept: "application/json", ...(options.headers || {}) } });
  } catch (err) {
    throw new ApiError("network", 0, null);
  } finally {
    clearTimeout(timer);
  }
  let body = null;
  try { body = await res.json(); } catch { /* not JSON */ }
  if (!res.ok) throw new ApiError(`HTTP ${res.status}`, res.status, body);
  return body;
}

function query(params) {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== "" && v != null) q.set(k, v);
  const s = q.toString();
  return s ? `?${s}` : "";
}

// ---------- status banner ----------

function showStatus(message, isError = false) {
  const el = $("#status");
  el.textContent = message;
  el.classList.toggle("is-error", isError);
  el.hidden = false;
}

function hideStatus() { $("#status").hidden = true; }

/** Work out why a request failed and say it in plain words. */
async function explainFailure() {
  try {
    const health = await api("/api/health");
    if (!health.database) return "The database is not available right now, so the numbers cannot be loaded. Please try again later.";
    return "Something went wrong while loading the data. Please refresh the page.";
  } catch {
    return "The ShipCarbon API is not reachable. Check that the server is running, then refresh the page.";
  }
}

// ---------- charts ----------

function baseOptions(horizontal) {
  const valueAxis = {
    beginAtZero: true,
    grid: { color: GRID, drawTicks: false },
    border: { display: false },
    ticks: { color: TICK, padding: 8, callback: (v) => nf0.format(v) },
  };
  const catAxis = {
    grid: { display: false },
    border: { color: "rgba(255,255,255,0.12)" },
    ticks: { color: horizontal ? "#b5bdd3" : TICK, padding: 6, autoSkip: false },
  };
  return {
    indexAxis: horizontal ? "y" : "x",
    responsive: true,
    maintainAspectRatio: false,
    animation: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? false : { duration: 500 },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: "#0b1122",
        borderColor: "#2b3860",
        borderWidth: 1,
        titleColor: "#f4f6fb",
        bodyColor: "#b5bdd3",
        padding: 12,
        displayColors: false,
        callbacks: { label: (ctx) => `${tonnes(ctx.raw * 1e6)} CO₂` },
      },
    },
    scales: horizontal ? { x: valueAxis, y: catAxis } : { x: catAxis, y: valueAxis },
  };
}

function drawBarChart(key, canvasId, labels, valuesMt, highlight, horizontal) {
  const colors = labels.map((l) => (highlight === "" || String(l) === String(highlight) ? ACCENT : ACCENT_DIM));
  if (typeof Chart === "undefined") return; // CDN blocked: the table and numbers still work
  if (charts[key]) {
    charts[key].data.labels = labels;
    charts[key].data.datasets[0].data = valuesMt;
    charts[key].data.datasets[0].backgroundColor = colors;
    charts[key].update();
    return;
  }
  Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
  charts[key] = new Chart($(canvasId), {
    type: "bar",
    data: {
      labels,
      datasets: [{
        data: valuesMt,
        backgroundColor: colors,
        hoverBackgroundColor: ACCENT,
        borderRadius: 4,
        borderSkipped: "start",
        maxBarThickness: horizontal ? 22 : 44,
        categoryPercentage: 0.8,
      }],
    },
    options: baseOptions(horizontal),
  });
}

function shortType(name) {
  return name.length > 22 && window.innerWidth < 640 ? name.slice(0, 20) + "…" : name;
}

// ---------- rendering ----------

function renderKpis(k) {
  const set = (key, text) => { $(`[data-kpi="${key}"]`).textContent = text; };
  set("co2", tonnes(k.co2_t));
  set("ships", nf0.format(k.ships));
  set("fuel", tonnes(k.fuel_t));
  set("perShip", tonnes(k.co2_per_ship_t));
}

function renderEmitters(rows) {
  const body = $("#emitters-body");
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="7" class="muted">No ships match these filters.</td></tr>`;
    return;
  }
  body.innerHTML = rows.map((r) => `
    <tr>
      <td class="rank">${r.rank}</td>
      <td class="name">${escapeHtml(r.name || "Unknown")}</td>
      <td class="muted">${escapeHtml(r.imo)}</td>
      <td>${escapeHtml(r.ship_type)}</td>
      <td>${escapeHtml(r.flag || "Unknown")}</td>
      <td>${r.year}</td>
      <td class="num">${nf0.format(r.co2_t)}</td>
    </tr>`).join("");
}

function renderNotes() {
  const yearText = state.year || "All years";
  const typeText = state.shipType || "All ship types";
  $("#scope").textContent = `${yearText} · ${typeText}`;
  $("#year-note").textContent = `Million tonnes · ${state.shipType || "all ship types"}${state.year ? ` · ${state.year} highlighted` : ""}`;
  $("#type-note").textContent = `Top 10 · million tonnes · ${state.year || "all years"}${state.shipType ? " · selected type highlighted" : ""}`;
  $("#emitters-note").textContent = state.year
    ? `Ships with the most CO₂ in ${state.year}.`
    : "Single ship-years with the most CO₂, all years ranked together.";
  $("#f-reset").disabled = !state.year && !state.shipType;
}

async function loadData() {
  const id = ++requestId;
  $("#kpis").classList.add("is-loading");
  $("#kpis").setAttribute("aria-busy", "true");
  renderNotes();
  const both = query({ year: state.year, ship_type: state.shipType });
  try {
    // The year chart keeps every year (filtered by ship type); the type chart keeps every type (filtered by year).
    const [kpis, byYear, byType, top] = await Promise.all([
      api(`/api/kpis${both}`),
      api(`/api/co2-by-year${query({ ship_type: state.shipType })}`),
      api(`/api/co2-by-type${query({ year: state.year })}`),
      api(`/api/top-emitters${both}`),
    ]);
    if (id !== requestId) return; // a newer filter change is already on its way
    hideStatus();
    renderKpis(kpis);
    drawBarChart("year", "#chart-year", byYear.map((r) => r.year), byYear.map((r) => r.co2_t / 1e6), state.year, false);
    const top10 = byType.slice(0, 10);
    drawBarChart("type", "#chart-type", top10.map((r) => shortType(r.ship_type)), top10.map((r) => r.co2_t / 1e6),
      state.shipType ? shortType(state.shipType) : "", true);
    renderEmitters(top);
  } catch (err) {
    if (id !== requestId) return;
    showStatus(await explainFailure(), true);
    $("#emitters-body").innerHTML = `<tr><td colspan="7" class="muted">Data not available.</td></tr>`;
  } finally {
    if (id === requestId) {
      $("#kpis").classList.remove("is-loading");
      $("#kpis").setAttribute("aria-busy", "false");
    }
  }
}

function fillSelect(select, values, keepFirst = true) {
  const first = keepFirst ? select.options[0].outerHTML : "";
  select.innerHTML = first + values.map((v) => `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`).join("");
}

async function loadMeta() {
  try {
    const meta = await api("/api/meta");
    fillSelect($("#f-year"), meta.years);
    fillSelect($("#f-type"), meta.ship_types);
    $("#f-year").disabled = $("#f-type").disabled = false;

    $("#p-type").innerHTML = `<option value="">Choose a ship type</option>` +
      meta.ship_types.map((t) => `<option${t === "Container ship" ? " selected" : ""}>${escapeHtml(t)}</option>`).join("");
    const latest = meta.years[meta.years.length - 1];
    $("#p-year").innerHTML = meta.years.slice().reverse().map((y) => `<option${y === latest ? " selected" : ""}>${y}</option>`).join("");
    fillSelect($("#p-flag"), meta.flags.filter((f) => f !== "Unknown"));
    return true;
  } catch {
    showStatus(await explainFailure(), true);
    $("#p-type").innerHTML = `<option value="">Not available</option>`;
    $("#emitters-body").innerHTML = `<tr><td colspan="7" class="muted">Data not available.</td></tr>`;
    $("#kpis").classList.remove("is-loading");
    return false;
  }
}

// ---------- prediction ----------

const FIELD_NAMES = {
  ship_type: "Ship type", time_at_sea_h: "Hours at sea", tech_eff_type: "Efficiency index type",
  tech_eff_value: "Efficiency value", flag: "Flag", ice_class: "Ice class", year: "Year",
};

function friendlyPredictError(err) {
  if (err.status === 0) return "The server could not be reached. Check your connection and try again.";
  if (err.status === 503) return "The prediction model is not available on the server right now.";
  if (err.status === 422 && err.body) {
    const detail = err.body.detail;
    if (Array.isArray(detail)) {
      return detail.map((d) => {
        const field = FIELD_NAMES[d.loc?.[d.loc.length - 1]] || "Input";
        if (d.type === "less_than_equal") return `${field} must be at most ${nf0.format(d.ctx.le)}.`;
        if (d.type === "greater_than_equal") return `${field} must be at least ${nf0.format(d.ctx.ge)}.`;
        if (d.type === "greater_than") return `${field} must be more than ${d.ctx.gt}.`;
        if (String(d.msg).includes("unknown ship type")) return "Please choose a ship type from the list.";
        return `${field}: ${String(d.msg).replace(/^Value error, /, "")}.`;
      }).join(" ");
    }
    if (typeof detail === "string") return detail;
  }
  return "Something went wrong with the prediction. Please try again.";
}

function readForm(form) {
  const f = new FormData(form);
  const errors = [];
  const ship = { ship_type: f.get("ship_type"), ice_class: f.get("ice_class") === "on" };
  if (!ship.ship_type) errors.push("Please choose a ship type.");

  const hours = Number(f.get("time_at_sea_h"));
  if (f.get("time_at_sea_h") === "" || !Number.isFinite(hours)) errors.push("Please enter the hours at sea.");
  else if (hours < 0 || hours > 8784) errors.push("Hours at sea must be between 0 and 8,784 (one year).");
  else ship.time_at_sea_h = hours;

  if (f.get("tech_eff_type")) ship.tech_eff_type = f.get("tech_eff_type");
  if (f.get("tech_eff_value") !== "") {
    const v = Number(f.get("tech_eff_value"));
    if (!Number.isFinite(v) || v <= 0 || v > 500) errors.push("Efficiency value must be more than 0 and at most 500.");
    else ship.tech_eff_value = v;
  }
  if (f.get("flag")) ship.flag = f.get("flag");
  if (f.get("year")) ship.year = Number(f.get("year"));
  return { ship, errors };
}

async function onPredict(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const button = $("#p-submit");
  const errorEl = $("#p-error");
  errorEl.hidden = true;

  const { ship, errors } = readForm(form);
  if (errors.length) {
    errorEl.textContent = errors.join(" ");
    errorEl.hidden = false;
    return;
  }
  button.disabled = true;
  button.textContent = "Predicting…";
  try {
    const out = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(ship),
    });
    $("#p-fuel").textContent = `${nf0.format(out.fuel_t)} t`;
    $("#p-co2").textContent = `${nf0.format(out.co2_t)} t`;
    $("#p-placeholder").hidden = true;
    $("#p-values").hidden = false;
  } catch (err) {
    errorEl.textContent = friendlyPredictError(err);
    errorEl.hidden = false;
  } finally {
    button.disabled = false;
    button.innerHTML = "Predict fuel and CO<sub>2</sub>";
  }
}

// ---------- fade-in ----------

function setupReveal() {
  const items = document.querySelectorAll(".reveal");
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduced || !("IntersectionObserver" in window)) {
    items.forEach((el) => el.classList.add("visible"));
    return;
  }
  const observer = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) {
        e.target.classList.add("visible");
        observer.unobserve(e.target);
      }
    }
  }, { threshold: 0.08, rootMargin: "0px 0px -40px 0px" });
  items.forEach((el) => observer.observe(el));
}

// ---------- start ----------

async function init() {
  setupReveal();
  showStatus("Loading data…");
  $("#kpis").classList.add("is-loading");

  $("#f-year").addEventListener("change", (e) => { state.year = e.target.value; loadData(); });
  $("#f-type").addEventListener("change", (e) => { state.shipType = e.target.value; loadData(); });
  $("#f-reset").addEventListener("click", () => {
    state.year = state.shipType = "";
    $("#f-year").value = $("#f-type").value = "";
    loadData();
  });
  $("#predict-form").addEventListener("submit", onPredict);

  if (await loadMeta()) await loadData();
}

document.addEventListener("DOMContentLoaded", init);

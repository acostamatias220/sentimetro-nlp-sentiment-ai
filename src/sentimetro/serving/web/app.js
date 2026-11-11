"use strict";

const EXAMPLES = {
  mix: [
    "Llegó antes de lo previsto y funciona perfecto. Muy buena relación calidad-precio.",
    "Está bien, cumple su función, aunque esperaba un poco más por lo que cuesta.",
    "Se rompió a la semana y el vendedor nunca respondió. No lo recomiendo.",
    "Me encantó, ya es la segunda vez que lo compro 😍",
  ],
  neg: [
    "El pedido llegó incompleto y nadie me da una solución.",
    "La batería no dura ni dos horas, una estafa.",
    "Talla mucho más chica de lo indicado, tuve que devolverlo.",
  ],
  hard: [
    "No es el mejor que probé, pero tampoco es malo.",
    "Genial, otra vez llegó tarde. Justo lo que necesitaba...",
    "El producto es excelente pero el envío fue un desastre.",
  ],
};

const $ = (selector) => document.querySelector(selector);
const textsEl = $("#texts");
const statusEl = $("#status");
const resultsEl = $("#results");
const summaryEl = $("#summary");
const analyzeBtn = $("#analyze");
let maxBatch = 64;

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

function percent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function renderResults(texts, data) {
  resultsEl.replaceChildren();
  const counts = {};
  const tpl = $("#result-tpl");
  data.predictions.forEach((prediction, i) => {
    counts[prediction.label] = (counts[prediction.label] || 0) + 1;
    const node = tpl.content.firstElementChild.cloneNode(true);
    node.classList.add(prediction.label);
    const badge = node.querySelector(".badge");
    badge.textContent = prediction.label;
    badge.classList.add(prediction.label);
    node.querySelector(".confidence").textContent = `confianza ${percent(prediction.score)}`;
    node.querySelector(".text").textContent = texts[i];
    const bars = node.querySelector(".bars");
    for (const [label, value] of Object.entries(prediction.scores)) {
      const row = document.createElement("div");
      row.className = "bar";
      row.innerHTML = `<span></span><div class="track"><div class="fill ${label}" style="width:0"></div></div><span class="value"></span>`;
      row.firstElementChild.textContent = label;
      row.querySelector(".value").textContent = percent(value);
      bars.appendChild(row);
      requestAnimationFrame(() => {
        row.querySelector(".fill").style.width = percent(value);
      });
    }
    resultsEl.appendChild(node);
  });
  summaryEl.replaceChildren(
    ...Object.entries(counts).map(([label, n]) => {
      const pill = document.createElement("span");
      pill.className = `pill ${label}`;
      pill.textContent = `${n} ${label}`;
      return pill;
    }),
  );
}

async function analyze() {
  const texts = textsEl.value
    .split("\n")
    .map((t) => t.trim())
    .filter(Boolean);
  if (texts.length === 0) {
    setStatus("Escribí al menos una reseña.", true);
    return;
  }
  if (texts.length > maxBatch) {
    setStatus(`Máximo ${maxBatch} reseñas por lote.`, true);
    return;
  }
  const headers = { "Content-Type": "application/json" };
  const apiKey = $("#api-key").value.trim();
  if (apiKey) headers["X-API-Key"] = apiKey;

  analyzeBtn.disabled = true;
  setStatus("Analizando…");
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers,
      body: JSON.stringify({ texts }),
    });
    const data = await response.json();
    if (!response.ok) {
      const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ") : data.detail;
      throw new Error(detail || `HTTP ${response.status}`);
    }
    renderResults(texts, data);
    setStatus(`${texts.length} reseña(s) en ${data.latency_ms.toFixed(1)} ms`);
  } catch (error) {
    setStatus(`Error: ${error.message}`, true);
  } finally {
    analyzeBtn.disabled = false;
  }
}

async function loadModelInfo() {
  try {
    const response = await fetch("/model-info");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const info = await response.json();
    maxBatch = info.limits.max_batch_size;
    $("#max-batch").textContent = maxBatch;
    const f1 = info.test_metrics.macro_f1 ?? info.validation_metrics.macro_f1;
    const parts = [`Modelo ${info.name} v${info.version}`, `backend ${info.backend}`];
    if (f1 !== undefined) parts.push(`macro-F1 ${f1.toFixed(3)}`);
    $("#model-info").textContent = parts.join(" · ");
  } catch {
    $("#model-info").textContent = "Modelo no disponible";
  }
}

document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => {
    textsEl.value = EXAMPLES[button.dataset.example].join("\n");
    textsEl.focus();
  });
});
$("#clear").addEventListener("click", () => {
  textsEl.value = "";
  resultsEl.innerHTML = '<li class="empty">Todavía no hay resultados.</li>';
  summaryEl.replaceChildren();
  setStatus("");
});
analyzeBtn.addEventListener("click", analyze);
textsEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) analyze();
});

textsEl.value = EXAMPLES.mix.join("\n");
loadModelInfo();

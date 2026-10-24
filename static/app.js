"use strict";

const MAX_LENGTH = 2000;
const RECENT_KEY = "recent-questions";
const LOADING_STEPS = [
  "Searching papers…",
  "Ranking keyword and vector matches…",
  "Reading abstracts…",
  "Checking whether full text is needed…",
  "Composing a cited answer…",
];
const STAGE_LABELS = {
  abstract: "From abstracts",
  full_text: "From full text",
  abstract_fallback: "Abstract fallback",
};

const $ = (id) => document.getElementById(id);
const els = {
  form: $("ask-form"),
  question: $("question"),
  counter: $("counter"),
  askBtn: $("ask-btn"),
  error: $("error"),
  loading: $("loading"),
  loadingText: $("loading-text"),
  loadingHint: $("loading-hint"),
  result: $("result"),
  empty: $("empty"),
  answer: $("answer"),
  stage: $("stage-badge"),
  papers: $("papers"),
  stats: $("stats"),
  timeline: $("timeline"),
  feedbackMsg: $("feedback-msg"),
  status: $("status"),
  recent: $("recent"),
  recentList: $("recent-list"),
  themeToggle: $("theme-toggle"),
};

let currentRequestId = null;
let loadingTimer = null;

/* ---------- storage (best effort) ---------- */
function storageGet(key) {
  try { return localStorage.getItem(key); } catch (_) { return null; }
}
function storageSet(key, value) {
  try { localStorage.setItem(key, value); } catch (_) { /* private mode */ }
}

/* ---------- theme ---------- */
function effectiveTheme() {
  const explicit = document.documentElement.dataset.theme;
  if (explicit) return explicit;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
els.themeToggle.addEventListener("click", () => {
  const next = effectiveTheme() === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  storageSet("theme", next);
});

/* ---------- API status ---------- */
const START_HINT = "The API isn't running. Start it with `uv run uvicorn api:app`, then this page reconnects automatically.";
let apiOnline = null;
let healthTimer = null;

function setApiState(online) {
  const text = els.status.querySelector(".status-text");
  els.status.dataset.state = online ? "up" : "down";
  text.textContent = online ? "API online" : "API offline";
  if (online === apiOnline) return;
  apiOnline = online;
  els.askBtn.disabled = !online;
  if (!online) {
    showError(START_HINT);
  } else if (els.error.textContent === START_HINT) {
    els.error.hidden = true;
  }
}

async function checkHealth() {
  try {
    const response = await fetch("/health", { cache: "no-store" });
    setApiState(response.ok);
  } catch (_) {
    setApiState(false);
  }
  clearTimeout(healthTimer);
  // Poll quickly while offline so the page recovers as soon as the server starts.
  healthTimer = setTimeout(checkHealth, apiOnline ? 30000 : 5000);
}

if (location.protocol === "file:") {
  els.status.dataset.state = "down";
  els.status.querySelector(".status-text").textContent = "Not served";
  els.askBtn.disabled = true;
  showError("This file must be opened through the API. Start `uv run uvicorn api:app` and open http://localhost:8000");
} else {
  checkHealth();
}

/* ---------- input ---------- */
function updateCounter() {
  const length = els.question.value.length;
  els.counter.textContent = `${length} / ${MAX_LENGTH}`;
  els.counter.classList.toggle("over", length >= MAX_LENGTH);
}
els.question.addEventListener("input", updateCounter);
els.question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    els.form.requestSubmit();
  }
});
document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    els.question.value = chip.textContent.trim();
    updateCounter();
    els.form.requestSubmit();
  });
});

/* ---------- recent questions ---------- */
function loadRecent() {
  try {
    const items = JSON.parse(storageGet(RECENT_KEY) || "[]");
    return Array.isArray(items) ? items.filter((item) => typeof item === "string") : [];
  } catch (_) {
    return [];
  }
}
function saveRecent(question) {
  const items = [question, ...loadRecent().filter((item) => item !== question)].slice(0, 5);
  storageSet(RECENT_KEY, JSON.stringify(items));
  renderRecent();
}
function renderRecent() {
  const items = loadRecent();
  els.recent.hidden = items.length === 0;
  els.recentList.replaceChildren(
    ...items.map((item) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "recent-item";
      button.textContent = item;
      button.title = item;
      button.addEventListener("click", () => {
        els.question.value = item;
        updateCounter();
        els.form.requestSubmit();
      });
      return button;
    })
  );
}
renderRecent();

/* ---------- UI states ---------- */
function showError(message, kind = "error") {
  els.error.textContent = message;
  els.error.classList.toggle("warn", kind === "warn");
  els.error.hidden = false;
}
function setLoading(isLoading) {
  els.askBtn.disabled = isLoading || apiOnline === false;
  els.askBtn.querySelector(".btn-label").textContent = isLoading ? "Thinking…" : "Ask";
  els.loading.hidden = !isLoading;
  clearInterval(loadingTimer);
  if (isLoading) {
    const started = Date.now();
    const tick = () => {
      const seconds = Math.floor((Date.now() - started) / 1000);
      const step = Math.min(Math.floor(seconds / 3), LOADING_STEPS.length - 1);
      els.loadingText.textContent = `${LOADING_STEPS[step]} ${seconds}s`;
      els.loadingHint.hidden = seconds < 12;
    };
    tick();
    loadingTimer = setInterval(tick, 1000);
  }
}

/* ---------- rendering ---------- */
// Abstract answers cite source papers by number; full-text answers cite PDF excerpts,
// so only abstract-stage citations link to the source cards.
function renderAnswer(markdown, paperCount, linkToPapers) {
  const html = DOMPurify.sanitize(marked.parse(markdown || "_No answer returned._"));
  els.answer.innerHTML = html;
  // Turn [1], [2] citations in text nodes into links to the matching source card.
  const walker = document.createTreeWalker(els.answer, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walker.nextNode()) {
    if (!walker.currentNode.parentElement.closest("a, code")) nodes.push(walker.currentNode);
  }
  for (const node of nodes) {
    const parts = node.nodeValue.split(/(\[\d{1,2}\])/g);
    if (parts.length === 1) continue;
    const fragment = document.createDocumentFragment();
    for (const part of parts) {
      const match = /^\[(\d{1,2})\]$/.exec(part);
      const number = match ? Number(match[1]) : 0;
      if (match && !linkToPapers) {
        const badge = document.createElement("span");
        badge.className = "cite static";
        badge.textContent = number;
        badge.title = `Full-text excerpt ${number}`;
        fragment.append(badge);
      } else if (number >= 1 && number <= paperCount) {
        const link = document.createElement("a");
        link.href = `#paper-${number}`;
        link.className = "cite";
        link.textContent = number;
        link.setAttribute("aria-label", `Source ${number}`);
        link.addEventListener("click", (event) => {
          event.preventDefault();
          highlightPaper(number);
        });
        fragment.append(link);
      } else if (part) {
        fragment.append(part);
      }
    }
    node.replaceWith(fragment);
  }
}

function highlightPaper(number) {
  const card = $(`paper-${number}`);
  if (!card) return;
  card.scrollIntoView({ behavior: "smooth", block: "center" });
  card.classList.add("highlight");
  setTimeout(() => card.classList.remove("highlight"), 1800);
}

function safeUrl(url) {
  try {
    const parsed = new URL(url);
    return parsed.protocol === "https:" || parsed.protocol === "http:" ? parsed.href : null;
  } catch (_) {
    return null;
  }
}

function renderPapers(papers) {
  els.papers.replaceChildren(
    ...papers.map((paper, index) => {
      const number = index + 1;
      const card = document.createElement("article");
      card.className = "paper";
      card.id = `paper-${number}`;

      const top = document.createElement("div");
      top.className = "paper-top";
      const num = document.createElement("span");
      num.className = "paper-num";
      num.textContent = number;
      top.append(num);
      if (paper.category) {
        const category = document.createElement("span");
        category.className = "paper-cat";
        category.textContent = paper.category;
        top.append(category);
      }
      if (paper.published) {
        const date = document.createElement("span");
        date.className = "paper-date";
        date.textContent = String(paper.published).slice(0, 10);
        top.append(date);
      }

      const title = document.createElement("h3");
      title.className = "paper-title";
      const url = safeUrl(paper.url);
      if (url) {
        const link = document.createElement("a");
        link.href = url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = paper.title || "Untitled paper";
        title.append(link);
      } else {
        title.textContent = paper.title || "Untitled paper";
      }

      card.append(top, title);
      return card;
    })
  );
}

function stat(label, value) {
  const wrapper = document.createElement("div");
  const term = document.createElement("dt");
  term.textContent = label;
  const detail = document.createElement("dd");
  detail.textContent = value;
  wrapper.append(term, detail);
  return wrapper;
}

function describeStep(step) {
  switch (step.name) {
    case "search_papers":
      return `${step.retrieved_count ?? 0} papers retrieved`;
    case "retrieve_full_text":
      return `${step.chunk_count ?? 0} chunks from ${step.selected_count ?? 0} papers, ${step.context_tokens ?? 0} tokens${step.success === false ? " (failed)" : ""}`;
    default:
      return [step.status, step.outcome].filter(Boolean).join(" · ");
  }
}

function renderDetails(result) {
  const usage = result.usage || {};
  const cost = Number(usage.estimated_cost_usd || 0);
  els.stats.replaceChildren(
    stat("Stage", STAGE_LABELS[result.answer_stage] || result.answer_stage || "unknown"),
    stat("Input tokens", (usage.input_tokens || 0).toLocaleString()),
    stat("Output tokens", (usage.output_tokens || 0).toLocaleString()),
    stat("Est. cost", `$${cost.toFixed(5)}`)
  );
  els.timeline.replaceChildren(
    ...(result.trajectory || []).map((step) => {
      const item = document.createElement("li");
      const name = document.createElement("span");
      name.className = "step-name";
      name.textContent = step.name;
      item.append(name);
      const meta = describeStep(step);
      if (meta) {
        const detail = document.createElement("span");
        detail.className = "step-meta";
        detail.textContent = ` — ${meta}`;
        item.append(detail);
      }
      return item;
    })
  );
}

function resetFeedback() {
  document.querySelectorAll(".fb-btn").forEach((button) => {
    button.disabled = false;
    button.classList.remove("selected");
  });
  els.feedbackMsg.textContent = "";
}

/* ---------- ask ---------- */
async function readError(response) {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail) && body.detail[0]?.msg) return body.detail[0].msg;
  } catch (_) { /* not JSON */ }
  return `The server returned ${response.status}.`;
}

els.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = els.question.value.trim();
  els.error.hidden = true;
  if (!question) {
    showError("Type a question first.", "warn");
    els.question.focus();
    return;
  }

  setLoading(true);
  els.result.hidden = true;
  els.empty.hidden = true;
  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (response.status === 429) {
      showError("You've reached the question limit for now. Please wait a minute and try again.", "warn");
      els.empty.hidden = false;
      return;
    }
    if (!response.ok) {
      showError(await readError(response));
      els.empty.hidden = false;
      return;
    }
    const result = await response.json();
    currentRequestId = result.request_id || null;
    const papers = result.retrieved_papers || [];
    els.stage.textContent = STAGE_LABELS[result.answer_stage] || "Answer";
    els.stage.className = `badge ${result.answer_stage || ""}`;
    renderAnswer(result.answer, papers.length, result.answer_stage !== "full_text");
    renderPapers(papers);
    renderDetails(result);
    resetFeedback();
    els.result.hidden = false;
    saveRecent(question);
  } catch (_) {
    els.empty.hidden = false;
    await checkHealth();
    if (apiOnline) showError("The connection dropped before the answer arrived. Please try again.");
  } finally {
    setLoading(false);
  }
});

/* ---------- feedback ---------- */
document.querySelectorAll(".fb-btn").forEach((button) => {
  button.addEventListener("click", async () => {
    if (!currentRequestId) return;
    const buttons = document.querySelectorAll(".fb-btn");
    buttons.forEach((b) => (b.disabled = true));
    try {
      const response = await fetch("/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ request_id: currentRequestId, rating: button.dataset.rating }),
      });
      if (response.ok || response.status === 409) {
        button.classList.add("selected");
        els.feedbackMsg.textContent = "Thanks for the feedback!";
      } else {
        els.feedbackMsg.textContent = await readError(response);
        buttons.forEach((b) => (b.disabled = false));
      }
    } catch (_) {
      els.feedbackMsg.textContent = "Couldn't send feedback.";
      buttons.forEach((b) => (b.disabled = false));
    }
  });
});

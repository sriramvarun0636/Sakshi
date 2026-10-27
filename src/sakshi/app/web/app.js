/* SAKSHI fieldroom: deliberately small, dependency-free presentation client. */
const state = {
  page: "briefing",
  bootstrap: null,
  event: "hurricane_maria",
  window: 0,
  sitrepWindow: 2,
  lens: "all",
  reviewLens: "pending",
  report: null,
  playing: false,
  speed: 1,
  timer: null,
  classify: null,
  explanation: null,
  featuredCase: null,
  briefingSitrep: null,
  briefingStep: 0,
  briefingAnalysis: null,
};
const view = document.getElementById("view");
const presentation = document.getElementById("presentation");
let requestVersion = 0;

function esc(value) {
  return String(value == null ? "" : value).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
}
function fmt(value) {
  return Number(value || 0).toLocaleString("en-US");
}
function pct(value) {
  return value == null ? "N/A" : Math.round(Number(value) * 100) + "%";
}
function displayName(value) {
  return String(value || "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}
function reviewReason(value) {
  return String(value || "Review requested")
    .replace(/^low_confidence:/i, "Low confidence · ")
    .replace(/_/g, " ");
}
function title(text, note) {
  return (
    '<div class="section-title"><h2>' +
    esc(text) +
    "</h2><span>" +
    esc(note || "") +
    "</span></div>"
  );
}
function pill(text, tone) {
  return (
    '<span class="pill ' + esc(tone || "muted") + '">' + esc(text) + "</span>"
  );
}
function emptyImage() {
  return '<div class="photo-missing">Image<br>unavailable</div>';
}
function loading(label) {
  view.innerHTML =
    '<div class="loading"><span class="spinner"></span><p>' +
    esc(label || "Reading the field record") +
    "</p></div>";
}
function notify(message) {
  const toast = document.getElementById("toast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => toast.classList.remove("show"), 3600);
}
function pageIndex(number, label) {
  return (
    '<div class="page-index"><span>' +
    esc(number) +
    " / " +
    esc(label) +
    "</span></div>"
  );
}
function notice(label, heading, text, command) {
  return (
    '<div class="notice"><div><span class="micro micro-orange">' +
    esc(label) +
    "</span><h3>" +
    esc(heading) +
    "</h3><p>" +
    esc(text) +
    "</p></div>" +
    (command ? "<code>" + esc(command) + "</code>" : "") +
    "</div>"
  );
}
async function api(path, options) {
  const response = await fetch(path, options);
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw Error("The local service did not return data.");
  }
  if (!response.ok) throw Error(payload.error || "Request failed");
  return payload;
}
async function post(path, data) {
  return api(path, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Sakshi-Client": "fieldroom",
    },
    body: JSON.stringify(data),
  });
}
function params(data) {
  const p = new URLSearchParams(data);
  p.set("presentation", presentation.checked ? "true" : "false");
  return p.toString();
}
function changePage(page) {
  if (!["briefing", "classify", "replay", "sitrep", "review", "results"].includes(page))
    page = "briefing";
  state.page = page;
  state.playing = false;
  clearInterval(state.timer);
  document
    .querySelectorAll(".nav-item")
    .forEach((b) => b.classList.toggle("active", b.dataset.page === page));
  document.getElementById("briefing-link").classList.toggle("active", page === "briefing");
  history.replaceState(null, "", "#" + page);
  window.scrollTo(0, 0);
  render();
}

function briefingModelHtml(result) {
  if (!result)
    return '<div class="briefing-model-await"><span>LIVE INFERENCE / LOCAL CPU</span><strong>Four decisions await the case.</strong><p>Run the pinned Maria post through the trained model. Its decisions and any abstention appear here.</p></div>';
  const decisions = result.heads.map((head, i) => {
    const label = head.status === "predicted" ? head.label || "Undecided" :
      head.status === "gated" ? "Gated" : "Not applicable";
    return '<div class="briefing-model-tile"><small>0' + (i + 1) + ' / ' + esc(head.name) +
      '</small><strong>' + esc(label) + '</strong><span>' +
      (head.status === "predicted" ? pct(head.probability) + " CALIBRATED" : "NO SCORE") +
      '</span></div>';
  }).join("");
  return '<div class="briefing-model-status ' + (result.abstain ? "is-abstained" : "") +
    '"><span>' + (result.abstain ? "ABSTAINED / HUMAN REVIEW" : "CLASSIFICATION RETURNED") +
    '</span><strong>' + esc(result.abstain ?
      reviewReason(result.abstain_reason) :
      result.model) +
    '</strong></div><div class="briefing-model-tiles">' + decisions + '</div>' +
    (result.origin ? '<div class="briefing-origin"><span>IMAGE ORIGIN / ' +
      esc(displayName(result.origin.status)) + '</span><strong>' +
      esc(result.origin.event ? "Earlier indexed in " + result.origin.event : "No earlier indexed match") +
      '</strong></div>' : "");
}

function briefingHtml(caseData, sitrep) {
  const hasCase = Boolean(caseData?.available);
  const sentences = sitrep?.available ? sitrep.sentences : [];
  const accepted = sentences.find((s) => s.accepted && s.id === "S3") || sentences.find((s) => s.accepted);
  const rejected = sentences.find((s) => !s.accepted && s.failed_gates?.some((g) => g.name === "number")) ||
    sentences.find((s) => !s.accepted);
  const originalCount = sentences.filter((s) => Number(s.round) === 0).length;
  const acceptedCount = sentences.filter((s) => s.accepted).length;
  const failedRewrites = sentences.filter((s) => !s.accepted && Number(s.round) > 0).length;
  const metrics = state.bootstrap.headline_metrics || [];
  const metricBars = metrics.map((metric, i) => {
    const value = Number(metric.value);
    return '<div class="briefing-score-row ' + (value < 0.5 ? "weak" : "") +
      '"><div><span>0' + (i + 1) + ' / ' + esc(metric.label) + '</span><strong>' +
      esc(value.toFixed(3)) + '</strong></div><i><b style="width:' +
      Math.max(0, Math.min(100, value * 100)).toFixed(1) + '%"></b></i></div>';
  }).join("");
  const photos = hasCase ? '<div class="briefing-photo-pair"><figure><img src="' +
    esc(caseData.current_image) + '" alt="Image indexed with a Hurricane Maria post"><figcaption><b>HURRICANE MARIA</b><span>' +
    esc(caseData.current_date) + '</span></figcaption></figure><figure><img src="' +
    esc(caseData.origin_image) + '" alt="Earlier indexed Hurricane Irma image"><figcaption><b>HURRICANE IRMA</b><span>' +
    esc(caseData.origin_date) + '</span></figcaption></figure></div>' :
    '<div class="briefing-photo-unavailable">The pinned image pair is unavailable on this machine.</div>';
  const sceneNav = '<nav class="briefing-chapters" aria-label="Briefing scenes"><span>FIELD BRIEFING / 04 SCENES</span>' +
    ['THE IMAGE', 'THE MODEL', 'THE CLAIM', 'THE RECORD'].map((label, i) =>
      '<button type="button" data-briefing-step="' + i + '" class="' +
      (state.briefingStep === i ? 'active' : '') + '">0' + (i + 1) + ' ' + label + '</button>').join('') +
    '<span>← → TO NAVIGATE</span></nav>';
  return '<div class="briefing">' + sceneNav +
    '<section class="briefing-scene briefing-cover" id="briefing-scene-0"><div class="briefing-cover-copy">' +
    '<div class="briefing-eyebrow"><span>CASE FILE / 01</span><span>SYSTEM FLAG · REVIEW PENDING</span></div>' +
    '<div><h1>One image.<br><em>Two events.</em></h1><p>The same flood photograph appears in a Maria post and an earlier indexed Irma record. The system flags reuse; a reviewer decides what the match means.</p></div>' +
    '<div class="briefing-cover-foot"><div><strong>07</strong><span>DAYS BETWEEN<br>INDEXED APPEARANCES</span></div><button type="button" data-briefing-step="1">FOLLOW THE EVIDENCE <b>↓</b></button></div></div>' +
    '<div class="briefing-cover-visual"><div class="briefing-visual-top"><span>FROZEN REPLAY / FACE-BLURRED</span><span>PHASH DISTANCE ' +
    (hasCase ? esc(caseData.phash_distance) : '—') + ' / 64</span></div>' + photos +
    '<div class="briefing-visual-bottom">EARLIER INDEXED APPEARANCE ≠ VERIFIED ORIGIN</div></div></section>' +
    '<section class="briefing-scene briefing-model" id="briefing-scene-1"><div class="briefing-section-head"><span>02 / LIVE MODEL</span><span>FOUR TASK HEADS · HUMAN REVIEW</span></div>' +
    '<div class="briefing-model-layout"><div class="briefing-model-copy"><h2>What does<br>the model <em>know?</em></h2><p>A classifier can help triage a post while remaining uncertain. Run this exact case through the local model and inspect every head.</p>' +
    '<button type="button" id="briefing-run" class="briefing-run" ' + (hasCase ? '' : 'disabled') + '>RUN THIS CASE LIVE <span>↗</span></button>' +
    '<small>THE FIRST RUN LOADS CACHED MODEL WEIGHTS.</small></div><div id="briefing-model-output">' +
    briefingModelHtml(state.briefingAnalysis) + '</div></div><button class="briefing-next" type="button" data-briefing-step="2">NEXT / THE CLAIM <span>↓</span></button></section>' +
    '<section class="briefing-scene briefing-claim" id="briefing-scene-2"><div class="briefing-section-head"><span>03 / VERIFIED GENERATION</span><span>HURRICANE MARIA · WINDOW 03</span></div>' +
    '<div class="briefing-claim-top"><div><h2>A claim must<br><em>earn its place.</em></h2><p>Every sentence needs a citation. Failed claims remain visible, along with the gate that stopped them.</p></div>' +
    '<div class="briefing-verdict"><span>AUTOMATED VERIFIER · NOT HUMAN RATED</span><strong>' +
    (sitrep?.available ? pct(sitrep.rates.final) : '—') + '</strong><small>FINAL ACCEPTANCE OF ORIGINAL CLAIMS</small><div><b>' +
    originalCount + ' ORIGINAL</b><b>' + acceptedCount + ' CLEARED</b><b>' + failedRewrites + ' FAILED REWRITES</b></div></div></div>' +
    '<div class="briefing-claim-pair"><article><span>CLAIM CLEARED / ' + esc(accepted?.id || '—') +
    '</span><p>' + esc(accepted?.text || 'No accepted claim in this window.') + '</p><small>CITED POST / ' +
    esc((accepted?.citations || []).join(', ') || 'NONE') + '</small></article><article class="stopped"><span>CLAIM STOPPED / ' +
    esc(rejected?.id || '—') + '</span><p>' + esc(rejected?.text || 'No rejected claim in this window.') +
    '</p><small>' + esc((rejected?.failed_gates || []).map((g) => g.name.toUpperCase() + ' GATE · ' + g.detail).join(' / ') || 'VERIFICATION FAILED') +
    '</small></article></div><button class="briefing-next" type="button" data-briefing-step="3">NEXT / THE RECORD <span>↓</span></button></section>' +
    '<section class="briefing-scene briefing-record" id="briefing-scene-3"><div class="briefing-section-head"><span>04 / FROZEN EVALUATION</span><span>OFFICIAL TEST · THREE SEEDS</span></div>' +
    '<div class="briefing-record-layout"><div><h2>The weak head<br>stays <em>visible.</em></h2><p>Mean macro-F1 across three frozen standard seeds. Each task has its own official test selector. Severity remains the hardest head.</p>' +
    '<a href="#results" class="briefing-record-link">OPEN THE FULL RESEARCH RECORD <span>↗</span></a></div><div class="briefing-scores">' + metricBars +
    '<small>0.0 <span>MEAN MACRO-F1 / NOT A HUMAN STUDY</span> 1.0</small></div></div>' +
    '<div class="briefing-exit"><span>THE EVIDENCE IS OPEN TO INSPECTION.</span><div><a href="#classify">INVESTIGATE ↗</a><a href="#replay">EVENT REPLAY ↗</a><a href="#sitrep">SITREP ↗</a><a href="#review">REVIEW ↗</a></div></div></section></div>';
}

function goBriefingStep(value) {
  const step = Math.max(0, Math.min(3, Number(value)));
  state.briefingStep = step;
  document.querySelectorAll('.briefing-chapters button').forEach((button) =>
    button.classList.toggle('active', Number(button.dataset.briefingStep) === step));
  document.getElementById('briefing-scene-' + step)?.scrollIntoView({behavior:'instant',block:'start'});
}

async function runBriefingCase() {
  const button = document.getElementById('briefing-run');
  if (!button) return;
  button.disabled = true;
  button.textContent = 'RUNNING LOCAL MODEL…';
  try {
    state.briefingAnalysis = await post('/api/featured-analysis', {presentation:presentation.checked});
    state.classify = state.briefingAnalysis;
    state.explanation = null;
    const output = document.getElementById('briefing-model-output');
    if (output) output.innerHTML = briefingModelHtml(state.briefingAnalysis);
  } catch (error) {
    notify(error.message);
  } finally {
    button.disabled = false;
    button.innerHTML = 'RUN THIS CASE LIVE <span>↗</span>';
  }
}

async function renderBriefing() {
  const version = ++requestVersion;
  loading('Opening the field briefing');
  try {
    const [caseData, sitrep] = await Promise.all([
      state.featuredCase || api('/api/featured-case'),
      state.briefingSitrep || api('/api/sitrep?' + params({event:'hurricane_maria',window:2})),
    ]);
    if (version !== requestVersion) return;
    state.featuredCase = caseData;
    state.briefingSitrep = sitrep;
    view.innerHTML = briefingHtml(caseData, sitrep);
    document.querySelectorAll('[data-briefing-step]').forEach((button) =>
      button.addEventListener('click', () => goBriefingStep(button.dataset.briefingStep)));
    document.getElementById('briefing-run')?.addEventListener('click',runBriefingCase);
    if (state.briefingStep) goBriefingStep(state.briefingStep);
  } catch (error) {
    view.innerHTML = notice('BRIEFING ERROR','Could not load the briefing',error.message,'');
  }
}

function radarArt() {
  return (
    '<svg viewBox="0 0 600 510" fill="none" preserveAspectRatio="xMidYMid slice" aria-hidden="true">' +
    '<path d="M-20 427C69 410 118 363 168 332S301 308 331 237 379 136 480 130 550 86 620 62" stroke="#A8C8FF" stroke-width="2" opacity=".75"/>' +
    '<path d="M-30 455C71 421 103 384 179 379S291 347 340 278 410 239 504 189 575 144 630 140" stroke="#4368FF" stroke-width="3"/>' +
    '<path d="M-20 379C53 362 127 300 190 296S305 287 341 199 391 104 493 100 551 54 620 26" stroke="#EEF3FB" opacity=".43"/>' +
    '<circle cx="340" cy="278" r="8" fill="#4368FF"/><circle cx="340" cy="278" r="23" stroke="#4368FF" opacity=".6"/>' +
    '<circle cx="504" cy="189" r="5" fill="#EEF3FB"/><circle cx="179" cy="379" r="5" fill="#EEF3FB"/>' +
    '<path d="M40 48h115M40 60h72M414 449h147M458 461h103" stroke="#A8C8FF" opacity=".7"/>' +
    '<text x="41" y="85" fill="#A8C8FF" font-size="10" letter-spacing="4">SIGNAL DETECTED</text><text x="420" y="435" fill="#A8C8FF" font-size="10" letter-spacing="4">MODEL / HUMAN</text></svg>' +
    '<span class="visual-cross"></span><span class="visual-caption">FIG. 01 / TRACE THE EVIDENCE</span>'
  );
}

function featuredCaseHtml(data) {
  if (!data?.available) return radarArt();
  return (
    '<div class="case-study"><div class="case-top"><span>FROM THE FROZEN PROVENANCE REPLAY</span><strong><i></i> SYSTEM FLAG / REVIEW PENDING</strong></div>' +
    '<div class="case-pair"><figure><div class="case-photo"><img src="' +
    esc(data.current_image) +
    '" alt="Image posted during Hurricane Maria"></div><figcaption><b>' +
    esc(data.current_event) +
    "</b><span>" +
    esc(data.current_date) +
    '</span></figcaption></figure><span class="case-arrow" aria-hidden="true">↔</span><figure><div class="case-photo"><img src="' +
    esc(data.origin_image) +
    '" alt="Earlier indexed image from Hurricane Irma"></div><figcaption><b>' +
    esc(data.origin_event) +
    "</b><span>" +
    esc(data.origin_date) +
    "</span></figcaption></figure></div>" +
    '<div class="case-bottom"><div><small>PERCEPTUAL HASH DISTANCE</small><strong>' +
    esc(data.phash_distance) +
    '<em> / 64</em></strong></div><p>The same flood photograph appears in two event records. The system flags the earlier indexed source for review.</p></div>' +
    '<div class="case-actions"><button type="button" id="analyze-featured" class="case-link">RUN LIVE ANALYSIS <span>↗</span></button><a class="case-link" href="#replay">OPEN EVENT REPLAY <span>↗</span></a></div></div>'
  );
}

async function renderFeaturedCase() {
  const host = document.getElementById("featured-case");
  if (!host) return;
  if (!state.featuredCase) {
    try {
      state.featuredCase = await api("/api/featured-case");
    } catch {
      state.featuredCase = { available: false };
    }
  }
  if (state.page !== "classify" || !host.isConnected) return;
  host.classList.toggle("is-case", Boolean(state.featuredCase.available));
  host.innerHTML = featuredCaseHtml(state.featuredCase);
  host.querySelector("#analyze-featured")?.addEventListener("click", analyzeFeaturedCase);
}

async function analyzeFeaturedCase() {
  const button = document.getElementById("analyze-featured");
  if (!button) return;
  button.disabled = true;
  button.textContent = "RUNNING LOCAL MODEL…";
  try {
    state.classify = await post("/api/featured-analysis", {
      presentation: presentation.checked,
    });
    state.explanation = null;
    const output = document.getElementById("classify-output");
    if (output) {
      output.innerHTML = classifyOutput(state.classify);
      bindExplanation();
      output.scrollIntoView({ behavior: "smooth" });
    }
  } catch (error) {
    notify(error.message);
  } finally {
    button.disabled = false;
    button.innerHTML = "RUN LIVE ANALYSIS <span>↗</span>";
  }
}

function classifyShell() {
  return (
    '<div class="page"><div class="page-inner">' +
    pageIndex("01", "INVESTIGATE") +
    '<section class="hero"><div class="hero-copy"><div class="hero-kicker">SITUATIONAL AWARENESS / LIVE ANALYSIS</div><div><h1>See the signal.<br><em>Trace</em> the source.</h1><p>One post, four decisions, and a visible path back to the evidence. Read the model; question the image.</p></div><div class="hero-foot"><span class="pulse"></span> CPU INFERENCE <span class="slash">/</span> LOCAL ONLY <span class="slash">/</span> HUMAN REVIEW</div></div><div class="hero-visual' +
    (state.featuredCase?.available ? " is-case" : "") +
    '" id="featured-case">' +
    featuredCaseHtml(state.featuredCase) +
    "</div></section>" +
    '<div class="ribbon"><div class="ribbon-cell"><small>01 / READ</small><strong>Four <em>decisions</em></strong></div><div class="ribbon-cell"><small>02 / TRACE</small><strong>Image <em>origin</em></strong></div><div class="ribbon-cell"><small>03 / EXPLAIN</small><strong>Evidence <em>map</em></strong></div><div class="ribbon-cell"><small>04 / ESCALATE</small><strong>Human <em>review</em></strong></div></div>' +
    title("The input desk", "TEXT / IMAGE / CONTEXT") +
    '<form id="classify-form"><div class="investigate-grid"><div class="investigate-text"><div class="field"><label for="post-text">01 / POST TEXT</label><textarea id="post-text" placeholder="Paste the crisis post here. Handles and links are removed from every displayed result."></textarea></div><div class="form-foot"><div class="form-help">Text-only and image-only posts are supported. This model was trained on English-language crisis data.</div><button class="button orange" type="submit">ANALYZE POST &nbsp; ↗</button></div></div><div class="investigate-media"><div><span class="field-label">02 / VISUAL EVIDENCE</span><label class="upload-drop" id="upload-drop"><input id="post-image" type="file" accept="image/jpeg,image/png,image/webp"><strong>Drop an image into the record.</strong><small>JPEG / PNG / WEBP · 10 MB MAXIMUM</small></label></div><div class="context-row"><div class="field"><label for="context-event">03 / EVENT CONTEXT</label><select id="context-event"><option value="">No event context</option>' +
    eventOptions() +
    '</select></div><div class="field"><label for="context-time">04 / POST TIME (UTC)</label><input id="context-time" type="datetime-local" value="2017-09-25T12:00"></div></div></div></div></form><div id="classify-output"></div></div></div>'
  );
}

function eventOptions() {
  return (state.bootstrap.events || [])
    .map(
      (e) => '<option value="' + esc(e.id) + '">' + esc(e.name) + "</option>",
    )
    .join("");
}

function classifyOutput(result) {
  if (!result) return "";
  const heads = result.heads
    .map(
      (head) =>
        '<div class="decision-tile"><small>' +
        esc(head.name) +
        "</small><strong>" +
        esc(
          head.status === "unavailable"
            ? "Not applicable"
            : head.status === "gated"
              ? "Gated by informative"
              : head.label || "Undecided",
        ) +
        "</strong><span>" +
        esc(
          head.status === "predicted"
            ? pct(head.probability) + " CALIBRATED"
            : "N/A",
        ) +
        "</span></div>",
    )
    .join("");
  const origin = result.origin
    ? '<div class="technical-card"><h3>IMAGE ORIGIN / INDEXED ARCHIVE</h3>' +
      pill(
        displayName(result.origin.status),
        result.origin.status === "recycled" ? "red" : "dark",
      ) +
      "<p>" +
      (result.origin.event
        ? "Earliest indexed match: <b>" +
          esc(result.origin.event) +
          "</b> / " +
          esc(result.origin.date || "date unavailable")
        : "No earlier indexed image was matched with the available context.") +
      "</p>" +
      (result.origin.image
        ? '<img class="heatmap-image" src="' +
          result.origin.image +
          '" alt="Earliest indexed visual match">'
        : "") +
      "</div>"
    : "";
  return (
    title("The model reading", esc(result.model) + " / CALIBRATED") +
    (result.non_english
      ? notice(
          "LANGUAGE CHECK",
          "Interpret with care",
          "The model was trained on English-language posts.",
          "",
        )
      : "") +
    '<div class="analysis-layout"><div class="evidence-portrait ' +
    (!result.image ? "empty" : "") +
    '">' +
    (result.image
      ? '<img src="' + result.image + '" alt="Uploaded visual evidence">'
      : "") +
    '<span class="corner">INPUT / 01</span></div><div><div style="margin-bottom:13px">' +
    pill(
      result.abstain
        ? "ABSTAINED · " + reviewReason(result.abstain_reason)
        : "CLASSIFICATION RETURNED",
      result.abstain ? "orange" : "dark",
    ) +
    '</div><div class="decision-columns">' +
    heads +
    "</div></div></div>" +
    (result.text
      ? '<div class="technical-card" style="margin-top:25px"><h3>CLEANED POST TEXT</h3><p>' +
        esc(result.text) +
        "</p></div>"
      : "") +
    '<div class="result-lower"><div class="technical-card"><h3>MODALITY RELIANCE / LEARNED FUSION GATE</h3><div class="reliance-bar"><span style="width:' +
    Math.round((result.modality.text || 0) * 100) +
    '%"></span><span style="width:' +
    Math.round((result.modality.image || 0) * 100) +
    '%"></span></div><div class="bar-key"><span>TEXT ' +
    pct(result.modality.text) +
    "</span><span>IMAGE " +
    pct(result.modality.image) +
    "</span></div></div>" +
    origin +
    "</div>" +
    title("What moved the decision", "ON DEMAND / CPU") +
    '<div class="notice"><div><span class="micro micro-orange">EVIDENCE MAP</span><h3>Inspect the model, not just the label.</h3><p>Word attribution and image focus are computed only when requested.</p></div><div><select id="explain-head" class="select">' +
    result.heads
      .filter((h) => h.status === "predicted")
      .map(
        (h) =>
          '<option value="' + esc(h.key) + '">' + esc(h.name) + "</option>",
      )
      .join("") +
    '</select><button id="explain-button" class="button" style="margin-top:9px;width:100%">REVEAL EVIDENCE →</button></div></div><div id="explain-output"></div>' +
    '<div style="margin:27px 0"><button class="button ghost small" disabled>GENERATE EVIDENCE RATIONALE</button><p class="micro micro-mute">REQUIRES SAME-MODEL PACKETS AND LOCAL QWEN / NLI WEIGHTS. SEE DOCS/GENERATION.MD.</p></div>'
  );
}

function explanationOutput(result) {
  if (!result) return "";
  const max = Math.max(0.001, ...result.words.map((w) => Math.max(0, w.score)));
  return (
    '<div class="explanation-grid" style="margin-top:25px"><div class="technical-card"><h3>01 / WORD ATTRIBUTION</h3>' +
    (result.words.length
      ? result.words
          .map(
            (w) =>
              '<span class="word-chip" style="background:rgba(67,104,255,' +
              ((Math.max(0, w.score) / max) * 0.38).toFixed(3) +
              ')">' +
              esc(w.text) +
              "</span>",
          )
          .join("")
      : "<p>No text attribution for this head.</p>") +
    '<p class="micro micro-mute">INTEGRATED GRADIENTS / ' +
    esc(result.steps || "—") +
    ' STEPS</p></div><div class="technical-card"><h3>02 / IMAGE FOCUS</h3>' +
    (result.heatmap
      ? '<img class="heatmap-image" src="' +
        result.heatmap +
        '" alt="Grad-CAM overlay">'
      : "<p>No image heatmap for this decision.</p>") +
    '<p class="micro micro-mute">GRAD-CAM / SOURCE IMAGE</p></div></div>'
  );
}

function bindClassify() {
  const form = document.getElementById("classify-form");
  const file = document.getElementById("post-image");
  file.addEventListener("change", () => {
    const drop = document.getElementById("upload-drop");
    if (file.files[0]) {
      drop.classList.add("has-image");
      drop.querySelector("strong").textContent = "Image staged for analysis.";
      drop.querySelector("small").textContent = file.files[0].name;
    } else {
      drop.classList.remove("has-image");
      drop.querySelector("strong").textContent =
        "Drop an image into the record.";
      drop.querySelector("small").textContent =
        "JPEG / PNG / WEBP · 10 MB MAXIMUM";
    }
  });
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const selected = file.files[0];
    if (selected && selected.size > 10 * 1024 * 1024) {
      notify("Image exceeds the 10 MB limit.");
      return;
    }
    let image = null;
    if (selected) {
      image = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result).split(",")[1]);
        reader.onerror = reject;
        reader.readAsDataURL(selected);
      });
    }
    const button = form.querySelector("button[type=submit]");
    button.disabled = true;
    button.textContent = "READING THE EVIDENCE…";
    try {
      state.classify = await post("/api/classify", {
        text: document.getElementById("post-text").value,
        image,
        filename: selected ? selected.name : null,
        event: document.getElementById("context-event").value,
        time: document.getElementById("context-time").value,
        presentation: presentation.checked,
      });
      state.explanation = null;
      document.getElementById("classify-output").innerHTML = classifyOutput(
        state.classify,
      );
      bindExplanation();
      document
        .getElementById("classify-output")
        .scrollIntoView({ behavior: "smooth" });
    } catch (error) {
      notify(error.message);
    } finally {
      button.disabled = false;
      button.innerHTML = "ANALYZE POST &nbsp; ↗";
    }
  });
  if (state.classify) {
    document.getElementById("classify-output").innerHTML = classifyOutput(
      state.classify,
    );
    bindExplanation();
  }
}

function bindExplanation() {
  const button = document.getElementById("explain-button");
  if (!button) return;
  button.addEventListener("click", async () => {
    button.disabled = true;
    button.textContent = "TRACING MODEL EVIDENCE…";
    try {
      state.explanation = await post("/api/explain", {
        token: state.classify.token,
        head: document.getElementById("explain-head").value,
        presentation: presentation.checked,
      });
      document.getElementById("explain-output").innerHTML = explanationOutput(
        state.explanation,
      );
    } catch (error) {
      notify(error.message);
    } finally {
      button.disabled = false;
      button.textContent = "REVEAL EVIDENCE →";
    }
  });
}

function replayHtml(data) {
  if (!data.available)
    return (
      '<div class="page"><div class="page-inner">' +
      pageIndex("02", "EVENT REPLAY") +
      notice(
        "ARTIFACT STATUS",
        "Event replay unavailable",
        data.message,
        data.command,
      ) +
      "</div></div>"
    );
  const maximum = Math.max(1, ...data.timeline.map((w) => w.count));
  const bars = data.timeline
    .map(
      (w, i) =>
        '<button class="' +
        (i === data.window ? "active" : w.recycled ? "recycled" : "") +
        '" data-window="' +
        i +
        '" title="' +
        esc(w.date) +
        " / " +
        fmt(w.count) +
        ' posts" style="height:' +
        Math.max(3, Math.round((w.count / maximum) * 100)) +
        '%" aria-label="Window ' +
        (i + 1) +
        '"></button>',
    )
    .join("");
  const feed = data.feed
    .map(
      (card, i) =>
        '<article class="feed-card ' +
        (card.recycled ? "recycled" : "") +
        '" style="animation-delay:' +
        i * 35 +
        'ms"><div class="photo">' +
        (card.image
          ? '<img src="' + card.image + '" alt="Post image">'
          : emptyImage()) +
        '</div><div class="copy"><div class="meta">' +
        esc(card.time) +
        " UTC / OBSERVED POST</div><p>" +
        esc(card.text) +
        '</p><div class="badges">' +
        pill(card.category, "muted") +
        (card.severity
          ? pill(
              card.severity,
              card.severity === "Severe damage" ? "orange" : "muted",
            )
          : "") +
        pill(card.status, card.recycled ? "red" : "dark") +
        "</div>" +
        (card.recycled
          ? '<div class="origin">REUSED IMAGE / EARLIEST INDEXED IN ' +
            esc(card.origin_event || "UNKNOWN EVENT") +
            " · " +
            esc(card.origin_date || "DATE UNKNOWN") +
            "</div>"
          : "") +
        "</div>" +
        (card.recycled
          ? '<div class="origin-image">' +
            (card.origin_image
              ? '<img src="' +
                card.origin_image +
                '" alt="Earlier indexed image">'
              : emptyImage()) +
            "</div>"
          : "") +
        "</article>",
    )
    .join("");
  return (
    '<div class="page"><div class="replay-stage"><div class="stage-inner">' +
    pageIndex("02", "EVENT REPLAY") +
    '<div class="replay-title-row"><h1 class="replay-title">' +
    esc(data.event.split(" ")[0]) +
    "<span>.</span><br>" +
    esc(data.event.split(" ").slice(1).join(" ")) +
    '</h1><div class="replay-subtitle">OBSERVE HOW A CRISIS UNFOLDS. FOLLOW THE SIGNAL; SEPARATE NEW EVIDENCE FROM RECYCLED IMAGERY.</div></div><div class="replay-overline"><span>EVENT CHRONOLOGY / ' +
    esc(data.model) +
    "</span><span>" +
    esc(data.window_count) +
    ' WINDOWS <span class="slash">/</span> ' +
    (data.evidence_ready ? "EVIDENCE READY" : "BASELINE PREVIEW") +
    '</span></div><div class="stage-controls"><div><label class="field-label" for="replay-event">SELECT EVENT</label><select id="replay-event">' +
    (state.bootstrap.events || [])
      .map(
        (e) =>
          '<option value="' +
          esc(e.id) +
          '" ' +
          (e.id === state.event ? "selected" : "") +
          ">" +
          esc(e.name) +
          "</option>",
      )
      .join("") +
    '</select></div><div><label class="field-label" for="replay-window">WINDOW ' +
    String(data.window + 1).padStart(2, "0") +
    " / " +
    String(data.window_count).padStart(2, "0") +
    '</label><input id="replay-window" type="range" min="0" max="' +
    (data.window_count - 1) +
    '" value="' +
    data.window +
    '"></div><div><span class="field-label">CURRENT WINDOW</span><div class="micro" style="margin-top:13px">' +
    esc(data.start) +
    '</div></div></div><div class="transport"><button class="button orange small" id="play-button">' +
    (state.playing ? "Ⅱ PAUSE" : "▶ PLAY") +
    '</button><div class="speed"><button data-speed="1" class="' +
    (state.speed === 1 ? "active" : "") +
    '">1×</button><button data-speed="2" class="' +
    (state.speed === 2 ? "active" : "") +
    '">2×</button><button data-speed="3" class="' +
    (state.speed === 3 ? "active" : "") +
    '">3×</button></div><span class="micro" style="color:#b7d4ff">' +
    esc(data.start) +
    " → " +
    esc(data.end) +
    '</span></div><div class="stage-timeline">' +
    bars +
    '</div><div class="stage-dates"><span>' +
    esc(data.timeline[0].date) +
    "</span><span>" +
    esc(data.timeline[data.timeline.length - 1].date) +
    '</span></div><p class="stage-status">' +
    (data.evidence_ready
      ? "These event windows have matching evidence packets."
      : "CLIP zero-shot baseline preview. Provenance decisions are real; these windows are not final classifier or sitrep evidence.") +
    '</p></div></div><div class="page-inner"><div class="replay-metrics"><div class="replay-metric"><small>POSTS OBSERVED</small><strong>' +
    fmt(data.counts.posts) +
    '</strong></div><div class="replay-metric"><small>INFORMATIVE</small><strong>' +
    fmt(data.counts.informative) +
    '</strong></div><div class="replay-metric"><small>RECYCLED IMAGES</small><strong>' +
    fmt(data.counts.recycled) +
    '</strong></div><div class="replay-metric"><small>AWAITING REVIEW</small><strong>' +
    fmt(data.counts.review) +
    "</strong></div></div>" +
    title(
      "The evidence stream",
      fmt(data.feed_count) + " POSTS IN THIS WINDOW",
    ) +
    '<div class="feed-toolbar"><div class="micro micro-mute">SHOWING ' +
    Math.min(8, data.feed_count) +
    " RECORDS / WINDOW " +
    (data.window + 1) +
    '</div><div class="segmented"><button data-lens="all" class="' +
    (state.lens === "all" ? "active" : "") +
    '">ALL POSTS</button><button data-lens="flagged" class="' +
    (state.lens === "flagged" ? "active" : "") +
    '">FLAGGED IMAGERY</button><button data-lens="severe" class="' +
    (state.lens === "severe" ? "active" : "") +
    '">SEVERE DAMAGE</button></div></div><div class="feed-list">' +
    (feed ||
      notice(
        "NO RECORDS",
        "No posts in this lens",
        "Move to another window or choose another feed lens.",
        "",
      )) +
    "</div></div></div>"
  );
}

async function renderReplay() {
  const version = ++requestVersion;
  const y = window.scrollY;
  loading("Reconstructing the event");
  try {
    const data = await api(
      "/api/replay?" +
        params({ event: state.event, window: state.window, lens: state.lens }),
    );
    if (version !== requestVersion) return;
    view.innerHTML = replayHtml(data);
    if (y > 200) window.scrollTo(0, y);
    document.getElementById("replay-event")?.addEventListener("change", (e) => {
      state.event = e.target.value;
      state.window = 0;
      renderReplay();
    });
    document
      .getElementById("replay-window")
      ?.addEventListener("change", (e) => {
        state.window = Number(e.target.value);
        renderReplay();
      });
    document.querySelectorAll("[data-window]").forEach((b) =>
      b.addEventListener("click", () => {
        state.window = Number(b.dataset.window);
        renderReplay();
      }),
    );
    document.querySelectorAll("[data-lens]").forEach((b) =>
      b.addEventListener("click", () => {
        state.lens = b.dataset.lens;
        renderReplay();
      }),
    );
    document.querySelectorAll("[data-speed]").forEach((b) =>
      b.addEventListener("click", () => {
        state.speed = Number(b.dataset.speed);
        startPlayback();
        renderReplay();
      }),
    );
    document.getElementById("play-button")?.addEventListener("click", () => {
      state.playing = !state.playing;
      startPlayback();
      document.getElementById("play-button").textContent = state.playing
        ? "Ⅱ PAUSE"
        : "▶ PLAY";
    });
  } catch (error) {
    view.innerHTML = notice(
      "REPLAY ERROR",
      "Could not open this event",
      error.message,
      "",
    );
  }
}
function startPlayback() {
  clearInterval(state.timer);
  if (state.playing)
    state.timer = setInterval(() => {
      const chosen = state.bootstrap.events.find((e) => e.id === state.event);
      state.window = (state.window + 1) % (chosen?.windows || 1);
      renderReplay();
    }, 1100 / state.speed);
}

function sitrepHtml(data) {
  const events = state.bootstrap.events || [];
  const selector =
    '<div class="field sitrep-event-field"><label for="sitrep-event">EVENT ARCHIVE</label><select id="sitrep-event">' +
    events
      .map(
        (e) =>
          '<option value="' +
          esc(e.id) +
          '" ' +
          (e.id === state.event ? "selected" : "") +
          ">" +
          esc(e.name) +
          "</option>",
      )
      .join("") +
    "</select></div>";
  const accepted = data.available ? data.sentences.filter((s) => s.accepted) : [];
  const rejected = data.available ? data.sentences.filter((s) => !s.accepted) : [];
  const originalCount = data.available ? data.sentences.filter((s) => Number(s.round) === 0).length : 0;
  const failedRewrites = rejected.filter((s) => Number(s.round) > 0).length;
  const verdict = data.available
    ? '<aside class="verifier-panel"><div class="verifier-top">AUTOMATED VERIFIER <span>NOT HUMAN RATED</span></div><div class="verifier-number"><strong>' +
      pct(data.rates.final) +
      '</strong><span>FINAL ACCEPTANCE<br>OF ORIGINAL CLAIMS</span></div><div class="verifier-ledger"><div><b>' +
      originalCount +
      '</b><small>ORIGINAL CLAIMS</small></div><div><b>' +
      accepted.length +
      '</b><small>CLAIMS CLEARED</small></div><div><b>' +
      failedRewrites +
      '</b><small>FAILED REWRITES</small></div></div><button type="button" id="jump-rejected" class="verifier-jump">INSPECT REJECTED CLAIMS <span>↓</span></button></aside>'
    : "";
  const system = (data.system_lines || [])
    .map(
      (line, i) =>
        '<div class="system-line"><b>' +
        String(i + 1).padStart(2, "0") +
        "</b><div>" +
        esc(line) +
        "</div></div>",
    )
    .join("");
  let body;
  if (!data.available) {
    body =
      '<div class="document-empty"><span class="micro micro-orange">ARTIFACT / PENDING</span><strong>No verified claim<br>leaves this desk.</strong><p>The event aggregate exists, but the matching evidence packets and verifier output do not. The lines below are computed from the baseline window; they are not generated claims.</p><code>' +
      esc(data.command) +
      "</code></div>" +
      title("Computed observations", "BASELINE / NO GENERATED CLAIMS") +
      (system || "<p>No system observations are available.</p>");
  } else {
    body =
      title("System observations", data.date) +
      system +
      title(
        "Claims that cleared the gates",
        "OPEN A CITATION TO INSPECT THE POST",
      ) +
      (accepted.length
        ? accepted.map((s) => claimHtml(s, data.citations)).join("")
        : notice(
            "VERIFICATION",
            "No claim cleared verification",
            "System observations remain available; no unsupported sentence is promoted.",
            "",
          )) +
      '<div id="rejected-claims">' +
      title("Stopped by verification", "REJECTED CLAIMS REMAIN VISIBLE") +
      (rejected.length
        ? rejected.map((s) => claimHtml(s, data.citations)).join("")
        : "<p>No rejected sentences in this window.</p>") +
      "</div>";
  }
  return (
    '<div class="page"><div class="page-inner">' +
    pageIndex("03", "SITUATION REPORT") +
    '<div class="document-page"><div class="document-rail"><span class="micro micro-orange">EVIDENCE<br>WITH RECEIPTS</span><span class="vertical-label">VERIFIED GENERATION / 03</span><span class="micro">SAKSHI<br>RESEARCH</span></div><div class="document-body"><div class="document-header"><span>FIELD DOCUMENT / 03</span><span>' +
    (data.available ? "AUTOMATED EVIDENCE CHECK" : "AWAITING EVIDENCE") +
    '</span></div><div class="document-intro"><div><h1>Claims,<br><em>with receipts.</em></h1><p class="lede">Each sentence needs an evidence trail. Rejected claims stay in view so the verifier can be questioned too.</p></div>' +
    verdict +
    '</div><div class="sitrep-controls">' +
    selector +
    (data.available
      ? '<div class="field sitrep-window-field"><label for="sitrep-window">WINDOW ' +
        (data.window + 1) +
        " / " +
        data.window_count +
        '</label><input id="sitrep-window" type="range" min="0" max="' +
        (data.window_count - 1) +
        '" value="' +
        data.window +
        '"></div></div><div class="sitrep-model micro micro-mute">' +
        esc(data.model_display) +
        '</div><div class="sitrep-note micro micro-mute">' +
        esc(data.verification_note) +
        "</div>"
      : "</div>") +
    body +
    "</div></div></div></div>"
  );
}
function claimHtml(s, citations) {
  return (
    '<article class="claim ' +
    (s.accepted ? "" : "rejected") +
    '"><div class="claim-id">' +
    esc(s.id) +
    "</div><div><p>" +
    esc(s.text) +
    "</p>" +
    (s.accepted
      ? (s.citations || [])
          .map(
            (pid) =>
              '<button class="citation" data-citation="' +
              esc(pid) +
              '">' +
              esc(pid) +
              " ↗</button>",
          )
          .join("")
      : (s.failed_gates || [])
          .map(
            (g) =>
              '<span class="gate">' +
              esc(g.name.toUpperCase()) +
              " GATE / " +
              esc(g.detail) +
              "</span>",
          )
          .join("")) +
    '<div class="citation-host"></div></div></article>'
  );
}
async function renderSitrep() {
  const version = ++requestVersion;
  loading("Opening the evidence sheet");
  try {
    const data = await api(
      "/api/sitrep?" + params({ event: state.event, window: state.sitrepWindow }),
    );
    if (version !== requestVersion) return;
    view.innerHTML = sitrepHtml(data);
    document.getElementById("sitrep-event")?.addEventListener("change", (e) => {
      state.event = e.target.value;
      state.sitrepWindow = e.target.value === "hurricane_maria" ? 2 : 0;
      renderSitrep();
    });
    document
      .getElementById("sitrep-window")
      ?.addEventListener("change", (e) => {
        state.sitrepWindow = Number(e.target.value);
        renderSitrep();
      });
    document.getElementById("jump-rejected")?.addEventListener("click", () =>
      document.getElementById("rejected-claims")?.scrollIntoView({ behavior: "smooth" }),
    );
    document.querySelectorAll("[data-citation]").forEach((b) =>
      b.addEventListener("click", () => {
        const row = data.citations[b.dataset.citation];
        const host = b.closest(".claim").querySelector(".citation-host");
        host.innerHTML = host.innerHTML
          ? ""
          : '<div class="citation-detail">' +
            (row?.image
              ? '<img src="' + row.image + '" alt="Cited post image">'
              : "") +
            "<p>" +
            esc(row?.text || "Cited post unavailable") +
            "</p></div>";
      }),
    );
  } catch (error) {
    view.innerHTML = notice(
      "REPORT ERROR",
      "Could not read this report",
      error.message,
      "",
    );
  }
}

function reviewHtml(data) {
  const cards = data.items
    .map(
      (item, i) =>
        '<article class="review-card"><div class="number">' +
        String(i + 1).padStart(2, "0") +
        "</div><div>" +
        (item.image
          ? '<img src="' + item.image + '" alt="Review post image">'
          : '<div class="photo-missing">NO IMAGE</div>') +
        '</div><div class="review-copy"><small>' +
        esc(item.event) +
        " / " +
        esc(item.kind) +
        "</small><p>" +
        esc(item.text) +
        "</p><small>REASON / " +
        esc(reviewReason(item.reason)) +
        '</small></div><div class="review-actions">' +
        (item.decision
          ? pill(item.decision, "dark")
          : '<button class="button small" data-token="' +
            esc(item.token) +
            '" data-decision="confirm">CONFIRM</button><button class="button ghost small" data-token="' +
            esc(item.token) +
            '" data-decision="dismiss">DISMISS</button>') +
        "</div></article>",
    )
    .join("");
  return (
    '<div class="page"><div class="page-inner">' +
    pageIndex("04", "HUMAN REVIEW") +
    '<div class="review-head"><div class="title"><span class="micro">THE HUMAN CHECKPOINT / 04</span><h1>Keep a human<br>in the loop.</h1><p>When confidence drops or an image origin is uncertain, the decision moves to a person. Every action stays in the local record.</p></div><div class="review-stamp"><span>↗</span></div></div><div class="review-statline"><div><strong>' +
    fmt(data.total) +
    "</strong><span>IN THE LANE</span></div><div><strong>" +
    fmt(data.pending) +
    "</strong><span>AWAITING DECISION</span></div><div><strong>" +
    fmt(data.reviewed) +
    "</strong><span>REVIEWED</span></div></div>" +
    title("The review lane", esc(data.source) + " / LOCAL RECORDS") +
    '<div class="feed-toolbar"><span class="micro micro-mute">SHOWING ' +
    Math.min(12, data.shown) +
    " OF " +
    data.shown +
    '</span><div class="segmented"><button data-review-lens="pending" class="' +
    (state.reviewLens === "pending" ? "active" : "") +
    '">PENDING</button><button data-review-lens="abstained" class="' +
    (state.reviewLens === "abstained" ? "active" : "") +
    '">ABSTAINED</button><button data-review-lens="origin" class="' +
    (state.reviewLens === "origin" ? "active" : "") +
    '">ORIGIN UNCERTAIN</button><button data-review-lens="all" class="' +
    (state.reviewLens === "all" ? "active" : "") +
    '">ALL</button></div></div><div class="review-list">' +
    (cards ||
      notice(
        "QUEUE CLEAR",
        "No records in this lane",
        "Choose another lens or return after new predictions are prepared.",
        "",
      )) +
    "</div>" +
    (data.source.includes("dev")
      ? '<p class="micro micro-mute" style="margin-top:25px">ABSTENTION SOURCE: SMALL TRAINED-MODEL DEVELOPMENT SAMPLE. FULL EVENT PREDICTIONS ARE PENDING.</p>'
      : "") +
    "</div></div>"
  );
}
async function renderReview() {
  const version = ++requestVersion;
  loading("Opening the review lane");
  try {
    const data = await api("/api/review?" + params({ lens: state.reviewLens }));
    if (version !== requestVersion) return;
    view.innerHTML = reviewHtml(data);
    document.querySelectorAll("[data-review-lens]").forEach((b) =>
      b.addEventListener("click", () => {
        state.reviewLens = b.dataset.reviewLens;
        renderReview();
      }),
    );
    document.querySelectorAll("[data-decision]").forEach((b) =>
      b.addEventListener("click", async () => {
        b.disabled = true;
        try {
          await post("/api/review", {
            token: b.dataset.token,
            decision: b.dataset.decision,
          });
          notify("Review decision saved to the local record.");
          renderReview();
        } catch (error) {
          notify(error.message);
          b.disabled = false;
        }
      }),
    );
  } catch (error) {
    view.innerHTML = notice(
      "REVIEW ERROR",
      "Could not open the lane",
      error.message,
      "",
    );
  }
}

function markdown(text) {
  const lines = String(text || "").split("\n");
  let html = "",
    table = [],
    list = [];
  function flush() {
    if (table.length) {
      const rows = table.filter(
        (row) =>
          !row
            .replace(/^\||\|$/g, "")
            .split("|")
            .every((cell) => /^:?-{3,}:?$/.test(cell.trim())),
      );
      html +=
        "<table><tbody>" +
        rows
          .map(
            (line, i) =>
              "<tr>" +
              line
                .replace(/^\||\|$/g, "")
                .split("|")
                .map(
                  (cell) =>
                    "<" +
                    (i === 0 ? "th" : "td") +
                    ">" +
                    esc(cell.trim()) +
                    "</" +
                    (i === 0 ? "th" : "td") +
                    ">",
                )
                .join("") +
              "</tr>",
          )
          .join("") +
        "</tbody></table>";
      table = [];
    }
    if (list.length) {
      html +=
        "<ul>" +
        list.map((item) => "<li>" + esc(item) + "</li>").join("") +
        "</ul>";
      list = [];
    }
  }
  for (const line of lines) {
    if (line.trim().startsWith("|")) {
      table.push(line);
      continue;
    }
    if (/^\s*[-*] /.test(line)) {
      list.push(line.replace(/^\s*[-*] /, ""));
      continue;
    }
    flush();
    if (/^###? /.test(line)) {
      const n = line.startsWith("### ") ? 3 : 2;
      html += "<h" + n + ">" + esc(line.slice(n + 1)) + "</h" + n + ">";
    } else if (line.startsWith("# ")) {
      html += "<h1>" + esc(line.slice(2)) + "</h1>";
    } else if (line.startsWith("```")) {
      html += "<hr>";
    } else if (line.trim()) {
      html += "<p>" + esc(line) + "</p>";
    }
  }
  flush();
  return html;
}
function resultChartHtml(metrics) {
  if (!Array.isArray(metrics) || metrics.length !== 4)
    return '<div class="diagram"><span>05</span></div>';
  return (
    '<div class="diagram metric-diagram"><div class="metric-chart-heading"><span>OFFICIAL TEST / THREE SEEDS</span><strong>MEAN MACRO-F1</strong></div><div class="metric-chart-rows">' +
    metrics
      .map((metric, i) => {
        const value = Number(metric.value);
        const width = Math.max(0, Math.min(100, value * 100));
        return (
          '<div class="metric-chart-row' +
          (value < 0.5 ? " weak" : "") +
          '"><div class="metric-row-label"><span>0' +
          (i + 1) +
          " / " +
          esc(metric.label) +
          "</span><b>" +
          esc(value.toFixed(3)) +
          '</b></div><div class="metric-track"><i style="width:' +
          width.toFixed(1) +
          '%"></i></div></div>'
        );
      })
      .join("") +
    '</div><div class="metric-chart-foot"><span>0.0</span><span>FROZEN RESULT · WEAK SEVERITY HEAD SHOWN</span><span>1.0</span></div></div>'
  );
}

function resultsHtml(data) {
  const list = data.reports
    .map(
      (report) =>
        '<button data-report="' +
        esc(report.id) +
        '" class="' +
        (report.id === data.selected ? "active" : "") +
        '"><small>' +
        esc(report.kind) +
        "</small>" +
        esc(report.title) +
        "</button>",
    )
    .join("");
  return (
    '<div class="page"><div class="page-inner">' +
    pageIndex("05", "RESEARCH RESULTS") +
    '<div class="results-head"><div class="copy"><span class="micro" style="color:var(--mint)">READ THE EVIDENCE / 05</span><h1>The record,<br>in view.</h1><p>Methods, findings, and limits. Development checks remain distinct from final research results.</p></div>' +
    resultChartHtml(state.bootstrap.headline_metrics) +
    '</div><div class="ribbon"><div class="ribbon-cell"><small>REPORTS AVAILABLE</small><strong>' +
    fmt(data.reports.length) +
    '</strong></div><div class="ribbon-cell"><small>PROVENANCE EVENTS</small><strong>' +
    fmt(state.bootstrap.provenance_events) +
    '</strong></div><div class="ribbon-cell"><small>AGGREGATE EVENTS</small><strong>' +
    fmt(state.bootstrap.events.length) +
    '</strong></div><div class="ribbon-cell"><small>VERIFIED SITREPS</small><strong>' +
    fmt(state.bootstrap.sitrep_events.length) +
    "</strong></div></div>" +
    title("Research record", "READ ONLY / PRECOMPUTED ARTIFACTS") +
    '<div class="report-layout"><aside class="report-list">' +
    list +
    '</aside><article class="report-paper">' +
    (data.content
      ? markdown(data.content)
      : notice(
          "ARTIFACT STATUS",
          "No result report yet",
          "This view reads committed reports and does not run experiments.",
          "sakshi-model evaluate --run <frozen-run>",
        )) +
    "</article></div>" +
    (data.figures?.length
      ? '<div class="figure-gallery">' +
        data.figures
          .map(
            (figure) =>
              '<div class="figure-card"><small>' +
              esc(figure.label) +
              '</small><img src="' +
              figure.src +
              '" alt="' +
              esc(figure.alt) +
              '"><p class="micro micro-mute">SOURCE: FROZEN REPORT ARTIFACT</p></div>',
          )
          .join("") +
        "</div>"
      : "") +
    "</div></div>"
  );
}
async function renderResults() {
  const version = ++requestVersion;
  loading("Opening the research record");
  try {
    const data = await api(
      "/api/results?" + params({ report: state.report || "" }),
    );
    if (version !== requestVersion) return;
    view.innerHTML = resultsHtml(data);
    document.querySelectorAll("[data-report]").forEach((b) =>
      b.addEventListener("click", () => {
        state.report = b.dataset.report;
        renderResults();
      }),
    );
  } catch (error) {
    view.innerHTML = notice(
      "RESULT ERROR",
      "Could not read the record",
      error.message,
      "",
    );
  }
}

function render() {
  requestVersion++;
  if (state.page === "briefing") renderBriefing();
  else if (state.page === "classify") {
    view.innerHTML = classifyShell();
    bindClassify();
    renderFeaturedCase();
  } else if (state.page === "replay") renderReplay();
  else if (state.page === "sitrep") renderSitrep();
  else if (state.page === "review") renderReview();
  else renderResults();
}
document
  .querySelectorAll(".nav-item")
  .forEach((button) =>
    button.addEventListener("click", () => changePage(button.dataset.page)),
  );
document.getElementById('briefing-link').addEventListener('click', () => changePage('briefing'));
document.addEventListener('keydown', (event) => {
  if (state.page !== 'briefing' || event.altKey || event.ctrlKey || event.metaKey ||
      ['INPUT','TEXTAREA','SELECT'].includes(document.activeElement?.tagName)) return;
  if (event.key === 'ArrowRight' || event.key === 'PageDown') {
    event.preventDefault();
    goBriefingStep(state.briefingStep + 1);
  } else if (event.key === 'ArrowLeft' || event.key === 'PageUp') {
    event.preventDefault();
    goBriefingStep(state.briefingStep - 1);
  } else if (event.key === 'Escape') {
    changePage('classify');
  }
});
presentation.addEventListener("change", () => {
  state.classify = null;
  state.explanation = null;
  state.briefingAnalysis = null;
  state.featuredCase = null;
  state.briefingSitrep = null;
  notify(
    presentation.checked
      ? "Safe view enabled. Re-analyze a post to refresh its images."
      : "Presentation mode off. Review images before sharing.",
  );
  render();
});
window.addEventListener("hashchange", () => changePage(location.hash.slice(1)));
(async () => {
  try {
    loading("Initializing the fieldroom");
    state.bootstrap = await api("/api/bootstrap");
    state.event =
      state.bootstrap.events.find((e) => e.id === "hurricane_maria")?.id ||
      state.bootstrap.events[0]?.id ||
      "hurricane_maria";
    changePage(location.hash.slice(1) || "briefing");
  } catch (error) {
    view.innerHTML = notice(
      "STARTUP ERROR",
      "Fieldroom unavailable",
      error.message,
      "sakshi-app",
    );
  }
})();

/* SAKSHI fieldroom: deliberately small, dependency-free presentation client. */
const state = {
  page: "classify",
  bootstrap: null,
  event: "hurricane_maria",
  window: 0,
  lens: "all",
  reviewLens: "pending",
  report: null,
  playing: false,
  speed: 1,
  timer: null,
  classify: null,
  explanation: null,
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
  if (!["classify", "replay", "sitrep", "review", "results"].includes(page))
    page = "classify";
  state.page = page;
  state.playing = false;
  clearInterval(state.timer);
  document
    .querySelectorAll(".nav-item")
    .forEach((b) => b.classList.toggle("active", b.dataset.page === page));
  history.replaceState(null, "", "#" + page);
  window.scrollTo(0, 0);
  render();
}

function radarArt() {
  return (
    '<svg viewBox="0 0 600 510" fill="none" preserveAspectRatio="xMidYMid slice" aria-hidden="true">' +
    '<path d="M-20 427C69 410 118 363 168 332S301 308 331 237 379 136 480 130 550 86 620 62" stroke="#A4CCB6" stroke-width="2" opacity=".75"/>' +
    '<path d="M-30 455C71 421 103 384 179 379S291 347 340 278 410 239 504 189 575 144 630 140" stroke="#EF613D" stroke-width="3"/>' +
    '<path d="M-20 379C53 362 127 300 190 296S305 287 341 199 391 104 493 100 551 54 620 26" stroke="#E9E8DF" opacity=".43"/>' +
    '<circle cx="340" cy="278" r="8" fill="#EF613D"/><circle cx="340" cy="278" r="23" stroke="#EF613D" opacity=".6"/>' +
    '<circle cx="504" cy="189" r="5" fill="#E9E8DF"/><circle cx="179" cy="379" r="5" fill="#E9E8DF"/>' +
    '<path d="M40 48h115M40 60h72M414 449h147M458 461h103" stroke="#B7D5BB" opacity=".7"/>' +
    '<text x="41" y="85" fill="#B7D5BB" font-size="10" letter-spacing="4">SIGNAL DETECTED</text><text x="420" y="435" fill="#B7D5BB" font-size="10" letter-spacing="4">MODEL / HUMAN</text></svg>' +
    '<span class="visual-cross"></span><span class="visual-caption">FIG. 01 / TRACE THE EVIDENCE</span>'
  );
}

function classifyShell() {
  return (
    '<div class="page"><div class="page-inner">' +
    pageIndex("01", "INVESTIGATE") +
    '<section class="hero"><div class="hero-copy"><div class="hero-kicker">SITUATIONAL AWARENESS / LIVE ANALYSIS</div><div><h1>See the signal.<br><em>Trace</em> the source.</h1><p>One post, four decisions, and a visible path back to the evidence. Read the model; question the image.</p></div><div class="hero-foot"><span class="pulse"></span> CPU INFERENCE <span class="slash">/</span> LOCAL ONLY <span class="slash">/</span> HUMAN REVIEW</div></div><div class="hero-visual">' +
    radarArt() +
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
        ? "ABSTAINED · " + (result.abstain_reason || "REVIEW")
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
              '<span class="word-chip" style="background:rgba(239,97,61,' +
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
    '">3×</button></div><span class="micro" style="color:#b6c6bb">' +
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
    '<div class="field" style="max-width:340px;margin:30px 0"><label for="sitrep-event">EVENT ARCHIVE</label><select id="sitrep-event">' +
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
    const accepted = data.sentences.filter((s) => s.accepted),
      rejected = data.sentences.filter((s) => !s.accepted);
    body =
      '<div class="ribbon" style="border:1px solid var(--ink);margin:40px 0"><div class="ribbon-cell"><small>ACCEPTED CLAIMS</small><strong>' +
      accepted.length +
      '</strong></div><div class="ribbon-cell"><small>REJECTED CLAIMS</small><strong>' +
      rejected.length +
      '</strong></div><div class="ribbon-cell"><small>INITIAL VERIFICATION</small><strong>' +
      pct(data.rates.initial) +
      '</strong></div><div class="ribbon-cell"><small>AFTER REWRITE</small><strong>' +
      pct(data.rates.final) +
      "</strong></div></div>" +
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
      title("Stopped by verification", "REJECTED CLAIMS REMAIN VISIBLE") +
      (rejected.length
        ? rejected.map((s) => claimHtml(s, data.citations)).join("")
        : "<p>No rejected sentences in this window.</p>");
  }
  return (
    '<div class="page"><div class="page-inner">' +
    pageIndex("03", "SITUATION REPORT") +
    '<div class="document-page"><div class="document-rail"><span class="micro micro-orange">EVIDENCE<br>WITH RECEIPTS</span><span class="vertical-label">VERIFIED GENERATION / 03</span><span class="micro">SAKSHI<br>RESEARCH</span></div><div class="document-body"><div class="document-header"><span>FIELD DOCUMENT / 03</span><span>' +
    (data.available ? "CITED & VERIFIED" : "AWAITING EVIDENCE") +
    '</span></div><h1>Claims,<br><em>with receipts.</em></h1><p class="lede">Each sentence needs an evidence trail. Rejected claims stay in view so the verifier can be questioned too.</p>' +
    selector +
    (data.available
      ? '<div class="field" style="max-width:340px"><label for="sitrep-window">WINDOW ' +
        (data.window + 1) +
        " / " +
        data.window_count +
        '</label><input id="sitrep-window" type="range" min="0" max="' +
        (data.window_count - 1) +
        '" value="' +
        data.window +
        '"></div><div class="micro micro-mute" style="margin-top:16px">' +
        esc(data.models.classifier) +
        " / " +
        esc(data.models.llm) +
        " / " +
        esc(data.models.nli) +
        "</div>"
      : "") +
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
      "/api/sitrep?" + params({ event: state.event, window: state.window }),
    );
    if (version !== requestVersion) return;
    view.innerHTML = sitrepHtml(data);
    document.getElementById("sitrep-event")?.addEventListener("change", (e) => {
      state.event = e.target.value;
      state.window = 0;
      renderSitrep();
    });
    document
      .getElementById("sitrep-window")
      ?.addEventListener("change", (e) => {
        state.window = Number(e.target.value);
        renderSitrep();
      });
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
        esc(item.reason) +
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
    '<div class="results-head"><div class="copy"><span class="micro" style="color:var(--mint)">READ THE EVIDENCE / 05</span><h1>The record,<br>in view.</h1><p>Methods, findings, and limits. Development checks remain distinct from final research results.</p></div><div class="diagram"><span>05</span></div></div><div class="ribbon"><div class="ribbon-cell"><small>REPORTS AVAILABLE</small><strong>' +
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
  if (state.page === "classify") {
    view.innerHTML = classifyShell();
    bindClassify();
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
presentation.addEventListener("change", () => {
  state.classify = null;
  state.explanation = null;
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
    changePage(location.hash.slice(1) || "classify");
  } catch (error) {
    view.innerHTML = notice(
      "STARTUP ERROR",
      "Fieldroom unavailable",
      error.message,
      "sakshi-app",
    );
  }
})();

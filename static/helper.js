/* ddd Mission Control — a floating helper overlaid on any page.
   Self-contained: injects its own launcher + panel, talks to /api/analyst,
   renders the same 'guide' answers (markdown + action chips) as /chat.
   Themed off ink-skin.css variables, so it follows light/dark automatically. */
(function () {
  if (window.__dddHelper) return;
  window.__dddHelper = true;

  var esc = function (s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  };
  var api = async function (p, o) {
    var r = await fetch(p, o);
    var ct = r.headers.get("content-type") || "";
    var b = ct.includes("json") ? await r.json() : await r.text();
    if (!r.ok) throw new Error((b && b.detail) || r.status);
    return b;
  };
  function mdInline(t) {
    var h = esc(t);
    h = h.replace(/\[([^\]]+)\]\(ddd:(gene|disease|pathway|target|mol)\)/g,
      '<a class="ddh-ent" data-kind="$2" onclick="window.dddEntity&&window.dddEntity(\'$2\', this.textContent)">$1</a>');
    h = h.replace(/`([^`]+)`/g, "<code>$1</code>");
    h = h.replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
    h = h.replace(/\[([^\]]+)\]\((https?:[^)]+|\/[^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
    return h;
  }
  function mdToHtml(md) {
    var L = String(md || "").split("\n"), out = [];
    for (var i = 0; i < L.length; i++) {
      var line = L[i];
      if (line.indexOf("```") === 0) {
        var buf = []; i++;
        while (i < L.length && L[i].indexOf("```") !== 0) buf.push(L[i++]);
        out.push("<pre>" + esc(buf.join("\n")) + "</pre>"); continue;
      }
      var hh = line.match(/^(#{1,4})\s+(.*)$/);
      if (hh) { var n = Math.min(hh[1].length, 3); out.push("<h" + n + ">" + mdInline(hh[2]) + "</h" + n + ">"); continue; }
      if (/^\s*\|.*\|\s*$/.test(line)) {
        var rows = [];
        while (i < L.length && /^\s*\|.*\|\s*$/.test(L[i])) rows.push(L[i++]); i--;
        var cells = function (r) { return r.trim().replace(/^\||\|$/g, "").split("|").map(function (c) { return c.trim(); }); };
        var body = rows.filter(function (r) { return !/^\s*\|[\s:|-]+\|\s*$/.test(r); });
        out.push("<table>" + body.map(function (r, k) {
          return "<tr>" + cells(r).map(function (c) { return k === 0 ? "<th>" + mdInline(c) + "</th>" : "<td>" + mdInline(c) + "</td>"; }).join("") + "</tr>";
        }).join("") + "</table>"); continue;
      }
      if (/^\s*[-*]\s+/.test(line)) {
        var items = [];
        while (i < L.length && /^\s*[-*]\s+/.test(L[i])) { items.push("<li>" + mdInline(L[i].replace(/^\s*[-*]\s+/, "")) + "</li>"); i++; } i--;
        out.push("<ul>" + items.join("") + "</ul>"); continue;
      }
      if (/^\s*\d+\.\s+/.test(line)) {
        var oi = [];
        while (i < L.length && /^\s*\d+\.\s+/.test(L[i])) { oi.push("<li>" + mdInline(L[i].replace(/^\s*\d+\.\s+/, "")) + "</li>"); i++; } i--;
        out.push("<ol>" + oi.join("") + "</ol>"); continue;
      }
      if (/^---+\s*$/.test(line)) { out.push("<hr>"); continue; }
      if (line.trim()) out.push("<p>" + mdInline(line) + "</p>");
    }
    return out.join("\n");
  }

  var CSS = '' +
'.ddh-fab{position:fixed;right:20px;bottom:20px;z-index:99998;display:flex;align-items:center;gap:8px;' +
  'font:600 13px/1 Inter,system-ui,sans-serif;padding:11px 15px;border-radius:22px;cursor:pointer;' +
  'background:var(--paper,#f6f6f4);color:var(--ink,#262626);border:1px solid var(--line,#e6e6e3);' +
  'box-shadow:0 6px 22px rgba(0,0,0,.14);transition:transform .12s ease,box-shadow .12s ease}' +
'.ddh-fab:hover{transform:translateY(-1px);box-shadow:0 9px 26px rgba(0,0,0,.18)}' +
'.ddh-fab .ic{color:var(--accent,#b07a1e);font-size:15px}' +
'.ddh-panel{position:fixed;right:20px;bottom:76px;z-index:99999;width:390px;max-width:calc(100vw - 32px);' +
  'height:68vh;max-height:620px;display:none;flex-direction:column;overflow:hidden;' +
  'background:var(--paper,#f6f6f4);color:var(--ink,#262626);border:1px solid var(--line,#e6e6e3);' +
  'border-radius:12px;box-shadow:0 18px 50px rgba(0,0,0,.26)}' +
'.ddh-panel.on{display:flex}' +
'.ddh-h{display:flex;align-items:center;gap:8px;padding:12px 14px;border-bottom:1px solid var(--line,#e6e6e3)}' +
'.ddh-h b{font:700 13.5px Inter,sans-serif}.ddh-h .sub{font:400 11.5px Inter,sans-serif;color:var(--muted,#6e6e6e)}' +
'.ddh-h .sp{flex:1}.ddh-x{border:0;background:none;color:var(--muted,#6e6e6e);font-size:19px;cursor:pointer;line-height:1;padding:2px 4px}' +
'.ddh-x:hover{color:var(--ink,#262626)}' +
'.ddh-dot{width:7px;height:7px;border-radius:50%;background:var(--accent,#b07a1e)}' +
'.ddh-scroll{flex:1;overflow:auto;padding:14px}' +
'.ddh-turn{margin-bottom:13px;font:400 13px/1.55 Inter,sans-serif}' +
'.ddh-turn.user{text-align:right}.ddh-turn.user .b{display:inline-block;background:var(--sel-bg,#f7e9cc);color:var(--sel-ink,#7a520a);' +
  'border:1px solid var(--sel-line,#e2c58c);border-radius:11px;padding:7px 11px;max-width:85%;text-align:left}' +
'.ddh-turn.bot .b{background:transparent}' +
'.ddh-turn.bot .b h1,.ddh-turn.bot .b h2,.ddh-turn.bot .b h3{font:700 13.5px Inter,sans-serif;margin:3px 0 7px}' +
'.ddh-turn.bot .b p{margin:0 0 8px}.ddh-turn.bot .b ul,.ddh-turn.bot .b ol{margin:0 0 8px;padding-left:18px}' +
'.ddh-turn.bot .b code{font:500 12px JetBrains Mono,monospace;background:var(--sel-bg,#f3efe6);padding:1px 4px;border-radius:3px}' +
'.ddh-turn.bot .b pre{background:#0000000a;padding:9px;border-radius:6px;overflow:auto;font:500 11.5px JetBrains Mono,monospace;margin:0 0 8px}' +
'.ddh-turn.bot .b table{border-collapse:collapse;width:100%;margin:0 0 8px;font-size:12px}' +
'.ddh-turn.bot .b th,.ddh-turn.bot .b td{border-bottom:1px solid var(--line,#e6e6e3);padding:5px 7px;text-align:left;vertical-align:top}' +
'.ddh-turn.bot .b th{color:var(--muted,#6e6e6e);font-weight:600}' +
'.ddh-turn.bot .b a{color:var(--accent,#b07a1e)}' +
'.ddh-ent{color:var(--accent,#b07a1e);cursor:pointer;border-bottom:1px dotted var(--accent,#b07a1e);white-space:nowrap}' +
'.ddh-ent:hover{background:var(--sel-bg,#f7e9cc);border-radius:3px}' +
'.ddh-acts{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}' +
'.ddh-chip{font:500 12px Inter,sans-serif;padding:6px 10px;border-radius:14px;cursor:pointer;' +
  'background:transparent;color:var(--ink,#262626);border:1px solid var(--line,#d9d9d6)}' +
'.ddh-chip:hover{background:var(--sel-bg,#f7e9cc);border-color:var(--sel-line,#e2c58c);color:var(--sel-ink,#7a520a)}' +
'.ddh-plan{margin-top:8px;border-left:2px solid var(--sel-line,#e2c58c);padding-left:10px}' +
'.ddh-plan .s{font-size:12px;color:var(--muted,#6e6e6e);margin:2px 0}' +
'.ddh-run{margin-top:8px;font-size:12px;color:var(--muted,#6e6e6e)}' +
'.ddh-err{color:#b4232a}' +
'.ddh-in{display:flex;gap:7px;padding:11px;border-top:1px solid var(--line,#e6e6e3)}' +
'.ddh-in textarea{flex:1;resize:none;border:1px solid var(--line,#d9d9d6);border-radius:9px;padding:9px 10px;' +
  'font:400 13px Inter,sans-serif;background:var(--card,#fff);color:var(--ink,#262626);max-height:110px}' +
'.ddh-send{border:1px solid var(--accent,#b07a1e);background:transparent;color:var(--accent,#b07a1e);' +
  'border-radius:9px;width:38px;cursor:pointer;font-size:15px}.ddh-send:hover{background:var(--sel-bg,#f7e9cc)}' +
'.ddh-send:disabled{opacity:.5;cursor:default}' +
'.ddh-dots:after{content:"";animation:ddhd 1.2s steps(4,end) infinite}@keyframes ddhd{0%{content:""}25%{content:"."}50%{content:".."}75%{content:"..."}}' +
'@media(max-width:560px){.ddh-panel{right:10px;left:10px;width:auto;bottom:70px;height:72vh}.ddh-fab{right:12px;bottom:12px}}';

  var st = document.createElement("style"); st.textContent = CSS; document.head.appendChild(st);

  var fab = document.createElement("button");
  fab.className = "ddh-fab"; fab.id = "ddhFab";
  fab.innerHTML = '<span class="ic">✦</span> Ask ddd';
  document.body.appendChild(fab);

  var panel = document.createElement("div");
  panel.className = "ddh-panel"; panel.id = "ddhPanel";
  panel.innerHTML =
    '<div class="ddh-h"><span class="ddh-dot"></span><b>Mission control</b>' +
      '<span class="sub">· ask ddd</span><span class="sp"></span>' +
      '<a href="/chat" title="Open full chat" style="color:var(--muted,#6e6e6e);text-decoration:none;font-size:14px;margin-right:4px">⤢</a>' +
      '<button class="ddh-x" id="ddhClose">×</button></div>' +
    '<div class="ddh-scroll" id="ddhScroll"></div>' +
    '<div class="ddh-in"><textarea id="ddhQ" rows="1" placeholder="Ask ddd anything…"></textarea>' +
      '<button class="ddh-send" id="ddhSend">↑</button></div>';
  document.body.appendChild(panel);

  var scroll = panel.querySelector("#ddhScroll");
  var qEl = panel.querySelector("#ddhQ");
  var sendBtn = panel.querySelector("#ddhSend");
  var LAST_CLARIFY = null, busy = false, greeted = false;

  function toBottom() { scroll.scrollTop = scroll.scrollHeight; }
  function addTurn(who, html) {
    var d = document.createElement("div");
    d.className = "ddh-turn " + who;
    d.innerHTML = who === "user" ? '<div class="b">' + html + "</div>" : '<div class="b">' + html + "</div>";
    scroll.appendChild(d); toBottom(); return d;
  }
  function actsHtml(a) {
    return (a && a.length) ? '<div class="ddh-acts">' + a.map(function (x) {
      return '<button class="ddh-chip" data-ask="' + esc(x.ask) + '">' + esc(x.label) + "</button>";
    }).join("") + "</div>" : "";
  }
  function planHtml(p) {
    return (p && p.length) ? '<div class="ddh-plan">' + p.map(function (s, i) {
      return '<div class="s">' + (i + 1) + ". " + esc(s) + "</div>";
    }).join("") + "</div>" : "";
  }
  function bindChips(el) {
    el.querySelectorAll(".ddh-chip").forEach(function (b) {
      b.addEventListener("click", function () { send(b.getAttribute("data-ask")); });
    });
  }

  function render(bot, r, q) {
    var el = bot.querySelector(".b");
    if (r.kind === "guide") { el.innerHTML = mdToHtml(r.say) + planHtml(r.plan) + actsHtml(r.actions); bindChips(el); toBottom(); return; }
    if (r.kind === "clarify") {
      LAST_CLARIFY = { pending: r.pending, q: q };
      el.innerHTML = mdInline(r.say) + '<div class="ddh-acts">' + r.options.map(function (o) {
        return '<button class="ddh-chip" data-choice="' + esc(o.value) + '">' + esc(o.label) + "</button>";
      }).join("") + "</div>";
      el.querySelectorAll("[data-choice]").forEach(function (b) {
        b.addEventListener("click", function () { choose(b.getAttribute("data-choice")); });
      });
      toBottom(); return;
    }
    if (r.kind === "answer") { el.innerHTML = mdToHtml(r.say); toBottom(); return; }
    if (r.kind === "ui") {
      var what = r.ui.action === "cell" ? "living cell" : (r.ui.action === "structure" ? "AlphaFold structure" : "graph");
      el.innerHTML = mdInline(r.say) + planHtml(r.plan) +
        '<div class="ddh-run">Open it in the <a href="/lab" target="_blank">Lab</a> to see the ' + what + ".</div>";
      toBottom(); return;
    }
    if (r.kind === "data") {
      el.innerHTML = mdInline(r.say) + planHtml(r.plan) +
        '<div class="ddh-run">Full result in the <a href="/chat" target="_blank">chat</a>.</div>';
      toBottom(); return;
    }
    if (r.kind === "run") {
      var note = r.playbook === "discovery_mission"
        ? '<div class="ddh-run">Agents working in tandem — <a href="/chat" target="_blank">open the full mission view</a>.</div>' : "";
      el.innerHTML = mdInline(r.say) + planHtml(r.plan) + note + '<div class="ddh-run ddh-dots" id="ro">Running</div>';
      watch(r.run_id, el); toBottom(); return;
    }
    el.innerHTML = mdInline(r.say || "Done."); toBottom();
  }

  async function watch(id, el) {
    try {
      var r = await api("/api/company/runs/" + id), out = el.querySelector("#ro");
      if (!out) return;
      if (r.status === "done" && r.report) {
        var txt = await (await fetch("/api/company/reports/" + encodeURIComponent(r.report))).text();
        out.className = "ddh-run"; out.innerHTML = mdToHtml(txt); toBottom();
      } else if (r.status === "error") {
        out.className = "ddh-run ddh-err"; out.textContent = (r.summary && r.summary.error) || "failed";
      } else {
        out.textContent = (r.log ? r.log.trim().split("\n").slice(-1)[0] : "Running");
        setTimeout(function () { watch(id, el); }, 2200);
      }
    } catch (e) { var o = el.querySelector("#ro"); if (o) { o.className = "ddh-run ddh-err"; o.textContent = e.message; } }
  }

  async function post(body) {
    return api("/api/analyst", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  }
  async function send(preset) {
    if (busy) return;
    var q = preset || qEl.value.trim(); if (!q) return;
    qEl.value = ""; qEl.style.height = "auto";
    addTurn("user", esc(q));
    busy = true; sendBtn.disabled = true;
    var bot = addTurn("bot", '<span class="ddh-dots" style="color:var(--muted,#6e6e6e)">Thinking</span>');
    try { var r = await post({ q: q }); render(bot, r, q); }
    catch (e) { bot.querySelector(".b").innerHTML = '<span class="ddh-err">' + esc(e.message) + "</span>"; }
    busy = false; sendBtn.disabled = false; qEl.focus();
  }
  async function choose(choice) {
    var c = LAST_CLARIFY; if (!c) return; LAST_CLARIFY = null;
    addTurn("user", esc(choice));
    var bot = addTurn("bot", '<span class="ddh-dots" style="color:var(--muted,#6e6e6e)">Working</span>');
    try { var r = await post({ q: c.q, choice: choice, pending: c.pending }); render(bot, r, c.q); }
    catch (e) { bot.querySelector(".b").innerHTML = '<span class="ddh-err">' + esc(e.message) + "</span>"; }
  }

  function greet() {
    if (greeted) return; greeted = true;
    var intro = "**Hi — I'm your mission control.** Ask me what's happening, what the world scout found, or what to do next. I can also run anything the bench does.";
    var acts = [
      { label: "What's happening right now?", ask: "What's happening right now?" },
      { label: "What has the world scout found?", ask: "What has the world scout found?" },
      { label: "What should I do next?", ask: "What should I do next?" },
      { label: "What can this app do?", ask: "What can this app do?" }
    ];
    var bot = addTurn("bot", mdToHtml(intro) + actsHtml(acts));
    bindChips(bot.querySelector(".b"));
  }

  function openPanel() {
    panel.classList.add("on"); try { localStorage.setItem("ddh-open", "1"); } catch (e) {}
    greet(); qEl.focus();
  }
  function closePanel() { panel.classList.remove("on"); try { localStorage.setItem("ddh-open", "0"); } catch (e) {} }

  fab.addEventListener("click", function () { panel.classList.contains("on") ? closePanel() : openPanel(); });
  panel.querySelector("#ddhClose").addEventListener("click", closePanel);
  sendBtn.addEventListener("click", function () { send(); });
  qEl.addEventListener("input", function () { qEl.style.height = "auto"; qEl.style.height = Math.min(qEl.scrollHeight, 110) + "px"; });
  qEl.addEventListener("keydown", function (e) { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });

  // open the panel and ask a question from anywhere on the page
  window.dddAsk = function (t) { openPanel(); send(t); };
  // a clicked entity in a report → the right drill-in question
  window.dddEntity = function (kind, name) {
    name = (name || "").trim();
    var map = {
      gene: "Is " + name + " worth pursuing?",
      target: "Is " + name + " worth pursuing?",
      disease: "Targets for " + name,
      pathway: "Literature on " + name,
      mol: "Describe " + name
    };
    window.dddAsk(map[kind] || name);
  };

  try { if (localStorage.getItem("ddh-open") === "1") openPanel(); } catch (e) {}
})();

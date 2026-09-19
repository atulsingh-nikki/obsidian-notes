// Lightweight force-directed connections graph on <canvas> — no external deps.

const urlMap = window.__POST_URLS__ || {};
const graphUrl = window.__GRAPH_URL__ || "assets/graph.json";

const canvas = document.getElementById("graphCanvas");
const wrap = canvas ? canvas.parentElement : null;
const tooltip = document.getElementById("graphTooltip");
const searchEl = document.getElementById("graphSearch");
if (canvas) init();

function accentFor(group) {
  // Deterministic pastel hue from the group string.
  let h = 0;
  for (let i = 0; i < group.length; i++) h = (h * 31 + group.charCodeAt(i)) % 360;
  return `hsl(${h}, 62%, 58%)`;
}

async function init() {
  const ctx = canvas.getContext("2d");
  let nodes = [];
  let links = [];
  try {
    const res = await fetch(graphUrl, { cache: "no-store" });
    const data = await res.json();
    nodes = data.nodes;
    links = data.links;
  } catch (e) {
    wrap.innerHTML = '<p class="bookshelf-status error">Could not load the graph data.</p>';
    return;
  }

  const byId = new Map();
  nodes.forEach((n) => {
    n.x = Math.random() * 800 - 400;
    n.y = Math.random() * 600 - 300;
    n.vx = 0;
    n.vy = 0;
    n.r = 4 + Math.min(10, (n.degree || 0) * 1.4);
    n.color = accentFor(n.group || "misc");
    byId.set(n.id, n);
  });
  const edges = links
    .map((l) => ({ s: byId.get(l.source), t: byId.get(l.target) }))
    .filter((e) => e.s && e.t);

  // Adjacency for hover highlighting.
  const adj = new Map();
  nodes.forEach((n) => adj.set(n.id, new Set()));
  edges.forEach((e) => { adj.get(e.s.id).add(e.t.id); adj.get(e.t.id).add(e.s.id); });

  let width = 0, height = 0, dpr = window.devicePixelRatio || 1;
  function resize() {
    width = wrap.clientWidth;
    height = wrap.clientHeight;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = width + "px";
    canvas.style.height = height + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  window.addEventListener("resize", resize);

  // View transform (pan/zoom).
  let scale = 1, offsetX = width / 2, offsetY = height / 2;
  let hovered = null, highlight = null;

  // Simple simulation.
  let alpha = 1;
  function tick() {
    // Repulsion (O(n^2), fine for ~100 nodes).
    for (let i = 0; i < nodes.length; i++) {
      const a = nodes[i];
      for (let j = i + 1; j < nodes.length; j++) {
        const b = nodes[j];
        let dx = a.x - b.x, dy = a.y - b.y;
        let d2 = dx * dx + dy * dy || 0.01;
        let f = 900 / d2;
        let d = Math.sqrt(d2);
        let fx = (dx / d) * f, fy = (dy / d) * f;
        a.vx += fx; a.vy += fy; b.vx -= fx; b.vy -= fy;
      }
    }
    // Springs.
    edges.forEach((e) => {
      let dx = e.t.x - e.s.x, dy = e.t.y - e.s.y;
      let d = Math.sqrt(dx * dx + dy * dy) || 0.01;
      let f = (d - 70) * 0.02;
      let fx = (dx / d) * f, fy = (dy / d) * f;
      e.s.vx += fx; e.s.vy += fy; e.t.vx -= fx; e.t.vy -= fy;
    });
    // Gravity to center + integrate.
    nodes.forEach((n) => {
      n.vx += -n.x * 0.002;
      n.vy += -n.y * 0.002;
      n.x += n.vx * alpha;
      n.y += n.vy * alpha;
      n.vx *= 0.85;
      n.vy *= 0.85;
    });
    alpha *= 0.995;
    if (alpha < 0.02) alpha = 0.02;
  }

  function draw() {
    ctx.clearRect(0, 0, width, height);
    ctx.save();
    ctx.translate(offsetX, offsetY);
    ctx.scale(scale, scale);

    const isDark = document.documentElement.classList.contains("theme-dark") ||
      (!document.documentElement.classList.contains("theme-light") &&
        window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
    const edgeColor = isDark ? "rgba(148,163,184,0.18)" : "rgba(100,116,139,0.22)";
    const edgeHi = isDark ? "rgba(129,140,248,0.7)" : "rgba(79,70,229,0.6)";

    edges.forEach((e) => {
      const on = highlight && (highlight === e.s.id || highlight === e.t.id);
      ctx.strokeStyle = on ? edgeHi : edgeColor;
      ctx.lineWidth = (on ? 1.6 : 0.7) / scale;
      ctx.beginPath();
      ctx.moveTo(e.s.x, e.s.y);
      ctx.lineTo(e.t.x, e.t.y);
      ctx.stroke();
    });

    nodes.forEach((n) => {
      const neighbors = highlight ? adj.get(highlight) : null;
      const dim = highlight && highlight !== n.id && !(neighbors && neighbors.has(n.id));
      ctx.globalAlpha = dim ? 0.25 : 1;
      ctx.fillStyle = n.color;
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fill();
      if (highlight === n.id) {
        ctx.lineWidth = 2 / scale;
        ctx.strokeStyle = isDark ? "#fff" : "#111827";
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    });
    ctx.restore();
  }

  function frame() {
    tick();
    draw();
    requestAnimationFrame(frame);
  }
  frame();

  // Coordinate helpers.
  function toWorld(px, py) {
    return { x: (px - offsetX) / scale, y: (py - offsetY) / scale };
  }
  function nodeAt(px, py) {
    const w = toWorld(px, py);
    let best = null, bestD = Infinity;
    nodes.forEach((n) => {
      const dx = n.x - w.x, dy = n.y - w.y;
      const d = dx * dx + dy * dy;
      const rr = (n.r + 6) * (n.r + 6);
      if (d < rr && d < bestD) { best = n; bestD = d; }
    });
    return best;
  }

  // Interaction: pan, zoom, hover, click.
  let dragging = false, dragMoved = false, lastX = 0, lastY = 0;
  canvas.addEventListener("mousedown", (e) => { dragging = true; dragMoved = false; lastX = e.offsetX; lastY = e.offsetY; });
  window.addEventListener("mouseup", () => { dragging = false; });
  canvas.addEventListener("mousemove", (e) => {
    if (dragging) {
      offsetX += e.offsetX - lastX;
      offsetY += e.offsetY - lastY;
      lastX = e.offsetX; lastY = e.offsetY;
      dragMoved = true;
      return;
    }
    const n = nodeAt(e.offsetX, e.offsetY);
    highlight = n ? n.id : null;
    hovered = n;
    canvas.style.cursor = n ? "pointer" : "grab";
    if (n && tooltip) {
      tooltip.hidden = false;
      tooltip.textContent = n.title;
      tooltip.style.left = e.offsetX + 12 + "px";
      tooltip.style.top = e.offsetY + 12 + "px";
    } else if (tooltip) {
      tooltip.hidden = true;
    }
  });
  canvas.addEventListener("click", (e) => {
    if (dragMoved) return;
    const n = nodeAt(e.offsetX, e.offsetY);
    if (n && urlMap[n.id]) window.location.href = urlMap[n.id];
  });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.1 : 0.9;
    const w = toWorld(e.offsetX, e.offsetY);
    scale = Math.max(0.2, Math.min(4, scale * factor));
    // keep cursor point stable
    offsetX = e.offsetX - w.x * scale;
    offsetY = e.offsetY - w.y * scale;
  }, { passive: false });

  if (searchEl) {
    searchEl.addEventListener("input", () => {
      const q = searchEl.value.toLowerCase().trim();
      if (!q) { highlight = null; return; }
      const hit = nodes.find((n) => n.title.toLowerCase().includes(q));
      if (hit) { highlight = hit.id; offsetX = width / 2 - hit.x * scale; offsetY = height / 2 - hit.y * scale; }
    });
  }
}

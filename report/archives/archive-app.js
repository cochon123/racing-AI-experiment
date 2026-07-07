/* Archive report renderer — extends main report with failure narrative + dual learning curves */

(function () {
  "use strict";

  const AGENTS = {
    time: { key: "time", label: "Time-only", color: "#22d3ee" },
    nobrakes: { key: "nobrakes", label: "No-brakes", color: "#fb923c" },
  };

  const PLOT_LAYOUT_BASE = {
    paper_bgcolor: "transparent",
    plot_bgcolor: "#0e1116",
    font: { family: "SF Mono, Cascadia Code, Consolas, monospace", color: "#8b949e", size: 11 },
    margin: { l: 52, r: 24, t: 36, b: 48 },
    xaxis: { gridcolor: "#2a2f3a", zerolinecolor: "#2a2f3a", tickfont: { color: "#8b949e" } },
    yaxis: { gridcolor: "#2a2f3a", zerolinecolor: "#2a2f3a", tickfont: { color: "#8b949e" } },
    legend: { orientation: "h", y: 1.14, x: 0, bgcolor: "transparent", font: { color: "#e6edf3" } },
  };

  async function fetchJSON(path) {
    try {
      const res = await fetch(path);
      if (!res.ok) throw new Error(`${res.status}`);
      return await res.json();
    } catch (e) {
      console.warn(path, e.message);
      return null;
    }
  }

  function fmtTime(sec) {
    if (sec == null || Number.isNaN(sec)) return "DNF";
    return sec.toFixed(2) + "s";
  }

  function fmtPct(v) {
    return v == null ? "—" : (v * 100).toFixed(1) + "%";
  }

  function fmtNum(v, d) {
    return v == null ? "—" : v.toFixed(d);
  }

  function videoSlot(src, alt) {
    const wrap = document.createElement("div");
    wrap.className = "video-slot";
    if (!src) {
      wrap.innerHTML = `<div class="video-placeholder"><div class="video-placeholder__icon">▶</div><div class="video-placeholder__text">Video not yet rendered</div></div>`;
      return wrap;
    }
    const video = document.createElement("video");
    video.controls = true;
    video.preload = "metadata";
    video.playsInline = true;
    video.setAttribute("aria-label", alt || "Replay");
    const source = document.createElement("source");
    source.src = src;
    source.type = "video/mp4";
    video.appendChild(source);
    video.addEventListener("error", () => {
      wrap.innerHTML = `<div class="video-placeholder"><div class="video-placeholder__icon">▶</div><div class="video-placeholder__text">Video not yet rendered</div></div>`;
    });
    wrap.appendChild(video);
    return wrap;
  }

  function renderHero(exp) {
    const badge = document.getElementById("archive-badge");
    const title = document.getElementById("archive-title");
    const subtitle = document.getElementById("archive-subtitle");
    const story = document.getElementById("archive-story");
    const lesson = document.getElementById("archive-lesson");
    if (badge && exp?.badge) {
      badge.textContent = exp.badge;
      badge.className = "archive-badge " + (exp.badge_class || "");
    }
    if (title && exp?.headline) title.textContent = exp.headline;
    if (subtitle && exp?.subtitle) subtitle.textContent = exp.subtitle;
    if (story && exp?.story) story.textContent = exp.story;
    if (lesson && exp?.lesson) lesson.textContent = exp.lesson;

    const meta = document.getElementById("archive-meta");
    if (meta && exp) {
      meta.innerHTML = `
        <span><strong>Tracks:</strong> ${exp.track_profile_label || "—"}</span>
        <span><strong>Decel penalty:</strong> ${exp.decel_penalty_label || "—"}</span>
        <span><strong>Train steps:</strong> ${(exp.train_steps || 0).toLocaleString()}</span>
        <span><strong>Final reward (time):</strong> ${fmtNum(exp.final_time_reward, 0)}</span>
        <span><strong>Final reward (no-brakes):</strong> ${fmtNum(exp.final_nobrakes_reward, 0)}</span>`;
    }

    const heroVid = document.getElementById("hero-video");
    if (heroVid) {
      heroVid.innerHTML = "";
      heroVid.appendChild(videoSlot("assets/videos/intro_overlay.mp4", "Hero overlay"));
    }
  }

  function renderStats(summary, extras) {
    const grid = document.getElementById("stats-grid");
    if (!grid || !summary?.agents) return;

    const metrics = [
      { label: "Mean lap time (paired)", key: "mean_lap_time", unit: "s", digits: 2 },
      { label: "Mean speed", key: "mean_speed", unit: "u/s", digits: 1 },
      { label: "Speed p5 → p95", key: "_range", unit: "u/s" },
      { label: "Drift fraction", key: "drift_fraction", pct: true },
      { label: "Hard brakes / lap", key: "hard_brake_events_per_lap", digits: 1 },
      { label: "DNF rate", key: "dnf_rate", pct: true },
      { label: "Brake input", key: "_brake", pct: true },
      { label: "Reverse driving", key: "_reverse", pct: true },
    ];

    grid.innerHTML = metrics.map((m) => {
      const vals = ["time", "nobrakes"].map((k) => {
        let display;
        if (m.key === "_range" && extras?.[k]) {
          display = `${fmtNum(extras[k].p5_speed, 1)} → ${fmtNum(extras[k].p95_speed, 1)}`;
        } else if (m.key === "_brake" && extras?.[k]) {
          display = fmtPct(extras[k].brake_input_fraction);
        } else if (m.key === "_reverse" && extras?.[k]) {
          display = fmtPct(extras[k].reverse_fraction);
        } else {
          const raw = summary.agents[k][m.key];
          display = m.pct ? fmtPct(raw) : fmtNum(raw, m.digits ?? 2);
        }
        return `<div class="stat-value"><div class="stat-value__agent ${k}">${summary.agents[k].label}</div><div class="stat-value__num">${display}</div>${m.unit && m.key !== "_range" ? `<div class="stat-value__unit">${m.unit}</div>` : ""}</div>`;
      }).join("");
      return `<div class="stat-card"><p class="stat-card__label">${m.label}</p><div class="stat-card__values">${vals}</div></div>`;
    }).join("");
  }

  function renderLearning(data, summary, elId) {
    const el = document.getElementById(elId);
    if (!el || !data?.time?.steps?.length) {
      if (el) el.innerHTML = `<div class="chart-empty">Learning data unavailable</div>`;
      return;
    }
    const traces = [];
    ["time", "nobrakes"].forEach((key) => {
      traces.push({
        type: "scatter", mode: "lines", name: (summary?.agents?.[key]?.label || AGENTS[key].label) + " reward",
        x: data[key].steps, y: data[key].reward,
        line: { color: summary?.agents?.[key]?.color || AGENTS[key].color, width: 2 },
        yaxis: "y",
      });
      if (data[key].ep_len?.length) {
        traces.push({
          type: "scatter", mode: "lines", name: (summary?.agents?.[key]?.label || AGENTS[key].label) + " ep len",
          x: data[key].steps, y: data[key].ep_len,
          line: { color: summary?.agents?.[key]?.color || AGENTS[key].color, width: 1.5, dash: "dot" },
          yaxis: "y2", opacity: 0.65,
        });
      }
    });
    Plotly.newPlot(el, traces, Object.assign({}, PLOT_LAYOUT_BASE, {
      title: { text: "Training progress", font: { size: 12, color: "#8b949e" } },
      xaxis: Object.assign({}, PLOT_LAYOUT_BASE.xaxis, { title: "Environment steps", tickformat: "~s" }),
      yaxis: Object.assign({}, PLOT_LAYOUT_BASE.yaxis, { title: "Mean episode reward", side: "left" }),
      yaxis2: { title: "Mean episode length (steps)", overlaying: "y", side: "right", gridcolor: "transparent", tickfont: { color: "#6e7681" } },
    }), { responsive: true, displayModeBar: false });
  }

  function smoothDensityFromBins(edges, densities, { points = 200, bandwidth = null } = {}) {
    const centers = edges.slice(0, -1).map((e, i) => (e + edges[i + 1]) / 2);
    const binWidth = edges.length > 1 ? edges[1] - edges[0] : 1;
    const bw = bandwidth ?? binWidth * 1.5;
    const xMin = edges[0];
    const xMax = edges[edges.length - 1];
    const xs = [];
    const ys = [];
    for (let p = 0; p < points; p++) {
      const x = xMin + ((xMax - xMin) * p) / (points - 1);
      let y = 0;
      for (let j = 0; j < centers.length; j++) {
        if (densities[j] <= 0) continue;
        const z = (x - centers[j]) / bw;
        y += densities[j] * Math.exp(-0.5 * z * z);
      }
      xs.push(x);
      ys.push(y);
    }
    return { x: xs, y: ys };
  }

  function hexToRgba(hex, alpha) {
    const h = hex.replace("#", "");
    return `rgba(${parseInt(h.slice(0, 2), 16)}, ${parseInt(h.slice(2, 4), 16)}, ${parseInt(h.slice(4, 6), 16)}, ${alpha})`;
  }

  function renderHistogram(data, summary) {
    const el = document.getElementById("chart-histogram");
    if (!el || !data?.bin_edges) {
      if (el) el.innerHTML = `<div class="chart-empty">No histogram data</div>`;
      return;
    }
    const edges = data.bin_edges;
    Plotly.newPlot(el, ["time", "nobrakes"].map((key) => {
      const color = summary?.agents?.[key]?.color || AGENTS[key].color;
      const curve = smoothDensityFromBins(edges, data[key]);
      return {
        type: "scatter", mode: "lines",
        name: summary?.agents?.[key]?.label || AGENTS[key].label,
        x: curve.x, y: curve.y,
        line: { color, width: 2.5, shape: "spline", smoothing: 1.1 },
        fill: "tozeroy", fillcolor: hexToRgba(color, 0.18),
      };
    }), Object.assign({}, PLOT_LAYOUT_BASE, {
      xaxis: Object.assign({}, PLOT_LAYOUT_BASE.xaxis, { title: "Speed (u/s)" }),
      yaxis: Object.assign({}, PLOT_LAYOUT_BASE.yaxis, { title: "Density" }),
    }), { responsive: true, displayModeBar: false });
  }

  function renderProfiles(profiles, summary, seed) {
    const el = document.getElementById("chart-profiles");
    if (!el || !profiles?.length) return;
    const track = profiles.find((p) => p.seed === seed) || profiles[0];
    Plotly.newPlot(el, ["time", "nobrakes"].map((key) => ({
      type: "scatter", mode: "lines", name: summary?.agents?.[key]?.label || AGENTS[key].label,
      x: track.s, y: track[key],
      line: { color: summary?.agents?.[key]?.color || AGENTS[key].color, width: 2 },
    })), Object.assign({}, PLOT_LAYOUT_BASE, {
      title: { text: `Track seed ${track.seed}`, font: { size: 12, color: "#8b949e" } },
      xaxis: Object.assign({}, PLOT_LAYOUT_BASE.xaxis, { title: "Track position", range: [0, 1] }),
      yaxis: Object.assign({}, PLOT_LAYOUT_BASE.yaxis, { title: "Speed (u/s)" }),
    }), { responsive: true, displayModeBar: false });
  }

  function renderReplays(summary, slug) {
    const grid = document.getElementById("replay-grid");
    if (!grid || !summary?.per_track) return;
    grid.innerHTML = "";
    summary.per_track.forEach((track) => {
      const card = document.createElement("article");
      card.className = "replay-card";
      const vid = document.createElement("div");
      vid.className = "replay-card__video";
      vid.appendChild(videoSlot(`assets/videos/track_${track.seed}_overlay.mp4`, `Track ${track.seed}`));
      const meta = document.createElement("div");
      meta.className = "replay-card__meta";
      meta.innerHTML = `
        <p class="replay-card__seed">seed ${track.seed}</p>
        <div class="replay-card__times">
          <div class="replay-time"><span class="label time">Time</span><span>${fmtTime(track.lap_time?.time)}</span></div>
          <div class="replay-time"><span class="label nobrakes">No-brakes</span><span>${fmtTime(track.lap_time?.nobrakes)}</span></div>
        </div>`;
      card.appendChild(vid);
      card.appendChild(meta);
      grid.appendChild(card);
    });

    const extras = document.getElementById("solo-videos");
    if (extras && slug === "reverse-exploit") {
      extras.innerHTML = "";
      ["time", "nobrakes"].forEach((mode) => {
        const box = document.createElement("div");
        box.className = "solo-video-box";
        box.innerHTML = `<h3>${mode === "time" ? "Time-only" : "No-brakes"} solo — seed 1003</h3>`;
        box.appendChild(videoSlot(`assets/videos/track_1003_${mode}_solo.mp4`, `${mode} solo`));
        extras.appendChild(box);
      });
    }
  }

  function renderPerTrack(summary) {
    const el = document.getElementById("chart-per-track");
    if (!el || !summary?.per_track) return;
    Plotly.newPlot(el, ["time", "nobrakes"].map((key) => ({
      type: "bar", name: summary.agents[key].label,
      x: summary.per_track.map((t) => String(t.seed)),
      y: summary.per_track.map((t) => t.lap_time?.[key] ?? null),
      marker: { color: summary.agents[key].color },
    })), Object.assign({}, PLOT_LAYOUT_BASE, {
      barmode: "group",
      xaxis: Object.assign({}, PLOT_LAYOUT_BASE.xaxis, { title: "Track seed" }),
      yaxis: Object.assign({}, PLOT_LAYOUT_BASE.yaxis, { title: "Lap time (s)" }),
    }), { responsive: true, displayModeBar: false });
  }

  async function init() {
    const slug = document.body.dataset.archive || "";
    const [summary, histograms, profiles, learning] = await Promise.all([
      fetchJSON("assets/data/summary.json"),
      fetchJSON("assets/data/speed_histograms.json"),
      fetchJSON("assets/data/speed_profiles.json"),
      fetchJSON("assets/data/learning_curves.json"),
    ]);

    renderHero(summary?.experiment);
    renderStats(summary, summary?.telemetry_extras);
    renderLearning(learning, summary, "chart-learning");
    renderHistogram(histograms, summary);

    const select = document.getElementById("profile-track-select");
    if (select && profiles?.length) {
      select.innerHTML = profiles.map((p) => `<option value="${p.seed}">${p.seed}</option>`).join("");
      const seed = profiles[0].seed;
      select.addEventListener("change", () => renderProfiles(profiles, summary, Number(select.value)));
      renderProfiles(profiles, summary, seed);
    }

    renderReplays(summary, slug);
    renderPerTrack(summary);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();

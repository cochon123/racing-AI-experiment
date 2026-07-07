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
    margin: { l: 52, r: 24, t: 32, b: 48 },
    xaxis: {
      gridcolor: "#2a2f3a",
      zerolinecolor: "#2a2f3a",
      tickfont: { color: "#8b949e" },
    },
    yaxis: {
      gridcolor: "#2a2f3a",
      zerolinecolor: "#2a2f3a",
      tickfont: { color: "#8b949e" },
    },
    legend: {
      orientation: "h",
      y: 1.12,
      x: 0,
      bgcolor: "transparent",
      font: { color: "#e6edf3" },
    },
  };

  const chartIds = [
    "chart-histogram",
    "chart-profiles",
    "chart-learning",
    "chart-per-track",
  ];

  let state = {
    summary: null,
    histograms: null,
    profiles: null,
    learning: null,
  };

  /* ── Fetch helper ── */
  async function fetchJSON(path) {
    try {
      const res = await fetch(path);
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      return await res.json();
    } catch (err) {
      console.warn(`Failed to load ${path}:`, err.message);
      return null;
    }
  }

  /* ── Utilities ── */
  function fmtTime(sec) {
    if (sec == null || Number.isNaN(sec)) return "DNF";
    return sec.toFixed(2) + "s";
  }

  function fmtPct(val) {
    if (val == null) return "—";
    return (val * 100).toFixed(1) + "%";
  }

  function fmtNum(val, digits) {
    if (val == null) return "—";
    return val.toFixed(digits);
  }

  function emptyChart(el, msg) {
    el.innerHTML = `<div class="chart-empty">${msg}</div>`;
  }

  function videoSlot(src, altText) {
    const wrap = document.createElement("div");
    wrap.className = "video-slot";

    if (!src) {
      wrap.innerHTML = `
        <div class="video-placeholder">
          <div class="video-placeholder__icon">▶</div>
          <div class="video-placeholder__text">Video not yet rendered</div>
        </div>`;
      return wrap;
    }

    const video = document.createElement("video");
    video.controls = true;
    video.preload = "metadata";
    video.playsInline = true;
    video.setAttribute("aria-label", altText || "Race replay");

    const source = document.createElement("source");
    source.src = src;
    source.type = "video/mp4";
    video.appendChild(source);

    video.addEventListener("error", () => {
      wrap.innerHTML = `
        <div class="video-placeholder">
          <div class="video-placeholder__icon">▶</div>
          <div class="video-placeholder__text">Video not yet rendered</div>
        </div>`;
    });

    wrap.appendChild(video);
    return wrap;
  }

  function plotLayout(extra) {
    return Object.assign({}, PLOT_LAYOUT_BASE, extra);
  }

  /* Smooth binned density into a KDE-like curve via Gaussian kernel convolution. */
  function smoothDensityFromBins(edges, densities, { points = 200, bandwidth = null } = {}) {
    const centers = [];
    for (let i = 0; i < edges.length - 1; i++) {
      centers.push((edges[i] + edges[i + 1]) / 2);
    }
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
    const r = parseInt(h.slice(0, 2), 16);
    const g = parseInt(h.slice(2, 4), 16);
    const b = parseInt(h.slice(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  function bindResize() {
    let timer;
    window.addEventListener("resize", () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        chartIds.forEach((id) => {
          const el = document.getElementById(id);
          if (el && el.querySelector(".js-plotly-plot")) {
            Plotly.Plots.resize(el);
          }
        });
      }, 150);
    });
  }

  /* ── Render: driver cards ── */
  function renderDriverCards(summary) {
    const container = document.getElementById("driver-cards");
    if (!container) return;

    const rewards = {
      time: "R = Δprogress",
      nobrakes: "R = Δprogress − λ·max(0, v<sub>t−1</sub> − v<sub>t</sub>)",
    };

    container.innerHTML = "";
    ["time", "nobrakes"].forEach((key) => {
      const agent = summary?.agents?.[key] || AGENTS[key];
      const card = document.createElement("article");
      card.className = "driver-card";
      card.style.setProperty("--agent-color", agent.color || AGENTS[key].color);
      card.innerHTML = `
        <p class="driver-card__label">${agent.label || AGENTS[key].label}</p>
        <p class="driver-card__reward">${rewards[key]}</p>`;
      container.appendChild(card);
    });
  }

  /* ── Render: hero video ── */
  function renderHeroVideo() {
    const container = document.getElementById("hero-video");
    if (!container) return;
    container.innerHTML = "";
    container.appendChild(
      videoSlot("assets/videos/intro_overlay.mp4", "Introduction race overlay")
    );
  }

  /* ── Render: key stats ── */
  function renderKeyStats(summary) {
    const grid = document.getElementById("stats-grid");
    if (!grid) return;

    if (!summary?.agents) {
      grid.innerHTML = `<div class="chart-empty">Summary data unavailable</div>`;
      return;
    }

    const metrics = [
      { label: "Mean lap time", key: "mean_lap_time", unit: "s", digits: 2, lowerBetter: true },
      { label: "Mean speed", key: "mean_speed", unit: "u/s", digits: 2, lowerBetter: false },
      { label: "Drift fraction", key: "drift_fraction", unit: "", digits: 1, pct: true },
      { label: "Hard brakes / lap", key: "hard_brake_events_per_lap", unit: "", digits: 1 },
      { label: "Off-track fraction", key: "offtrack_fraction", unit: "", digits: 1, pct: true },
      { label: "DNF rate", key: "dnf_rate", unit: "", digits: 1, pct: true },
    ];

    grid.innerHTML = metrics
      .map((m) => {
        const vals = ["time", "nobrakes"].map((k) => {
          const raw = summary.agents[k][m.key];
          const display = m.pct ? fmtPct(raw) : fmtNum(raw, m.digits);
          return `
            <div class="stat-value">
              <div class="stat-value__agent ${k}">${summary.agents[k].label}</div>
              <div class="stat-value__num">${display}</div>
              ${m.unit ? `<div class="stat-value__unit">${m.unit}</div>` : ""}
            </div>`;
        });
        return `
          <div class="stat-card">
            <p class="stat-card__label">${m.label}</p>
            <div class="stat-card__values">${vals.join("")}</div>
          </div>`;
      })
      .join("");
  }

  /* ── Render: speed histogram ── */
  function renderSpeedHistogram(data, summary) {
    const el = document.getElementById("chart-histogram");
    if (!el) return;

    if (!data?.bin_edges || !data.time || !data.nobrakes) {
      emptyChart(el, "Histogram data unavailable");
      return;
    }

    const edges = data.bin_edges;

    const traces = ["time", "nobrakes"].map((key) => {
      const color = summary?.agents?.[key]?.color || AGENTS[key].color;
      const curve = smoothDensityFromBins(edges, data[key]);
      return {
        type: "scatter",
        mode: "lines",
        name: summary?.agents?.[key]?.label || AGENTS[key].label,
        x: curve.x,
        y: curve.y,
        line: { color, width: 2.5, shape: "spline", smoothing: 1.1 },
        fill: "tozeroy",
        fillcolor: hexToRgba(color, 0.18),
      };
    });

    Plotly.newPlot(
      el,
      traces,
      plotLayout({
        title: { text: "Speed density", font: { size: 12, color: "#8b949e" } },
        xaxis: Object.assign({}, PLOT_LAYOUT_BASE.xaxis, {
          title: "Speed (u/s)",
        }),
        yaxis: Object.assign({}, PLOT_LAYOUT_BASE.yaxis, {
          title: "Density",
        }),
      }),
      { responsive: true, displayModeBar: false }
    );
  }

  /* Annotate the deepest speed dip of the time-only agent as the braking zone. */
  function brakingZoneAnnotation(track) {
    const speeds = track.time;
    if (!Array.isArray(speeds) || !speeds.length) return [];
    let minIdx = 0;
    for (let i = 1; i < speeds.length; i++) {
      if (speeds[i] != null && speeds[i] < speeds[minIdx]) minIdx = i;
    }
    if (speeds[minIdx] == null) return [];
    return [
      {
        x: track.s[minIdx],
        y: speeds[minIdx],
        xref: "x",
        yref: "y",
        text: "braking zone",
        showarrow: true,
        arrowhead: 2,
        ax: 40,
        ay: -30,
        font: { size: 10, color: "#8b949e" },
        arrowcolor: "#2a2f3a",
      },
    ];
  }

  /* ── Render: speed profiles ── */
  function renderSpeedProfiles(profiles, summary, seed) {
    const el = document.getElementById("chart-profiles");
    if (!el) return;

    if (!profiles?.length) {
      emptyChart(el, "Speed profile data unavailable");
      return;
    }

    const track = profiles.find((p) => p.seed === seed) || profiles[0];
    if (!track) {
      emptyChart(el, "No track selected");
      return;
    }

    const traces = ["time", "nobrakes"].map((key) => ({
      type: "scatter",
      mode: "lines",
      name: summary?.agents?.[key]?.label || AGENTS[key].label,
      x: track.s,
      y: track[key],
      line: {
        color: summary?.agents?.[key]?.color || AGENTS[key].color,
        width: 2,
      },
    }));

    Plotly.newPlot(
      el,
      traces,
      plotLayout({
        title: { text: `Track seed ${track.seed}`, font: { size: 12, color: "#8b949e" } },
        xaxis: Object.assign({}, PLOT_LAYOUT_BASE.xaxis, {
          title: "Track position (0 → 1)",
          range: [0, 1],
        }),
        yaxis: Object.assign({}, PLOT_LAYOUT_BASE.yaxis, {
          title: "Speed (u/s)",
        }),
        annotations: brakingZoneAnnotation(track),
      }),
      { responsive: true, displayModeBar: false }
    );
  }

  function setupProfileSelect(profiles, summary) {
    const select = document.getElementById("profile-track-select");
    if (!select || !profiles?.length) return;

    select.innerHTML = profiles
      .map((p) => `<option value="${p.seed}">Seed ${p.seed}</option>`)
      .join("");

    const initial = profiles[0].seed;
    select.value = String(initial);

    select.addEventListener("change", () => {
      renderSpeedProfiles(profiles, summary, Number(select.value));
    });

    renderSpeedProfiles(profiles, summary, initial);
  }

  /* ── Render: learning curves ── */
  function renderLearningCurves(data, summary) {
    const el = document.getElementById("chart-learning");
    if (!el) return;

    if (!data?.time?.steps || !data?.nobrakes?.steps) {
      emptyChart(el, "Learning curve data unavailable");
      return;
    }

    const traces = ["time", "nobrakes"].map((key) => ({
      type: "scatter",
      mode: "lines",
      name: summary?.agents?.[key]?.label || AGENTS[key].label,
      x: data[key].steps,
      y: data[key].reward,
      line: {
        color: summary?.agents?.[key]?.color || AGENTS[key].color,
        width: 2,
      },
    }));

    Plotly.newPlot(
      el,
      traces,
      plotLayout({
        title: { text: "Episode reward", font: { size: 12, color: "#8b949e" } },
        xaxis: Object.assign({}, PLOT_LAYOUT_BASE.xaxis, {
          title: "Environment steps",
          tickformat: "~s",
        }),
        yaxis: Object.assign({}, PLOT_LAYOUT_BASE.yaxis, {
          title: "Mean reward",
        }),
      }),
      { responsive: true, displayModeBar: false }
    );
  }

  /* ── Render: race replays ── */
  function renderReplays(summary) {
    const grid = document.getElementById("replay-grid");
    if (!grid) return;

    const tracks = summary?.per_track;
    if (!tracks?.length) {
      grid.innerHTML = `<div class="chart-empty">No replay data</div>`;
      return;
    }

    grid.innerHTML = "";
    tracks.forEach((track) => {
      const card = document.createElement("article");
      card.className = "replay-card";

      const videoWrap = document.createElement("div");
      videoWrap.className = "replay-card__video";
      videoWrap.appendChild(
        videoSlot(
          track.video,
          `Track ${track.seed} overlay replay`
        )
      );

      const timeVal = track.lap_time?.time;
      const nobrakesVal = track.lap_time?.nobrakes;

      const meta = document.createElement("div");
      meta.className = "replay-card__meta";
      meta.innerHTML = `
        <p class="replay-card__seed">seed ${track.seed}</p>
        <div class="replay-card__times">
          <div class="replay-time">
            <span class="label time">Time-only</span>
            <span>${fmtTime(timeVal)}</span>
          </div>
          <div class="replay-time">
            <span class="label nobrakes">No-brakes</span>
            <span class="${nobrakesVal == null ? "dnf" : ""}">${fmtTime(nobrakesVal)}</span>
          </div>
        </div>`;

      card.appendChild(videoWrap);
      card.appendChild(meta);
      grid.appendChild(card);
    });
  }

  /* ── Render: per-track bar chart ── */
  function renderPerTrackChart(summary) {
    const el = document.getElementById("chart-per-track");
    if (!el) return;

    const tracks = summary?.per_track;
    if (!tracks?.length) {
      emptyChart(el, "Per-track data unavailable");
      return;
    }

    const seeds = tracks.map((t) => String(t.seed));

    const traces = ["time", "nobrakes"].map((key) => ({
      type: "bar",
      name: summary.agents[key].label,
      x: seeds,
      y: tracks.map((t) => t.lap_time?.[key] ?? null),
      marker: { color: summary.agents[key].color },
    }));

    Plotly.newPlot(
      el,
      traces,
      plotLayout({
        title: { text: "Lap time by track seed", font: { size: 12, color: "#8b949e" } },
        barmode: "group",
        xaxis: Object.assign({}, PLOT_LAYOUT_BASE.xaxis, { title: "Track seed" }),
        yaxis: Object.assign({}, PLOT_LAYOUT_BASE.yaxis, { title: "Lap time (s)" }),
      }),
      { responsive: true, displayModeBar: false }
    );
  }

  /* ── Render: meta / footer ── */
  function renderMeta(summary) {
    const dateEl = document.getElementById("experiment-date");
    const nTracks = document.getElementById("n-tracks");
    if (dateEl && summary?.experiment?.date) {
      dateEl.textContent = summary.experiment.date;
    }
    if (nTracks && summary?.experiment?.n_eval_tracks) {
      nTracks.textContent = String(summary.experiment.n_eval_tracks);
    }
  }

  /* ── Init ── */
  async function init() {
    const [summary, histograms, profiles, learning] = await Promise.all([
      fetchJSON("assets/data/summary.json"),
      fetchJSON("assets/data/speed_histograms.json"),
      fetchJSON("assets/data/speed_profiles.json"),
      fetchJSON("assets/data/learning_curves.json"),
    ]);

    state = { summary, histograms, profiles, learning };

    renderDriverCards(summary);
    renderHeroVideo();
    renderKeyStats(summary);
    renderSpeedHistogram(histograms, summary);
    setupProfileSelect(profiles, summary);
    renderLearningCurves(learning, summary);
    renderReplays(summary);
    renderPerTrackChart(summary);
    renderMeta(summary);
    bindResize();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

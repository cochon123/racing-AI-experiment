# Racing AI Experiment — Reward Shaping Comparison

A little experiment on how learning objectives shape AI behavior.

Two identical PPO agents learn to drive a top-down 2D racing game on procedurally
generated tracks. The **only** difference between them is the reward function:

| Agent | Color | Reward |
|---|---|---|
| `time` | cyan | progress along track + lap bonus − small time penalty |
| `nobrakes` | orange | same, **minus** an aggressive penalty proportional to any deceleration |

Hypothesis: both learn to race, but their speed distributions diverge — `time`
brakes hard into corners, `nobrakes` carries speed and drifts instead of slowing
down, possibly at the cost of lap time on tight tracks.

## Setup

Requires Python 3.12+ with a CUDA-enabled `torch` already installed, then:

```bash
pip install -r requirements.txt
```

## Commands

```bash
python -m racing play --seed 7          # drive yourself (arrows, R reset, ESC quit)
python -m racing race --vs nobrakes     # you vs a trained AI, same track
python -m racing watch --seed 3         # AI vs AI, live overlay
python -m racing newtrack --seed 42     # preview a generated track
python -m racing train --agent both     # train both agents (PPO)
python -m racing evaluate               # run held-out tracks, build dataset/
python -m racing report                 # render videos + charts, serve web report
python -m racing selftest               # headless sanity checks
```

## Layout

- `racing/` — game core (track generator, drift physics, renderer), gym env, training, evaluation
- `runs/` — model checkpoints and training logs
- `dataset/` — evaluation telemetry (`telemetry.csv`, `summary.json`)
- `report/` — static web report (open via `python -m racing report`)

## Reports (GitHub Pages)

After deploy, the static reports are served from the `report/` folder:

- [Main experiment — reward shaping comparison](https://cochon123.github.io/racing-AI-experiment/)
- [Archive: flat tracks (null result)](https://cochon123.github.io/racing-AI-experiment/archives/flat-tracks/)
- [Archive: reverse exploit (reward hack)](https://cochon123.github.io/racing-AI-experiment/archives/reverse-exploit/)

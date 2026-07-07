"""PPO training entrypoint for the racing reward comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from racing.agents import AGENT_SPECS, make_env
from racing.env import DEFAULT_DECEL_COEF, REWARD_CONSTANTS


PPO_HYPERPARAMS: dict[str, Any] = {
    "policy": "MlpPolicy",
    "policy_kwargs": {"net_arch": [256, 256]},
    "n_steps": 1024,
    "batch_size": 4096,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "ent_coef": 0.003,
    "learning_rate": 3e-4,
}


def train(
    reward_mode: str,
    total_steps: int = 1_500_000,
    n_envs: int = 12,
    seed: int = 42,
    device: str = "auto",
    runs_dir: str = "runs",
) -> PPO:
    """Train one PPO agent and save checkpoints, final model, CSV logs, and config."""

    if reward_mode not in AGENT_SPECS:
        raise ValueError("reward_mode must be 'time' or 'nobrakes'")
    if n_envs < 1:
        raise ValueError("n_envs must be at least 1")

    run_dir = Path(runs_dir) / reward_mode
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [make_env(reward_mode) for _ in range(n_envs)]
    vec_env = VecMonitor(SubprocVecEnv(env_fns, start_method="fork"))
    vec_env.seed(seed)

    logger = configure(str(run_dir), ["stdout", "csv"])
    model = PPO(
        PPO_HYPERPARAMS["policy"],
        vec_env,
        policy_kwargs=PPO_HYPERPARAMS["policy_kwargs"],
        n_steps=PPO_HYPERPARAMS["n_steps"],
        batch_size=PPO_HYPERPARAMS["batch_size"],
        gamma=PPO_HYPERPARAMS["gamma"],
        gae_lambda=PPO_HYPERPARAMS["gae_lambda"],
        ent_coef=PPO_HYPERPARAMS["ent_coef"],
        learning_rate=PPO_HYPERPARAMS["learning_rate"],
        seed=seed,
        device=device,
        verbose=1,
    )
    model.set_logger(logger)

    config = {
        "reward_mode": reward_mode,
        "reward_params": {"decel_coef": DEFAULT_DECEL_COEF, **REWARD_CONSTANTS},
        "total_steps": int(total_steps),
        "n_envs": int(n_envs),
        "seed": int(seed),
        "device": device,
        "ppo_hyperparams": PPO_HYPERPARAMS,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    checkpoint_callback = CheckpointCallback(
        save_freq=max(250_000 // n_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix="ppo_racing",
    )
    try:
        model.learn(total_timesteps=int(total_steps), callback=checkpoint_callback, progress_bar=False)
        model.save(run_dir / "model.zip")
    finally:
        vec_env.close()
    return model

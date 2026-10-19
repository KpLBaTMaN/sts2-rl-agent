"""Masked PPO for the SpireFormer entity policy (CleanRL-style, single file).

SB3's MaskablePPO only accepts fixed-size vector inputs, so the entity policy
gets its own small trainer.  Hyperparameters come from the same experiment
config as the flat agent (learning rate, n_steps, batch size, epochs, gamma,
entropy), and SB3's defaults are used for the rest, so the two can be compared
fairly.  Results are written with the same file names as the flat runner.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from sts2_env.entity.model import SpireFormer, SpireFormerConfig, masked_logits
from sts2_env.entity.observation import OBS_SPEC
from sts2_env.training.environment import make_env

# SB3 PPO defaults, kept identical so only the policy differs from the flat baseline.
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
MODEL_FORMAT = "spireformer-v2"


def stack_observations(observations: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.stack([obs[key] for obs in observations]) for key in OBS_SPEC}


def to_tensors(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value, device=device) for key, value in obs.items()}


class EntityAgent:
    """Wraps a SpireFormer with the ``predict`` API the runner's evaluator uses."""

    def __init__(self, model: SpireFormer, device: torch.device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def predict(self, obs, action_masks, deterministic=True):
        batch = to_tensors(stack_observations([obs]), self.device)
        mask = torch.as_tensor(np.asarray(action_masks, dtype=bool)[None], device=self.device)
        logits, _ = self.model(batch)
        logits = masked_logits(logits, mask)
        action = logits.argmax(-1) if deterministic else Categorical(logits=logits).sample()
        return int(action.item()), None

    def save(self, path: Path, num_timesteps: int, optimizer: torch.optim.Optimizer | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format": MODEL_FORMAT,
            "model_config": self.model.describe(),
            "state_dict": self.model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "num_timesteps": num_timesteps,
        }, path)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> tuple["EntityAgent", dict]:
        device = torch.device(device)
        saved = torch.load(path, map_location=device, weights_only=False)
        if saved.get("format") != MODEL_FORMAT:
            raise ValueError(f"{path} is not a {MODEL_FORMAT} SpireFormer model (retrain older models)")
        cfg = {k: saved["model_config"][k] for k in ("d_model", "layers", "heads", "dropout", "n_actions")}
        model = SpireFormer(SpireFormerConfig(**cfg)).to(device)
        model.load_state_dict(saved["state_dict"])
        model.eval()
        return cls(model, device), saved


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def train_entity(config: dict, directory: Path, evaluate, save_evaluation, project_path) -> EntityAgent:
    """Train SpireFormer with masked PPO; mirrors the flat runner's outputs."""
    train, evaluation, spec = config["training"], config["evaluation"], config["model"]
    device = _resolve_device(train["device"])
    torch.manual_seed(train["model_seed"])
    np.random.seed(train["model_seed"])

    n_envs, n_steps = train["n_envs"], train["n_steps"]
    envs = [make_env(config, worker=i) for i in range(n_envs)]
    n_actions = int(envs[0].action_space.n)
    model = SpireFormer(SpireFormerConfig(d_model=spec["d_model"], layers=spec["layers"],
                                          heads=spec["heads"], n_actions=n_actions)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train["learning_rate"], eps=1e-5)
    agent = EntityAgent(model, device)
    num_timesteps = 0
    if config["resume_from"]:
        loaded, saved = EntityAgent.load(project_path(config["resume_from"]), device)
        if loaded.model.config != model.config:
            raise ValueError(f"Resume model {loaded.model.config} does not match config {model.config}")
        model.load_state_dict(loaded.model.state_dict())
        if saved.get("optimizer"):
            optimizer.load_state_dict(saved["optimizer"])
            for group in optimizer.param_groups:
                group["lr"] = train["learning_rate"]
        num_timesteps = saved["num_timesteps"]
    print(f"SpireFormer: {model.describe()} on {device}", flush=True)

    writer = SummaryWriter(str(directory / "tb_logs" / "SpireFormer_1"))
    buffers = {key: np.zeros((n_steps, n_envs, *shape), dtype) for key, (shape, dtype) in OBS_SPEC.items()}
    masks = np.zeros((n_steps, n_envs, n_actions), dtype=bool)
    actions = np.zeros((n_steps, n_envs), dtype=np.int64)
    log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
    rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
    dones = np.zeros((n_steps, n_envs), dtype=np.float32)
    values = np.zeros((n_steps, n_envs), dtype=np.float32)

    current = [env.reset()[0] for env in envs]
    current_done = np.zeros(n_envs, dtype=np.float32)
    recent: list[tuple[float, float, float]] = []  # (success, episode length, floor)
    episode_lengths = np.zeros(n_envs, dtype=np.int64)
    best_score = (-1.0, -1.0)  # (win rate, average floor): floors break ties before any wins
    next_eval = num_timesteps + evaluation["every_steps"]
    next_checkpoint = num_timesteps + train["checkpoint_every"]
    target = num_timesteps + train["total_timesteps"]
    started = time.perf_counter()
    start_steps = num_timesteps
    interrupted = False

    try:
        while num_timesteps < target:
            model.eval()
            for step in range(n_steps):
                obs = stack_observations(current)
                step_masks = np.stack([env.env.action_masks().astype(bool) for env in envs])
                for key in OBS_SPEC:
                    buffers[key][step] = obs[key]
                masks[step] = step_masks
                dones[step] = current_done
                with torch.no_grad():
                    logits, value = model(to_tensors(obs, device))
                    dist = Categorical(logits=masked_logits(logits, torch.as_tensor(step_masks, device=device)))
                    action = dist.sample()
                actions[step] = action.cpu().numpy()
                log_probs[step] = dist.log_prob(action).cpu().numpy()
                values[step] = value.cpu().numpy()

                for i, env in enumerate(envs):
                    next_obs, reward, terminated, truncated, info = env.step(int(actions[step, i]))
                    rewards[step, i] = reward
                    episode_lengths[i] += 1
                    current_done[i] = float(terminated or truncated)
                    if terminated or truncated:
                        recent.append((float(info["is_success"]), float(episode_lengths[i]), float(info["floor"])))
                        episode_lengths[i] = 0
                        next_obs, _ = env.reset()
                    current[i] = next_obs
                num_timesteps += n_envs

            with torch.no_grad():
                _, last_value = model(to_tensors(stack_observations(current), device))
                last_value = last_value.cpu().numpy()
            advantages = np.zeros_like(rewards)
            gae = np.zeros(n_envs, dtype=np.float32)
            for step in reversed(range(n_steps)):
                next_value = last_value if step == n_steps - 1 else values[step + 1]
                next_nonterminal = 1.0 - (current_done if step == n_steps - 1 else dones[step + 1])
                delta = rewards[step] + train["gamma"] * next_value * next_nonterminal - values[step]
                gae = delta + train["gamma"] * GAE_LAMBDA * next_nonterminal * gae
                advantages[step] = gae
            returns = advantages + values

            flat = {key: torch.as_tensor(value.reshape(n_steps * n_envs, *value.shape[2:]), device=device)
                    for key, value in buffers.items()}
            b_masks = torch.as_tensor(masks.reshape(-1, n_actions), device=device)
            b_actions = torch.as_tensor(actions.reshape(-1), device=device)
            b_log_probs = torch.as_tensor(log_probs.reshape(-1), device=device)
            b_advantages = torch.as_tensor(advantages.reshape(-1), device=device)
            b_returns = torch.as_tensor(returns.reshape(-1), device=device)

            model.train()
            stats = []
            total = n_steps * n_envs
            for _ in range(train["n_epochs"]):
                order = torch.randperm(total, device=device)
                for start in range(0, total, train["batch_size"]):
                    idx = order[start:start + train["batch_size"]]
                    logits, value = model({key: tensor[idx] for key, tensor in flat.items()})
                    dist = Categorical(logits=masked_logits(logits, b_masks[idx]))
                    new_log_prob = dist.log_prob(b_actions[idx])
                    ratio = (new_log_prob - b_log_probs[idx]).exp()
                    adv = b_advantages[idx]
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                    policy_loss = torch.max(-adv * ratio, -adv * ratio.clamp(1 - CLIP_RANGE, 1 + CLIP_RANGE)).mean()
                    value_loss = 0.5 * (b_returns[idx] - value).pow(2).mean()
                    entropy = dist.entropy().mean()
                    loss = policy_loss + VALUE_COEF * value_loss - train["ent_coef"] * entropy
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    optimizer.step()
                    with torch.no_grad():
                        clipped = ((ratio - 1).abs() > CLIP_RANGE).float().mean()
                    stats.append((policy_loss.item(), value_loss.item(), entropy.item(), clipped.item()))
            model.eval()

            policy_loss, value_loss, entropy, clip_fraction = np.mean(stats, axis=0)
            window = recent[-200:]
            win_rate = float(np.mean([w for w, _, _ in window])) if window else 0.0
            mean_length = float(np.mean([n for _, n, _ in window])) if window else 0.0
            mean_floor = float(np.mean([f for _, _, f in window])) if window else 0.0
            fps = (num_timesteps - start_steps) / (time.perf_counter() - started)
            for name, value in (("train/policy_loss", policy_loss), ("train/value_loss", value_loss),
                                ("train/entropy", entropy), ("train/clip_fraction", clip_fraction),
                                ("rollout/success_rate", win_rate), ("rollout/ep_len_mean", mean_length),
                                ("rollout/floor_mean", mean_floor),
                                ("time/fps", fps)):
                writer.add_scalar(name, value, num_timesteps)
            floor_text = "" if config["task"] == "combat" else f"| floor {mean_floor:5.2f} "
            print(f"steps {num_timesteps:>9,} | train win {win_rate:6.1%} {floor_text}| ep len {mean_length:6.1f} "
                  f"| entropy {entropy:.3f} | {fps:,.0f} steps/s", flush=True)

            if num_timesteps >= next_checkpoint:
                agent.save(directory / "checkpoints" / f"spireformer_{num_timesteps}_steps.pt", num_timesteps, optimizer)
                next_checkpoint += train["checkpoint_every"]
            if num_timesteps >= next_eval:
                summary, _ = evaluate(config, evaluation["episodes"], model=agent)
                writer.add_scalar("eval/success_rate", summary["win_rate"], num_timesteps)
                print(f"  eval: {summary['wins']}/{summary['episodes']} wins"
                      + ("" if config["task"] == "combat" else f", average floor {summary['average_floor']:.2f}"), flush=True)
                score = (summary["win_rate"], summary["average_floor"])
                if score > best_score:
                    best_score = score
                    agent.save(directory / "best_model" / "best_model.pt", num_timesteps)
                next_eval += evaluation["every_steps"]
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted; saving the current model.", flush=True)
    finally:
        writer.close()
        for env in envs:
            env.close()

    agent.save(directory / ("interrupted_model.pt" if interrupted else "final_model.pt"), num_timesteps, optimizer)
    if interrupted:
        return agent

    summary, rows = evaluate(config, evaluation["final_episodes"], model=agent)
    summary.update(training_seconds=time.perf_counter() - started,
                   requested_timesteps=train["total_timesteps"],
                   actual_new_timesteps=num_timesteps - start_steps,
                   model="final_model.pt", evaluation_mode=evaluation["mode"],
                   policy="spireformer", parameters=model.describe()["parameters"])
    save_evaluation(directory, "final-results", summary, rows)
    best = directory / "best_model" / "best_model.pt"
    if best.exists():
        best_agent, _ = EntityAgent.load(best, device)
        best_summary, best_rows = evaluate(config, evaluation["final_episodes"], model=best_agent)
        save_evaluation(directory, "best-results", best_summary, best_rows)
    (directory / "model.json").write_text(json.dumps(model.describe(), indent=2), encoding="utf-8")
    print(f"\nFinal: {summary['wins']}/{summary['episodes']} wins ({summary['win_rate']:.1%})", flush=True)
    print(f"Results and models: {directory}", flush=True)
    return agent

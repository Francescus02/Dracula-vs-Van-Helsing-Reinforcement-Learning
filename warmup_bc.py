"""
Behavioral Cloning warm-up for Dracula vs Van Helsing.

Pipeline:
  1. Collect demonstrations from an ordered sequence of teachers:
       Greedy (10k) -> V1 (20k) -> V2 (30k) -> V3 (100k) -> V4 (800k)
  2. Train a MaskablePPO policy with supervised cross-entropy on (obs, mask, action).
  3. Evaluate the BC policy against all 6 benchmark bots.

Design notes:
  - Architecture [512, 256] (2 layers). The 3-layer version [512, 512, 256] was
    oversized for ~1M samples (2.1 samples/param vs the recommended 5-10). The
    same architecture will be used for PPO fine-tuning, so weights transfer.
  - Train/val split 90/10 gives honest accuracy. Training accuracy alone is
    misleading for a 216k-parameter model on 1M samples.
  - Sequential phases with decreasing LR preserve weaker teachers' priors while
    installing the stronger target policy.
  - Demonstrations are cached to .npz to avoid re-collecting on debug re-runs.

Usage:
    python warmup_bc.py --role 0 --eval-games 10000
    python warmup_bc.py --role 1 --eval-games 10000
    python warmup_bc.py --role 0 --skip-eval  # fast smoke test
"""
import argparse
import copy
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sb3_contrib import MaskablePPO

from dvvh_env import DraculaVsVanHelsingEnv
from bots import (
    BaseBot, RandomBot, GreedyHeuristicBot, TacticalRefinedBot,
    TacticalRefinedBot_V2, TacticalRefinedBot_V3, TacticalRefinedBot_V4,
)
from rl_wrappers import DraculaVsVanHelsingMaskEnv, SB3PolicyBotAdapter


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class WarmupConfig:
    ego_player: int = 0

    # Demonstration collection
    n_steps_per_teacher: List[int] = None

    # BC training
    bc_epochs_per_teacher: List[int] = None
    bc_lr_per_teacher: List[float] = None
    bc_batch_size: int = 256
    bc_weight_decay: float = 1e-5
    bc_val_fraction: float = 0.10
    bc_grad_clip: float = 0.5
    bc_early_stop_patience: int = 10

    # Architecture (must match train_curriculum.py for weight transfer)
    net_arch: List[int] = None
    activation: str = "relu"

    # Evaluation
    eval_games_per_bot: int = 10000

    # Paths
    seed: int = 42
    save_root: str = "./checkpoints"
    cache_root: str = "./bc_cache"
    history_root: str = "./bc_history"

    def __post_init__(self):
        if self.n_steps_per_teacher is None:
            self.n_steps_per_teacher = [10_000, 20_000, 30_000, 100_000, 800_000]
        if self.bc_epochs_per_teacher is None:
            self.bc_epochs_per_teacher = [30, 30, 30, 40, 60]
        if self.bc_lr_per_teacher is None:
            self.bc_lr_per_teacher = [3e-4, 3e-4, 3e-4, 2.5e-4, 2e-4]
        if self.net_arch is None:
            self.net_arch = [512, 256]


# =============================================================================
# HELPERS
# =============================================================================
def _activation_module(name: str) -> type:
    return {"relu": nn.ReLU, "tanh": nn.Tanh}[name]


def _teacher_sequence(seed: int) -> List[Tuple[str, BaseBot]]:
    """Ordered list of (name, bot). Weakest first, strongest last."""
    return [
        ("Greedy", GreedyHeuristicBot(seed=seed + 102)),
        ("V1",     TacticalRefinedBot(seed=seed + 103)),
        ("V2",     TacticalRefinedBot_V2(seed=seed + 104)),
        ("V3",     TacticalRefinedBot_V3(seed=seed + 106)),
        ("V4",     TacticalRefinedBot_V4(seed=seed + 105)),
    ]


def _opponent_pool(seed: int, exclude: str) -> Dict[str, BaseBot]:
    """All bots except the current teacher. Used as opponents during collection."""
    pool = {
        "Random": RandomBot(seed=seed + 1),
        "Greedy": GreedyHeuristicBot(seed=seed + 2),
        "V1":     TacticalRefinedBot(seed=seed + 3),
        "V2":     TacticalRefinedBot_V2(seed=seed + 4),
        "V3":     TacticalRefinedBot_V3(seed=seed + 6),
        "V4":     TacticalRefinedBot_V4(seed=seed + 5),
    }
    pool.pop(exclude, None)
    return pool


# =============================================================================
# DEMONSTRATION COLLECTION
# =============================================================================
def collect_demonstrations(
    teacher_name: str,
    teacher_bot: BaseBot,
    n_steps: int,
    ego_player: int,
    seed: int,
    opponent_pool: Dict[str, BaseBot],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    env = DraculaVsVanHelsingEnv(seed=seed, track_stats=False)
    rng = np.random.default_rng(seed)
    opp_names = list(opponent_pool.keys())

    obs_buf, mask_buf, act_buf = [], [], []
    collected = 0
    ep_count = 0

    while collected < n_steps:
        opp_name = opp_names[rng.integers(len(opp_names))]
        opp_bot = opponent_pool[opp_name]
        obs, info = env.reset()
        ep_count += 1

        while not env.done and collected < n_steps:
            p = info["current_player"]
            if p == ego_player:
                mask = info["action_mask"].astype(bool)
                action = teacher_bot.select_action(env, obs, info)
                assert mask[action], (
                    f"Teacher {teacher_name} produced illegal action {action}"
                )
                obs_buf.append(obs.copy())
                mask_buf.append(mask.copy())
                act_buf.append(int(action))
                collected += 1
            else:
                action = opp_bot.select_action(env, obs, info)
            obs, _, _, info = env.step(action)

    print(
        f"  [{teacher_name:8s}] collected {collected:7d} steps "
        f"over {ep_count:5d} episodes"
    )
    return (
        np.stack(obs_buf).astype(np.float32),
        np.stack(mask_buf).astype(bool),
        np.array(act_buf, dtype=np.int64),
    )


def collect_or_load(
    cfg: WarmupConfig,
    teacher_name: str,
    teacher_bot: BaseBot,
    n_steps: int,
    phase_idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load demonstrations from cache if available; otherwise collect and save."""
    role = "Dracula" if cfg.ego_player == 0 else "VanHelsing"
    cache_dir = os.path.join(cfg.cache_root, role)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir, f"teacher_{teacher_name}_{n_steps}.npz"
    )

    if os.path.exists(cache_path):
        print(f"\n[CACHE HIT] {cache_path}")
        data = np.load(cache_path)
        return data["obs"], data["masks"], data["acts"]

    print(f"\n[COLLECT] Teacher = {teacher_name} | target = {n_steps} steps")
    opp_pool = _opponent_pool(cfg.seed, exclude=teacher_name)
    obs, masks, acts = collect_demonstrations(
        teacher_name=teacher_name,
        teacher_bot=teacher_bot,
        n_steps=n_steps,
        ego_player=cfg.ego_player,
        seed=cfg.seed + 1000 * (phase_idx + 1),
        opponent_pool=opp_pool,
    )

    np.savez_compressed(cache_path, obs=obs, masks=masks, acts=acts)
    print(f"  [CACHE SAVE] {cache_path}")
    return obs, masks, acts


# =============================================================================
# BC TRAINING
# =============================================================================
def build_policy_for_bc(cfg: WarmupConfig) -> MaskablePPO:
    """Create an untrained MaskablePPO whose policy we will train with BC."""
    dummy_env = DraculaVsVanHelsingMaskEnv(ego_player=cfg.ego_player, seed=cfg.seed)
    act_fn = _activation_module(cfg.activation)
    model = MaskablePPO(
        "MlpPolicy",
        dummy_env,
        policy_kwargs=dict(net_arch=cfg.net_arch, activation_fn=act_fn),
        device="cpu",
        verbose=0,
        seed=cfg.seed,
    )
    return model


@th.no_grad()
def _evaluate_loader(policy, loader, device: str) -> Tuple[float, float]:
    """Returns (mean loss, accuracy) on the given loader."""
    policy.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for b_obs, b_mask, b_act in loader:
        b_obs = b_obs.to(device)
        b_mask = b_mask.to(device)
        b_act = b_act.to(device)
        dist = policy.get_distribution(b_obs, action_masks=b_mask)
        log_prob = dist.log_prob(b_act)
        loss = -log_prob.mean()

        pred = dist.distribution.probs.argmax(dim=-1)
        total_correct += (pred == b_act).sum().item()
        total_loss += loss.item() * b_obs.shape[0]
        total_n += b_obs.shape[0]
    policy.train()
    return total_loss / max(1, total_n), total_correct / max(1, total_n)


def _snapshot_state(policy) -> dict:
    return {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}


def _restore_state(policy, state: dict, device) -> None:
    policy.load_state_dict({k: v.to(device) for k, v in state.items()})


def train_bc_phase(
    model: MaskablePPO,
    obs: np.ndarray,
    masks: np.ndarray,
    acts: np.ndarray,
    cfg: WarmupConfig,
    phase_name: str,
    epochs: int,
    lr: float,
) -> dict:
    """Supervised cross-entropy with train/val split, LR schedule, early stopping."""
    policy = model.policy
    device = next(policy.parameters()).device
    policy.train()

    # Reproducible train/val split
    n = len(obs)
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(n)
    n_val = int(cfg.bc_val_fraction * n)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    train_ds = TensorDataset(
        th.as_tensor(obs[train_idx]),
        th.as_tensor(masks[train_idx]),
        th.as_tensor(acts[train_idx]),
    )
    val_ds = TensorDataset(
        th.as_tensor(obs[val_idx]),
        th.as_tensor(masks[val_idx]),
        th.as_tensor(acts[val_idx]),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.bc_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.bc_batch_size, shuffle=False)

    optimizer = th.optim.Adam(
        policy.parameters(), lr=lr, weight_decay=cfg.bc_weight_decay
    )
    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_state = _snapshot_state(policy)
    epochs_since_improvement = 0
    history = {
        "phase": phase_name, "epochs": [], "train_loss": [],
        "train_acc": [], "val_loss": [], "val_acc": [], "lr": [],
    }

    for epoch in range(epochs):
        policy.train()
        total_loss, total_correct, total_n = 0.0, 0, 0
        for b_obs, b_mask, b_act in train_loader:
            b_obs = b_obs.to(device)
            b_mask = b_mask.to(device)
            b_act = b_act.to(device)

            dist = policy.get_distribution(b_obs, action_masks=b_mask)
            log_prob = dist.log_prob(b_act)
            loss = -log_prob.mean()

            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(policy.parameters(), cfg.bc_grad_clip)
            optimizer.step()

            pred = dist.distribution.probs.argmax(dim=-1)
            total_correct += (pred == b_act).sum().item()
            total_loss += loss.item() * b_obs.shape[0]
            total_n += b_obs.shape[0]

        train_loss = total_loss / total_n
        train_acc = total_correct / total_n
        val_loss, val_acc = _evaluate_loader(policy, val_loader, device)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["epochs"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = _snapshot_state(policy)
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"    [{phase_name:6s}] ep {epoch+1:2d}/{epochs} | "
                f"train {train_loss:.4f}/{train_acc*100:5.2f}% | "
                f"val {val_loss:.4f}/{val_acc*100:5.2f}% | "
                f"lr {current_lr:.2e}"
            )

        if epochs_since_improvement >= cfg.bc_early_stop_patience:
            print(
                f"    [{phase_name:6s}] early stop at epoch {epoch+1} "
                f"(no val improvement for {cfg.bc_early_stop_patience} epochs)"
            )
            break

    _restore_state(policy, best_state, device)
    print(f"    [{phase_name:6s}] best val accuracy: {best_val_acc*100:.2f}%")
    return history


# =============================================================================
# EVALUATION
# =============================================================================
def evaluate_bc_agent(
    model: MaskablePPO,
    ego_player: int,
    eval_games_per_bot: int,
    seed: int,
) -> Dict[str, float]:
    env = DraculaVsVanHelsingEnv(seed=seed + 999, track_stats=False)
    agent = SB3PolicyBotAdapter(model, deterministic=True, name="BC_Agent")
    opponents = {
        "Random": RandomBot(seed=seed + 1),
        "Greedy": GreedyHeuristicBot(seed=seed + 2),
        "V1":     TacticalRefinedBot(seed=seed + 3),
        "V2":     TacticalRefinedBot_V2(seed=seed + 4),
        "V3":     TacticalRefinedBot_V3(seed=seed + 6),
        "V4":     TacticalRefinedBot_V4(seed=seed + 5),
    }
    results = {}
    for name, bot in opponents.items():
        wins = 0
        for _ in range(eval_games_per_bot):
            obs, info = env.reset()
            done = False
            while not done:
                p = info["current_player"]
                if p == ego_player:
                    action = agent.select_action(env, obs, info)
                else:
                    action = bot.select_action(env, obs, info)
                obs, _, done, info = env.step(action)
            if env.winner == ego_player:
                wins += 1
        wr = wins / eval_games_per_bot
        results[name] = wr
        print(f"    vs {name:7s}: {wr*100:6.2f}% ({wins}/{eval_games_per_bot})")
    return results


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", type=int, choices=[0, 1], required=True)
    parser.add_argument("--eval-games", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-root", type=str, default="./checkpoints")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    cfg = WarmupConfig(
        ego_player=args.role,
        eval_games_per_bot=args.eval_games,
        seed=args.seed,
        save_root=args.save_root,
    )
    role_name = "Dracula" if cfg.ego_player == 0 else "VanHelsing"

    print("=" * 80)
    print(f"BC WARM-UP — {role_name}")
    print(f"Architecture: {cfg.net_arch} ({cfg.activation})")
    print(f"Teachers:     Greedy -> V1 -> V2 -> V3 -> V4")
    print(f"Steps:        {cfg.n_steps_per_teacher}")
    print(f"Epochs:       {cfg.bc_epochs_per_teacher}")
    print(f"LRs:          {cfg.bc_lr_per_teacher}")
    print("=" * 80)

    # 1. Collect or load demonstrations
    teacher_seq = _teacher_sequence(cfg.seed)
    datasets = []
    for i, ((name, bot), n_steps) in enumerate(
        zip(teacher_seq, cfg.n_steps_per_teacher)
    ):
        obs, masks, acts = collect_or_load(cfg, name, bot, n_steps, phase_idx=i)
        datasets.append((name, obs, masks, acts))

    # 2. Build policy and train sequentially
    model = build_policy_for_bc(cfg)
    all_history = []
    for i, (name, obs, masks, acts) in enumerate(datasets):
        epochs = cfg.bc_epochs_per_teacher[i]
        lr = cfg.bc_lr_per_teacher[i]
        print(
            f"\n[BC TRAIN] Phase = {name} | samples = {len(obs)} | "
            f"epochs = {epochs} | lr = {lr}"
        )
        history = train_bc_phase(
            model, obs, masks, acts, cfg,
            phase_name=name, epochs=epochs, lr=lr,
        )
        all_history.append(history)

    # 3. Save final model
    out_dir = os.path.join(cfg.save_root, role_name)
    os.makedirs(out_dir, exist_ok=True)
    bc_path = os.path.join(out_dir, f"bc_warmup_{role_name}_model")
    model.save(bc_path)
    print(f"\n[SAVE] BC model -> {bc_path}")

    # 4. Save training history
    os.makedirs(cfg.history_root, exist_ok=True)
    hist_path = os.path.join(cfg.history_root, f"bc_{role_name}_history.json")
    with open(hist_path, "w") as f:
        json.dump(all_history, f, indent=2)
    print(f"[SAVE] Training history -> {hist_path}")

    # 5. Evaluate
    if not args.skip_eval:
        print(f"\n[EVAL] {cfg.eval_games_per_bot} games per bot ({role_name})")
        results = evaluate_bc_agent(
            model,
            ego_player=cfg.ego_player,
            eval_games_per_bot=cfg.eval_games_per_bot,
            seed=cfg.seed,
        )
        print("\n[EVAL SUMMARY]")
        for name, wr in results.items():
            print(f"  {name:7s}: {wr*100:6.2f}%")

        eval_path = os.path.join(
            cfg.history_root, f"bc_{role_name}_eval.json"
        )
        with open(eval_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[SAVE] Eval results -> {eval_path}")
    else:
        print("\n[SKIP] Evaluation skipped")


if __name__ == "__main__":
    main()
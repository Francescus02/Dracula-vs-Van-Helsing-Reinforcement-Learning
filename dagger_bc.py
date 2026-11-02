"""
DAgger (Dataset Aggregation) fine-tuning for Dracula vs Van Helsing.

DAgger solves the compounding-error problem of Behavioral Cloning:
  - BC is trained on states visited by the EXPERT (d_pi_expert).
  - At inference, the learner visits d_pi_learner != d_pi_expert.
  - Errors compound quadratically with the horizon.

DAgger iteratively:
  1. Rolls out the LEARNER (visits d_pi_learner).
  2. Queries the EXPERT on every learner-visited state (label a*).
  3. Aggregates (s, a*) into the training dataset.
  4. Retrains (warm-start, decreasing LR).

Reuses warmup_bc.py's training and evaluation infrastructure.

Usage:
    python dagger_bc.py --role 0
    python dagger_bc.py --role 0 --iterations 2 --steps-per-iter 200000 --eval-games 5000
    python dagger_bc.py --role 1 --iterations 2
"""
import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sb3_contrib import MaskablePPO

from dvvh_env import DraculaVsVanHelsingEnv
from bots import (
    BaseBot, RandomBot, GreedyHeuristicBot, TacticalRefinedBot,
    TacticalRefinedBot_V2, TacticalRefinedBot_V3, TacticalRefinedBot_V4,
)
from rl_wrappers import SB3PolicyBotAdapter

# Reuse BC training infrastructure from warmup_bc.py
from warmup_bc import (
    WarmupConfig, train_bc_phase, _opponent_pool,
)


# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class DAggerConfig:
    ego_player: int = 0
    seed: int = 42
    save_root: str = "./checkpoints"
    cache_root: str = "./bc_cache"
    history_root: str = "./bc_history"

    # DAgger loop
    dagger_iterations: int = 2
    dagger_steps_per_iter: int = 200_000
    dagger_epochs: int = 40
    dagger_lr_schedule: List[float] = None
    dagger_min_improvement: float = 0.01  # stop if < 1pp improvement per iteration

    # Expert
    expert_name: str = "V4"

    # Evaluation
    eval_games_per_bot: int = 5000
    eval_bots: List[str] = None

    # Architecture / BC hyperparams (must match warmup_bc.py)
    net_arch: List[int] = None
    activation: str = "relu"
    bc_batch_size: int = 256
    bc_weight_decay: float = 1e-5
    bc_val_fraction: float = 0.10
    bc_grad_clip: float = 0.5
    bc_early_stop_patience: int = 10

    # BC checkpoint path (auto-derived if None)
    bc_checkpoint: Optional[str] = None

    def __post_init__(self):
        if self.dagger_lr_schedule is None:
            self.dagger_lr_schedule = [1e-4, 5e-5, 3e-5]
        if self.eval_bots is None:
            self.eval_bots = ["Greedy", "V1", "V2", "V3", "V4"]
        if self.net_arch is None:
            self.net_arch = [512, 256]
        if self.bc_checkpoint is None:
            role = "Dracula" if self.ego_player == 0 else "VanHelsing"
            self.bc_checkpoint = os.path.join(
                self.save_root, role, f"bc_warmup_{role}_model"
            )


# =============================================================================
# HELPERS
# =============================================================================
def _role_name(ego_player: int) -> str:
    return "Dracula" if ego_player == 0 else "VanHelsing"


def _make_eval_pool(seed: int) -> Dict[str, BaseBot]:
    return {
        "Random": RandomBot(seed=seed + 1),
        "Greedy": GreedyHeuristicBot(seed=seed + 2),
        "V1":     TacticalRefinedBot(seed=seed + 3),
        "V2":     TacticalRefinedBot_V2(seed=seed + 4),
        "V3":     TacticalRefinedBot_V3(seed=seed + 6),
        "V4":     TacticalRefinedBot_V4(seed=seed + 5),
    }


def _make_expert(cfg: DAggerConfig) -> BaseBot:
    pool = _make_eval_pool(cfg.seed + 500)
    return pool[cfg.expert_name]


def _make_bc_config(cfg: DAggerConfig) -> WarmupConfig:
    """Build a WarmupConfig with DAgger hyperparams to pass to train_bc_phase."""
    return WarmupConfig(
        ego_player=cfg.ego_player,
        seed=cfg.seed,
        save_root=cfg.save_root,
        cache_root=cfg.cache_root,
        history_root=cfg.history_root,
        net_arch=cfg.net_arch,
        activation=cfg.activation,
        bc_batch_size=cfg.bc_batch_size,
        bc_weight_decay=cfg.bc_weight_decay,
        bc_val_fraction=cfg.bc_val_fraction,
        bc_grad_clip=cfg.bc_grad_clip,
        bc_early_stop_patience=cfg.bc_early_stop_patience,
    )


# =============================================================================
# DATASET LOADING (BC cache)
# =============================================================================
def load_bc_dataset(cfg: DAggerConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all .npz files from the BC cache directory and concatenate."""
    role = _role_name(cfg.ego_player)
    cache_dir = os.path.join(cfg.cache_root, role)
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(
            f"BC cache not found at {cache_dir}. "
            f"Run 'python warmup_bc.py --role {cfg.ego_player}' first."
        )

    files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".npz"))
    if not files:
        raise FileNotFoundError(
            f"No .npz files in {cache_dir}. Run warmup_bc.py first."
        )

    print(f"\n[LOAD] BC cache from {cache_dir}")
    all_obs, all_masks, all_acts = [], [], []
    for fname in files:
        data = np.load(os.path.join(cache_dir, fname))
        n = len(data["obs"])
        all_obs.append(data["obs"])
        all_masks.append(data["masks"])
        all_acts.append(data["acts"])
        print(f"  {fname:40s} -> {n:7d} samples")

    obs = np.concatenate(all_obs, axis=0)
    masks = np.concatenate(all_masks, axis=0)
    acts = np.concatenate(all_acts, axis=0)
    print(f"  Total BC dataset: {len(obs):,} samples")
    return obs, masks, acts


# =============================================================================
# DAgger ROLLOUT
# =============================================================================
def collect_dagger_states(
    learner_model: MaskablePPO,
    expert_bot: BaseBot,
    n_steps: int,
    ego_player: int,
    seed: int,
    opponent_pool: Dict[str, BaseBot],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rollout the LEARNER (visits d_pi_learner). On every learner decision state s:
      - label a* = pi_expert(s)   [query the expert]
      - execute a_exec = pi_learner(s)   [transitions to next learner state]
      - record (s, mask, a*) in the dataset
    """
    env = DraculaVsVanHelsingEnv(seed=seed, track_stats=False)
    rng = np.random.default_rng(seed)
    opp_names = list(opponent_pool.keys())

    agent_adapter = SB3PolicyBotAdapter(
        learner_model, deterministic=False, name="DAgger_Learner"
    )

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
                # Label: expert's action on learner's state
                expert_action = expert_bot.select_action(env, obs, info)
                assert mask[expert_action], (
                    f"Expert {expert_bot.__class__.__name__} produced "
                    f"illegal action {expert_action} in phase {info['phase']}"
                )
                obs_buf.append(obs.copy())
                mask_buf.append(mask.copy())
                act_buf.append(int(expert_action))
                collected += 1
                # Execute: learner's action (visits d_pi_learner)
                exec_action = agent_adapter.select_action(env, obs, info)
            else:
                exec_action = opp_bot.select_action(env, obs, info)
            obs, _, _, info = env.step(exec_action)

    print(
        f"  [DAgger] collected {collected:7d} states over {ep_count:5d} episodes"
    )
    return (
        np.stack(obs_buf).astype(np.float32),
        np.stack(mask_buf).astype(bool),
        np.array(act_buf, dtype=np.int64),
    )


# =============================================================================
# QUICK EVAL
# =============================================================================
def quick_eval(
    model: MaskablePPO,
    ego_player: int,
    bot_names: List[str],
    games_per_bot: int,
    seed: int,
) -> Dict[str, float]:
    env = DraculaVsVanHelsingEnv(seed=seed + 999, track_stats=False)
    agent = SB3PolicyBotAdapter(model, deterministic=True, name="DAgger_Agent")
    pool = _make_eval_pool(seed)

    results = {}
    for name in bot_names:
        bot = pool[name]
        wins = 0
        for _ in range(games_per_bot):
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
        wr = wins / games_per_bot
        results[name] = wr
        print(f"    vs {name:7s}: {wr*100:6.2f}% ({wins}/{games_per_bot})")
    return results


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", type=int, choices=[0, 1], required=True)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--steps-per-iter", type=int, default=200_000)
    parser.add_argument("--eval-games", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-root", type=str, default="./checkpoints")
    parser.add_argument("--bc-checkpoint", type=str, default=None)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    cfg = DAggerConfig(
        ego_player=args.role,
        seed=args.seed,
        save_root=args.save_root,
        dagger_iterations=args.iterations,
        dagger_steps_per_iter=args.steps_per_iter,
        eval_games_per_bot=args.eval_games,
        bc_checkpoint=args.bc_checkpoint,
    )
    role = _role_name(cfg.ego_player)

    print("=" * 80)
    print(f"DAgger FINE-TUNING — {role}")
    print(f"BC checkpoint:      {cfg.bc_checkpoint}")
    print(f"Expert:             {cfg.expert_name}")
    print(f"Iterations:         {cfg.dagger_iterations}")
    print(f"Steps per iter:     {cfg.dagger_steps_per_iter:,}")
    print(f"Epochs per iter:    {cfg.dagger_epochs}")
    print(f"LR schedule:        {cfg.dagger_lr_schedule}")
    print(f"Architecture:       {cfg.net_arch} ({cfg.activation})")
    print("=" * 80)

    # Verify BC checkpoint exists (handle missing .zip extension)
    bc_path = cfg.bc_checkpoint
    if not os.path.exists(bc_path) and os.path.exists(bc_path + ".zip"):
        bc_path = bc_path + ".zip"
    if not os.path.exists(bc_path):
        raise FileNotFoundError(
            f"BC checkpoint not found: {bc_path}. "
            f"Run 'python warmup_bc.py --role {cfg.ego_player}' first."
        )

    # Load BC model
    print(f"\n[LOAD] BC model: {bc_path}")
    model = MaskablePPO.load(bc_path, device="cpu")
    bc_cfg = _make_bc_config(cfg)

    # Load BC dataset
    init_obs, init_masks, init_acts = load_bc_dataset(cfg)

    # Setup expert and opponent pool (excludes the expert)
    expert = _make_expert(cfg)
    opp_pool = _opponent_pool(cfg.seed, exclude=cfg.expert_name)
    print(f"[SETUP] Expert = {cfg.expert_name}")
    print(f"[SETUP] Opponents during rollout: {list(opp_pool.keys())}")

    # Baseline evaluation
    print(f"\n[BASELINE EVAL] {cfg.eval_games_per_bot} games per bot")
    baseline = quick_eval(
        model, cfg.ego_player, cfg.eval_bots, cfg.eval_games_per_bot, cfg.seed
    )
    baseline_wr_greedy = baseline.get("Greedy", 0.0)
    print(f"[BASELINE] vs Greedy: {baseline_wr_greedy*100:.2f}%")

    # DAgger loop
    out_dir = os.path.join(cfg.save_root, role)
    os.makedirs(out_dir, exist_ok=True)
    history = {"role": role, "baseline": baseline, "iterations": []}
    baseline_v4 = baseline.get("V4", 0.0)
    prev_primary_metric = baseline_v4
    stopped_early = False

    for it in range(1, cfg.dagger_iterations + 1):
        print(f"\n{'=' * 80}\n[DAgger ITER {it}/{cfg.dagger_iterations}]\n{'=' * 80}")

        # Rollout: collect states + expert labels
        new_obs, new_masks, new_acts = collect_dagger_states(
            learner_model=model,
            expert_bot=expert,
            n_steps=cfg.dagger_steps_per_iter,
            ego_player=cfg.ego_player,
            seed=cfg.seed + 10000 * it,
            opponent_pool=opp_pool,
        )

        DAGGER_OVERSAMPLE = 2
        agg_obs = np.concatenate(
            [init_obs] + [new_obs] * DAGGER_OVERSAMPLE, axis=0
        )
        agg_masks = np.concatenate(
            [init_masks] + [new_masks] * DAGGER_OVERSAMPLE, axis=0
        )
        agg_acts = np.concatenate(
            [init_acts] + [new_acts] * DAGGER_OVERSAMPLE, axis=0
        )
        dagger_pct = (
            DAGGER_OVERSAMPLE * len(new_obs) / len(agg_obs) * 100
        )
        print(
            f"  [AGGREGATE] new: {len(new_obs):,} "
            f"(x{DAGGER_OVERSAMPLE} = {DAGGER_OVERSAMPLE*len(new_obs):,}, "
            f"{dagger_pct:.1f}% of {len(agg_obs):,} total)"
        )

        # Retrain (warm-start, decreasing LR)
        lr = cfg.dagger_lr_schedule[min(it - 1, len(cfg.dagger_lr_schedule) - 1)]
        print(f"  [RETRAIN] epochs={cfg.dagger_epochs}, lr={lr:.2e}")
        phase_hist = train_bc_phase(
            model, agg_obs, agg_masks, agg_acts,
            bc_cfg, phase_name=f"DAgger_iter{it}",
            epochs=cfg.dagger_epochs, lr=lr,
        )

        # Save per-iteration checkpoint
        ckpt_path = os.path.join(out_dir, f"dagger_{role}_iter{it}")
        model.save(ckpt_path)
        print(f"  [SAVE] {ckpt_path}.zip")

        # Evaluate
        if not args.skip_eval:
            print(f"  [EVAL] {cfg.eval_games_per_bot} games per bot")
            results = quick_eval(
                model, cfg.ego_player, cfg.eval_bots,
                cfg.eval_games_per_bot, cfg.seed,
            )
        else:
            results = {}

        iter_record = {
            "iteration": it,
            "lr": lr,
            "dataset_size": int(len(agg_obs)),
            "eval": results,
            "phase_history_summary": {
                "epochs": phase_hist["epochs"],
                "best_val_acc": float(max(phase_hist["val_acc"])),
                "final_train_acc": float(phase_hist["train_acc"][-1]),
            },
        }
        history["iterations"].append(iter_record)

        if results and "V4" in results:
            current_metric = results["V4"]
            improvement = current_metric - prev_primary_metric
            print(
                f"  [ITER {it}] vs V4 (primary): {current_metric*100:.2f}% "
                f"(improvement: {improvement*100:+.2f} pp, "
                f"threshold: {cfg.dagger_min_improvement*100:+.2f} pp)"
            )
            if "Greedy" in results:
                print(
                    f"  [ITER {it}] vs Greedy (diag): "
                    f"{results['Greedy']*100:.2f}%"
                )
            if improvement < cfg.dagger_min_improvement:
                print(
                    f"  [EARLY STOP] V4 improvement {improvement*100:+.2f} pp "
                    f"below threshold {cfg.dagger_min_improvement*100:.2f} pp."
                )
                stopped_early = True
                prev_primary_metric = current_metric
                break
            prev_primary_metric = current_metric

    # Save final checkpoint
    final_path = os.path.join(out_dir, f"dagger_{role}_final")
    model.save(final_path)
    print(f"\n[SAVE] Final DAgger model -> {final_path}.zip")

    # Save history
    os.makedirs(cfg.history_root, exist_ok=True)
    hist_path = os.path.join(cfg.history_root, f"dagger_{role}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[SAVE] History -> {hist_path}")

    # Summary
    print("\n" + "=" * 80)
    print(f"DAgger SUMMARY — {role}")
    print("=" * 80)
    print(f"  Baseline vs Greedy: {baseline_wr_greedy*100:.2f}%")
    for rec in history["iterations"]:
        it = rec["iteration"]
        parts = [f"  Iter {it}: "]
        if "Greedy" in rec["eval"]:
            parts.append(f"vs Greedy {rec['eval']['Greedy']*100:6.2f}% | ")
        if "V1" in rec["eval"]:
            parts.append(f"vs V1 {rec['eval']['V1']*100:6.2f}% | ")
        parts.append(
            f"dataset {rec['dataset_size']:>9,} | "
            f"best val {rec['phase_history_summary']['best_val_acc']*100:5.2f}%"
        )
        print("".join(parts))
    if stopped_early:
        print("  [Stopped early — improvement below threshold]")
    print("=" * 80)


if __name__ == "__main__":
    main()
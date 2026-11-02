"""
Adaptive curriculum callback with state persistence for resume.

Adds:
  - state_path: optional JSON sidecar for ema_winrates / elo / streak / best_v4.
  - Auto-load on __init__ if file exists; auto-save on every eval.
  - Everything else unchanged from the previous working version.
"""
import json
import os
from typing import Dict, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from dvvh_env import DraculaVsVanHelsingEnv
from bots import (
    BaseBot,
    RandomBot,
    GreedyHeuristicBot,
    TacticalRefinedBot,
    TacticalRefinedBot_V2,
    TacticalRefinedBot_V3,
    TacticalRefinedBot_V4,
)
from rl_wrappers import SB3PolicyBotAdapter


class AdaptiveLR:
    """
    Callable LR schedule driven by the (ratcheted) EMA win rate vs V4.

        wr <= ramp_start            : lr_high
        ramp_start < wr <= target   : geometric ramp  lr_high -> lr_high * mid_ratio
        wr > target                 : lr_high * mid_ratio * exp(-(wr-target)/tau), floored at lr_min

    SB3 calls it as f(progress_remaining); the argument is ignored.
    """

    def __init__(self, lr_high=5e-5, lr_min=5e-6, ramp_start=0.50, target=0.56,
                 mid_ratio=0.30, tau=0.03):
        self.lr_high, self.lr_min = lr_high, lr_min
        self.ramp_start, self.target = ramp_start, target
        self.mid_ratio, self.tau = mid_ratio, tau
        self.value = lr_high
        self.boost = 1.0  # >1 only during critic warm-up

    def update(self, wr_v4: float) -> float:
        if wr_v4 <= self.ramp_start:
            v = self.lr_high
        elif wr_v4 <= self.target:
            f = (wr_v4 - self.ramp_start) / max(1e-9, self.target - self.ramp_start)
            v = self.lr_high * (self.mid_ratio ** f)
        else:
            v = self.lr_high * self.mid_ratio * float(np.exp(-(wr_v4 - self.target) / self.tau))
        self.value = float(max(self.lr_min, v))
        return self.value

    def __call__(self, progress_remaining: float) -> float:
        return self.value * self.boost


class AdaptiveCurriculumAndEloCallback(BaseCallback):
    """
    Master RL Training Callback for 'Dracula vs Van Helsing'.

    Mechanisms:
      1. Reward annealing (linear decay of shaping_weight to 0).
      2. Multi-baseline evaluation every `eval_freq` steps.
      3. Progressive curriculum scheduler (tier i unlocked when tier i-1 mastered).
      4. Consecutive-mastery graduation + best-checkpoint saving.
      5. Optional JSON state persistence across resume.
    """

    ANCHOR_ELOS = {
        "Random": 1000.0,
        "Greedy": 1300.0,
        "V1_Tactical": 1400.0,
        "V2_Bayesian": 1450.0,
        "V3_Asymmetric": 1525.0,
        "V4_Minimax": 1560.0,
    }

    def __init__(
        self,
        ego_player: int,
        total_timesteps: int,
        eval_freq: int = 50_000,
        eval_games_per_bot: int = 250,
        initial_shaping_weight: float = 1.0,
        annealing_fraction: float = 0.55,
        pfsp_alpha: float = 1.5,
        min_prob_floor: float = 0.02,
        promotion_winrate_threshold: float = 0.52,
        required_consecutive_passes: int = 5,
        save_dir: str = "./checkpoints",
        deterministic_eval: bool = True,
        verbose: int = 1,
        state_path: Optional[str] = None,
        lr_controller: Optional[AdaptiveLR] = None,
        v4_eval_games: int = 500,
        unlock_thresholds: Optional[List[float]] = None,
        bc_wr_gating: bool = False,
        initial_ema: Optional[Dict[str, float]] = None,
        train_exclude: Optional[List[str]] = None,
        v4_pool_k: int = 3,
        bc_full_below: float = 0.25,
        bc_zero_above: float = 0.40,
    ):
        super().__init__(verbose)
        self.lr_controller = lr_controller
        self.v4_eval_games = v4_eval_games
        self.unlock_thresholds = unlock_thresholds or [1.00, 0.40, 0.28, 0.24, 0.20, 0.18]
        self.bc_wr_gating = bc_wr_gating
        self.bc_full_below = bc_full_below
        self.bc_zero_above = bc_zero_above
        self.unlocked_latch: Dict[str, bool] = {}
        self.best_v4_ema: float = 0.0  # kept for state-file compatibility
        self.train_exclude = set(train_exclude or [])
        self.v4_pool_k = v4_pool_k
        self.v4_hist: List[List[float]] = []  # [wins, games] per eval
        self.ego_player = ego_player
        self.role_name = "Dracula" if ego_player == 0 else "VanHelsing"
        self.total_timesteps = max(1, total_timesteps)
        self.eval_freq = eval_freq
        self.eval_games_per_bot = eval_games_per_bot
        self.deterministic_eval = deterministic_eval

        # Reward annealing
        self.initial_shaping_weight = initial_shaping_weight
        self.annealing_steps = int(self.total_timesteps * annealing_fraction)
        self.current_shaping_weight = initial_shaping_weight

        # PFSP & curriculum
        self.pfsp_alpha = pfsp_alpha
        self.min_prob_floor = min_prob_floor
        self.promotion_threshold = promotion_winrate_threshold
        self.required_consecutive_passes = required_consecutive_passes

        # Eval env & opponents
        self.eval_env = DraculaVsVanHelsingEnv(seed=999, track_stats=False)
        self.eval_env.return_obs_from_step = False  # adapter/bots recompute what they need
        self.eval_opponents: Dict[str, BaseBot] = {
            "Random": RandomBot(seed=901),
            "Greedy": GreedyHeuristicBot(seed=905),
            "V1_Tactical": TacticalRefinedBot(seed=906),
            "V2_Bayesian": TacticalRefinedBot_V2(seed=902),
            "V3_Asymmetric": TacticalRefinedBot_V3(seed=903),
            "V4_Minimax": TacticalRefinedBot_V4(seed=904),
        }

        # Tracking state
        self.ema_winrates: Dict[str, float] = {
            "Random": 0.30,
            "Greedy": 0.10,
            "V1_Tactical": 0.05,
            "V2_Bayesian": 0.05,
            "V3_Asymmetric": 0.03,
            "V4_Minimax": 0.02,
        }
        if initial_ema:
            for k, v in initial_ema.items():
                if k in self.ema_winrates:
                    self.ema_winrates[k] = float(v)
            if "V4_Minimax" in initial_ema:
                self.v4_hist = [[initial_ema["V4_Minimax"] * 1000.0, 1000.0]]
        self.agent_elo: float = 1200.0
        self.consecutive_v4_passes: int = 0
        self.best_v4_winrate: float = 0.0
        self.curriculum_graduated: bool = False

        self.save_dir = os.path.join(save_dir, self.role_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self._last_eval_step = 0
        self._last_anneal_step = 0

        # State persistence
        self.state_path = state_path
        if self.state_path and os.path.exists(self.state_path):
            self._load_state()

    # -------------------------------------------------------------------------
    # STATE PERSISTENCE
    # -------------------------------------------------------------------------
    def _save_state(self) -> None:
        if not self.state_path:
            return
        state = {
            "ema_winrates": dict(self.ema_winrates),
            "agent_elo": float(self.agent_elo),
            "consecutive_v4_passes": int(self.consecutive_v4_passes),
            "best_v4_winrate": float(self.best_v4_winrate),
            "curriculum_graduated": bool(self.curriculum_graduated),
            "current_shaping_weight": float(self.current_shaping_weight),
            "last_eval_step": int(self._last_eval_step),
            "last_anneal_step": int(self._last_anneal_step),
            "unlocked_latch": dict(self.unlocked_latch),
            "best_v4_ema": float(self.best_v4_ema),
            "v4_hist": [list(map(float, h)) for h in self.v4_hist],
            "lr_value": float(self.lr_controller.value) if self.lr_controller else None,
        }
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, self.state_path)

    def _load_state(self) -> None:
        with open(self.state_path) as f:
            state = json.load(f)
        for name, val in state.get("ema_winrates", {}).items():
            if name in self.ema_winrates:
                self.ema_winrates[name] = float(val)
        self.agent_elo = float(state.get("agent_elo", self.agent_elo))
        self.consecutive_v4_passes = int(state.get("consecutive_v4_passes", 0))
        self.best_v4_winrate = float(state.get("best_v4_winrate", 0.0))
        self.curriculum_graduated = bool(state.get("curriculum_graduated", False))
        self.current_shaping_weight = float(
            state.get("current_shaping_weight", self.initial_shaping_weight)
        )
        self._last_eval_step = int(state.get("last_eval_step", 0))
        self._last_anneal_step = int(state.get("last_anneal_step", 0))
        self.unlocked_latch = {k: bool(v) for k, v in state.get("unlocked_latch", {}).items()}
        self.best_v4_ema = float(state.get("best_v4_ema", 0.0))
        if state.get("v4_hist"):
            self.v4_hist = [list(map(float, h)) for h in state["v4_hist"]]
        if self.lr_controller is not None and self.v4_hist:
            self.lr_controller.update(self._pooled_v4())
        print(f"[CALLBACK] Restored state from {self.state_path}")

    # -------------------------------------------------------------------------
    # PER-STEP
    # -------------------------------------------------------------------------
    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_anneal_step) >= 1000:
            self._last_anneal_step = self.num_timesteps
            progress = min(1.0, self.num_timesteps / max(1, self.annealing_steps))
            self.current_shaping_weight = float(
                self.initial_shaping_weight * (1.0 - progress)
            )
            self.training_env.env_method(
                "set_shaping_weight", self.current_shaping_weight
            )
            self.logger.record(
                "curriculum/shaping_weight", self.current_shaping_weight
            )

        if (self.num_timesteps - self._last_eval_step) >= self.eval_freq:
            self._last_eval_step = self.num_timesteps
            self._run_evaluation_and_update_curriculum()

        return True

    # -------------------------------------------------------------------------
    # EVALUATION
    # -------------------------------------------------------------------------
    def _run_evaluation_and_update_curriculum(self) -> None:
        policy_adapter = SB3PolicyBotAdapter(
            self.model, deterministic=self.deterministic_eval, name="CurrentPolicy"
        )

        current_winrates: Dict[str, float] = {}

        for opp_name, opp_bot in self.eval_opponents.items():
            wins = 0
            n_games = self.v4_eval_games if opp_name == "V4_Minimax" else self.eval_games_per_bot
            for _ in range(n_games):
                obs, info = self.eval_env.reset()
                done = False
                while not done:
                    p = info["current_player"]
                    if p == self.ego_player:
                        action = policy_adapter.select_action(self.eval_env, obs, info)
                    else:
                        action = opp_bot.select_action(self.eval_env, obs, info)
                    obs, _, done, info = self.eval_env.step(action)
                if self.eval_env.winner == self.ego_player:
                    wins += 1

            wr = wins / float(n_games)
            if opp_name == "V4_Minimax":
                self.v4_hist.append([float(wins), float(n_games)])
                self.v4_hist = self.v4_hist[-20:]
            current_winrates[opp_name] = wr
            self.ema_winrates[opp_name] = 0.60 * wr + 0.40 * self.ema_winrates[opp_name]

            anchor_elo = self.ANCHOR_ELOS.get(opp_name, 1400.0)
            expected_score = 1.0 / (
                1.0 + 10.0 ** ((anchor_elo - self.agent_elo) / 400.0)
            )
            self.agent_elo += 32.0 * (wr - expected_score)

            self.logger.record(f"eval_winrate/vs_{opp_name}", wr)
            self.logger.record(
                f"eval_winrate_ema/vs_{opp_name}", self.ema_winrates[opp_name]
            )

        self.logger.record("curriculum/agent_elo", self.agent_elo)

        new_probs = self._compute_pfsp_probabilities()
        self.training_env.env_method("set_sampling_probs", new_probs.tolist())
        for idx, opp_name in enumerate(self.eval_opponents.keys()):
            self.logger.record(f"pfsp_spawn_prob/{opp_name}", new_probs[idx])

        v4_wr = current_winrates["V4_Minimax"]

        # Pooled V4 estimate over the last k evals (SE ~1pp instead of ~2.2pp)
        pooled_v4 = self._pooled_v4()
        self.best_v4_ema = max(self.best_v4_ema, self.ema_winrates["V4_Minimax"])
        if self.lr_controller is not None:
            new_lr = self.lr_controller.update(pooled_v4)
            self.logger.record("curriculum/adaptive_lr", new_lr)
        if self.bc_wr_gating and hasattr(self.model, "bc_scale"):
            span = max(1e-9, self.bc_zero_above - self.bc_full_below)
            self.model.bc_scale = float(np.clip((self.bc_zero_above - pooled_v4) / span, 0.0, 1.0))
            self.logger.record("curriculum/bc_scale", self.model.bc_scale)
        self.logger.record("curriculum/v4_pooled", pooled_v4)

        if pooled_v4 >= self.promotion_threshold:
            self.consecutive_v4_passes += 1
        else:
            self.consecutive_v4_passes = 0
        self.logger.record(
            "curriculum/consecutive_v4_passes", self.consecutive_v4_passes
        )

        if pooled_v4 >= self.best_v4_winrate:
            self.best_v4_winrate = pooled_v4
            best_path = os.path.join(self.save_dir, f"best_{self.role_name}_model")
            self.model.save(best_path)

        snap_path = os.path.join(
            self.save_dir, f"snapshot_{self.role_name}_{self.num_timesteps}_steps"
        )
        self.model.save(snap_path)

        if (
            not self.curriculum_graduated
            and self.consecutive_v4_passes >= self.required_consecutive_passes
        ):
            self.curriculum_graduated = True
            grad_path = os.path.join(
                self.save_dir, f"graduated_{self.role_name}_model"
            )
            self.model.save(grad_path)
            if self.verbose >= 1:
                print(
                    f"\n>>> [CURRICULUM GRADUATION!] {self.role_name} beat V4 "
                    f"(>={self.promotion_threshold*100:.1f}%) for "
                    f"{self.consecutive_v4_passes} consecutive evals. "
                    f"Saved to {grad_path} <<<\n"
                )

        # Persist callback state
        self._save_state()

        if self.verbose >= 1:
            prob_str = " | ".join(
                f"{k[:6]}:{new_probs[i]*100:4.1f}%"
                for i, k in enumerate(self.eval_opponents.keys())
            )
            expl_var = self._read_logged_scalar("train/explained_variance")
            val_loss = self._read_logged_scalar("train/value_loss")
            ent_loss = self._read_logged_scalar("train/entropy_loss")
            approx_kl = self._read_logged_scalar("train/approx_kl")

            print(
                f"[Step {self.num_timesteps:8,}] {self.role_name:10s} | "
                f"Elo: {self.agent_elo:6.1f} | "
                f"ShapeW: {self.current_shaping_weight:.3f} | "
                f"Streak: {self.consecutive_v4_passes}/{self.required_consecutive_passes}"
            )
            print(
                f"    WR -> Rnd:{current_winrates['Random']*100:5.1f}% | "
                f"Gr: {current_winrates['Greedy']*100:5.1f}% | "
                f"V1: {current_winrates['V1_Tactical']*100:5.1f}% | "
                f"V2: {current_winrates['V2_Bayesian']*100:5.1f}% | "
                f"V3: {current_winrates['V3_Asymmetric']*100:5.1f}% |"
                f"V4: {v4_wr*100:5.1f}% (pool{self.v4_pool_k}: {pooled_v4*100:5.1f}%)"
            )

            print(f"    Next Spawn -> [{prob_str}]")
            ep_rew = self._read_ep_info_stat("ep_rew_mean")
            ep_len = self._read_ep_info_stat("ep_len_mean")
            print(
                f"    Diag -> ExplVar: {expl_var:+.3f} | "
                f"ValLoss: {val_loss:.4f} | "
                f"EntLoss: {ent_loss:+.3f} | "
                f"KL: {approx_kl:.4f}"
            )
            print(
                f"    Rew  -> ep_rew_mean: {ep_rew:+.3f} | "
                f"ep_len_mean: {ep_len:.1f}"
            )

    def _pooled_v4(self) -> float:
        h = self.v4_hist[-self.v4_pool_k:]
        if not h:
            return self.ema_winrates.get("V4_Minimax", 0.0)
        w = sum(x[0] for x in h)
        g = sum(x[1] for x in h)
        return float(w / max(1.0, g))

    def _read_logged_scalar(self, key: str, default: float = float("nan")) -> float:
        val = self.logger.name_to_value.get(key, None)
        if val is None:
            return default
        try:
            return float(np.squeeze(val))
        except (TypeError, ValueError):
            return default

    def _read_ep_info_stat(self, key: str, default: float = float("nan")) -> float:
        buffer = getattr(self.model, "ep_info_buffer", None)
        if not buffer:
            return default
        try:
            if key == "ep_rew_mean":
                return float(np.mean([ep["r"] for ep in buffer]))
            if key == "ep_len_mean":
                return float(np.mean([ep["l"] for ep in buffer]))
        except (KeyError, TypeError):
            pass
        return default

    # -------------------------------------------------------------------------
    # CURRICULUM SCHEDULER
    # -------------------------------------------------------------------------
    def _compute_pfsp_probabilities(self) -> np.ndarray:
        names = list(self.eval_opponents.keys())
        n = len(names)

        unlock_thresholds = self.unlock_thresholds

        # LATCH: once an opponent is unlocked it stays unlocked (no flip-flop)
        unlocked = [True] + [False] * (n - 1)
        for i in range(1, n):
            if self.unlocked_latch.get(names[i], False) or (
                self.ema_winrates[names[i - 1]] >= unlock_thresholds[i]
            ):
                unlocked[i] = True
                self.unlocked_latch[names[i]] = True

        for i, name in enumerate(names):
            if name in self.train_exclude:
                unlocked[i] = False  # evaluated, never sampled for training
        raw_weights = np.zeros(n, dtype=np.float64)
        for i, name in enumerate(names):
            if not unlocked[i]:
                raw_weights[i] = 0.0
                continue
            p = np.clip(self.ema_winrates[name], 0.02, 0.98)
            raw_weights[i] = p * ((1.0 - p) ** 2.0)

        if np.sum(raw_weights) <= 1e-9:
            raw_weights[[i for i, nm in enumerate(names) if nm not in self.train_exclude][0]] = 1.0

        probs = raw_weights / np.sum(raw_weights)

        n_unlocked = sum(unlocked)
        if n_unlocked > 0:
            floor_scale = self.min_prob_floor * n_unlocked
            for i in range(n):
                if unlocked[i]:
                    probs[i] = (1.0 - floor_scale) * probs[i] + self.min_prob_floor
                else:
                    probs[i] = 0.0

        return probs / np.sum(probs)
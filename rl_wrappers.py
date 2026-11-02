from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from functools import partial

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


class SB3PolicyBotAdapter(BaseBot):
    """
    Wraps a trained or frozen Stable-Baselines3 / sb3-contrib model (e.g., MaskablePPO)
    into the standard `BaseBot` interface so past neural network checkpoints can sit
    inside the exact same Opponent Pool alongside Heuristic Bots during Self-Play.
    """

    def __init__(
        self,
        model: Any,
        deterministic: bool = False,
        name: str = "FrozenRLCheckpoint",
    ):
        self.model = model
        self.deterministic = deterministic
        self.name = name

    def select_action(
        self, env: DraculaVsVanHelsingEnv, obs: np.ndarray, info: Dict
    ) -> int:
        active_p = info["current_player"]
        assert active_p == env.current_player, (
            f"Adapter called for player {active_p} but env.current_player={env.current_player}"
        )
        mask = info["action_mask"].astype(bool)
        opp_obs = env.get_observation(active_p)
        action, _ = self.model.predict(
            opp_obs, action_masks=mask, deterministic=self.deterministic
        )
        return int(action)


class DraculaVsVanHelsingMaskEnv(gym.Env):
    """
    Single-Agent Perspective Gymnasium Wrapper for 'Dracula vs Van Helsing'.
    Compatible with `sb3-contrib` (MaskablePPO / MaskableDQN) and `SubprocVecEnv`.

    Features:
      1. Automatic Opponent Turn Execution:
         - When `ego_player == 0` (Dracula), Van Helsing (`player 1`) is played automatically
           inside `reset()` and `step()` until control returns to Dracula or the match ends.
         - When `ego_player == 1` (Van Helsing), Dracula (`player 0`) is played automatically.
      2. Multi-Step Reward Aggregation & Role-Specific Reward Annealing:
         - Accumulates intermediate shaping deltas (`step_vampires_added`, `step_hp_lost`)
           across all internal transitions (including opponent turns that trigger round end).
         - Applies linear/cosine annealing weight `shaping_weight` (1.0 -> 0.0) controlled
           externally by the training callback across multiprocess workers.
      3. Dynamic Opponent Pool Sampling (PFSP Ready):
         - Samples an opponent from `opponent_pool` at the start of every episode according
           to `sampling_probs`, which can be updated live via `set_sampling_probs()`.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        ego_player: int = 0,
        opponent_dict: Optional[Dict[str, BaseBot]] = None,
        initial_probs: Optional[List[float]] = None,
        shaping_weight: float = 1.0,
        vampire_shaping_coef: float = 0.06,
        hp_shaping_coef: float = 0.04,
        seed: Optional[int] = None,
        always_dracula_first: bool = False,
        track_internal_stats: bool = False,
    ):
        """
        Args:
            ego_player: 0 to train Dracula, 1 to train Van Helsing.
            opponent_dict: Mapping of opponent names to `BaseBot` instances.
            initial_probs: Initial sampling distribution over `opponent_dict.keys()`.
            shaping_weight: Initial annealing multiplier gamma(t) in [0.0, 1.0].
            vampire_shaping_coef: Intermediate reward per vampire added (positive for Dracula, negative for VH).
            hp_shaping_coef: Intermediate reward per HP lost by Dracula (negative for Dracula, positive for VH).
            seed: Random seed for reproducibility across parallel workers.
        """
        super().__init__()
        assert ego_player in (0, 1), "ego_player must be 0 (Dracula) or 1 (Van Helsing)."

        self.ego_player = ego_player
        self.opp_player = 1 - ego_player
        self.rng = np.random.default_rng(seed)

        self.env = DraculaVsVanHelsingEnv(
            seed=seed,
            always_dracula_first=always_dracula_first,
            track_stats=track_internal_stats,
        )

        # The wrapper never needs the observation returned by the raw env.step()
        self.env.return_obs_from_step = False

        # Define Gymnasium Spaces
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._probe_obs_dim(),),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(DraculaVsVanHelsingEnv.ACTION_DIM)

        # Configure Opponent Pool
        if opponent_dict is None:
            opponent_dict = self._build_default_curriculum_pool(seed)

        self.opponent_names: List[str] = list(opponent_dict.keys())
        self.opponent_bots: List[BaseBot] = list(opponent_dict.values())

        if initial_probs is None:
            n = len(self.opponent_bots)
            self.sampling_probs = np.ones(n, dtype=np.float64) / max(1, n)
        else:
            probs = np.array(initial_probs, dtype=np.float64)
            self.sampling_probs = probs / np.sum(probs)

        self.current_opp_idx: int = 0
        self.current_opp_bot: BaseBot = self.opponent_bots[0]
        self.current_opp_name: str = self.opponent_names[0]

        # Reward Shaping & Annealing Parameters
        self.shaping_weight = float(shaping_weight)
        self.vampire_shaping_coef = float(vampire_shaping_coef)
        self.hp_shaping_coef = float(hp_shaping_coef)

        # Episode-level trackers for SB3 Monitor / TensorBoard
        self.ep_shaping_reward = 0.0
        self.ep_terminal_reward = 0.0
        self.ep_ego_steps = 0

    def _probe_obs_dim(self) -> int:
        return int(self.env.get_observation(0).shape[0])

    @staticmethod
    def _build_default_curriculum_pool(seed: Optional[int] = None) -> Dict[str, BaseBot]:
        base_s = seed if seed is not None else 42
        return {
            "Random": RandomBot(seed=base_s + 1),
            "Greedy": GreedyHeuristicBot(seed=base_s + 5),
            "V1_Tactical": TacticalRefinedBot(seed=base_s + 6),
            "V2_Bayesian": TacticalRefinedBot_V2(seed=base_s + 2),
            "V3_Asymmetric": TacticalRefinedBot_V3(seed=base_s + 3),
            "V4_Minimax": TacticalRefinedBot_V4(seed=base_s + 4),
        }

    # =========================================================================
    # INTER-PROCESS CONTROL METHODS (CALLED VIA VEC_ENV.ENV_METHOD)
    # =========================================================================
    def set_shaping_weight(self, new_weight: float) -> None:
        """Updates the reward annealing factor gamma(t) in [0.0, 1.0]."""
        self.shaping_weight = float(np.clip(new_weight, 0.0, 1.0))

    def set_sampling_probs(self, new_probs: Union[List[float], np.ndarray]) -> None:
        """Updates the PFSP opponent sampling distribution dynamically during training."""
        arr = np.array(new_probs, dtype=np.float64)
        assert len(arr) == len(self.opponent_bots), "Probability length mismatch!"
        total = np.sum(arr)
        if total > 0:
            self.sampling_probs = arr / total
        else:
            self.sampling_probs = np.ones(len(arr), dtype=np.float64) / len(arr)

    def add_or_update_opponent(
        self, name: str, bot: BaseBot, prob_weight: float = 0.20
    ) -> None:
        """
        Inserts a new opponent (e.g., a past Self-Play checkpoint wrapped in SB3PolicyBotAdapter)
        into the pool and re-normalizes sampling probabilities.
        """
        if name in self.opponent_names:
            idx = self.opponent_names.index(name)
            self.opponent_bots[idx] = bot
        else:
            self.opponent_names.append(name)
            self.opponent_bots.append(bot)
            appended = np.append(self.sampling_probs * (1.0 - prob_weight), prob_weight)
            self.sampling_probs = appended / np.sum(appended)

    def get_opponent_names(self) -> List[str]:
        return list(self.opponent_names)

    # =========================================================================
    # REQUIRED BY SB3-CONTRIB (MASKABLE PPO / MASKABLE DQN)
    # =========================================================================
    def action_masks(self) -> np.ndarray:
        """
        Returns a boolean array of shape (10,) indicating legal actions for `ego_player`.
        Must be called when `self.env.current_player == self.ego_player` (or terminal).
        """
        if not self.env.done:
            assert self.env.current_player == self.ego_player, (
                f"action_masks called for ego={self.ego_player} "
                f"but current_player={self.env.current_player}"
            )
        mask = self.env.get_legal_actions()
        if self.env.done or np.sum(mask) == 0:
            fallback = np.zeros(DraculaVsVanHelsingEnv.ACTION_DIM, dtype=bool)
            fallback[0] = True
            return fallback
        return mask.astype(bool)

    # =========================================================================
    # GYMNASIUM CORE INTERFACE: RESET & STEP
    # =========================================================================
    def _compute_shaping_delta(self, vamps_added: int, hp_lost: int) -> float:
        """
        Computes zero-sum intermediate reward aligned with each role's objective:
          - Dracula gains from converting vampires and loses slightly from losing HP.
          - Van Helsing gains from dealing HP damage and loses from allowing vampires.
        """
        if self.shaping_weight <= 1e-6 or (vamps_added == 0 and hp_lost == 0):
            return 0.0

        drac_delta = (
            (vamps_added * self.vampire_shaping_coef)
            - (hp_lost * self.hp_shaping_coef)
        )
        role_delta = drac_delta if self.ego_player == 0 else -drac_delta
        return float(self.shaping_weight * role_delta)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Sample opponent for this match from the active PFSP distribution
        self.current_opp_idx = int(
            self.rng.choice(len(self.opponent_bots), p=self.sampling_probs)
        )
        self.current_opp_bot = self.opponent_bots[self.current_opp_idx]
        self.current_opp_name = self.opponent_names[self.current_opp_idx]

        self.ep_shaping_reward = 0.0
        self.ep_terminal_reward = 0.0
        self.ep_ego_steps = 0

        obs, info = self.env.reset(seed=seed)

        # Fast-forward through opponent turns if opponent acts first at Round 1 start
        while not self.env.done and self.env.current_player != self.ego_player:
            opp_obs = self.env.get_observation(self.opp_player)
            opp_action = self.current_opp_bot.select_action(self.env, opp_obs, info)
            obs, _, _, info = self.env.step(opp_action)

        ego_obs = self.env.get_observation(self.ego_player)
        return ego_obs, self._build_wrapper_info(info)

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executes `ego_player`'s action, then automatically steps `opp_player` until
        either `ego_player` is active again or the game terminates.
        """
        assert not self.env.done, "Cannot call step() on a terminated environment!"
        assert self.env.current_player == self.ego_player, (
            f"Turn desynchronization: expected player {self.ego_player}, got {self.env.current_player}"
        )

        self.ep_ego_steps += 1
        accumulated_shaping = 0.0

        # 1. Execute ego_player's action
        _, _, done, info = self.env.step(int(action))
        accumulated_shaping += self._compute_shaping_delta(
            info["step_vampires_added"], info["step_hp_lost"]
        )

        # 2. Automatically loop opponent turns until control returns to ego_player or game ends
        while not done and self.env.current_player != self.ego_player:
            opp_obs = self.env.get_observation(self.opp_player)
            opp_action = self.current_opp_bot.select_action(self.env, opp_obs, info)
            _, _, done, info = self.env.step(opp_action)
            accumulated_shaping += self._compute_shaping_delta(
                info["step_vampires_added"], info["step_hp_lost"]
            )

        # 3. Compute terminal reward (+1.0 for ego win, -1.0 for ego loss)
        terminal_reward = 0.0
        if done:
            terminal_reward = float(info["rewards_both"][self.ego_player])

        self.ep_shaping_reward += accumulated_shaping
        self.ep_terminal_reward += terminal_reward
        total_step_reward = terminal_reward + accumulated_shaping

        ego_obs = self.env.get_observation(self.ego_player)
        wrapper_info = self._build_wrapper_info(info)

        return ego_obs, float(total_step_reward), bool(done), False, wrapper_info

    def _build_wrapper_info(self, raw_info: Dict[str, Any]) -> Dict[str, Any]:
        info = dict(raw_info)
        info["ego_player"] = self.ego_player
        info["opponent_name"] = self.current_opp_name
        info["shaping_weight"] = self.shaping_weight
        if self.env.done:
            info["ego_won"] = bool(self.env.winner == self.ego_player)
            info["ep_shaping_reward"] = self.ep_shaping_reward
            info["ep_terminal_reward"] = self.ep_terminal_reward
            info["ep_ego_steps"] = self.ep_ego_steps
        return info


# =============================================================================
# FACTORY HELPER FOR SUBPROC_VEC_ENV / DUMMY_VEC_ENV
# =============================================================================

def _make_env_impl(rank, ego_player, shaping_weight, base_seed, initial_probs):
    worker_seed = base_seed + rank * 1000
    return DraculaVsVanHelsingMaskEnv(
        ego_player=ego_player,
        initial_probs=initial_probs,
        shaping_weight=shaping_weight,
        seed=worker_seed,
    )

def make_masked_dvvh_env(
    rank: int,
    ego_player: int = 0,
    shaping_weight: float = 1.0,
    base_seed: int = 42,
    initial_probs: Optional[List[float]] = None,
) -> Callable[[], gym.Env]:
    """
    Returns a picklable initializer thunk for SubprocVecEnv / DummyVecEnv.
    Uses functools.partial so it works with any multiprocessing start_method.
    """
    return partial(
        _make_env_impl,
        rank=rank,
        ego_player=ego_player,
        shaping_weight=shaping_weight,
        base_seed=base_seed,
        initial_probs=initial_probs,
    )


# =============================================================================
# SELF-DIAGNOSTIC TEST (VERIFIES GYMNASIUM & MASKING COMPLIANCE)
# =============================================================================
if __name__ == "__main__":
    print("Running self-diagnostic check on DraculaVsVanHelsingMaskEnv...")
    for role_id, role_name in [(0, "Dracula"), (1, "Van Helsing")]:
        env = DraculaVsVanHelsingMaskEnv(ego_player=role_id, seed=123)
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape, f"Invalid obs shape: {obs.shape}"
        done = False
        steps = 0
        while not done:
            mask = env.action_masks()
            legal = np.flatnonzero(mask)
            action = int(np.random.choice(legal))
            obs, rew, done, trunc, info = env.step(action)
            steps += 1
        print(
            f"  [OK] Role={role_name:11s} | Opponent={info['opponent_name']:13s} | "
            f"Ego Steps={steps:2d} | Ego Won={str(info['ego_won']):5s} | "
            f"TermRew={info['ep_terminal_reward']:+.1f} | ShapeRew={info['ep_shaping_reward']:+.3f}"
        )
    print("All wrapper checks passed 100%!")
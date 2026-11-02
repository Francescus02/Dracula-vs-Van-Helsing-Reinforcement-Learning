"""
Shared utilities for the self-play phase (league training, probes, final validation).

* Player specs   : plain dicts that are picklable and can be rebuilt inside any worker process
      {"kind": "bot",   "bot": "V4" | "V4_eps10" | "Mix_V2V3V4" | ...}
      {"kind": "model", "path": "<zip>", "deterministic": True/False}
* matchup()      : plays Dracula-spec vs Van-Helsing-spec games, optionally on several processes
* LeagueMaskEnv  : gym env whose opponent pool can be replaced live (league members = bots or models)
"""
import math
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from dvvh_env import DraculaVsVanHelsingEnv
from rl_wrappers import DraculaVsVanHelsingMaskEnv
from robustness_eval import make_condition  # bots, noisy bots, mixtures

ROLE_NAMES = {0: "Dracula", 1: "VanHelsing"}


# ----------------------------------------------------------------------------
# misc helpers
# ----------------------------------------------------------------------------
class ConstLR:
    """Picklable constant learning-rate schedule."""

    def __init__(self, value: float):
        self.value = float(value)

    def __call__(self, progress_remaining: float) -> float:
        return self.value


def load_model(path: str):
    """Load a MaskablePPO checkpoint without un-pickling the (custom) LR/clip schedule objects."""
    from sb3_contrib import MaskablePPO

    if not (os.path.exists(path) or os.path.exists(path + ".zip")):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return MaskablePPO.load(
        path,
        device="cpu",
        custom_objects={
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.1,
        },
    )


_MODEL_CACHE: Dict[str, Any] = {}


def build_player(spec: Dict[str, Any], seed: int = 0):
    """Instantiate a bot-like object with .select_action(env, obs, info) from a spec dict."""
    if spec["kind"] == "bot":
        return make_condition(spec["bot"], seed)
    import torch

    torch.set_num_threads(1)
    from rl_wrappers import SB3PolicyBotAdapter

    path = spec["path"]
    if path not in _MODEL_CACHE:
        _MODEL_CACHE[path] = load_model(path)
    return SB3PolicyBotAdapter(
        _MODEL_CACHE[path], deterministic=bool(spec.get("deterministic", True)), name=spec.get("id", path)
    )


# ----------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------
def ci95(p: float, n: int) -> float:
    return 1.96 * math.sqrt(max(p * (1 - p), 1e-12) / max(n, 1))


def z_diff(p1: float, n1: int, p0: float, n0: int) -> float:
    """z statistic of p1 - p0 for two independent proportions."""
    se = math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    return (p1 - p0) / se if se > 0 else 0.0


def diff_ci95(p1: float, n1: int, p0: float, n0: int) -> float:
    return 1.96 * math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)


# ----------------------------------------------------------------------------
# matchups (Dracula spec vs Van Helsing spec)
# ----------------------------------------------------------------------------
def _play_chunk(args):
    spec_d, spec_v, games, seed = args
    env = DraculaVsVanHelsingEnv(seed=seed, track_stats=False)
    env.return_obs_from_step = False
    d_bot = build_player(spec_d, seed + 11)
    v_bot = build_player(spec_v, seed + 12)
    wins = conv = steps = 0
    rounds: Dict[int, int] = {}
    for _ in range(games):
        _, info = env.reset()
        n = 0
        while not env.done:
            bot = d_bot if env.current_player == 0 else v_bot
            _, _, _, info = env.step(bot.select_action(env, None, info))
            n += 1
        wins += int(env.winner == 0)
        conv += int("convert" in str(env.win_reason).lower())
        steps += n
        rounds[env.round_num] = rounds.get(env.round_num, 0) + 1
    return wins, games, steps, conv, rounds


def matchup(spec_d, spec_v, games: int, seed: int = 1234, workers: int = 1) -> Dict[str, Any]:
    """Returns dict(d_wins, games, d_wr, steps, conv, rounds). Dracula = spec_d, Van Helsing = spec_v."""
    workers = max(1, min(workers, games))
    per = math.ceil(games / workers)
    jobs, left = [], games
    for i in range(workers):
        g = min(per, left)
        if g <= 0:
            break
        jobs.append((spec_d, spec_v, g, seed + 1000 * i))
        left -= g
    if len(jobs) == 1:
        results = [_play_chunk(jobs[0])]
    else:
        with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
            results = list(ex.map(_play_chunk, jobs))
    wins = sum(r[0] for r in results)
    n = sum(r[1] for r in results)
    rounds: Dict[int, int] = {}
    for r in results:
        for k, v in r[4].items():
            rounds[k] = rounds.get(k, 0) + v
    return dict(d_wins=wins, games=n, d_wr=wins / n, steps=sum(r[2] for r in results) / n,
                conv=sum(r[3] for r in results) / n, rounds=rounds)


def role_wr(spec_role, spec_other, role: int, games: int, seed: int, workers: int) -> Dict[str, Any]:
    """Win rate of `spec_role` playing `role` against `spec_other` playing the other seat."""
    if role == 0:
        res = matchup(spec_role, spec_other, games, seed, workers)
        wr, w = res["d_wr"], res["d_wins"]
    else:
        res = matchup(spec_other, spec_role, games, seed, workers)
        wr, w = 1.0 - res["d_wr"], res["games"] - res["d_wins"]
    return dict(wr=wr, wins=w, games=res["games"], ci=ci95(wr, res["games"]))


# ----------------------------------------------------------------------------
# league environment
# ----------------------------------------------------------------------------
class LeagueMaskEnv(DraculaVsVanHelsingMaskEnv):
    """DraculaVsVanHelsingMaskEnv whose opponent pool is described by picklable specs."""

    def __init__(self, ego_player: int, league_specs: List[Dict], probs: Sequence[float], seed: int):
        placeholder = {"_init": build_player({"kind": "bot", "bot": "V3"}, seed)}
        super().__init__(ego_player=ego_player, opponent_dict=placeholder, shaping_weight=0.0, seed=seed)
        self._cache: Dict[str, Any] = {}
        self._seed0 = seed
        self.set_league(league_specs, probs)

    def set_league(self, specs: List[Dict], probs: Sequence[float]) -> None:
        names, bots, new_cache = [], [], {}
        for i, sp in enumerate(specs):
            key = f"{sp['id']}|{sp.get('path', sp.get('bot'))}|{sp.get('deterministic', '')}"
            bot = self._cache.get(key)
            if bot is None:
                bot = build_player(sp, self._seed0 + 17 * (i + 1))
            new_cache[key] = bot
            names.append(sp["id"])
            bots.append(bot)
        self._cache = new_cache  # drops members that left the league
        self.opponent_names, self.opponent_bots = names, bots
        arr = np.asarray(probs, dtype=np.float64)
        self.sampling_probs = arr / arr.sum()
        # the episode currently running keeps its opponent; the new pool applies from the next reset


def _make_league_env(rank, ego_player, specs, probs, base_seed):
    return LeagueMaskEnv(ego_player, specs, probs, seed=base_seed + rank * 1000)


def make_league_env(rank: int, ego_player: int, specs: List[Dict], probs: Sequence[float], base_seed: int):
    return partial(_make_league_env, rank=rank, ego_player=ego_player, specs=specs,
                   probs=list(probs), base_seed=base_seed)


# ----------------------------------------------------------------------------
# PFSP sampling with floors
# ----------------------------------------------------------------------------
def pfsp_probs(win_rates: Sequence[float], floors: Sequence[float], min_weight: float = 0.05) -> np.ndarray:
    """weight_i ∝ (1 - wr_i), then every member gets at least floors[i] probability mass."""
    w = np.maximum(1.0 - np.asarray(win_rates, dtype=np.float64), min_weight)
    p = w / w.sum()
    f = np.asarray(floors, dtype=np.float64)
    if f.sum() >= 1.0:
        return f / f.sum()
    for _ in range(50):
        low = p < f - 1e-12
        if not low.any():
            break
        p[low] = f[low]
        rest = ~low
        if rest.any():
            p[rest] *= (1.0 - p[low].sum()) / p[rest].sum()
    return p / p.sum()
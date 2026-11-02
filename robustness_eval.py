"""
Generalisation / robustness test: does the agent keep its edge over V4 when the OPPONENT is
not exactly the deterministic bot it was trained against?

For every opponent condition the SAME opponent is played by (a) the trained agent and (b) V4
(as a role-matched baseline). The relevant quantity is the DIFFERENCE agent - V4:
  * V4 clean condition      -> the V4 mirror (is the agent really above V4?)
  * perturbed conditions    -> if diff stays >= 0 the agent is not just exploiting a deterministic bot.

    python robustness_eval.py --role 0 --model ./checkpoints/Dracula/ppo_Dracula_final.zip --games 3000
    python robustness_eval.py --role 0 --agent-bot V4 --games 50 --conditions V4   # self-test
"""
import argparse, math
import numpy as np
from dvvh_env import DraculaVsVanHelsingEnv
from bots import (GreedyHeuristicBot, TacticalRefinedBot, TacticalRefinedBot_V2,
                  TacticalRefinedBot_V3, TacticalRefinedBot_V4)

BOTS = {"Greedy": GreedyHeuristicBot, "V1": TacticalRefinedBot, "V2": TacticalRefinedBot_V2,
        "V3": TacticalRefinedBot_V3, "V4": TacticalRefinedBot_V4}


class NoisyBot:
    """With probability eps plays a uniformly random legal action, else the base bot."""
    def __init__(self, base, eps, seed):
        self.base, self.eps, self.rng = base, eps, np.random.default_rng(seed)

    def select_action(self, env, obs, info):
        if self.rng.random() < self.eps:
            return int(self.rng.choice(np.flatnonzero(env.get_legal_actions())))
        return self.base.select_action(env, obs, info)


class MixBot:
    """Picks one of the bots at every decision (assumes the bots are stateless given env)."""
    def __init__(self, bots, seed):
        self.bots, self.rng = bots, np.random.default_rng(seed)

    def select_action(self, env, obs, info):
        return self.bots[int(self.rng.integers(len(self.bots)))].select_action(env, obs, info)


def make_condition(name, seed):
    if name in BOTS:
        return BOTS[name](seed=seed)
    if name.startswith("V4_eps"):
        return NoisyBot(BOTS["V4"](seed=seed), int(name[6:]) / 100.0, seed + 1)
    if name.startswith("V3_eps"):
        return NoisyBot(BOTS["V3"](seed=seed), int(name[6:]) / 100.0, seed + 1)
    if name == "Mix_V3V4":
        return MixBot([BOTS["V3"](seed=seed), BOTS["V4"](seed=seed + 1)], seed + 2)
    if name == "Mix_V2V3V4":
        return MixBot([BOTS["V2"](seed=seed), BOTS["V3"](seed=seed + 1), BOTS["V4"](seed=seed + 2)], seed + 3)
    raise ValueError(name)


def play(ego_bot, role, opp_bot, games, seed):
    env = DraculaVsVanHelsingEnv(seed=seed, track_stats=False)
    env.return_obs_from_step = False
    wins = 0
    for _ in range(games):
        _, info = env.reset()
        while not env.done:
            bot = ego_bot if env.current_player == role else opp_bot
            _, _, _, info = env.step(bot.select_action(env, None, info))
        wins += int(env.winner == role)
    return wins


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", type=int, default=0)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--agent-bot", type=str, default=None, help="use a heuristic bot as the 'agent' (self-test)")
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--conditions", nargs="+",
                    default=["V4", "V3", "V4_eps10", "V4_eps25", "Mix_V3V4", "Mix_V2V3V4"])
    a = ap.parse_args()

    if a.agent_bot:
        agent = BOTS[a.agent_bot](seed=1)
    else:
        from sb3_contrib import MaskablePPO
        from rl_wrappers import SB3PolicyBotAdapter
        agent = SB3PolicyBotAdapter(MaskablePPO.load(a.model, device="cpu"), deterministic=True, name="Agent")
    baseline = BOTS["V4"](seed=1)

    print(f"{'condition':12s} | {'agent':>8s} | {'V4 base':>8s} | {'diff (95% CI)':>20s} | z")
    for i, cond in enumerate(a.conditions):
        w_a = play(agent, a.role, make_condition(cond, 100 + i), a.games, a.seed + i)
        w_b = play(baseline, a.role, make_condition(cond, 100 + i), a.games, a.seed + 50 + i)
        pa, pb = w_a / a.games, w_b / a.games
        se = math.sqrt(pa * (1 - pa) / a.games + pb * (1 - pb) / a.games)
        d = pa - pb
        print(f"{cond:12s} | {pa*100:7.2f}% | {pb*100:7.2f}% | {d*100:+6.2f} +/- {1.96*se*100:4.2f} pp | {d/se if se else 0:+.2f}", flush=True)
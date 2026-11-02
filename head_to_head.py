"""
Head-to-head: trained Dracula agent (player 0) vs trained Van Helsing agent (player 1).

Why: both agents were only ever evaluated against heuristic bots. Before self-play we need to know
(1) how they compare with each other and (2) whether strength composes "transitively" through V4:

    predicted Dracula WR = a + (1 - b) - m
        a = WR of the Dracula agent vs V4 (as Van Helsing)
        b = WR of the Van Helsing agent vs V4 (as Dracula)
        m = WR of V4 (Dracula) vs V4 (Van Helsing)   [mirror]

A large deviation from the prediction means non-transitive strength (one agent exploits V4-specific
features that the other does not share) -> the self-play league must then be more diverse.

    python head_to_head.py --dracula ./checkpoints/Dracula/FROZEN_Dracula_v1.zip \
        --vh ./checkpoints/VanHelsing/FROZEN_VH_v1.zip --games 5000 --mode both \
        --a 0.5291 --b 0.5244 --m 0.5091 --pred-sd 0.008
"""
import argparse, math
import numpy as np
from dvvh_env import DraculaVsVanHelsingEnv
from bots import (GreedyHeuristicBot, TacticalRefinedBot, TacticalRefinedBot_V2,
                  TacticalRefinedBot_V3, TacticalRefinedBot_V4)

BOTS = {"Greedy": GreedyHeuristicBot, "V1": TacticalRefinedBot, "V2": TacticalRefinedBot_V2,
        "V3": TacticalRefinedBot_V3, "V4": TacticalRefinedBot_V4}


def run(dracula, vh, games, seed):
    env = DraculaVsVanHelsingEnv(seed=seed, track_stats=False)
    env.return_obs_from_step = False
    wins, steps, conv, rounds = 0, 0, 0, {}
    for _ in range(games):
        _, info = env.reset()
        n = 0
        while not env.done:
            bot = dracula if env.current_player == 0 else vh
            _, _, _, info = env.step(bot.select_action(env, None, info))
            n += 1
        wins += int(env.winner == 0)
        steps += n
        conv += int("convert" in str(env.win_reason).lower())
        rounds[env.round_num] = rounds.get(env.round_num, 0) + 1
    return wins, steps / games, conv / games, rounds


def report(tag, wins, games, steps, conv, rounds, pred, pred_sd):
    p = wins / games
    se = math.sqrt(p * (1 - p) / games)
    print(f"\n[{tag}] Dracula agent WR vs Van Helsing agent: {p*100:.2f}% +/- {1.96*se*100:.2f} (95% CI, {games} games)")
    print(f"   avg decisions/game {steps:.1f} | ended by conversion {conv*100:.1f}% | end-round dist "
          + ", ".join(f"R{k}:{v/games*100:.0f}%" for k, v in sorted(rounds.items())))
    if pred is not None:
        z = (p - pred) / math.sqrt(se ** 2 + pred_sd ** 2)
        verdict = "consistent with transitivity" if abs(z) < 3 else "NON-TRANSITIVE: deviation is significant"
        print(f"   predicted {pred*100:.2f}% | observed - predicted = {(p-pred)*100:+.2f} pp | z = {z:+.2f} -> {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dracula", type=str, default=None)
    ap.add_argument("--vh", type=str, default=None)
    ap.add_argument("--dracula-bot", type=str, default=None, help="self-test: heuristic bot as Dracula")
    ap.add_argument("--vh-bot", type=str, default=None, help="self-test: heuristic bot as Van Helsing")
    ap.add_argument("--games", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=31337)
    ap.add_argument("--mode", choices=["det", "stoch", "both"], default="both",
                    help="det = argmax actions; stoch = sampled actions (how the policies are trained)")
    ap.add_argument("--a", type=float, default=None, help="Dracula agent WR vs V4")
    ap.add_argument("--b", type=float, default=None, help="Van Helsing agent WR vs V4")
    ap.add_argument("--m", type=float, default=None, help="V4 vs V4 mirror, Dracula WR")
    ap.add_argument("--pred-sd", type=float, default=0.008)
    a = ap.parse_args()

    pred = (a.a + (1 - a.b) - a.m) if None not in (a.a, a.b, a.m) else None
    if a.dracula_bot or a.vh_bot:
        d_bot = BOTS[a.dracula_bot](seed=1)
        v_bot = BOTS[a.vh_bot](seed=2)
        w, st, cv, rd = run(d_bot, v_bot, a.games, a.seed)
        report("bots", w, a.games, st, cv, rd, pred, a.pred_sd)
    else:
        from sb3_contrib import MaskablePPO
        from rl_wrappers import SB3PolicyBotAdapter
        md = MaskablePPO.load(a.dracula, device="cpu")
        mv = MaskablePPO.load(a.vh, device="cpu")
        modes = ["det", "stoch"] if a.mode == "both" else [a.mode]
        for mode in modes:
            det = mode == "det"
            d_bot = SB3PolicyBotAdapter(md, deterministic=det, name="Dracula")
            v_bot = SB3PolicyBotAdapter(mv, deterministic=det, name="VanHelsing")
            w, st, cv, rd = run(d_bot, v_bot, a.games, a.seed)
            report(mode, w, a.games, st, cv, rd, pred if det else None, a.pred_sd)
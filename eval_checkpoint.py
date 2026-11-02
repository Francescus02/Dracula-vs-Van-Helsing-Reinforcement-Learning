"""
Precise evaluation of one or more PPO/DAgger checkpoints (default 5000 games vs V4 and V3).
Use it to choose the final checkpoint: the in-training 'best_' file is picked on noisy
evaluations (winner's curse), this script gives a 95% CI.

    python eval_checkpoint.py --role 0 --models ./checkpoints/Dracula/ppo_Dracula_final.zip \
        ./checkpoints/Dracula/snapshot_Dracula_1000000_steps.zip --games 5000 --opponents V4 V3
"""
import argparse, math
import numpy as np
from sb3_contrib import MaskablePPO
from dvvh_env import DraculaVsVanHelsingEnv
from bots import RandomBot, GreedyHeuristicBot, TacticalRefinedBot, TacticalRefinedBot_V2, \
    TacticalRefinedBot_V3, TacticalRefinedBot_V4
from rl_wrappers import SB3PolicyBotAdapter

BOTS = {"Random": RandomBot, "Greedy": GreedyHeuristicBot, "V1": TacticalRefinedBot,
        "V2": TacticalRefinedBot_V2, "V3": TacticalRefinedBot_V3, "V4": TacticalRefinedBot_V4}


def evaluate(model_path, role, opp_name, games, seed):
    model = MaskablePPO.load(model_path, device="cpu")
    agent = SB3PolicyBotAdapter(model, deterministic=True, name="Agent")
    opp = BOTS[opp_name](seed=seed + 1)
    env = DraculaVsVanHelsingEnv(seed=seed, track_stats=False)
    env.return_obs_from_step = False
    wins = 0
    for _ in range(games):
        _, info = env.reset()
        while not env.done:
            bot = agent if env.current_player == role else opp
            _, _, _, info = env.step(bot.select_action(env, None, info))
        wins += env.winner == role
    p = wins / games
    return p, 1.96 * math.sqrt(p * (1 - p) / games)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", type=int, default=0)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=5000)
    ap.add_argument("--opponents", nargs="+", default=["V4", "V3"])
    ap.add_argument("--seed", type=int, default=2024)
    a = ap.parse_args()
    for m in a.models:
        for o in a.opponents:
            p, ci = evaluate(m, a.role, o, a.games, a.seed)
            print(f"{m} | vs {o}: {p*100:.2f}% +/- {ci*100:.2f} (95% CI, {a.games} games)", flush=True)
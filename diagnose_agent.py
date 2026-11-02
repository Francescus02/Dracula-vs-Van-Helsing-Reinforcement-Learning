"""
Comprehensive diagnostic for a trained MaskablePPO agent.

Answers:
  A. Observation sufficiency: can a model predict the winner from obs?
  B. Policy identifiability: can a model predict the agent's action from obs?
  C. Behavioral gap: does the agent behave like V4 per phase?
  D. Outcome characterization: does the agent win by turtling or by conversion?
  E. Phase-by-phase action distributions.

Usage:
    python diagnose_agent.py --model ./checkpoints/Dracula/snapshot_Dracula_300000_steps.zip
"""
import argparse
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sb3_contrib import MaskablePPO

from dvvh_env import DraculaVsVanHelsingEnv
from bots import RandomBot, TacticalRefinedBot_V3, TacticalRefinedBot_V4
from rl_wrappers import SB3PolicyBotAdapter

PHASE_NAMES = {
    0: "TURN_START", 1: "CHOOSE_DISCARD",
    2: "EFFECT_1_REV_OWN", 3: "EFFECT_3_REV_OPP",
    4: "EFFECT_4_SWAP_OWN", 5: "EFFECT_6_SWAP_OPP",
    6: "EFFECT_7_TRUMP",
}


def collect(policy, opponent, n_eps, ego_player, seed=42, expert=None):
    env = DraculaVsVanHelsingEnv(seed=seed, track_stats=False)
    records, ep_results = [], []
    for _ in range(n_eps):
        obs, info = env.reset()
        ep_recs = []
        while not env.done:
            if env.current_player == ego_player:
                act = policy.select_action(env, obs, info)
                rec = {"obs": obs.copy(), "action": int(act), "phase": int(info["phase"])}
                if expert is not None:  # what would V4 do in THIS state?
                    rec["v4_action"] = int(expert.select_action(env, obs, info))
                ep_recs.append(rec)
            else:
                act = opponent.select_action(env, obs, info)
            obs, _, _, info = env.step(act)
        won = (env.winner == ego_player)
        for r in ep_recs:
            r["ego_won"] = won
        records.extend(ep_recs)
        ep_results.append({
            "won": won, "reason": env.win_reason,
            "final_round": env.round_num, "final_hp": int(env.dracula_hp),
            "max_vamps": int(np.max(env.vampires)),
            "ego_steps": len(ep_recs),
        })
    return records, ep_results


def obs_sufficiency(records, label):
    X = np.stack([r["obs"] for r in records])
    y = np.array([int(r["ego_won"]) for r in records])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=42, stratify=y)
    base = max(y.mean(), 1 - y.mean())
    lr = LogisticRegression(max_iter=3000, C=1.0).fit(X_tr, y_tr)
    mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500,
                        random_state=42).fit(X_tr, y_tr)
    acc_lr, acc_mlp = lr.score(X_te, y_te), mlp.score(X_te, y_te)
    print(f"\n[{label}] OBS SUFFICIENCY  (obs -> winner)")
    print(f"  Baseline:    {base*100:5.2f}%")
    print(f"  LogisticReg: {acc_lr*100:5.2f}%  (gain {(acc_lr-base)*100:+5.2f} pp)")
    print(f"  MLP(128):    {acc_mlp*100:5.2f}%  (gain {(acc_mlp-base)*100:+5.2f} pp)")
    return acc_lr, acc_mlp, base


def policy_bc(records, label):
    X = np.stack([r["obs"] for r in records])
    y = np.array([int(r["action"]) for r in records])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500,
                        random_state=42).fit(X_tr, y_tr)
    acc = mlp.score(X_te, y_te)
    c = Counter(y); tot = len(y)
    dist = ", ".join(f"A{k}:{v/tot*100:.0f}%" for k, v in sorted(c.items()))
    print(f"\n[{label}] POLICY IDENTIFIABILITY  (obs -> action)")
    print(f"  BC accuracy: {acc*100:5.2f}%  (uniform = 10%)")
    print(f"  Action dist: {dist}")


def action_by_phase(records, label):
    by_phase = {}
    for r in records:
        by_phase.setdefault(r["phase"], Counter())[r["action"]] += 1
    print(f"\n[{label}] ACTION DISTRIBUTION BY PHASE")
    for ph, c in sorted(by_phase.items()):
        tot = sum(c.values())
        dist = ", ".join(f"A{k}:{v/tot*100:.0f}%" for k, v in sorted(c.items()))
        print(f"  [{ph}] {PHASE_NAMES.get(ph, ph):20s} ({tot:5d}) : {dist}")


def compare_actions(agent_rec, v4_rec):
    print(f"\n[COMPARISON] Agent vs V4 action distributions per phase")
    a_by, v_by = {}, {}
    for r in agent_rec: a_by.setdefault(r["phase"], Counter())[r["action"]] += 1
    for r in v4_rec:    v_by.setdefault(r["phase"], Counter())[r["action"]] += 1
    for ph in sorted(set(a_by) | set(v_by)):
        a, v = a_by.get(ph, Counter()), v_by.get(ph, Counter())
        a_t, v_t = max(1, sum(a.values())), max(1, sum(v.values()))
        print(f"  [{ph}] {PHASE_NAMES.get(ph, ph)}:")
        for act in sorted(set(a) | set(v)):
            ap, vp = a.get(act, 0)/a_t*100, v.get(act, 0)/v_t*100
            mark = " <<<" if abs(ap-vp) > 15 else ""
            print(f"    A{act}: agent {ap:5.1f}% | V4 {vp:5.1f}%{mark}")


def agreement_report(records, label):
    """Per-phase agreement with V4 on the states the AGENT actually visits (conditional, not marginal)."""
    by, won_l, lost_l = {}, [], []
    for r in records:
        if "v4_action" not in r:
            continue
        ok = r["action"] == r["v4_action"]
        by.setdefault(r["phase"], []).append(ok)
        (won_l if r["ego_won"] else lost_l).append(ok)
    allv = [x for v in by.values() for x in v]
    print(f"\n[{label}] AGREEMENT WITH V4 ON AGENT-VISITED STATES  (overall {np.mean(allv)*100:.2f}%, n={len(allv)})")
    for ph, v in sorted(by.items()):
        print(f"  [{ph}] {PHASE_NAMES.get(ph, ph):20s} ({len(v):6d}) : {np.mean(v)*100:5.1f}%")
    if won_l and lost_l:
        print(f"  agreement in games WON {np.mean(won_l)*100:.2f}% | LOST {np.mean(lost_l)*100:.2f}%")


def outcome_analysis(eps, label):
    n = len(eps)
    wins = sum(1 for e in eps if e["won"])
    print(f"\n[{label}] OUTCOME ANALYSIS  ({n} games, WR {wins/n*100:.1f}%)")

    win_r = Counter(e["reason"][:60] for e in eps if e["won"])
    loss_r = Counter(e["reason"][:60] for e in eps if not e["won"])
    print("  Win reasons:")
    for k, c in win_r.most_common(5):
        print(f"    [{c/n*100:5.1f}%] {k}")
    print("  Loss reasons:")
    for k, c in loss_r.most_common(5):
        print(f"    [{c/n*100:5.1f}%] {k}")

    wl = [e["ego_steps"] for e in eps if e["won"]]
    ll = [e["ego_steps"] for e in eps if not e["won"]]
    if wl: print(f"  Steps — wins avg {np.mean(wl):.1f}, losses avg {np.mean(ll):.1f}")

    print("  End-round distribution (wins / losses):")
    for rnd in range(1, 6):
        w = sum(1 for e in eps if e["won"] and e["final_round"] == rnd)
        l = sum(1 for e in eps if not e["won"] and e["final_round"] == rnd)
        print(f"    R{rnd}: W {w:3d} | L {l:3d}")

    print("  Max vampires reached (per game):")
    mv = Counter(e["max_vamps"] for e in eps)
    for k in sorted(mv):
        print(f"    max_vamps={k}: {mv[k]/n*100:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--ego", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--opp", choices=["Random", "V3", "V4"], default="Random",
                    help="opponent used to collect states (use V4 for informative diagnostics)")
    args = ap.parse_args()

    model = MaskablePPO.load(args.model, device="cpu")
    agent = SB3PolicyBotAdapter(model, deterministic=True, name="Agent")
    v4 = TacticalRefinedBot_V4(seed=args.seed + 1)
    rnd = {"Random": RandomBot, "V3": TacticalRefinedBot_V3, "V4": TacticalRefinedBot_V4}[args.opp](seed=args.seed + 2)

    print(f"Collecting {args.episodes} games with AGENT vs {args.opp}...")
    a_rec, a_eps = collect(agent, rnd, args.episodes, args.ego, args.seed, expert=v4)
    print(f"Collecting {args.episodes} games with V4 vs {args.opp}...")
    v_rec, v_eps = collect(v4, rnd, args.episodes, args.ego, args.seed + 1)

    outcome_analysis(a_eps, "AGENT")
    outcome_analysis(v_eps, "V4")
    obs_sufficiency(a_rec, "AGENT")
    obs_sufficiency(v_rec, "V4")
    policy_bc(a_rec, "AGENT")
    policy_bc(v_rec, "V4")
    agreement_report(a_rec, "AGENT")
    action_by_phase(a_rec, "AGENT")
    compare_actions(a_rec, v_rec)


if __name__ == "__main__":
    main()
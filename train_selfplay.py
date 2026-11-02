"""
STAGE 3 - League self-play for 'Dracula vs Van Helsing'.

Design (all choices are justified in the accompanying write-up):
  * Two independent PPO learners (one per role) => role-conditioned critics/advantages by construction
    (this is what RAE does in SPIRAL; no extra machinery needed).
  * Alternating phases: `phase_steps` of Dracula vs League-VH, then `phase_steps` of VH vs League-Dracula.
  * League for role R = opponents of the OPPOSITE role:
        FROZEN_v1 of that role, V4, V3 (+ two perturbed-V4 anchors for style diversity)
        + gated snapshots of the opponent role's own training.
  * PFSP: weight ∝ (1 - win_rate) with per-member probability floors (V4 >= 12%, others >= 4-10%).
  * Training is STOCHASTIC (sampled actions); all evaluations are DETERMINISTIC.
  * No BC / DAgger / teacher loss: V4 is only an anchor opponent and an external yardstick.
  * Per phase: snapshot gate, metric (a) vs FROZEN_v1, metric (b) vs neural league members,
    optional metric (c) from exploit_probe.py results, plus a V4 regression check.

    python train_selfplay.py --cycles 10 --phase-steps 100000
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer

from sp_common import (ROLE_NAMES, ci95, diff_ci95, load_model, make_league_env, pfsp_probs,
                       role_wr, z_diff)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def make_schedule(v: float):
    try:
        from stable_baselines3.common.utils import FloatSchedule
        return FloatSchedule(v)
    except ImportError:  # older SB3
        from stable_baselines3.common.utils import get_schedule_fn
        return get_schedule_fn(v)


def model_spec(path: str, deterministic: bool = True, sid: str = "") -> Dict:
    return {"kind": "model", "path": str(path), "deterministic": deterministic, "id": sid or str(path)}


def bot_spec(name: str) -> Dict:
    return {"kind": "bot", "bot": name, "id": name}


def anchor_specs(role: int, frozen: Dict[int, str], noisy: bool) -> List[Dict]:
    opp = 1 - role
    specs = [
        {**model_spec(frozen[opp], False, f"FROZEN_{ROLE_NAMES[opp]}_v1"), "floor": 0.10, "anchor": True, "prior": 0.50},
        {**bot_spec("V4"), "floor": 0.12, "anchor": True, "prior": 0.52},
        {**bot_spec("V3"), "floor": 0.08, "anchor": True, "prior": 0.55},
    ]
    if noisy:
        specs += [
            {**bot_spec("V4_eps10"), "floor": 0.04, "anchor": True, "prior": 0.68},
            {**bot_spec("Mix_V2V3V4"), "floor": 0.04, "anchor": True, "prior": 0.58},
        ]
    return specs


def league_specs(role: int, state: Dict, frozen: Dict[int, str], noisy: bool) -> List[Dict]:
    specs = anchor_specs(role, frozen, noisy)
    for snap in state["admitted"][str(1 - role)]:  # snapshots of the OPPONENT role
        specs.append({**model_spec(snap["path"], False, snap["id"]), "floor": 0.05, "anchor": False, "prior": 0.50})
    return specs


class LeagueStatsCallback(BaseCallback):
    """Collects per-opponent results of the training episodes and refreshes the PFSP distribution."""

    def __init__(self, ids, floors, prior, update_every=25_000, k_prior=100):
        super().__init__(0)
        self.ids, self.floors, self.prior = list(ids), list(floors), np.asarray(prior, float)
        self.update_every, self.k_prior = update_every, k_prior
        self.wins = {i: 0 for i in self.ids}
        self.games = {i: 0 for i in self.ids}
        self._last = 0

    def estimates(self) -> np.ndarray:
        return np.array([(self.wins[i] + self.prior[j] * self.k_prior) / (self.games[i] + self.k_prior)
                         for j, i in enumerate(self.ids)])

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "ego_won" in info and "opponent_name" in info and info["opponent_name"] in self.games:
                self.games[info["opponent_name"]] += 1
                self.wins[info["opponent_name"]] += int(bool(info["ego_won"]))
        if self.num_timesteps - self._last >= self.update_every:
            self._last = self.num_timesteps
            probs = pfsp_probs(self.estimates(), self.floors)
            self.training_env.env_method("set_sampling_probs", probs.tolist())
        return True


def configure_learner(model, env, args, log_dir: Path):
    model.set_env(env)
    sched = make_schedule(args.lr)
    model.learning_rate = float(args.lr)
    model.lr_schedule = sched
    model.clip_range = make_schedule(args.clip)
    model.ent_coef, model.target_kl = args.ent_coef, args.target_kl
    model.n_steps, model.batch_size, model.n_epochs = args.n_steps, args.batch_size, args.n_epochs
    model.gamma, model.gae_lambda = args.gamma, args.gae_lambda
    model.rollout_buffer = MaskableRolloutBuffer(
        model.n_steps, model.observation_space, model.action_space, model.device,
        gamma=model.gamma, gae_lambda=model.gae_lambda, n_envs=model.n_envs,
    )
    if getattr(model, "_logger", None) is None or not getattr(model, "_custom_logger", False):
        log_dir.mkdir(parents=True, exist_ok=True)
        model.set_logger(configure(str(log_dir), ["csv"]))


def read_exploit(out: Path, role: int):
    """Latest and baseline exploiter results written by exploit_probe.py for targets of `role`."""
    d = out / "exploit"
    if not d.exists():
        return None, None
    base, latest = None, None
    for f in sorted(d.glob(f"{ROLE_NAMES[role]}_*.json"), key=lambda x: x.stat().st_mtime):
        r = json.load(open(f))
        if r.get("tag") == "baseline":
            base = r
        else:
            latest = r
    return base, latest


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen-dracula", default="./checkpoints/Dracula/FROZEN_Dracula_v1.zip")
    ap.add_argument("--frozen-vh", default="./checkpoints/VanHelsing/FROZEN_VH_v1.zip")
    ap.add_argument("--out-dir", default="./selfplay")
    ap.add_argument("--cycles", type=int, default=10, help="1 cycle = phase_steps per role")
    ap.add_argument("--phase-steps", type=int, default=100_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4), help="processes for evaluation games")
    # PPO (same validated values as stage 2, except LR which is constant)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--ent-coef", type=float, default=0.003)
    ap.add_argument("--clip", type=float, default=0.10)
    ap.add_argument("--target-kl", type=float, default=0.010)
    ap.add_argument("--n-steps", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--n-epochs", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    # league
    ap.add_argument("--no-noisy-anchors", action="store_true")
    ap.add_argument("--max-snapshots", type=int, default=5)
    ap.add_argument("--admit-rule", choices=["noninferior", "strict", "always"], default="noninferior")
    ap.add_argument("--admit-every", type=int, default=2, help="min cycles between admissions of the same role")
    ap.add_argument("--noninferior-margin", type=float, default=0.02)
    ap.add_argument("--pfsp-every", type=int, default=25_000)
    # evaluation
    ap.add_argument("--eval-games", type=int, default=2000, help="games per gate / metric (a) evaluation")
    ap.add_argument("--v4-eval-games", type=int, default=2000)
    ap.add_argument("--v4-eval-every", type=int, default=2)
    ap.add_argument("--b-lo", type=float, default=0.49)
    ap.add_argument("--b-hi", type=float, default=0.51)
    # measured baselines of the FROZEN v1 agents (from head_to_head.py / eval_checkpoint.py / robustness_eval.py)
    ap.add_argument("--base-h2h-d", type=float, default=0.4944, help="FROZEN_Dracula vs FROZEN_VH, Dracula WR (det)")
    ap.add_argument("--base-h2h-n", type=int, default=5000)
    ap.add_argument("--base-v4-d", type=float, default=0.5291)
    ap.add_argument("--base-v4-d-n", type=int, default=15000)
    ap.add_argument("--base-v4-vh", type=float, default=0.5244)
    ap.add_argument("--base-v4-vh-n", type=int, default=13000)
    args = ap.parse_args()

    out = Path(args.out_dir)
    for r in (0, 1):
        (out / ROLE_NAMES[r] / "snapshots").mkdir(parents=True, exist_ok=True)
    frozen = {0: args.frozen_dracula, 1: args.frozen_vh}
    base_h2h = {0: (args.base_h2h_d, args.base_h2h_n), 1: (1 - args.base_h2h_d, args.base_h2h_n)}
    base_v4 = {0: (args.base_v4_d, args.base_v4_d_n), 1: (args.base_v4_vh, args.base_v4_vh_n)}

    state_path = out / "state.json"
    if args.resume and state_path.exists():
        state = json.load(open(state_path))
        print(f"[RESUME] phase_index={state['phase_index']}")
    else:
        state = {"phase_index": 0, "admitted": {"0": [], "1": []},
                 "champion": {str(r): {"path": frozen[r], "wr": base_h2h[r][0], "cycle": 0} for r in (0, 1)},
                 "last_admit": {"0": -99, "1": -99}, "wr_prior": {"0": {}, "1": {}}}

    models = {}
    for r in (0, 1):
        last = out / ROLE_NAMES[r] / "last.zip"
        src = last if (args.resume and last.exists()) else Path(frozen[r])
        models[r] = load_model(str(src))
        print(f"[LOAD] {ROLE_NAMES[r]} learner <- {src}")

    total_phases = args.cycles * 2
    t0 = time.time()
    while state["phase_index"] < total_phases:
        pi = state["phase_index"]
        cycle, role = pi // 2, pi % 2
        rn, on = ROLE_NAMES[role], ROLE_NAMES[1 - role]
        model = models[role]
        specs = league_specs(role, state, frozen, not args.no_noisy_anchors)
        ids = [s["id"] for s in specs]
        floors = [s["floor"] for s in specs]
        prior_map = state["wr_prior"][str(role)]
        prior = [prior_map.get(s["id"], s["prior"]) for s in specs]
        probs = pfsp_probs(prior, floors)
        print("\n" + "=" * 100)
        print(f"[CYCLE {cycle+1}/{args.cycles}] training {rn} vs League-{on} | {len(specs)} members | "
              f"{args.phase_steps:,} steps | learner timesteps so far {model.num_timesteps:,}")
        print("   league: " + " | ".join(f"{i[:18]}:{p*100:4.1f}%" for i, p in zip(ids, probs)))

        seed = args.seed + 7919 * pi
        env = VecMonitor(SubprocVecEnv([make_league_env(k, role, specs, probs, seed) for k in range(args.envs)]),
                         info_keywords=("ego_won",))
        cb = LeagueStatsCallback(ids, floors, prior, args.pfsp_every)
        try:
            configure_learner(model, env, args, out / rn / "log")
            model.learn(total_timesteps=args.phase_steps, callback=cb, reset_num_timesteps=False)
        finally:
            env.close()

        est = cb.estimates()
        for i, e in zip(ids, est):
            prior_map[i] = float(e)
        model.save(str(out / rn / "last"))
        snap = out / rn / "snapshots" / f"snap_c{cycle+1:03d}.zip"
        model.save(str(snap))

        # ---- metric (b): WR vs neural league members (training episodes, stochastic play)
        neural = [s["id"] for s in specs if s["kind"] == "model"]
        gb = sum(cb.games[i] for i in neural)
        wb = sum(cb.wins[i] for i in neural)
        gall = sum(cb.games.values())
        wall = sum(cb.wins.values())
        wr_b = wb / gb if gb else float("nan")
        ci_b = ci95(wr_b, gb) if gb else float("nan")
        pass_b = None if gb < 100 else bool(wr_b + ci_b >= args.b_lo and wr_b - ci_b <= args.b_hi)

        # ---- metric (a): snapshot vs FROZEN_v1 of the other role (deterministic, fresh seeds)
        res = role_wr(model_spec(str(snap), True), model_spec(frozen[1 - role], True), role,
                      args.eval_games, seed + 1, args.workers)
        p0, n0 = base_h2h[role]
        z_a = z_diff(res["wr"], res["games"], p0, n0)
        pass_a = bool(z_a >= 1.96 and res["wr"] > 0.5)
        a_user = bool(res["wr"] > 0.52 and res["wr"] - res["ci"] > 0.50)

        # ---- metric (c): from exploit_probe.py (optional)
        pb, pl = read_exploit(out, role)
        pass_c = None
        if pb and pl:
            pass_c = bool(z_diff(pl["exploiter_wr"], pl["games"], pb["exploiter_wr"], pb["games"]) < 1.64)

        pos = [x for x in (pass_a, pass_b, pass_c) if x is not None]
        if pass_c is None:
            improved = bool(pass_a and pass_b)
        else:
            improved = sum(bool(x) for x in (pass_a, pass_b, pass_c)) >= 2
        overfit = bool(pass_a and pass_c is False)

        # ---- V4 regression check
        v4_txt = ""
        v4_rec = None
        if (cycle + 1) % args.v4_eval_every == 0 or cycle + 1 == args.cycles:
            rv = role_wr(model_spec(str(snap), True), bot_spec("V4"), role, args.v4_eval_games, seed + 2, args.workers)
            bp, bn = base_v4[role]
            zv = z_diff(rv["wr"], rv["games"], bp, bn)
            v4_rec = dict(wr=rv["wr"], ci=rv["ci"], games=rv["games"], z_vs_frozen=zv)
            v4_txt = (f"vs V4 {rv['wr']*100:.2f}% +/-{rv['ci']*100:.2f} (FROZEN {bp*100:.2f}%, z={zv:+.2f}"
                      f"{' REGRESSION' if zv < -2 else ''})")

        # ---- snapshot admission into the league used by the OTHER role + champion
        admitted = False
        spaced = (cycle + 1) - state["last_admit"][str(role)] >= args.admit_every
        if args.admit_rule == "always":
            admitted = spaced
        elif args.admit_rule == "strict":
            admitted = bool(z_a >= 1.64 and res["wr"] > 0.5)
        else:
            admitted = bool(spaced and res["wr"] >= p0 - args.noninferior_margin)
        if admitted:
            state["admitted"][str(role)].append({"id": f"{rn}_snap_c{cycle+1:03d}", "path": str(snap),
                                                 "cycle": cycle + 1, "wr": res["wr"]})
            state["admitted"][str(role)] = state["admitted"][str(role)][-args.max_snapshots:]
            state["last_admit"][str(role)] = cycle + 1
        champ = state["champion"][str(role)]
        if pass_a and res["wr"] > champ["wr"]:
            state["champion"][str(role)] = {"path": str(snap), "wr": res["wr"], "cycle": cycle + 1}
            shutil.copy(snap, out / rn / "champion.zip")

        rec = dict(cycle=cycle + 1, role=rn, steps=model.num_timesteps, vs_frozen_wr=res["wr"], vs_frozen_ci=res["ci"],
                   z_vs_frozen_baseline=z_a, a_user=a_user, pass_a=pass_a, wr_neural_league=wr_b, pass_b=pass_b,
                   wr_all_league_train=(wall / gall if gall else None), pass_c=pass_c, improved=improved,
                   overfit_flag=overfit, admitted=admitted, v4=v4_rec,
                   per_member={i: [cb.wins[i], cb.games[i]] for i in ids}, minutes=(time.time() - t0) / 60)
        with open(out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

        print(f"[RESULT {rn} c{cycle+1}] vs FROZEN_{on}: {res['wr']*100:.2f}% +/-{res['ci']*100:.2f} "
              f"(baseline {p0*100:.2f}%, z={z_a:+.2f}; user-rule >52%&CI>50%: {a_user})")
        print(f"   (a)={pass_a} | (b) neural-league WR {wr_b*100:.1f}% +/-{ci_b*100:.1f} ({gb} eps) -> {pass_b} | "
              f"(c)={pass_c} | improved={improved}{' | OVERFITTING FLAG' if overfit else ''}")
        print(f"   train WR vs whole league (stochastic) {100*wall/max(gall,1):.1f}% ({gall} eps) | {v4_txt}")
        print("   per member (WR est.): " + " | ".join(f"{i[:14]}:{e*100:.0f}%({cb.games[i]})" for i, e in zip(ids, est)))
        print(f"   admitted to League-{rn}: {admitted} | league sizes: "
              f"{len(league_specs(0, state, frozen, not args.no_noisy_anchors))}/"
              f"{len(league_specs(1, state, frozen, not args.no_noisy_anchors))}")

        state["phase_index"] += 1
        tmp = str(state_path) + ".tmp"
        json.dump(state, open(tmp, "w"), indent=2)
        os.replace(tmp, state_path)

    for r in (0, 1):
        cf = out / ROLE_NAMES[r] / "champion.zip"
        print(f"[DONE] {ROLE_NAMES[r]}: last = {out / ROLE_NAMES[r] / 'last.zip'} | champion.zip "
              + ("exists" if cf.exists() else "was NOT created (no snapshot passed criterion (a)) -> use last.zip"))
    print("Validate with final_validation.py (see instructions).")


if __name__ == "__main__":
    main()
"""
Final validation of the self-play agents against (1) V4 and (2) the FROZEN_v1 agents, plus balance and
robustness checks. Pre-registered decision rules are printed at the end.

    # full validation
    python final_validation.py --new-dracula ./selfplay/Dracula/last.zip --new-vh ./selfplay/VanHelsing/last.zip --games 10000
    # only the held-out robustness block (opponents that were NOT in the training league)
    python final_validation.py --new-dracula ... --new-vh ... --only-robust --robust-games 3000 \
        --robust-conditions Greedy V1 V2 V4_eps25 V3_eps25

champion.zip is created for a role only if one of its snapshots passed criterion (a); otherwise use last.zip.
All games are DETERMINISTIC, on fresh seeds, split across processes (--workers).
"""
import argparse, json, os, time

from sp_common import ROLE_NAMES, ci95, diff_ci95, matchup, role_wr, z_diff

IN_LEAGUE = {"V4", "V3", "V4_eps10", "Mix_V2V3V4"}  # opponents used as league anchors in train_selfplay.py
NEAR_LEAGUE = {"Mix_V3V4"}


def ms(path, det=True):
    return {"kind": "model", "path": path, "deterministic": det, "id": path}


def bs(name):
    return {"kind": "bot", "bot": name, "id": name}


def fmt(p, n):
    return f"{p*100:6.2f}% +/-{ci95(p, n)*100:.2f}"


def tag(c):
    return "IN-LEAGUE" if c in IN_LEAGUE else ("near-league" if c in NEAR_LEAGUE else "HELD-OUT")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-dracula", required=True)
    ap.add_argument("--new-vh", required=True)
    ap.add_argument("--frozen-dracula", default="./checkpoints/Dracula/FROZEN_Dracula_v1.zip")
    ap.add_argument("--frozen-vh", default="./checkpoints/VanHelsing/FROZEN_VH_v1.zip")
    ap.add_argument("--games", type=int, default=10000)
    ap.add_argument("--robust-games", type=int, default=3000)
    ap.add_argument("--robust-conditions", nargs="*", default=["V4_eps10", "Mix_V3V4"])
    ap.add_argument("--only-robust", action="store_true", help="skip V4 / FROZEN / new-vs-new blocks")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--seed", type=int, default=9001)
    ap.add_argument("--json", default="./selfplay/final_validation.json")
    # measured FROZEN_v1 baselines (p, n)
    ap.add_argument("--base-h2h-d", type=float, default=0.4944)
    ap.add_argument("--base-h2h-n", type=int, default=5000)
    ap.add_argument("--base-v4-d", type=float, default=0.5291)
    ap.add_argument("--base-v4-d-n", type=int, default=15000)
    ap.add_argument("--base-v4-vh", type=float, default=0.5244)
    ap.add_argument("--base-v4-vh-n", type=int, default=13000)
    a = ap.parse_args()

    for label, p in (("new Dracula", a.new_dracula), ("new Van Helsing", a.new_vh),
                     ("baseline Dracula", a.frozen_dracula), ("baseline Van Helsing", a.frozen_vh)):
        if not os.path.exists(p):
            hint = " (champion.zip only exists for a role if one of its snapshots passed criterion (a): use last.zip)" \
                if "champion" in os.path.basename(p) else ""
            raise SystemExit(f"[ERROR] {label} checkpoint not found: {p}{hint}")

    new = {0: a.new_dracula, 1: a.new_vh}
    frozen = {0: a.frozen_dracula, 1: a.frozen_vh}
    base_v4 = {0: (a.base_v4_d, a.base_v4_d_n), 1: (a.base_v4_vh, a.base_v4_vh_n)}
    base_h2h = {0: (a.base_h2h_d, a.base_h2h_n), 1: (1 - a.base_h2h_d, a.base_h2h_n)}
    t0 = time.time()
    R, verdict = {}, {}

    print("=" * 96)
    for r in (0, 1):
        rn, on = ROLE_NAMES[r], ROLE_NAMES[1 - r]
        print(f"\n### {rn}: {new[r]}")
        v = {}
        if not a.only_robust:
            v4 = role_wr(ms(new[r]), bs("V4"), r, a.games, a.seed + r, a.workers)
            bp, bn = base_v4[r]
            z4, d4 = z_diff(v4["wr"], v4["games"], bp, bn), diff_ci95(v4["wr"], v4["games"], bp, bn)
            print(f" vs V4            : {fmt(v4['wr'], v4['games'])} | baseline {bp*100:.2f}% | diff {(v4['wr']-bp)*100:+.2f} +/-{d4*100:.2f} pp (z={z4:+.2f})")
            fr = role_wr(ms(new[r]), ms(frozen[1 - r]), r, a.games, a.seed + 10 + r, a.workers)
            p0, n0 = base_h2h[r]
            zf, df = z_diff(fr["wr"], fr["games"], p0, n0), diff_ci95(fr["wr"], fr["games"], p0, n0)
            a_user = fr["wr"] > 0.52 and fr["wr"] - fr["ci"] > 0.50
            print(f" vs {on}-baseline : {fmt(fr['wr'], fr['games'])} | baseline {p0*100:.2f}% | diff {(fr['wr']-p0)*100:+.2f} +/-{df*100:.2f} pp (z={zf:+.2f}) | rule '>52% & CI>50%': {a_user}")
            R.setdefault(rn, {}).update(path=new[r], vs_v4=v4, vs_frozen=fr, z_v4=z4, z_frozen=zf)
            v.update(no_v4_regression=bool(z4 >= -2.0), v4_improved=bool(z4 >= 1.96),
                     beats_frozen=bool(zf >= 1.96 and fr["wr"] > 0.5), user_rule_52=bool(a_user))
        rob_ok, rob, held_pos, held_n = True, {}, 0, 0
        for c in a.robust_conditions:
            pn = role_wr(ms(new[r]), bs(c), r, a.robust_games, a.seed + 20 + r, a.workers)
            pf = role_wr(ms(frozen[r]), bs(c), r, a.robust_games, a.seed + 30 + r, a.workers)
            zc = z_diff(pn["wr"], pn["games"], pf["wr"], pf["games"])
            dc = diff_ci95(pn["wr"], pn["games"], pf["wr"], pf["games"])
            bad = (pn["wr"] - pf["wr"]) <= -0.03 and zc <= -2
            rob_ok &= not bad
            if tag(c) == "HELD-OUT":
                held_n += 1
                held_pos += int(pn["wr"] >= pf["wr"])
            rob[c] = dict(new=pn["wr"], baseline=pf["wr"], diff=pn["wr"] - pf["wr"], z=zc, kind=tag(c))
            print(f" robustness {c:11s} [{tag(c):11s}]: new {pn['wr']*100:6.2f}% | baseline agent {pf['wr']*100:6.2f}% | diff {(pn['wr']-pf['wr'])*100:+.2f} +/-{dc*100:.2f} pp (z={zc:+.2f}){'  <-- WORSE' if bad else ''}")
        R.setdefault(rn, {}).update(path=new[r], robust=rob)
        v.update(robust_ok=bool(rob_ok), heldout_not_worse=f"{held_pos}/{held_n}")
        verdict[rn] = v

    if not a.only_robust:
        print("\n### New Dracula vs New Van Helsing")
        hh = {}
        for mode, det in (("det", True), ("stoch", False)):
            m = matchup(ms(new[0], det), ms(new[1], det), a.games // 2, a.seed + 40, a.workers)
            hh[mode] = m["d_wr"]
            print(f" {mode:5s}: Dracula WR {fmt(m['d_wr'], m['games'])} | avg decisions {m['steps']:.1f} | conversion endings {m['conv']*100:.1f}%")
        R["new_vs_new"] = hh
        d4, v4h = verdict["Dracula"], verdict["VanHelsing"]
        a_ = R["Dracula"]["vs_v4"]["wr"]; b_ = R["VanHelsing"]["vs_v4"]["wr"]
        print(f" transitivity check (det): predicted Dracula WR = a + (1-b) - m with a={a_:.4f}, b={b_:.4f}, m=0.5091 -> "
              f"{a_ + (1 - b_) - 0.5091:.4f}; observed {hh['det']:.4f}")

    print("\n" + "=" * 96 + "\nPRE-REGISTERED DECISION RULES (per role)")
    ok_all = True
    for rn, v in verdict.items():
        if a.only_robust:
            ok = v["robust_ok"]
            print(f" {rn:10s}: {'ROBUST' if ok else 'NOT ROBUST'}  {v}")
        else:
            ok = v["no_v4_regression"] and v["beats_frozen"] and v["robust_ok"]
            print(f" {rn:10s}: {'PASS' if ok else 'FAIL'}  {v}")
        ok_all &= ok
    if a.only_robust:
        print("\n ROBUST = no condition worse than the baseline agent by >=3pp with z<=-2.")
    else:
        print("\n PASS = no V4 regression (z>=-2) AND beats baseline (diff z>=1.96, WR>50%) AND robustness not worse by >=3pp")
        print(f" OVERALL: {'IMPROVEMENT CONFIRMED' if ok_all else 'NOT CONFIRMED'}")
    print(f" ({(time.time()-t0)/60:.1f} min)")
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    json.dump(dict(results=R, verdict=verdict), open(a.json, "w"), indent=2, default=float)
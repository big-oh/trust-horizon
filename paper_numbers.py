"""
paper_numbers.py — check every simulated or computed number quoted in paper/main_ieee.tex
against the JSON files written by validate.py, experiments.py, ablation.py and figures.py.

Each row is (where in the paper, value as printed in the paper, value recomputed here, decimals).
A row passes if the recomputed value, rounded to the printed precision, equals the printed value.
Exit status is 1 if any row fails.
"""
import json, sys
import numpy as np
from trust_horizon import *

V = json.load(open("validation.json"))
X = json.load(open("results.json"))
A = json.load(open("ablation.json"))
RP = json.load(open("returns_pts.json"))
abl = {d["name"]: d for d in A["abl"]}
P = X["pareto"]
def cfg(v, m, L): return [p for p in P if p["v"] == v and p["m"] == m and p["L"] == L][0]
def best(v): return max([p for p in P if p["v"] == v and p["L"] == 0], key=lambda z: z["succ"])

rows = []
def row(where, printed, value, dec): rows.append((where, printed, value, dec))

# --- Section IV closed forms
row("Intro: 0.95^10", 0.60, 0.95 ** 10, 2)
row("Intro: 0.99^100", 0.37, 0.99 ** 100, 2)
row("Prop. 1: n0.8/n0.5", 0.322, ratio_const(0.8), 3)
row("Prop. 1: n0.99/n0.5", 0.0145, ratio_const(0.99), 4)
for p, printed in [(0.99, 22), (0.995, 44), (0.999, 223)]:
    row(f"Fig. 2 caption: n0.8 at p={p}", printed, np.floor(horizon_const(0.8, 1 - p)), 0)
for key, printed in [("0.5|10", 93.30), ("0.5|100", 99.31), ("0.5|1000", 99.931), ("0.8|10", 97.79), ("0.8|100", 99.78),
                     ("0.8|1000", 99.978), ("0.99|10", 99.90), ("0.99|100", 99.990), ("0.99|1000", 99.9990)]:
    dec = len(str(printed).split(".")[1]) if printed != 99.9990 else 4
    row(f"Table II: alpha|n={key}", printed, 100 * X["nines"][key], dec)
row("Prop. 2: T0.8/T0.5 at k=1", 0.25, frailty_horizon(0.8, 1, 1) / frailty_horizon(0.5, 1, 1), 2)
for name, (p1, p4, iid4, kap, p8, at8) in {"GPT-4o · retail": (0.604, 0.383, 0.133, 0.51, 0.30, 0.87),
                                           "Claude 3.5 Sonnet · retail": (0.692, 0.462, 0.229, 0.45, 0.36, 0.94),
                                           "Claude 3.5 Sonnet · airline": (0.460, 0.225, 0.045, 0.47, 0.15, 0.79)}.items():
    t = X["tau"][name]
    row(f"Table III {name}: iid p^4", iid4, t["iid4"], 3)
    row(f"Table III {name}: kappa", kap, t["icc"], 2)
    row(f"Table III {name}: pass^8", p8, t["pred8"], 2)
    row(f"Table III {name}: pass@8", at8, t["at8"], 2)
t = X["tau"]["GPT-4o · retail"]
row("Sec. IV-C: iid pass^8 (GPT-4o retail)", 0.018, t["iid8"], 3)
row("Sec. IV-C: iid pass@8 (%)", 99.94, 100 * t["iidat8"], 2)
row("Sec. IV-C: Beta pass@8 (%)", 87, 100 * t["at8"], 0)
row("Sec. IV-D: G at r=0.9,f=0.05", 9.5, 0.95 / 0.1, 1)
row("Sec. IV-D: G at r=0.5,f=0.02", 2, 0.98 / 0.5, 0)
eps, r, f, phi, lam = 0.05, 0.9, 0.05, 0.02, 10
a, b, c, _ = verifier_step(eps, r, f, 1)
row("Sec. IV-E: m* example", 2.7, 1 + np.log((1 - phi) * a / (lam * phi * (1 - r))) / np.log(r / c), 1)
row("Sec. IV-F: L* bare (ref. setting)", 20, reset_Lstar(0.005, 0.01, 0.01), 0)
row("Sec. IV-F: L* verified (ref. setting)", 62, verified_Lstar(0.005, 0.01, 0.01, 0.9, 0.05), 0)
row("Sec. IV-G: checkpoint L* (p=.99,v=5)", 20, checkpoint_Lstar(0.99, 5), 0)
row("Sec. IV-G: W(1000) with checkpoints", 1530, checkpoint_cost(1000, 0.99, checkpoint_Lstar(0.99, 5), 5), -1)
row("Sec. IV-G: restart cost n=1000 (x1e7)", 2.3, restart_cost(1000, 0.99, 5) / 1e7, 1)
row("Sec. IV-H: transition recall", 0.96, 1 - 2 * 0.01 * 0.01 * 0.95 / 0.005, 2)

# --- Table V (validation)
row("Table V: R(100) predicted", 0.366, V["E0"]["R100_analytic"], 3)
row("Table V: R(100) simulated", 0.363, V["E0"]["R100_sim"], 3)
row("Table V: frailty ratio simulated", 0.240, V["E1"]["ratio_sim"], 3)
for d, (pr, sm) in zip(V["E2"], [(43.4, 43), (105.0, 107), (210.0, 209), (662.8, 653)]):
    row(f"Table V: n0.8 predicted r={d['r']}", pr, d["n80_analytic"], 1)
    row(f"Table V: n0.8 simulated r={d['r']}", sm, d["n80_sim"], 0)
row("Table V: laundering predicted", 0.469, A["launder"]["analytic"], 3)
row("Table V: laundering simulated", 0.471, A["launder"]["sim"], 3)
for d, pr, sm in zip(V["E3"]["rows"], [7.10, 6.23, 5.97, 6.23, 7.10], [7.10, 6.24, 6.01, 6.22, 7.11]):
    row(f"Table V: h(L={d['L']}) x1e3 predicted", pr, 1e3 * d["h_analytic"], 2)
    row(f"Table V: h(L={d['L']}) x1e3 simulated", sm, 1e3 * d["h_sim"], 2)
sim_the = {0.5: RP[0][1], 0.9: RP[2][1], 0.99: RP[4][1]}
for rr, pr, sm in [(0.5, 66, 67), (0.9, 262, 271), (0.99, 1438, 1533)]:
    row(f"Table V: THE n0.8 predicted r={rr}", pr, trust_horizon_equation(0.8, 0.005, 0.01, 0.01, rr, 0.05), 0)
    row(f"Table V: THE n0.8 simulated r={rr}", sm, sim_the[rr], 0)
disc = max(abs(sim_the[rr] / trust_horizon_equation(0.8, 0.005, 0.01, 0.01, rr, 0.05) - 1) for rr in sim_the)
row("Sec. V-B: largest discrepancy (%)", 7, 100 * disc, 0)

# --- Ablation (Sec. V-C)
row("Ablation: bare n0.5", 95, abl["Bare loop"]["n50"], 0)
row("Ablation: resets n0.5", 118, abl["+ Resets"]["n50"], 0)
row("Ablation: bare n0.8", 38, abl["Bare loop"]["n80"], 0)
row("Ablation: resets n0.8", 39, abl["+ Resets"]["n80"], 0)
row("Ablation: verifier n0.8", 184, abl["+ Verifier"]["n80"], 0)
row("Ablation: verifier gain", 4.8, abl["+ Verifier"]["n80"] / abl["Bare loop"]["n80"], 1)
row("Ablation: verifier+resets (L=20) n0.8", 180, abl["+ Verifier + resets (L tuned for bare loop)"]["n80"], 0)
row("Ablation: verifier+resets (L=62) n0.8", 224, abl["+ Verifier + resets (L re-tuned)"]["n80"], 0)
row("Ablation: better handoff n0.8", 267, abl["+ Better handoff (rho/5)"]["n80"], 0)
row("Ablation: full-harness gain", 7.0, abl["+ Better handoff (rho/5)"]["n80"] / abl["Bare loop"]["n80"], 1)
row("Ablation: L re-tuned", 62, A["L"][1], 0)
ci_ok = all(max(d["n80"] - d["ci"][0], d["ci"][1] - d["n80"]) <= (2 if d["n80"] < 50 else 0.05 * d["n80"]) for d in A["abl"])
row("Table IV: bootstrap half-width <= 2 steps (n0.8<50) or 5%", 1, float(ci_ok), 0)
worst_pred = max(abs(d["n80"] / d["pred"] - 1) for d in A["abl"])
row("Ablation: Eq. (THE) ticks within 7%", 1, float(worst_pred <= 0.07), 0)

# --- Pareto (Sec. V-D)
row("Pareto: number of configurations", 34, len(P), 0)
row("Pareto: max success with m=1 and a sensor (%) <= 10", 1, float(max(p["succ"] for p in P if p["m"] == 1 and p["v"] != "none") <= 0.10), 0)
row("Pareto: judge m=1 success (%)", 0, 100 * cfg("judge", 1, 0)["succ"], 0)
row("Pareto: no sensor success (%)", 26.5, 100 * cfg("none", 1, 0)["succ"], 1)
t3 = cfg("tests", 3, 0); none = cfg("none", 1, 0)
row("Pareto: tests m=3 cost-per-success reduction (%)", 30, 100 * (1 - t3["cps"] / none["cps"]), -1)
tests_ok = [p for p in P if p["v"] == "tests" and p["m"] >= 3]
row("Pareto: tests (m>=3) min success (%)", 71, 100 * min(p["succ"] for p in tests_ok), 0)
row("Pareto: tests (m>=3) max success (%)", 75, 100 * max(p["succ"] for p in tests_ok), 0)
row("Pareto: tests (m>=3) cost per success (x ideal)", 1.6, np.mean([p["cps"] / 150 for p in tests_ok]), 1)
row("Pareto: tests+judge best success (%)", 88.5, 100 * best("tests+judge")["succ"], 1)
row("Pareto: no-sensor silent share (%)", 73.5, 100 * best("none")["silent"], 1)
row("Pareto: tests+judge silent share (%)", 10.8, 100 * best("tests+judge")["silent"], 1)
row("Pareto: tests+judge cost (x ideal)", 2.3, best("tests+judge")["cps"] / 150, 1)
row("Pareto: lint m=1 plotted (success > 2%)", 1, float(min(cfg("lint", 1, L)["succ"] for L in (0, 20)) > 0.02), 0)

# per-step cost of tests with m=3, recomputed from the same seeds as experiments.py
seed = 100; menu = {"none": (0, 0, 0), "lint": (0.40, 0.01, 0.05), "tests": (0.75, 0.03, 0.25), "judge": (0.80, 0.10, 0.60), "tests+judge": (0.92, 0.127, 0.85)}
for vn, (r_, f_, v_) in menu.items():
    for m_ in ([1] if vn == "none" else [1, 2, 3, 5]):
        for L_ in [None, X["Ls"]]:
            if (vn, m_, L_) == ("tests", 3, None):
                res = simulate(Harness(eps0=0.005, gamma=0.01, r=r_, f=f_, m=m_, v=v_, reset_cost=2), 150, 8000, seed)
                per_step = res["cost"].sum() / np.minimum(res["fail_step"], 150).sum()
            seed += 1
row("Pareto: tests m=3 per-step cost increase (%)", 30, 100 * (per_step - 1), -1)

bad = 0
w = max(len(r_[0]) for r_ in rows)
for where, printed, value, dec in rows:
    ok = round(float(value), dec) == round(float(printed), dec)
    bad += not ok
    print(f"{'ok ' if ok else 'BAD'}  {where:<{w}}  paper={printed:<10}  code={float(value):.6g}")
print(f"\n{len(rows) - bad}/{len(rows)} numbers match")
sys.exit(1 if bad else 0)

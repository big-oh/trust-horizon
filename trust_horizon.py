"""
trust_horizon.py — reproducibility code for
"Lusser's Law for Agents: The Compounding-Reliability Gap in Long-Horizon Agent Harnesses"
A. Siddiqui, WebridgeAI Research, preprint v0.2 (Sep 2026).

Everything here is a *synthetic* model of an agent loop. No LLM is called.
The simulator exists to (1) check the closed-form results in the paper and
(2) explore regimes where the closed forms are only approximations.

Model of one task (n steps):
  * per-attempt error at step s since the last context reset:  eps(s) = eps0 * (1 + gamma * s)
  * optional context reset every L steps; each reset loses critical state with prob rho (silent error)
  * optional verifier ("sensor") with recall r = P(flag | wrong) and false-alarm f = P(flag | correct)
  * up to m attempts per step; a flagged final attempt escalates (detected failure)
  * optional "stuck" steps (prob phi): every attempt at that step repeats the same wrong output
  * optional task heterogeneity: eps0 ~ Gamma(shape=k, mean=eps0) across tasks ("frailty"; k is nu in the paper)
Outcome per task: success, silent failure (a wrong result was accepted), or detected failure (escalation).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

# ----------------------------------------------------------------------------- analytic results

def horizon_const(alpha, eps):
    """Trust horizon for constant per-step error eps: n_alpha = ln(1/alpha) / -ln(1-eps)."""
    return np.log(1 / alpha) / -np.log1p(-eps)

def ratio_const(alpha, beta=0.5):
    """n_alpha / n_beta under any constant hazard: ln(alpha)/ln(beta). Invariant to model quality."""
    return np.log(alpha) / np.log(beta)

def frailty_survival(t, T50, k):
    """Population survival when each task has constant hazard lambda ~ Gamma(k, .).
    S(t) = (1 + t/theta)^(-k), with theta chosen so that S(T50)=0.5."""
    theta = T50 / (2 ** (1 / k) - 1)
    return (1 + t / theta) ** (-k)

def frailty_horizon(alpha, T50, k):
    theta = T50 / (2 ** (1 / k) - 1)
    return theta * (alpha ** (-1 / k) - 1)

def weibull_survival(t, T50, k):
    lam = T50 / np.log(2) ** (1 / k)
    return np.exp(-(t / lam) ** k)

def weibull_horizon(alpha, T50, k):
    lam = T50 / np.log(2) ** (1 / k)
    return lam * np.log(1 / alpha) ** (1 / k)

def beta_from_pass(p1, pk, k):
    """Fit Beta(a,b) per-task success distribution to pass^1 and pass^k (method of moments on E[theta], E[theta^k])."""
    from scipy.optimize import brentq
    def g(s):
        a = p1 * s
        return np.prod([(a + j) / (s + j) for j in range(k)]) - pk
    s = brentq(g, 1e-4, 1e6)
    return p1 * s, (1 - p1) * s

def pass_hat_k(a, b, k):
    """pass^k = E[theta^k] under theta ~ Beta(a,b)."""
    return np.prod([(a + j) / (a + b + j) for j in range(k)])

def pass_at_k(a, b, k):
    """pass@k = 1 - E[(1-theta)^k]."""
    return 1 - np.prod([(b + j) / (a + b + j) for j in range(k)])

def verifier_step(eps, r, f, m):
    """One step with verifier + up to m attempts (no stuck). Returns (P_correct, P_silent, P_escalate, E[attempts])."""
    a = (1 - eps) * (1 - f)          # accepted & correct
    b = eps * (1 - r)                # accepted & wrong  (silent)
    c = (1 - eps) * f + eps * r      # rejected -> retry
    geo = (1 - c ** m) / (1 - c)
    return a * geo, b * geo, c ** m, geo

def eff_error(eps, r, f):
    """Silent-error rate per step in the unlimited-retry limit: b/(a+b)."""
    a = (1 - eps) * (1 - f); b = eps * (1 - r)
    return b / (a + b)

def reset_hazard(L, eps0, gamma, rho):
    """Approximate per-step hazard with context reset every L steps (first-order): eps0 + eps0*gamma*(L-1)/2 + rho/L."""
    return eps0 + eps0 * gamma * (L - 1) / 2 + rho / L

def reset_Lstar(eps0, gamma, rho):
    """Square-root law for the optimal reset interval."""
    return np.sqrt(2 * rho / (eps0 * gamma))

def silent_rate(eps0, r, f):
    """eps~ : per-step silent-error rate after verification (small-error limit of b/(a+b))."""
    return eps0 * (1 - r) / (1 - f)

def trust_horizon_equation(alpha, eps0, gamma, rho, r, f, m=np.inf):
    """Trust Horizon Equation (small-error regime, reset interval re-optimised for the verified loop):
         n_alpha ≈ ln(1/alpha) / [ eps~ + sqrt(2 eps~ gamma rho) + f^m ],   eps~ = eps0 (1-r)/(1-f)
       - eps~            : errors that slip past the step verifier
       - sqrt(2 eps~ g rho): the price of context rot + lossy handoffs at the optimal reset interval
                          (handoff loss is invisible to a step-level verifier, so it is NOT discounted by r)
       - f^m             : detected-failure floor from false alarms exhausting the retry budget"""
    et = silent_rate(eps0, r, f)
    floor = 0.0 if not np.isfinite(m) else f ** m
    rot = np.sqrt(2 * et * gamma * rho) if (gamma > 0 and rho > 0) else 0.0
    return np.log(1 / alpha) / (et + rot + floor)

def verified_Lstar(eps0, gamma, rho, r, f):
    return reset_Lstar(silent_rate(eps0, r, f), gamma, rho)

def restart_cost(n, p, v=0.0):
    """Expected steps when errors stay silent until an end-of-task check (cost v); any failure forces a full rerun:
       E = (n + v) / p^n  — exponential in n."""
    return (n + v) / p ** n

def checkpoint_cost(n, p, L, v):
    """Verified checkpoints every L steps (check cost v, perfect segment check, rollback to last checkpoint):
       E = (n/L) (L + v) / p^L  — linear in n."""
    return (n / L) * (L + v) / p ** L

def checkpoint_Lstar(p, v):
    """Minimiser of (L+v)/(L p^L):  L* = [-v + sqrt(v^2 + 4v/h)]/2 ≈ sqrt(v/h), h = -ln p (Young-type square-root law)."""
    h = -np.log(p)
    return (-v + np.sqrt(v * v + 4 * v / h)) / 2

# ----------------------------------------------------------------------------- simulator

@dataclass
class Harness:
    eps0: float = 0.005
    gamma: float = 0.0
    L: int | None = None       # reset interval (None = never)
    rho: float = 0.0           # handoff loss probability per reset
    r: float = 0.0             # verifier recall
    f: float = 0.0             # verifier false-alarm rate
    m: int = 1                 # max attempts per step
    phi: float = 0.0           # stuck-step probability
    frailty_k: float | None = None   # Gamma shape (nu in the paper) for task heterogeneity (None = homogeneous)
    v: float = 0.0             # verifier cost per attempt (in generation-attempt units)
    reset_cost: float = 0.0    # cost of one context reset

def simulate(h: Harness, n_max: int, trials: int = 20000, seed: int = 0):
    """Simulate `trials` tasks for up to n_max steps. Returns dict with first-failure step and type,
    so that the survival curve R(n) for every n <= n_max comes from one run."""
    rng = np.random.default_rng(seed)
    T = trials
    if h.frailty_k is None:
        eps0 = np.full(T, h.eps0)
    else:
        eps0 = rng.gamma(h.frailty_k, h.eps0 / h.frailty_k, size=T)
    alive = np.ones(T, bool)
    fail_step = np.full(T, n_max + 1)       # step index at which task failed (1-based), n_max+1 = survived
    fail_type = np.zeros(T, np.int8)        # 0 none, 1 silent, 2 detected
    cost = np.zeros(T)
    since = np.zeros(T)
    verifier = h.r > 0 or h.f > 0
    for step in range(1, n_max + 1):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        # context reset before this step
        if h.L is not None and step > 1 and (step - 1) % h.L == 0:
            since[idx] = 0
            cost[idx] += h.reset_cost
            lost = rng.random(idx.size) < h.rho
            if lost.any():
                j = idx[lost]; alive[j] = False; fail_step[j] = step; fail_type[j] = 1
                idx = idx[~lost]
        eps = np.minimum(eps0[idx] * (1 + h.gamma * since[idx]), 1.0)
        stuck = rng.random(idx.size) < h.phi
        pending = np.ones(idx.size, bool)
        outcome = np.zeros(idx.size, np.int8)  # 0 ok, 1 silent, 2 escalate
        for att in range(1, h.m + 1):
            pi = np.flatnonzero(pending)
            if pi.size == 0:
                break
            cost[idx[pi]] += 1 + (h.v if verifier else 0)
            wrong = stuck[pi] | (rng.random(pi.size) < eps[pi])
            if verifier:
                flag = np.where(wrong, rng.random(pi.size) < h.r, rng.random(pi.size) < h.f)
            else:
                flag = np.zeros(pi.size, bool)
            acc = ~flag
            outcome[pi[acc & wrong]] = 1
            pending[pi[acc]] = False
            if att == h.m:
                outcome[pi[flag]] = 2
                pending[pi[flag]] = False
        bad = outcome > 0
        if bad.any():
            j = idx[bad]; alive[j] = False; fail_step[j] = step; fail_type[j] = outcome[bad]
        since[idx] += 1
    return dict(fail_step=fail_step, fail_type=fail_type, cost=cost, n_max=n_max)

def survival(res):
    n = np.arange(0, res["n_max"] + 1)
    fs = res["fail_step"]
    return n, np.array([(fs > k).mean() for k in n])

def horizon_from_sim(res, alpha):
    n, R = survival(res)
    below = np.flatnonzero(R < alpha)
    return n[below[0]] - 1 if below.size else np.nan

def wilson(p, N, z=1.96):
    den = 1 + z * z / N
    c = (p + z * z / (2 * N)) / den
    hw = z * np.sqrt(p * (1 - p) / N + z * z / (4 * N * N)) / den
    return c - hw, c + hw

if __name__ == "__main__":
    print("ratio n80/n50 (const hazard):", ratio_const(0.8))
    print("ratio n99/n50 (const hazard):", ratio_const(0.99))
    print("frailty k=1 ratio:", frailty_horizon(0.8, 1, 1))

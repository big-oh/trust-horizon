import numpy as np, json, time
from trust_horizon import *
out = {}
t0=time.time()
# E0: constant hazard sanity
res = simulate(Harness(eps0=0.01), 400, 40000, 1)
n,R = survival(res)
print("E0 R(100) sim", R[100], "analytic", 0.99**100, " n80 sim", horizon_from_sim(res,0.8), "analytic", horizon_const(0.8,0.01))
out['E0']=dict(R100_sim=float(R[100]),R100_analytic=0.99**100,n80_sim=float(horizon_from_sim(res,0.8)),n80_analytic=float(horizon_const(0.8,0.01)))
# E1 frailty k=1: ratio
res = simulate(Harness(eps0=0.01, frailty_k=1.0), 3000, 40000, 2)
n50 = horizon_from_sim(res,0.5); n80 = horizon_from_sim(res,0.8)
print("E1 frailty k=1: n50",n50,"n80",n80,"ratio",n80/n50)
out['E1']=dict(n50_sim=float(n50),n80_sim=float(n80),ratio_sim=float(n80/n50),ratio_analytic=0.25)
# E2 verifier gain
out['E2']=[]
for r,f in [(0.5,0.02),(0.8,0.05),(0.9,0.05),(0.97,0.1)]:
    res = simulate(Harness(eps0=0.01, r=r,f=f,m=50), 3000, 20000, 3)
    ns = horizon_from_sim(res,0.8); na = horizon_const(0.8, eff_error(0.01,r,f))
    print(f"E2 r={r} f={f}: n80 sim {ns} analytic {na:.1f} gain {na/horizon_const(0.8,0.01):.2f}")
    out['E2'].append(dict(r=r,f=f,n80_sim=float(ns),n80_analytic=float(na),gain=float(na/horizon_const(0.8,0.01))))
# E3 square root law
eps0,gamma,rho = 0.005,0.01,0.01
print("L* analytic", reset_Lstar(eps0,gamma,rho))
out['E3']=dict(Lstar=float(reset_Lstar(eps0,gamma,rho)),rows=[])
for L in [5,10,20,40,80]:
    res = simulate(Harness(eps0=eps0,gamma=gamma,L=L,rho=rho), 400, 30000, 4)
    n,R = survival(res); hs = -np.log(R[400])/400
    print(f"E3 L={L}: hazard sim {hs:.5f} approx {reset_hazard(L,eps0,gamma,rho):.5f}")
    out['E3']['rows'].append(dict(L=L,h_sim=float(hs),h_analytic=float(reset_hazard(L,eps0,gamma,rho))))
json.dump(out,open('validation.json','w'),indent=1)
print("time", time.time()-t0)

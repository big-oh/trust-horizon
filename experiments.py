import numpy as np, json
from trust_horizon import *
R = {}
# --- tau-bench beta fits
tau = {"GPT-4o · retail":(0.604,0.383,4),"Claude 3.5 Sonnet · retail":(0.692,0.462,4),"Claude 3.5 Sonnet · airline":(0.460,0.225,4)}
R['tau']={}
for k,(p1,pk,K) in tau.items():
    a,b = beta_from_pass(p1,pk,K)
    R['tau'][k]=dict(p1=p1,p4=pk,a=a,b=b,icc=1/(a+b+1),iid4=p1**4,
        pred8=pass_hat_k(a,b,8), iid8=p1**8, at8=pass_at_k(a,b,8), iidat8=1-(1-p1)**8)
    print(k, {kk:round(v,4) for kk,v in R['tau'][k].items()})
# --- heterogeneity horizons (T50=1)
R['het']={}
for name,fn in [("exp",lambda a: np.log(1/a)/np.log(2)),("frailty_k1",lambda a: frailty_horizon(a,1,1)),("weibull_0.6",lambda a: weibull_horizon(a,1,0.6)),("weibull_0.37",lambda a: weibull_horizon(a,1,0.37))]:
    R['het'][name]={str(a):fn(a) for a in [0.8,0.9,0.99]}
    print(name,{a:round(v,4) for a,v in R['het'][name].items()})
# --- nines table
R['nines']={f"{a}|{n}":a**(1/n) for a in [0.5,0.8,0.99] for n in [10,100,1000]}
for k,v in R['nines'].items(): print(k, f"{v:.6f}", f"err={1-v:.2e}")
# --- reference reset interval (the full ablation is in ablation.py)
eps0,gamma,rho=0.005,0.01,0.01
Ls=int(round(reset_Lstar(eps0,gamma,rho)))
R['Ls']=Ls
# --- Pareto
menu = {"none":(0,0,0),"lint":(0.40,0.01,0.05),"tests":(0.75,0.03,0.25),"judge":(0.80,0.10,0.60),"tests+judge":(0.92,0.127,0.85)}
pts=[]; N=150; seed=100
for vn,(r,f,v) in menu.items():
    for m in ([1] if vn=="none" else [1,2,3,5]):
        for L in [None,Ls]:
            h=Harness(eps0=eps0,gamma=gamma,L=L,rho=rho if L else 0,r=r,f=f,m=m,v=v,reset_cost=2)
            res=simulate(h,N,8000,seed); seed+=1
            succ=(res['fail_step']>N).mean(); cost=res['cost'].mean()
            silent=((res['fail_type']==1)).mean(); det=(res['fail_type']==2).mean()
            pts.append(dict(v=vn,m=m,L=L or 0,succ=float(succ),cost=float(cost),silent=float(silent),det=float(det),cps=float(cost/max(succ,1e-9))))
R['pareto']=pts
for p in sorted(pts,key=lambda x:x['cost']): print(p)
json.dump(R,open('results.json','w'),indent=1,default=float)

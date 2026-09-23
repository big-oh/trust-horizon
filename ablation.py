import numpy as np, json
from trust_horizon import *
eps0,gamma,rho=0.005,0.01,0.01
r,f,m,v=0.9,0.05,3,0.3
L0=int(round(reset_Lstar(eps0,gamma,rho)))
L1=int(round(verified_Lstar(eps0,gamma,rho,r,f)))
L2=int(round(verified_Lstar(eps0,gamma,0.002,r,f)))
print("L0",L0,"L1",L1,"L2",L2)
cfg = [("Bare loop",Harness(eps0=eps0,gamma=gamma)),
       ("+ Resets",Harness(eps0=eps0,gamma=gamma,L=L0,rho=rho,reset_cost=2)),
       ("+ Verifier",Harness(eps0=eps0,gamma=gamma,r=r,f=f,m=m,v=v)),
       ("+ Verifier + resets (L tuned for bare loop)",Harness(eps0=eps0,gamma=gamma,L=L0,rho=rho,r=r,f=f,m=m,v=v,reset_cost=2)),
       ("+ Verifier + resets (L re-tuned)",Harness(eps0=eps0,gamma=gamma,L=L1,rho=rho,r=r,f=f,m=m,v=v,reset_cost=2)),
       ("+ Better handoff (rho/5)",Harness(eps0=eps0,gamma=gamma,L=L2,rho=0.002,r=r,f=f,m=m,v=v,reset_cost=2))]
out=[];curves={}
def quad(e,g,extra=0,a=0.8):
    A=e*g/2; B=e+extra; C=-np.log(1/a); return (-B+np.sqrt(B*B-4*A*C))/(2*A)
et=silent_rate(eps0,r,f); fl=f**m
pred=[quad(eps0,gamma), np.log(1.25)/reset_hazard(L0,eps0,gamma,rho), quad(et,gamma,fl),
      np.log(1.25)/(reset_hazard(L0,et,gamma,rho)+fl), np.log(1.25)/(reset_hazard(L1,et,gamma,rho)+fl),
      np.log(1.25)/(reset_hazard(L2,et,gamma,0.002)+fl)]
for i,(k,h) in enumerate(cfg):
    res=simulate(h,4000,20000,20+i); n,S=survival(res); fs=res['fail_step']
    n80=horizon_from_sim(res,0.8); n50=horizon_from_sim(res,0.5)
    rng=np.random.default_rng(7); bs=[]
    for _ in range(150):
        s=np.sort(rng.choice(fs,fs.size)); # R(n)=P(fs>n); n80 = largest n with R>=0.8
        q=s[int(np.floor(0.2*fs.size))]-1; bs.append(q)
    cps=res['cost'].sum()/np.minimum(fs-1,4000).sum()
    sil=((res['fail_type']==1)&(fs<=n80+1)).sum(); det=((res['fail_type']==2)&(fs<=n80+1)).sum()
    out.append(dict(name=k,n80=int(n80),n50=int(n50),ci=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],pred=float(pred[i]),cps=float(cps)))
    curves[k]=S[:1201].tolist()
    print(out[-1])
print("THE (m=3):",trust_horizon_equation(0.8,eps0,gamma,rho,r,f,m), " THE better handoff:",trust_horizon_equation(0.8,eps0,gamma,0.002,r,f,m))
# stuck-step laundering check
res=simulate(Harness(eps0=0.0,r=0.9,f=0.0,m=6,phi=1.0),1,50000,5)
launder=dict(sim=float((res['fail_type']==1).mean()),analytic=1-0.9**6)
print("P(silent|stuck,m=6) sim",launder['sim'],"analytic",launder['analytic'])
json.dump(dict(abl=out,L=[L0,L1,L2],launder=launder),open('ablation.json','w'),indent=1)
json.dump(curves,open('abl_curves.json','w'))

import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from trust_horizon import *

INK="#000000"; PAPER="#ffffff"; BLUE="#1F3A5F"; FLARE="#B03A2E"; PERI="#7F95B5"; MUTE="#555555"; GRID=(0,0,0,0.08)
plt.rcParams.update({
    "font.family":"STIXGeneral","mathtext.fontset":"stix","font.size":8,"axes.edgecolor":INK,"axes.labelcolor":INK,
    "xtick.color":INK,"ytick.color":INK,"text.color":INK,"axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.6,
    "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.5,"axes.facecolor":"white","figure.facecolor":"white","savefig.facecolor":"white",
    "legend.frameon":False,"xtick.major.width":0.6,"ytick.major.width":0.6,"xtick.major.size":2.5,"ytick.major.size":2.5,
    "axes.titlesize":8,"axes.labelsize":8,"legend.fontsize":7,"xtick.labelsize":7.5,"ytick.labelsize":7.5,"lines.linewidth":1.2,"pdf.fonttype":42})
W1=7.0  # full text width (in)

def tag(ax,t,x=-0.12,y=1.04):
    ax.text(x,y,f"({t.lower()})",transform=ax.transAxes,fontsize=9,fontweight="bold",color=INK,va="bottom")

FIGDIR="paper/figs_ieee"
def save(fig,name):
    fig.savefig(f"{FIGDIR}/{name}.pdf",bbox_inches="tight",pad_inches=0.04,metadata={"CreationDate":None}); plt.close(fig); print(name)

import os; os.makedirs(FIGDIR,exist_ok=True)

# ------------------------------------------------ F2 Lusser curves
fig,ax=plt.subplots(1,2,figsize=(W1,2.55),gridspec_kw=dict(width_ratios=[1.35,1],wspace=0.32))
n=np.arange(0,501)
for p,c,lab,ls in [(0.99,FLARE,"p = 0.99","-"),(0.995,BLUE,"p = 0.995","--"),(0.999,PERI,"p = 0.999","-.")]:
    ax[0].plot(n,p**n,color=c,label=lab,ls=ls)
    n80=horizon_const(0.8,1-p); ax[0].plot([n80],[0.8],"o",ms=4.5,color=c,mec=INK,mew=0.6)
for a,ls in [(0.8,"--"),(0.5,":")]:
    ax[0].axhline(a,color=INK,lw=0.8,ls=ls,alpha=0.7)
ax[0].text(503,0.81,"trust (80%)",fontsize=7.2,va="bottom",ha="right",color=MUTE)
ax[0].text(503,0.51,"capability (50%)",fontsize=7.2,va="bottom",ha="right",color=MUTE)
ax[0].set_xlabel("task length n (steps)"); ax[0].set_ylabel("end-to-end reliability  R(n) = $p^n$")
ax[0].set_xlim(0,500); ax[0].set_ylim(0,1.02); ax[0].legend(loc="lower left"); tag(ax[0],"A")
# nines needed
ns=np.array([10,30,100,300,1000,3000])
for a,c in [(0.5,PERI),(0.8,BLUE),(0.99,FLARE)]:
    ax[1].plot(ns,-np.log10(1-a**(1/ns)),"o-",color=c,ms=3.5,label=f"{int(a*100)}% end-to-end")
ax[1].set_xscale("log"); ax[1].set_xlabel("task length n (steps)"); ax[1].set_ylabel("per-step 'nines' required")
ax[1].set_yticks([1,2,3,4,5]); ax[1].set_yticklabels(["90%","99%","99.9%","99.99%","99.999%"]); ax[1].legend(loc="upper left"); tag(ax[1],"B")
save(fig,"fig_lusser")

# ------------------------------------------------ F3 harness swings
data=[("Gemini 3.1 Pro","Terminal-Bench 2.0",59.4,80.2,"Gemini CLI","TongAgents"),
      ("Claude Opus 4.6","Terminal-Bench 2.0",58.0,76.4,"Claude Code","Meta-Harness"),
      ("GPT-5.3-Codex","Terminal-Bench 2.0",64.7,78.4,"Terminus 2","SageAgent"),
      ("Claude Opus 4.5","SWE-bench Pro",45.9,55.4,"SEAL scaffold","Claude Code")]
fig,ax=plt.subplots(figsize=(W1,2.25))
for i,(m,b,lo,hi,hl,hh) in enumerate(data):
    y=len(data)-1-i
    ax.plot([lo,hi],[y,y],color=INK,lw=2.2,alpha=0.25,solid_capstyle="round")
    ax.plot(lo,y,"o",ms=8,color=PAPER,mec=INK,mew=1.4); ax.plot(hi,y,"o",ms=8,color=BLUE,mec=INK,mew=0.8)
    ax.text(lo-1.0,y,f"{hl}  {lo:.1f}",ha="right",va="center",fontsize=7.4,color=MUTE)
    ax.text(hi+1.0,y,f"{hi:.1f}  {hh}",ha="left",va="center",fontsize=7.4,color=INK)
    ax.text((lo+hi)/2,y+0.25,f"+{hi-lo:.1f} pts",ha="center",va="bottom",fontsize=7.6,fontweight="normal",fontstyle="italic",color=FLARE)
ax.set_yticks(range(len(data))); ax.set_yticklabels([f"{d[0]}\n{d[1]}" for d in data][::-1],fontsize=7.6)
ax.set_xlim(28,100); ax.set_ylim(-0.6,len(data)-0.2); ax.set_xlabel("task success (%) — identical model weights, different harness")
ax.grid(axis="y",visible=False); ax.spines["left"].set_visible(False); ax.tick_params(axis="y",length=0)
save(fig,"fig_swings")

# ------------------------------------------------ F4 pass^k
R=json.load(open("results.json"))
fig,ax=plt.subplots(1,2,figsize=(W1,2.5),gridspec_kw=dict(wspace=0.3))
cols=[BLUE,FLARE,PERI]; ks=np.arange(1,9)
for (name,d),c in zip(R["tau"].items(),cols):
    a,b=d["a"],d["b"]
    ax[0].plot(ks,[pass_hat_k(a,b,k) for k in ks],color=c,label=name)
    ax[0].plot(ks,d["p1"]**ks,color=c,ls=":",lw=1.3)
    ax[0].plot([1,4],[d["p1"],d["p4"]],"o",color=c,mec=INK,mew=0.7,ms=5)
ax[0].plot([8],[0.25],marker="v",color=INK,ms=5)
ax[0].set_xlabel("k (independent repeats of the same task)"); ax[0].set_ylabel("pass$^k$  (all k succeed)")
ax[0].set_ylim(0,0.75); ax[0].legend(loc="upper right",fontsize=6.6,handlelength=1.4); tag(ax[0],"A")
# pass@k: heterogeneity wall
d=R["tau"]["GPT-4o · retail"]; kk=np.arange(1,21)
ax[1].plot(kk,[1-(1-d["p1"])**k for k in kk],color=INK,ls=":",lw=1.4,label="if failures were i.i.d.")
ax[1].plot(kk,[pass_at_k(d["a"],d["b"],k) for k in kk],color=BLUE,label="Beta fit (heterogeneous tasks)")
ax[1].fill_between(kk,[pass_at_k(d["a"],d["b"],k) for k in kk],[1-(1-d["p1"])**k for k in kk],color=FLARE,alpha=0.18,lw=0)
ax[1].text(9,0.9,"retry gap",color=FLARE,fontsize=7.6,fontweight="bold")
ax[1].set_xlabel("k (attempts with a perfect selector)"); ax[1].set_ylabel("pass@k (≥1 succeeds)"); ax[1].set_ylim(0.55,1.01)
ax[1].legend(loc="lower right"); tag(ax[1],"B")
save(fig,"fig_passk")

# ------------------------------------------------ F5 frailty / hazard shapes
t=np.logspace(-2.5,1.3,400)
fig,ax=plt.subplots(1,2,figsize=(W1,2.45),gridspec_kw=dict(wspace=0.32))
curves=[("Constant hazard",lambda t: 0.5**t,INK,":"),
        ("Gamma frailty, $\\nu$=1  (log-logistic)",lambda t: frailty_survival(t,1,1),BLUE,"-"),
        ("Weibull $\\nu$=0.6 (agents)",lambda t: weibull_survival(t,1,0.6),FLARE,"-"),
        ("Weibull $\\nu$=0.37 (humans)",lambda t: weibull_survival(t,1,0.37),PERI,"--")]
for lab,fn,c,ls in curves: ax[0].plot(t,fn(t),color=c,ls=ls,label=lab)
ax[0].set_xscale("log"); ax[0].axhline(0.8,color=INK,lw=0.6,ls="--",alpha=0.5); ax[0].axvline(1,color=INK,lw=0.6,alpha=0.4)
ax[0].set_xlabel("task length / $T_{50}$"); ax[0].set_ylabel("success probability"); ax[0].legend(loc="lower left",fontsize=6.3,frameon=True,facecolor="white",edgecolor="none",framealpha=1); tag(ax[0],"A")
# hazard
tt=np.logspace(-2,1.3,300)
ax[1].plot(tt,np.full_like(tt,np.log(2)),color=INK,ls=":")
th=1/(2-1); ax[1].plot(tt,1/(th+tt),color=BLUE)
lam=1/np.log(2)**(1/0.6); ax[1].plot(tt,0.6/lam*(tt/lam)**(-0.4),color=FLARE)
lam=1/np.log(2)**(1/0.37); ax[1].plot(tt,0.37/lam*(tt/lam)**(-0.63),color=PERI,ls="--")
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("task length / $T_{50}$"); ax[1].set_ylabel("population hazard h(t)")
ax[1].text(0.012,0.055,"declining population hazard:\nlearning — or selection?",fontsize=7,color=INK); ax[1].set_ylim(0.03,6); tag(ax[1],"B")
save(fig,"fig_frailty")

# ------------------------------------------------ F7 gain heatmap
rr=np.linspace(0,0.99,200); ff=np.linspace(0,0.4,200); RR,FF=np.meshgrid(rr,ff)
G=0.01/eff_error(0.01,RR,FF)
cmap=LinearSegmentedColormap.from_list("wb",["#ffffff","#C9D3E0",BLUE,"#0d1a2b"])
fig,ax=plt.subplots(figsize=(3.45,2.5))
im=ax.pcolormesh(RR,FF,np.log10(G),cmap=cmap,shading="auto",rasterized=True,vmin=0,vmax=2)
cs=ax.contour(RR,FF,G,levels=[2,5,10,20,50],colors=[FLARE],linewidths=0.9)
ax.clabel(cs,fmt=lambda x:f"{x:.0f}×",fontsize=7,colors=FLARE)
ax.set_xlabel("verifier recall r  (P flag | wrong)"); ax.set_ylabel("false-alarm rate f"); ax.grid(False)
cb=fig.colorbar(im,ax=ax,pad=0.02); cb.set_ticks([0,1,2]); cb.set_ticklabels(["1×","10×","100×"]); cb.outline.set_visible(False); cb.set_label("trust-horizon gain G",fontsize=7.6)
for (r,f,l) in [(0.4,0.01,"lint"),(0.75,0.03,"tests"),(0.80,0.10,"LLM judge"),(0.92,0.127,"tests+judge")]:
    ax.plot(r,f,"o",color=FLARE,mec=INK,ms=4.5,mew=0.6); ax.text(r-0.02,f+0.012,l,fontsize=6.8,color=INK,ha="right",bbox=dict(boxstyle="round,pad=0.15",fc=PAPER,ec="none",alpha=0.85))
save(fig,"fig_gain")

# ------------------------------------------------ F8 square-root law
eps0,gamma,rho=0.005,0.01,0.01
Ls=np.arange(3,301)
fig,ax=plt.subplots(figsize=(3.45,2.45))
for e,c,lab,Lpts in [(eps0,FLARE,"bare loop ($\\varepsilon_0$ = 0.005)",[5,10,20,40,80,160]),(silent_rate(eps0,0.9,0.05),BLUE,"verified loop ($\\tilde{\\varepsilon}$ = 0.00053)",[10,20,40,62,120,240])]:
    ax.plot(Ls,[reset_hazard(L,e,gamma,rho)*1e3 for L in Ls],color=c,label=lab)
    Lst=reset_Lstar(e,gamma,rho); ax.axvline(Lst,color=c,lw=0.7,ls="--"); ax.text(Lst*1.04,{FLARE:9.3,BLUE:3.1}[c],f"L* = {Lst:.0f}",color=c,fontsize=7.2,fontweight="bold")
    sim=[]
    for L in Lpts:
        res=simulate(Harness(eps0=e,gamma=gamma,L=L,rho=rho),600,20000,int(L)); n_,S=survival(res); sim.append(-np.log(S[600])/600*1e3)
    ax.plot(Lpts,sim,"o",color=c,mec=INK,mew=0.6,ms=4)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("context-reset interval L (steps)"); ax.set_ylabel("per-step hazard ($\\times 10^{-3}$)")
ax.set_yticks([0.5,1,2,5,10,20]); ax.set_yticklabels(["0.5","1","2","5","10","20"]); ax.legend(loc="upper right",fontsize=6.6,frameon=True,facecolor="white",edgecolor="none",framealpha=1)
ax.set_ylim(0.3,60)
save(fig,"fig_sqrt")

# ------------------------------------------------ F9 laundering
m=np.arange(1,11); eps,r,f=0.05,0.9,0.05
ok=[];sil=[];esc=[];stS=[];stE=[];cost=[];cst=[]
for mm in m:
    a_,b_,c_,g_=verifier_step(eps,r,f,mm); ok.append(a_);sil.append(b_);esc.append(c_)
    stS.append(1-r**mm); stE.append(r**mm); cost.append(g_); cst.append((1-r**mm)/(1-r))
fig,ax=plt.subplots(1,2,figsize=(W1,2.45),gridspec_kw=dict(wspace=0.32))
ax[0].plot(m,ok,"o-",color=BLUE,ms=3.5,label="correct")
ax[0].plot(m,np.array(sil)*10,"o-",color=FLARE,ms=3.5,label="silent × 10")
ax[0].plot(m,esc,"o-",color=INK,ms=3.5,label="escalated")
ax[0].set_xlabel("retry budget m"); ax[0].set_ylabel("probability (ordinary step)"); ax[0].legend(); ax[0].set_title("ordinary step",loc="left",fontsize=8.5,fontstyle="italic"); tag(ax[0],"A",y=1.1)
ax[1].plot(m,stS,"o-",color=FLARE,ms=3.5,label="silent (error laundered)")
ax[1].plot(m,stE,"o-",color=INK,ms=3.5,label="escalated (caught)")
ax2=ax[1].twinx(); ax2.plot(m,cst,color=PERI,lw=1.3,ls="--"); ax2.set_ylabel("expected attempts",color=PERI); ax2.tick_params(colors=PERI); ax2.grid(False); ax2.spines["right"].set_visible(True); ax2.spines["right"].set_color(PERI)
ax[1].set_xlabel("retry budget m"); ax[1].set_ylabel("probability (stuck step)"); ax[1].legend(loc="lower center",fontsize=7); ax[1].set_title("stuck step",loc="left",fontsize=8.5,fontstyle="italic"); tag(ax[1],"B",y=1.1)
save(fig,"fig_laundering")

# ------------------------------------------------ F10 checkpoint cost
p=0.99; v=5; nn=np.arange(5,1001); Lc=checkpoint_Lstar(p,v)
fig,ax=plt.subplots(figsize=(3.45,2.4))
ax.plot(nn,restart_cost(nn,p,v),color=FLARE,label="check once at the end, rerun on failure")
ax.plot(nn,checkpoint_cost(nn,p,Lc,v),color=BLUE,label=f"verified checkpoints every $L_c^*$ ≈ {Lc:.0f}")
ax.plot(nn,nn,color=INK,ls=":",lw=1.1,label="ideal (no failures)")
ax.set_yscale("log"); ax.set_xlabel("task length n (steps)"); ax.set_ylabel("expected steps executed"); ax.legend(loc="lower right",fontsize=6.6)
ax.text(620,4e3,"exponential",color=FLARE,fontsize=7.6,fontweight="bold",rotation=28); ax.text(700,1.35e3,"linear",color=BLUE,fontsize=7.6,fontweight="bold")
ax.set_ylim(4,1e6)
save(fig,"fig_checkpoint")

# ------------------------------------------------ F11 ablation
A=json.load(open("ablation.json")); C=json.load(open("abl_curves.json"))
fig,ax=plt.subplots(1,2,figsize=(W1,2.7),gridspec_kw=dict(width_ratios=[1,1.15],wspace=0.35))
pal=["#9e9e9e","#cfcfcf",PERI,"#c8d2e0",BLUE,FLARE]
for (k,S),c in zip(C.items(),pal):
    ax[0].plot(np.arange(len(S)),S,color=c,lw=1.5)
ax[0].axhline(0.8,color=INK,lw=0.7,ls="--"); ax[0].set_xlim(0,1200); ax[0].set_ylim(0,1.01)
ax[0].set_xlabel("task length n (steps)"); ax[0].set_ylabel("simulated reliability R(n)"); tag(ax[0],"A")
names=["Bare loop","+ Resets","+ Verifier","+ Verifier + resets\n(L tuned for bare)","+ Verifier + resets\n(L re-tuned)","+ Better handoff\n($\\rho$ ÷ 5)"]
y=np.arange(len(A["abl"]))[::-1]
for yi,d,c,nm in zip(y,A["abl"],pal,names):
    ax[1].barh(yi,d["n80"],color=c,edgecolor=INK,lw=0.6,height=0.66)
    ax[1].errorbar(d["n80"],yi,xerr=[[d["n80"]-d["ci"][0]],[d["ci"][1]-d["n80"]]],color=INK,lw=0.8,capsize=2)
    ax[1].plot(d["pred"],yi,marker="|",ms=11,mew=1.8,color=INK)
    ax[1].text(max(d["ci"][1],d["pred"])+8,yi,f"{d['n80']}",va="center",fontsize=7.4,fontweight="bold")
ax[1].set_yticks(y); ax[1].set_yticklabels(names,fontsize=6.9); ax[1].set_xlabel("80% trust horizon $n_{80}$ (steps)")
ax[1].grid(axis="y",visible=False); ax[1].set_xlim(0,330); ax[1].tick_params(axis="y",length=0); tag(ax[1],"B",x=-0.55)
save(fig,"fig_ablation")

# ------------------------------------------------ F12 diminishing returns
miss=np.logspace(-3,0,200)  # 1-r
fig,ax=plt.subplots(figsize=(3.45,2.45))
for rho_,c,lab in [(0.0,INK,"no handoff loss ($\\rho$ = 0)"),(0.002,BLUE,"$\\rho$ = 0.002"),(0.01,FLARE,"$\\rho$ = 0.01")]:
    et=eps0*miss/(1-0.05)
    h=et+ (np.sqrt(2*et*gamma*rho_) if rho_>0 else 0)
    if rho_==0: h=et+eps0*gamma*0  # pure verifier regime (ignores rot for reference)
    ax.plot(1/miss,np.log(1.25)/h,color=c,label=lab,ls=":" if rho_==0 else "-")
# sim points for rho=0.01
pts=[]
for r_ in [0.5,0.8,0.9,0.97,0.99]:
    L_=int(round(verified_Lstar(eps0,gamma,0.01,r_,0.05)))
    res=simulate(Harness(eps0=eps0,gamma=gamma,L=L_,rho=0.01,r=r_,f=0.05,m=200),2600,8000,int(r_*100))
    pts.append((float(1/(1-r_)),float(horizon_from_sim(res,0.8))))
ax.plot(*zip(*pts),"o",color=FLARE,mec=INK,mew=0.6,ms=4.5)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("verifier quality 1/(1 − r)"); ax.set_ylabel("trust horizon $n_{80}$ (steps)")
ax.text(40,2.2e4,"slope 1",fontsize=7,color=INK,rotation=33); ax.text(150,700,"slope 1/2",fontsize=7.4,color=FLARE,fontweight="bold",rotation=17)
ax.legend(loc="upper left",fontsize=6.9); ax.set_ylim(20,1e5)
json.dump(pts,open("returns_pts.json","w"))
save(fig,"fig_returns")

# ------------------------------------------------ F13 pareto
P=R["pareto"]; colmap={"none":"#444444","lint":"#bdbdbd","tests":BLUE,"judge":PERI,"tests+judge":FLARE}
fig,ax=plt.subplots(1,2,figsize=(W1,2.75),gridspec_kw=dict(width_ratios=[1.5,1],wspace=0.33))
Q=[p_ for p_ in P if p_["succ"]>0.02]
for p_ in Q:
    mk="o" if p_["L"]==0 else "s"
    ax[0].scatter(p_["cps"]/150,p_["succ"]*100,s=14+p_["m"]*8,color=colmap[p_["v"]],edgecolor=INK,lw=0.5,marker=mk,zorder=3)
pts=sorted([(p_["cps"]/150,p_["succ"]*100) for p_ in Q]); fr=[]; best=-1
for x,y_ in pts:
    if y_>best: fr.append((x,y_)); best=y_
ax[0].plot(*zip(*fr),color=INK,lw=0.9,ls="--",zorder=2,alpha=0.7)
for k,c in colmap.items(): ax[0].scatter([],[],color=c,edgecolor=INK,lw=0.5,s=28,label=k)
ax[0].scatter([],[],color="white",edgecolor=INK,marker="s",s=28,label="+ resets")
ax[0].legend(loc="upper right",fontsize=6.5,title="sensor  (marker size = retry budget m)",title_fontsize=6.5,ncol=3,columnspacing=0.8,handletextpad=0.2)
ax[0].set_xscale("log"); ax[0].set_xlabel("cost per successful task  (× ideal cost)"); ax[0].set_ylabel("task success, n = 150 (%)")
ax[0].set_xticks([1.5,2,3,5,7,10]); ax[0].set_xticklabels(["1.5","2","3","5","7","10"]); ax[0].xaxis.set_minor_formatter(plt.NullFormatter()); ax[0].set_ylim(0,100); tag(ax[0],"A")
# silent share
cats=["none","lint","tests","judge","tests+judge"]
sil=[max([p_ for p_ in P if p_["v"]==c and p_["L"]==0],key=lambda z:z["succ"])["silent"]*100 for c in cats]
succ=[max([p_ for p_ in P if p_["v"]==c and p_["L"]==0],key=lambda z:z["succ"])["succ"]*100 for c in cats]
x=np.arange(len(cats))
ax[1].bar(x,succ,color=[colmap[c] for c in cats],edgecolor=INK,lw=0.6,label="success")
ax[1].bar(x,sil,bottom=succ,color="none",edgecolor=FLARE,hatch="////",lw=0.8,label="silent failure")
ax[1].set_xticks(x); ax[1].set_xticklabels(["none","lint","tests","judge","tests\n+judge"],fontsize=6.9)
ax[1].set_ylabel("share of tasks (%)"); ax[1].legend(loc="upper center",fontsize=6.5,ncol=2,bbox_to_anchor=(0.5,1.13)); ax[1].grid(axis="x",visible=False); ax[1].set_ylim(0,100); tag(ax[1],"B",x=-0.2)
save(fig,"fig_pareto")

import sys; sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__), '..'))
import numpy as np, itertools
from scipy.optimize import linprog
from rpascap import navmodels as nm
from rpascap.geometry import DEFAULT_MENU
from rpascap.model.two_stage import ConfigData, Instance
from rpascap.model.fast import lane_capacity_lp
def cap(K,n,mf,S=None):
    A=[];b=[]
    for l in range(n):
        if S is not None and not S[l]: continue
        A.append(mf*0.02*K[l]/1e-6); b.append(1-0.2); A.append(mf*K[l]/1e-5); b.append(1)
    ub=[(0,450 if (S is None or S[l]) else 0) for l in range(n)]
    r=linprog(-np.ones(n),A_ub=np.array(A),b_ub=b,bounds=ub); return -r.fun
for nav,m in [('E1-precise',0),('E2-measured',0.9),('E2-measured',0),('E4-KimBae',0.99)]:
    out=[]
    for cfg in DEFAULT_MENU[1:]:
        n=cfg.n_lanes; K=nm.kappa_matrix(cfg,nm.MODELS[nav],nm.Kinematics(),5.0,0.1)
        a=cap(K,n,1-m); b=max(cap(K,n,1-m,np.array(S,bool)) for S in itertools.product([0,1],repeat=n) if any(S))
        out.append(f"{cfg.name}:{a:.1f}/{b:.1f}")
    print(nav,m,' '.join(out))

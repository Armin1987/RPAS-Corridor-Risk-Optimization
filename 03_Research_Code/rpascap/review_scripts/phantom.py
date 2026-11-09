import sys; sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__), '..'))
import numpy as np
from scipy.optimize import linprog
from rpascap import navmodels as nm
from rpascap.geometry import DEFAULT_MENU
from rpascap.model.two_stage import ConfigData, Instance
from rpascap.model.fast import lane_capacity_lp
cfg=DEFAULT_MENU[4]; K=nm.kappa_matrix(cfg,nm.MODELS['E1-precise'],nm.Kinematics(),5.0,0.1)
n=4; U=np.full((1,1,n),3600*15/120.); C=np.full((1,1,n),0.02)
cd=ConfigData(n,4.0,U,C,K[None,None]); ins=Instance([[cd]],np.zeros((1,1,1)),np.zeros((1,1)),np.ones(1),np.zeros(1,int),np.ones(1))
F,x=lane_capacity_lp(cd,0,0,ins); print('code F*',round(F,1),np.round(x,1))
# true per-aircraft semantics: rows only for lanes actually used; enumerate supports
best=0
import itertools
for S in itertools.product([0,1],repeat=n):
    S=np.array(S,bool)
    if not S.any(): continue
    A=[];b=[]
    for l in np.flatnonzero(S):
        A.append(0.02*K[l]/1e-6); b.append(1-1e-5*0.02/1e-6); A.append(K[l]/1e-5); b.append(1)
    r=linprog(-np.ones(n),A_ub=np.array(A),b_ub=b,bounds=[(0,450 if s else 0) for s in S])
    if -r.fun>best: best=-r.fun; bx=r.x
print('rows only on used lanes F*',round(best,1),np.round(bx,1))

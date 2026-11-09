import sys; sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__), '..'))
import numpy as np
from rpascap.model import two_stage as ts
from rpascap.model.fast import solve_fast
from tests.helpers import random_instance
worst=0; n=0
for seed in range(60):
    rng=np.random.default_rng(seed+999)
    ins=random_instance(seed=seed, K=1, configs_per_k=4, lanes=(0,2,3,1), T=8, B=4, W=3, theta_A=[1e-5,2e-6,5e-7][seed%3])
    for cl in ins.configs:
        for cd in cl:
            cd.C[:] = rng.uniform(0.0, 0.15, cd.C.shape)   # some lanes lambda_f*C > theta_G
    ins.max_switches=seed%4; ins.switch_cost=[0.0,5.0,0.5][seed%3]
    ins.mitigation=[0.0,0.9,1.0][seed%3]
    w=rng.random(3); ins.p=w/w.sum()
    a=ts.solve(ins, mip_rel_gap=1e-9); b=solve_fast(ins)
    d=abs(a.obj-b.obj); worst=max(worst,d); n+=1
    if d>1e-5*max(1,abs(a.obj)): print('MISMATCH',seed,a.obj,b.obj)
print('instances',n,'max absdiff',worst)

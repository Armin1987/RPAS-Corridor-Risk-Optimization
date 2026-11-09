import sys; sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__), '..'))
import yaml, numpy as np
from pathlib import Path
from rpascap.study import Study
from rpascap.geometry import lanes, lane_points
import os, sys as _s
ROOT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')
cfg=yaml.safe_load(open(os.path.join(ROOT,'configs','study.yaml')))
P=Path(_s.argv[1])   # processed inputs directory
st=Study.__new__(Study); st.cfg=cfg; st.log=print; st.processed=P
import json; g=json.load(open(P/'grid_meta.json')); st.lat0,st.lon0,st.cell=g['LAT0'],g['LON0'],g['CELL_m']
st._exposure(); st._corridors()
print('coverage frac of grid', st.coverage.mean(), 'boxes', len(st.boxes))
for cor in st.corridors:
    for cfgm in st.menu[1:]:
        for L in lanes(cor,cfgm):
            xs,ys=lane_points(cor,L['offset'],st.cell)
            rr=(np.floor(ys/st.cell)-st.r0).astype(int); cc=(np.floor(xs/st.cell)-st.c0).astype(int)
            inside=st.coverage[rr,cc]
            if (~inside).any(): print(cor.name,cfgm.name,L['offset'],'pts outside coverage',(~inside).sum(),'/',len(inside))
print(st.boxes[:3])

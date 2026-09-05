#!/usr/bin/env python3
"""A1 vs CTRL bootstrap 稳定性验证（多次大样本）"""
import json
import random
 
A1_PATHS = {42: '/root/data/alfworld/opd/a1/d3_eval/134/a1_strong_134.json',
            7: '/root/data/alfworld/opd/a1/a3/eval/134/a1_seed7_134.json',
            123: '/root/data/alfworld/opd/a1/a3/eval/134/a1_seed123_134.json',
            2024: '/root/data/alfworld/opd/a1/a3/eval/134/a1_seed2024_134.json'}
CTRL_PATHS = {42: '/root/data/alfworld/opd/a1/a3/eval/134/ctrl/ctrl_seed42_134.json',
              7: '/root/data/alfworld/opd/a1/a3/eval/134/ctrl/ctrl_seed7_134.json',
              123: '/root/data/alfworld/opd/a1/a3/eval/134/ctrl/ctrl_seed123_134.json',
              2024: '/root/data/alfworld/opd/a1/a3/eval/134/ctrl/ctrl_seed2024_134.json'}
 
def load(p):
    return json.load(open(p))['per_game']
 
a1_per, ctrl_per = [], []
for s in [42, 7, 123, 2024]:
    g1, g2 = load(A1_PATHS[s]), load(CTRL_PATHS[s])
    assert set(g1) == set(g2)
    a1_per.append(g1); ctrl_per.append(g2)
games = sorted(a1_per[0])
 
for run in range(5):
    rng = random.Random(1000 + run)
    diffs = []
    for _ in range(50000):
        gs = [rng.choice(games) for _ in games]
        ds = []
        for i in range(4):
            w1 = sum(a1_per[i][g] for g in gs) / len(gs)
            w2 = sum(ctrl_per[i][g] for g in gs) / len(gs)
            ds.append(w1 - w2)
        diffs.append(sum(ds) / 4)
    diffs.sort()
    lo, hi = diffs[1250], diffs[48750]
    print(f'run{run}: A1-CTRL CI [{lo*100:+.2f}pp, {hi*100:+.2f}pp]', '排除0' if lo > 0 else '跨0')

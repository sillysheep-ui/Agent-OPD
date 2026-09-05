#!/usr/bin/env python3
"""A1 vs CTRL bootstrap——统一 A1-CTRL 方向（修复符号）"""
import json
import random
 
A1_PATHS = {
    42: '/root/data/alfworld/opd/a1/d3_eval/134/a1_strong_134.json',
    7: '/root/data/alfworld/opd/a1/a3/eval/134/a1_seed7_134.json',
    123: '/root/data/alfworld/opd/a1/a3/eval/134/a1_seed123_134.json',
    2024: '/root/data/alfworld/opd/a1/a3/eval/134/a1_seed2024_134.json',
}
CTRL_PATHS = {
    42: '/root/data/alfworld/opd/a1/a3/eval/134/ctrl/ctrl_seed42_134.json',
    7: '/root/data/alfworld/opd/a1/a3/eval/134/ctrl/ctrl_seed7_134.json',
    123: '/root/data/alfworld/opd/a1/a3/eval/134/ctrl/ctrl_seed123_134.json',
    2024: '/root/data/alfworld/opd/a1/a3/eval/134/ctrl/ctrl_seed2024_134.json',
}
 
def load(p):
    d = json.load(open(p))
    return d['per_game']
 
a1_per, ctrl_per = [], []
for s in [42, 7, 123, 2024]:
    g1 = load(A1_PATHS[s]); g2 = load(CTRL_PATHS[s])
    assert set(g1) == set(g2)
    a1_per.append(g1); ctrl_per.append(g2)
 
games = sorted(a1_per[0])
means = []
for i in range(4):
    m1 = sum(a1_per[i].values()) / len(games)
    m2 = sum(ctrl_per[i].values()) / len(games)
    means.append(m1 - m2)   # A1 - CTRL per seed
    print(f'seed{ [42,7,123,2024][i] }: A1={m1*100:.1f} CTRL={m2*100:.1f} A1-CTRL={ (m1-m2)*100:+.2f}pp')
print(f'mean A1-CTRL: {sum(means)/4*100:+.2f}pp')
 
# paired bootstrap: diff = A1 - CTRL（每 seed 内 A1 胜率 - CTRL 胜率，再对 seed 平均）
rng = random.Random(42)
diffs = []
for _ in range(10000):
    gs = [rng.choice(games) for _ in games]
    ds = []
    for i in range(4):
        w1 = sum(a1_per[i][g] for g in gs) / len(gs)
        w2 = sum(ctrl_per[i][g] for g in gs) / len(gs)
        ds.append(w1 - w2)
    diffs.append(sum(ds) / 4)
diffs.sort()
lo, hi = diffs[250], diffs[9750]
print(f'A1-CTRL 95% CI: [{lo*100:+.2f}pp, {hi*100:+.2f}pp]')
print(f'排除 0: {"是" if lo > 0 else "否——边缘跨 0"}')

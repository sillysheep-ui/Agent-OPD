#!/usr/bin/env python3
"""matched-count 最终分析: A1_mc vs CTRL（B=50000 game-level paired bootstrap）
A1_mc = A1 下采样到 CTRL 样本数（127/129/125/128——subsample seed 0 固定）"""
import json
import random
 
A1MC_PATHS = {
    42: '/root/data/alfworld/opd/a1/a3/eval/134/a1_mc/a1_mc_seed42_134.json',
    7: '/root/data/alfworld/opd/a1/a3/eval/134/a1_mc/a1_mc_seed7_134.json',
    123: '/root/data/alfworld/opd/a1/a3/eval/134/a1_mc/a1_mc_seed123_134.json',
    2024: '/root/data/alfworld/opd/a1/a3/eval/134/a1_mc/a1_mc_seed2024_134.json',
}
CTRL_PATHS = {
    42: '/root/data/alfworld/opd/a1/a3/eval/134/ctrl/ctrl_seed42_134.json',
    7: '/root/data/alfworld/opd/a1/a3/eval/134/ctrl/ctrl_seed7_134.json',
    123: '/root/data/alfworld/opd/a1/a3/eval/134/ctrl/ctrl_seed123_134.json',
    2024: '/root/data/alfworld/opd/a1/a3/eval/134/ctrl/ctrl_seed2024_134.json',
}
A1_PATHS = {  # 原始 A1（对照）
    42: '/root/data/alfworld/opd/a1/d3_eval/134/a1_strong_134.json',
    7: '/root/data/alfworld/opd/a1/a3/eval/134/a1_seed7_134.json',
    123: '/root/data/alfworld/opd/a1/a3/eval/134/a1_seed123_134.json',
    2024: '/root/data/alfworld/opd/a1/a3/eval/134/a1_seed2024_134.json',
}
 
def load(p):
    d = json.load(open(p))
    return d['per_game']
 
amc, ctrl, a1 = [], [], []
for s in [42, 7, 123, 2024]:
    g1 = load(A1MC_PATHS[s]); g2 = load(CTRL_PATHS[s]); g3 = load(A1_PATHS[s])
    assert set(g1) == set(g2) == set(g3), f'seed{s} games mismatch'
    amc.append(g1); ctrl.append(g2); a1.append(g3)
 
games = sorted(amc[0])
print('=== 单 seed 成功率 ===')
for i, s in enumerate([42, 7, 123, 2024]):
    m1 = sum(a1[i].values()) / len(games)
    m2 = sum(amc[i].values()) / len(games)
    m3 = sum(ctrl[i].values()) / len(games)
    print(f'seed{s}: A1={m1*100:.1f} A1mc={m2*100:.1f} CTRL={m3*100:.1f} | A1mc-CTRL={(m2-m3)*100:+.2f}pp')
 
def boot(x, y, rng_seed, n=50000):
    rng = random.Random(rng_seed)
    diffs = []
    for _ in range(n):
        gs = [rng.choice(games) for _ in games]
        ds = []
        for i in range(4):
            w1 = sum(x[i][g] for g in gs) / len(gs)
            w2 = sum(y[i][g] for g in gs) / len(gs)
            ds.append(w1 - w2)
        diffs.append(sum(ds) / 4)
    diffs.sort()
    return diffs[1250], diffs[48750], sum(diffs) / len(diffs)
 
print('\n=== matched-count 判定（A1mc vs CTRL——样本数对齐）===')
lo, hi, mean = boot(amc, ctrl, 42)
print(f'A1mc-CTRL = {mean*100:+.2f}pp, 95% CI [{lo*100:+.2f}, {hi*100:+.2f}]pp')
print('✅ 排除 0（显著）' if lo > 0 else '❌ 跨 0')
 
print('\n=== 对照（原始 A1 vs CTRL——B=50000 同口径）===')
lo, hi, mean = boot(a1, ctrl, 42)
print(f'A1-CTRL = {mean*100:+.2f}pp, 95% CI [{lo*100:+.2f}, {hi*100:+.2f}]pp')
 
print('\n=== A1 vs A1mc（下采样损失检验）===')
lo, hi, mean = boot(a1, amc, 42)
print(f'A1-A1mc = {mean*100:+.2f}pp, 95% CI [{lo*100:+.2f}, {hi*100:+.2f}]pp')

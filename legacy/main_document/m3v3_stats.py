#!/usr/bin/env python3
"""M3 v3 统计协议:
Primary:   T ~ G + S + G×S + X（X=action_type/technical/repeat_obs/sim_topK）——game-level cluster bootstrap
Secondary: pooled S 分位 5 桶（统一 cut points）——每桶 A1 vs A3
"""
import json
import random
from collections import defaultdict
import numpy as np
 
BASE = '/root/data/alfworld/opd/a1/a3/run1/'
rows = []
for g in ['A1', 'A3']:
    rows += [json.loads(l) for l in open(f'{BASE}m3v3_result_{g}_k10_both.jsonl')]
for r in rows:
    r['G'] = 1 if r['group'] == 'A3' else 0
print(f'loaded: A1={sum(1 for r in rows if r["group"]=="A1")}, A3={sum(1 for r in rows if r["group"]=="A3")}')
 
# game-level 聚合（每 game 平均——cluster 单位）
by_game = defaultdict(list)
for r in rows:
    by_game[r['gamefile']].append(r)
 
game_rows = []
for g, rs in by_game.items():
    n = len(rs)
    game_rows.append({
        'game': g, 'n': n,
        'S': sum(r['S_i'] for r in rs) / n,
        'T': sum(r['T_i'] for r in rs) / n,
        'G': sum(r['G'] for r in rs) / n,     # 组混合比例（A1=0, A3=1）
        'sim': sum(r['sim_topK'] for r in rs) / n,
        'technical': sum(r['technical'] for r in rs) / n,
        'repeat_obs': sum(r['repeat_obs'] for r in rs) / n,
        'action_type_look': sum(1 for r in rs if r['action_type'] == 'look') / n,
    })
 
def ols(rows, cols):
    X = np.array([[1] + [r[c] for c in cols] for r in rows], dtype=float)
    y = np.array([r['T'] for r in rows], dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta
 
print('=== Primary: T ~ 1 + S + G + S×G + sim + technical + repeat_obs + action_type_look ===')
cols = ['S', 'G', 'sim', 'technical', 'repeat_obs', 'action_type_look']
# 交互项
for r in game_rows:
    r['SxG'] = r['S'] * r['G']
beta = ols(game_rows, cols + ['SxG'])
names = ['intercept'] + cols + ['SxG']
for n_, b in zip(names, beta):
    print(f'  {n_:<18}: {b:+.4f}')
 
# game-level cluster bootstrap for γ (G 系数) 和 SxG
rng = random.Random(42)
games = [r['game'] for r in game_rows]
B = 5000
gammas, sxgs = [], []
for _ in range(B):
    gs = [rng.choice(games) for _ in games]
    sub = [next(r for r in game_rows if r['game'] == g) for g in gs]
    b = ols(sub, cols + ['SxG'])
    gammas.append(b[2])
    sxgs.append(b[6])
gammas.sort(); sxgs.sort()
print(f'\n  γ (G) 95% CI: [{gammas[125]:+.4f}, {gammas[4975]:+.4f}]', '✅ 显著' if gammas[125] > 0 or gammas[4975] < 0 else '❌ 不显著')
print(f'  S×G 95% CI: [{sxgs[125]:+.4f}, {sxgs[4975]:+.4f}]')
 
# 无控制变量的简单版本（对比）
print('\n=== Primary (无控制): T ~ 1 + S + G ===')
beta2 = ols(game_rows, ['S', 'G'])
print(f'  intercept {beta2[0]:+.4f}, S {beta2[1]:+.4f}, G {beta2[2]:+.4f}')
gammas2 = []
for _ in range(B):
    gs = [rng.choice(games) for _ in games]
    sub = [next(r for r in game_rows if r['game'] == g) for g in gs]
    gammas2.append(ols(sub, ['S', 'G'])[2])
gammas2.sort()
print(f'  γ 95% CI: [{gammas2[125]:+.4f}, {gammas2[4975]:+.4f}]', '✅' if gammas2[125] > 0 or gammas2[4975] < 0 else '❌')
 
# Secondary: pooled S 分位 5 桶
print('\n=== Secondary: pooled S 分位 5 桶 ===')
all_S = sorted(r['S_i'] for r in rows)
cuts = [all_S[int(len(all_S) * q)] for q in [0.2, 0.4, 0.6, 0.8]]
print(f'cut points: {[f"{c:.3f}" for c in cuts]}')
buckets = [(-1e9, cuts[0]), (cuts[0], cuts[1]), (cuts[1], cuts[2]), (cuts[2], cuts[3]), (cuts[3], 1e9)]
for bi, (lo, hi) in enumerate(buckets):
    sub = [r for r in rows if lo <= r['S_i'] < hi]
    a1 = [r for r in sub if r['group'] == 'A1']
    a3 = [r for r in sub if r['group'] == 'A3']
    if not a1 or not a3:
        print(f'  bin{bi+1} [{lo:.3f},{hi:.3f}): A1 n={len(a1)} A3 n={len(a3)}——空桶')
        continue
    t1 = sum(r['T_i'] for r in a1) / len(a1)
    t3 = sum(r['T_i'] for r in a3) / len(a3)
    s1 = sum(r['S_i'] for r in a1) / len(a1)
    s3 = sum(r['S_i'] for r in a3) / len(a3)
    print(f'  bin{bi+1} [{lo:.3f},{hi:.3f}): A1 n={len(a1)} T={t1:.4f} S={s1:.4f} | A3 n={len(a3)} T={t3:.4f} S={s3:.4f} | ΔT={t3-t1:+.4f}')
 
# sim 与 T 的关系（sanity check）
print('\n=== sanity: T vs sim（pooled） ===')
sims = sorted(r['sim_topK'] for r in rows)
mid = sims[len(sims) // 2]
lo_s = [r for r in rows if r['sim_topK'] < mid]
hi_s = [r for r in rows if r['sim_topK'] >= mid]
print(f'low-sim: T={sum(r["T_i"] for r in lo_s)/len(lo_s):.4f} (n={len(lo_s)})')
print(f'high-sim: T={sum(r["T_i"] for r in hi_s)/len(hi_s):.4f} (n={len(hi_s)})')
print('→ 越相似 transfer 越高？', '✅ 是' if sum(r['T_i'] for r in hi_s)/len(hi_s) > sum(r['T_i'] for r in lo_s)/len(lo_s) else '❌ 否')

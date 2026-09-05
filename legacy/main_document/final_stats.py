#!/usr/bin/env python3
"""Untouched held-out ID final test 统计（预冻结协议: B=50000 固定 rng 42）"""
import json
import random
from collections import defaultdict
 
D = '/root/data/alfworld/opd/final_test/'
def load(name):
    return json.load(open(f'{D}{name}_id140.json'))['per_game']
 
sft = load('sft'); a1 = load('a1r42'); ctrl = load('ctrl42'); a1mc = load('a1mc42')
games = sorted(sft)
assert all(set(x) == set(games) for x in [a1, ctrl, a1mc])
 
def sr(d):
    return sum(d.values()) / len(games) * 100
print('=== overall success（140 ID held-out）===')
for n, d in [('SFT', sft), ('A1R42', a1), ('CTRL42', ctrl), ('A1mc42', a1mc)]:
    print(f'{n}: {sr(d):.1f}%')
 
def boot(x, y, rng_seed=42, n=50000):
    rng = random.Random(rng_seed)
    diffs = [sum(x[g] - y[g] for g in [rng.choice(games) for _ in games]) / len(games)
             for _ in range(n)]
    diffs.sort()
    return diffs[1250], diffs[48750], sum(diffs) / len(diffs)
 
print('\n=== 预冻结主检验（paired bootstrap B=50000）===')
for label, x, y in [('H1: A1R42 > SFT', a1, sft),
                    ('H2: A1mc42 > CTRL42', a1mc, ctrl)]:
    lo, hi, mean = boot(x, y)
    print(f'{label}: Δ={mean*100:+.2f}pp, 95% CI [{lo*100:+.2f}, {hi*100:+.2f}]pp ->',
          '✅ 排除0' if lo > 0 else '❌ 跨0')
 
print('\n=== secondary（不预设显著）===')
for label, x, y in [('A1R42-CTRL42', a1, ctrl), ('A1mc42-A1R42', a1mc, a1)]:
    lo, hi, mean = boot(x, y)
    print(f'{label}: Δ={mean*100:+.2f}pp, CI [{lo*100:+.2f}, {hi*100:+.2f}]pp')
 
print('\n=== 6 类 task breakdown（descriptive）===')
def by_task(d):
    r = defaultdict(list)
    for g, v in d.items():
        tt = g.split('/')[-3]
        r[tt].append(v)
    return r
bt = {n: by_task(d) for n, d in [('SFT', sft), ('A1R42', a1), ('CTRL42', ctrl), ('A1mc42', a1mc)]}
tasks = sorted(set().union(*[set(b.keys()) for b in bt.values()]))
print(f'{"task":<32}{"SFT":>8}{"A1R42":>8}{"CTRL42":>8}{"A1mc42":>8}')
for t in tasks:
    print(f'{t:<32}' + ''.join(f'{sum(bt[m][t])/len(bt[m][t])*100:>8.1f}' for m in ['SFT','A1R42','CTRL42','A1mc42']))

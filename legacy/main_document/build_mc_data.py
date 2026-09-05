#!/usr/bin/env python3
"""matched-count 下采样: A1 parquet → CTRL 的 n（subsample_seed=0——固定规则）"""
import json
import random
import pandas as pd
 
A1_PARQUETS = {
    42: '/root/data/alfworld/opd/a1/run1/a1_ce_train.parquet',
    7: '/root/data/alfworld/opd/a1/a3/run1/a1_seed7_train.parquet',
    123: '/root/data/alfworld/opd/a1/a3/run1/a1_seed123_train.parquet',
    2024: '/root/data/alfworld/opd/a1/a3/run1/a1_seed2024_train.parquet',
}
 
for s in [42, 7, 123, 2024]:
    a1 = pd.read_parquet(A1_PARQUETS[s])
    ctrl = pd.read_parquet(f'/root/data/alfworld/opd/a1/a3/run1/ctrl_seed{s}_train.parquet')
    n_ctrl = len(ctrl)
    n_a1 = len(a1)
    # 固定 subsample seed 0
    rng = random.Random(0)
    idx = sorted(rng.sample(range(n_a1), n_ctrl))
    sub = a1.iloc[idx].reset_index(drop=True)
    out = f'/root/data/alfworld/opd/a1/a3/run1/a1_mc_seed{s}_train.parquet'
    sub.to_parquet(out, index=False)
    print(f'seed{s}: A1 {n_a1} -> mc {len(sub)} (CTRL {n_ctrl})')

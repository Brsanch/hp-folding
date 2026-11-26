"""Test whether 'more seeds' actually expands coverage or just plateaus.

Run ONE method (PERM) with many seeds, watching coverage as a function of
seed count. If coverage plateaus well below 100%, more seeds doesn't help —
the method itself is algorithmically biased.

Compare to ALL methods × fewer seeds (same total runs). If method-diversity
beats seed-diversity at the same compute budget, that's evidence the seeds
are exploring the same biased subspace.
"""
from __future__ import annotations
from pathlib import Path
import sys, time
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from exclusion_analyzer import (
    canonicalize_fold, enumerate_all_optima,
    run_perm, run_basin_hop, run_replica_exchange, run_saddle_hop, run_crossover_ga,
)

SEQ = 'PPHPPHPHPHHPPHPH'

print('Enumerating all optima...', flush=True)
best, all_optima, _ = enumerate_all_optima(SEQ, time_limit=60)
print(f'  optimum={best}, total optima={len(all_optima)}\n', flush=True)

CHECKPOINTS = [10, 25, 50, 100, 200, 400]
METHOD_RUNNERS = {
    'PERM': run_perm,
    'basin_hop': run_basin_hop,
    'rep_exch': run_replica_exchange,
    'saddle_hop': run_saddle_hop,
}

# ── Experiment A: PERM alone, growing seed count ──
print('=== Experiment A: PERM × N seeds (single method, more dice) ===', flush=True)
print(f'{"N seeds":>8s}  {"unique opts":>12s}  {"coverage":>8s}', flush=True)
reached_perm_only = set()
t0 = time.time()
for seed in range(CHECKPOINTS[-1]):
    try:
        c, canon = run_perm(SEQ, seed, 2.5)
        if c == best:
            reached_perm_only.add(canon)
    except Exception: pass
    if (seed + 1) in CHECKPOINTS:
        cov = len(reached_perm_only & all_optima)
        print(f'{seed+1:>8d}  {cov:>9d}/{len(all_optima)}  {100*cov/len(all_optima):>6.1f}%', flush=True)
print(f'  total wall: {time.time()-t0:.0f}s', flush=True)

# ── Experiment B: 4 methods × N/4 seeds (method diversity) ──
print('\n=== Experiment B: 4 methods × (N/4) seeds (method diversity, same total budget) ===', flush=True)
print(f'{"total runs":>10s}  {"unique opts":>12s}  {"coverage":>8s}', flush=True)
reached_multi = set()
t0 = time.time()
total_runs = 0
for seed in range(CHECKPOINTS[-1] // 4):
    for mname, mfn in METHOD_RUNNERS.items():
        try:
            c, canon = mfn(SEQ, seed, 2.5)
            if c == best:
                reached_multi.add(canon)
        except Exception: pass
        total_runs += 1
        if total_runs in CHECKPOINTS:
            cov = len(reached_multi & all_optima)
            print(f'{total_runs:>10d}  {cov:>9d}/{len(all_optima)}  {100*cov/len(all_optima):>6.1f}%', flush=True)
print(f'  total wall: {time.time()-t0:.0f}s', flush=True)

# Compare final
print('\n=== COMPARISON at equal compute (400 runs each) ===', flush=True)
print(f'PERM-only:    {len(reached_perm_only & all_optima)}/{len(all_optima)} = '
      f'{100*len(reached_perm_only & all_optima)/len(all_optima):.1f}%', flush=True)
print(f'4-method:     {len(reached_multi & all_optima)}/{len(all_optima)} = '
      f'{100*len(reached_multi & all_optima)/len(all_optima):.1f}%', flush=True)

# Check: is PERM-only really finding NEW optima at high seed count?
# Compute incremental gain: if at N=200 we have X, at N=400 we have Y, gain=Y-X
# If gain is small, plateau is hit.
print('\n(If PERM-only plateaus well below the 4-method run, seeds ≠ exploration.)', flush=True)

"""Test whether sprawl_sa actually reaches the optima our standard methods miss.

  1. Re-enumerate all 182 optima for L16_pseudo_S6.
  2. Identify the ones our standard methods systematically miss (by re-running
     a quick PERM/basin_hop/RE pass and subtracting reached set).
  3. Run sprawl_sa with diverse seeds.
  4. Report: does sprawl_sa reach previously-missed optima?
"""
from __future__ import annotations
from pathlib import Path
import sys, time, random
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from hp_folding import sprawl_sa, perm, search
from exclusion_analyzer import (
    canonicalize_fold, enumerate_all_optima, NBRS,
    run_perm, run_basin_hop, run_replica_exchange,
)

SEQ = 'PPHPPHPHPHHPPHPH'   # L16_pseudo_S6

print('Step 1: enumerate all canonical optima')
t0 = time.time()
best, all_optima, n_total = enumerate_all_optima(SEQ, time_limit=60)
print(f'  optimum={best}, total optima={len(all_optima)}, '
      f'enumerated {n_total} folds in {time.time()-t0:.1f}s')

print('\nStep 2: standard-methods baseline (50 seeds × 3 fast methods, 3s budget)')
standard_reached = set()
for seed in range(50):
    for runner in (run_perm, run_basin_hop, run_replica_exchange):
        try:
            c, canon = runner(SEQ, seed, 3.0)
            if c == best:
                standard_reached.add(canon)
        except Exception:
            pass
standard_reached &= all_optima
missed_by_standard = all_optima - standard_reached
print(f'  standard methods reached {len(standard_reached)}/{len(all_optima)} '
      f'({100*len(standard_reached)/len(all_optima):.1f}%)')
print(f'  {len(missed_by_standard)} optima missed by standard methods')

print('\nStep 3: sprawl_sa runs with varying alpha/beta')
configs = [
    {'alpha': 0.3, 'beta': 0.1, 'label': 'mild_sprawl'},
    {'alpha': 0.5, 'beta': 0.2, 'label': 'moderate_sprawl'},
    {'alpha': 0.8, 'beta': 0.4, 'label': 'strong_sprawl'},
    {'alpha': 1.5, 'beta': 0.6, 'label': 'aggressive_sprawl'},
]
sprawl_reached_all = set()
for cfg in configs:
    cfg_reached = set()
    for seed in range(40):
        try:
            res = sprawl_sa.sprawl_sa(SEQ, iterations=80_000, t_start=3.0, t_end=0.02,
                                        alpha=cfg['alpha'], beta=cfg['beta'], seed=seed)
            if res.contacts == best:
                cfg_reached.add(canonicalize_fold(res.fold.coords))
        except Exception:
            pass
    cfg_reached &= all_optima
    new_optima = cfg_reached - standard_reached
    print(f'  {cfg["label"]:20s} (α={cfg["alpha"]}, β={cfg["beta"]}): '
          f'reached {len(cfg_reached)}/{len(all_optima)} optima, '
          f'NEW (vs standard) = {len(new_optima)}')
    sprawl_reached_all |= cfg_reached

new_overall = sprawl_reached_all - standard_reached
print(f'\n=== sprawl_sa overall ===')
print(f'  reached {len(sprawl_reached_all)}/{len(all_optima)} '
      f'({100*len(sprawl_reached_all)/len(all_optima):.1f}%)')
print(f'  reached {len(new_overall)} optima that standard methods MISSED')

# Combined coverage
combined = standard_reached | sprawl_reached_all
print(f'\n=== combined (standard + sprawl_sa) ===')
print(f'  reached {len(combined)}/{len(all_optima)} '
      f'({100*len(combined)/len(all_optima):.1f}%)')
print(f'  improvement vs standard alone: '
      f'+{100*(len(combined)-len(standard_reached))/len(all_optima):.1f}%')

if len(new_overall) > 0:
    print(f'\nVERDICT: sprawl_sa reached {len(new_overall)} previously-excluded optima.')
    print('         Bias fix works on this synthetic.')
else:
    print(f'\nVERDICT: sprawl_sa did NOT reach any new optima.')
    print('         The bias may need different intervention.')

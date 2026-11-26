#!/usr/bin/env bash
# Launch ~24 parallel record-attempt runs (8 per sequence × S6/S9/S10).
# Each run gets 6h budget, different config angle to escape the current best basin.
#
# Configs per sequence:
#   re_macro_seed{11,22}    — replica exchange, macro move set, mid-T (0.10..3.5)
#   re_hot_seed{33,44}      — replica exchange, macro+, HOT-T (0.20..6.0)
#   re_random_seed{55,66}   — replica exchange from RANDOM init (escape current basin)
#   wl_seed{77,88}          — Wang-Landau (one from best, one from random)
set -u
cd "$(dirname "$0")"
PYTHON="/opt/homebrew/bin/python3"
mkdir -p record_attempt_results

run_one() {
  local sid="$1" tag="$2" seed="$3" mode="$4" tmin="$5" tmax="$6" from_best="$7"
  $PYTHON -u -c "
import sys, json, time, random, os, math
sys.path.insert(0, 'src')
from hp_folding import sequences, search, fold as fold_mod, verifier, moves, wang_landau
sid='$sid'; tag='$tag'; seed=$seed; mode='$mode'; tmin=$tmin; tmax=$tmax; from_best=$from_best
s = sequences.get(sid)
seq = s.sequence

# Find best known fold for this sequence across all results dirs.
best_path=None; best_c=-1
for d in ['replica_exchange_results','basin_hop_results','perm_results']:
    if not os.path.isdir(d): continue
    for f in os.listdir(d):
        if not f.startswith(sid + '_') or not f.endswith('.json'): continue
        try:
            data = json.load(open(f'{d}/{f}'))
            c = data.get('contacts', 0)
            if c > best_c: best_c = c; best_path = f'{d}/{f}'
        except: pass

print(f'{sid}/{tag} mode={mode} from_best={from_best} (best known fold {best_c} at {best_path})', flush=True)

# Build closures for moves that need sequence
def cp(c, r): return moves.compact_pivot(c, r, seq)
def sr(c, r): return moves.subchain_reflect(c, r, seq)
def cm(c, r): return moves.cluster_merge(c, r, seq)

def macro_plus(coords, rng):
    u = rng.random()
    if   u < 0.20: return moves.pull_move_v2(coords, rng)
    elif u < 0.32: return moves.lmw_pull_full(coords, rng)
    elif u < 0.45: return moves.pivot(coords, rng)
    elif u < 0.55: return moves.local_rebridge(coords, rng)
    elif u < 0.65: return cp(coords, rng)
    elif u < 0.73: return moves.bond_rebridge(coords, rng)
    elif u < 0.79: return cm(coords, rng)
    elif u < 0.85: return moves.fragment_random_walk(coords, rng)
    elif u < 0.91: return sr(coords, rng)
    elif u < 0.96: return moves.crankshaft(coords, rng)
    else:          return moves.end_flip(coords, rng)

t0 = time.time()
TIME_BUDGET = 21600  # 6 hours

if mode == 're':
    # Build temperature ladder
    n_temps = 8
    log_tmin = math.log(tmin); log_tmax = math.log(tmax)
    temps = tuple(math.exp(log_tmin + (log_tmax-log_tmin)*i/(n_temps-1)) for i in range(n_temps))

    if from_best and best_path:
        seed_data = json.load(open(best_path))
        seed_fold = fold_mod.Fold(sequence=seq, coords=[tuple(c) for c in seed_data['coords']])
        initial_folds = [seed_fold.copy() for _ in temps]
    else:
        initial_folds = None  # straight-line default

    res = search.replica_exchange(seq, temperatures=temps, iterations=20_000_000,
                                    exchange_every=1000, seed=seed,
                                    move_fn=macro_plus, initial_folds=initial_folds)
    print(f'{sid}/{tag} DONE  contacts={res.contacts}/-{s.best_known}  wall={time.time()-t0:.0f}s', flush=True)
    out = res.fold.coords
    contacts = res.contacts
else:  # wang_landau
    if from_best and best_path:
        seed_data = json.load(open(best_path))
        initial_coords = [tuple(c) for c in seed_data['coords']]
    else:
        initial_coords = None
    res = wang_landau.wang_landau(seq, initial_coords=initial_coords,
                                    time_budget_seconds=TIME_BUDGET,
                                    rng_seed=seed)
    contacts = -res.best_energy if hasattr(res, 'best_energy') else res.contacts
    out = res.best_coords if hasattr(res, 'best_coords') else res.coords
    print(f'{sid}/{tag} DONE  contacts={contacts}/-{s.best_known}  wall={time.time()-t0:.0f}s', flush=True)

with open(f'record_attempt_results/{sid}_{tag}.json','w') as f:
    json.dump({'sequence_id':sid,'sequence':seq,'n_residues':s.length,
               'n_h_residues':seq.count('H'),'contacts':int(contacts),
               'coords':[list(c) for c in out],
               'note':f'mode={mode} seed={seed} T={tmin}-{tmax} from_best={from_best}'}, f, indent=2)
" > record_attempt_results/${sid}_${tag}.log 2>&1 &
  disown
  echo "  $sid/$tag → PID $!"
}

for sid in S6 S9 S10; do
  echo "=== $sid ==="
  run_one $sid re_macro_s11   11 re 0.10 3.5 1
  run_one $sid re_macro_s22   22 re 0.10 3.5 1
  run_one $sid re_hot_s33     33 re 0.20 6.0 1
  run_one $sid re_hot_s44     44 re 0.20 6.0 1
  run_one $sid re_random_s55  55 re 0.10 3.5 0
  run_one $sid re_random_s66  66 re 0.20 6.0 0
  run_one $sid wl_best_s77    77 wl 0.0 0.0 1
  run_one $sid wl_random_s88  88 wl 0.0 0.0 0
done

sleep 5
echo ""
echo "=== alive ==="
pgrep -f "from hp_folding import sequences" | wc -l

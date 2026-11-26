"""Single Wang-Landau run on a sequence, optionally seeded from current best."""
from pathlib import Path
import sys, json, time, os
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from hp_folding import sequences, wang_landau

if len(sys.argv) != 4:
    print("usage: wl_one.py <sid> <seed> <from_best:0|1>", file=sys.stderr)
    sys.exit(1)
sid, seed, from_best = sys.argv[1], int(sys.argv[2]), bool(int(sys.argv[3]))
s = sequences.get(sid)

best_path = None; best_c = -1
for d in ['replica_exchange_results', 'basin_hop_results', 'perm_results']:
    full = str(Path(__file__).resolve().parent / d)
    if not os.path.isdir(full):
        continue
    for f in os.listdir(full):
        if not f.startswith(sid + '_') or not f.endswith('.json'):
            continue
        try:
            data = json.load(open(f'{full}/{f}'))
            c = data.get('contacts', 0)
            if c > best_c:
                best_c = c
                best_path = f'{full}/{f}'
        except Exception:
            pass

print(f'{sid}/wl_seed{seed}_fb{int(from_best)} starting (best known {best_c} from {best_path})', flush=True)
init = None
if from_best and best_path:
    init = [tuple(c) for c in json.load(open(best_path))['coords']]

t0 = time.time()
res = wang_landau.wang_landau(s.sequence, initial_coords=init,
                               time_budget_seconds=21600, rng_seed=seed)
contacts = res.best_contacts
coords = res.best_coords
print(f'{sid}/wl_seed{seed}_fb{int(from_best)} DONE  contacts={contacts}/-{s.best_known}  '
      f'wall={time.time()-t0:.0f}s', flush=True)

out_dir = str(Path(__file__).resolve().parent / 'record_attempt_results')
fb_suffix = 'best' if from_best else 'random'
with open(f'{out_dir}/{sid}_wl_{fb_suffix}_s{seed}_clean.json', 'w') as f:
    json.dump({'sequence_id': sid, 'sequence': s.sequence,
               'n_residues': s.length, 'n_h_residues': s.sequence.count('H'),
               'contacts': int(contacts),
               'coords': [list(c) for c in coords],
               'note': f'wang_landau seed={seed} from_best={from_best}'}, f, indent=2)

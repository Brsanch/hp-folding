"""Bidirectional scout: backward direction (hcore-space exploration with CSP).

Reads a shared knowledge file `scout_kb_<sid>.json` listing known feasible
hcore polyominoes for a sequence. Picks one (weighted by contact count),
mutates it via the chosen strategy, runs solve_csp on the variant. If feasible:
- adds the variant to the knowledge file (atomic append)
- if contact count exceeds current best, saves the fold

Run many scouts in parallel — each picks a strategy and works in its own niche.
The shared knowledge file is the bidirectional channel: forward scouts (existing
saddle-hops, RE, etc.) deposit their hcores here too, so backward scouts have
seeds beyond what they'd find alone.
"""
from __future__ import annotations
import sys, json, os, random, time, hashlib, argparse
from pathlib import Path
import fcntl

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from hp_folding import sequences, hcore, hcore_csp

NBRS = ((1,0),(-1,0),(0,1),(0,-1))


def hcore_hash(cells):
    return hashlib.sha1(repr(sorted(hcore._canon_translate(cells))).encode()).hexdigest()[:16]


def kb_path(sid: str) -> Path:
    base = Path(__file__).resolve().parent / 'scout_kb'
    base.mkdir(parents=True, exist_ok=True)
    return base / f'kb_{sid}.json'


def kb_read(sid: str) -> list[dict]:
    p = kb_path(sid)
    if not p.exists(): return []
    try:
        with open(p) as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            data = json.load(f)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return data
    except Exception:
        return []


def kb_append(sid: str, entry: dict):
    """Atomic append (read-update-write under exclusive lock)."""
    p = kb_path(sid)
    with open(p, 'a+') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.seek(0)
        try:
            data = json.load(f)
        except Exception:
            data = []
        # Dedup by hash before adding
        existing_hashes = {e['hash'] for e in data}
        if entry['hash'] not in existing_hashes:
            data.append(entry)
            f.seek(0); f.truncate()
            json.dump(data, f, separators=(',', ':'))
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return True
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return False


def is_connected(cells):
    if not cells: return True
    cells = set(cells)
    start = next(iter(cells))
    seen = {start}; stack = [start]
    while stack:
        x, y = stack.pop()
        for dx, dy in NBRS:
            n = (x+dx, y+dy)
            if n in cells and n not in seen:
                seen.add(n); stack.append(n)
    return len(seen) == len(cells)


# ── Mutation strategies ──────────────────────────────────────

def mut_single_swap(cells, rng):
    cells = set(cells)
    candidates = list(cells)
    rng.shuffle(candidates)
    for r in candidates:
        small = cells - {r}
        if not is_connected(small): continue
        bnds = set()
        for x,y in small:
            for dx,dy in NBRS:
                n = (x+dx, y+dy)
                if n not in small and n != r: bnds.add(n)
        for a in rng.sample(list(bnds), min(len(bnds), 6)):
            yield hcore._canon_translate(frozenset(small | {a}))


def mut_double_swap(cells, rng):
    cells = set(cells)
    cands_r1 = list(cells); rng.shuffle(cands_r1)
    for r1 in cands_r1[:6]:
        cells_minus = cells - {r1}
        if not is_connected(cells_minus): continue
        cands_r2 = list(cells_minus); rng.shuffle(cands_r2)
        for r2 in cands_r2[:5]:
            after_remove = cells_minus - {r2}
            if not is_connected(after_remove): continue
            bnds = set()
            for x,y in after_remove:
                for dx,dy in NBRS:
                    n = (x+dx, y+dy)
                    if n not in after_remove: bnds.add(n)
            bnds_l = list(bnds); rng.shuffle(bnds_l)
            for a1 in bnds_l[:4]:
                for a2 in bnds_l[:4]:
                    if a1 == a2: continue
                    new = frozenset(after_remove | {a1, a2})
                    if is_connected(new):
                        yield hcore._canon_translate(new)


def mut_sprawl(cells, rng):
    """Add a cell at a concave corner; remove a cell that's locally most-redundant."""
    cells = set(cells)
    # Find concave corners (boundary cells with 2+ neighbors in cells)
    bnds = set()
    for x,y in cells:
        for dx,dy in NBRS:
            n = (x+dx, y+dy)
            if n not in cells: bnds.add(n)
    concave = [b for b in bnds if sum(1 for dx,dy in NBRS if (b[0]+dx, b[1]+dy) in cells) >= 2]
    rng.shuffle(concave)
    for a in concave[:5]:
        cands_r = sorted(cells, key=lambda c: -sum(1 for dx,dy in NBRS if (c[0]+dx, c[1]+dy) in cells))[:5]
        rng.shuffle(cands_r)
        for r in cands_r[:3]:
            new = (cells - {r}) | {a}
            if is_connected(new):
                yield hcore._canon_translate(frozenset(new))


def mut_random_walk_perturb(cells, rng):
    """Apply 5-10 random remove+add operations in sequence (large jump)."""
    cur = set(cells)
    for _ in range(rng.randint(5, 10)):
        # Remove
        cands_r = [c for c in cur if is_connected(cur - {c})]
        if not cands_r: break
        cur.discard(rng.choice(cands_r))
        # Add at boundary
        bnds = set()
        for x,y in cur:
            for dx,dy in NBRS:
                n = (x+dx, y+dy)
                if n not in cur: bnds.add(n)
        if not bnds: break
        cur.add(rng.choice(list(bnds)))
    yield hcore._canon_translate(frozenset(cur))


STRATEGIES = {
    'single': mut_single_swap,
    'double': mut_double_swap,
    'sprawl': mut_sprawl,
    'walk':   mut_random_walk_perturb,
}


# ── Main scout loop ──────────────────────────────────────────

def scout_loop(sid: str, strategy: str, scout_id: int, time_budget: float):
    s = sequences.get(sid)
    rng = random.Random(scout_id * 7919 + hash(strategy) % 1000)
    n_h = s.sequence.count('H')
    target = s.best_known
    mutator = STRATEGIES[strategy]

    out_dir = Path(__file__).resolve().parent / 'scout_results'
    out_dir.mkdir(exist_ok=True)
    log_path = out_dir / f'{sid}_{strategy}_s{scout_id}.log'

    deadline = time.monotonic() + time_budget
    n_tried = 0
    n_feasible = 0
    n_new_added = 0
    best_seen = 0

    def log(msg):
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    log(f'=== scout {sid}/{strategy}/s{scout_id} target=-{target} n_h={n_h} ===')

    while time.monotonic() < deadline:
        # Read knowledge base, pick a seed weighted by contacts
        kb = kb_read(sid)
        if not kb:
            time.sleep(2)
            continue
        # Sample weighted by contacts (higher = more likely)
        weights = [max(1, e['contacts']) ** 2 for e in kb]
        total = sum(weights)
        r = rng.random() * total
        acc = 0; pick = kb[0]
        for e, w in zip(kb, weights):
            acc += w
            if r <= acc: pick = e; break

        seed_cells = frozenset(tuple(c) for c in pick['cells'])
        # Apply mutation strategy (yields 1+ candidates)
        for variant in mutator(seed_cells, rng):
            if time.monotonic() > deadline: break
            if len(variant) != n_h: continue
            n_tried += 1
            h = hcore_hash(variant)
            # Check if already known
            kb_now = kb_read(sid)
            if any(e['hash'] == h for e in kb_now): continue
            # Try to thread the chain
            sol = hcore_csp.solve_csp(s.sequence, variant, time_budget_seconds=4.0)
            if sol is None: continue
            n_feasible += 1
            if sol.contacts > best_seen:
                best_seen = sol.contacts
            entry = {
                'hash': h,
                'cells': sorted(list(variant)),
                'contacts': int(sol.contacts),
                'source': f'{strategy}_s{scout_id}',
                'parent': pick.get('hash', '?'),
            }
            added = kb_append(sid, entry)
            if added:
                n_new_added += 1
                log(f'  feasible variant: contacts={sol.contacts}  parent={pick.get("contacts")}  '
                    f'(tried={n_tried} feasible={n_feasible} new={n_new_added})')
                # Save high-contact folds
                if sol.contacts > 15:  # arbitrary threshold for "interesting"
                    fold_dir = out_dir / 'folds'
                    fold_dir.mkdir(exist_ok=True)
                    with open(fold_dir / f'{sid}_c{sol.contacts}_{strategy}_s{scout_id}_t{int(time.time())}.json','w') as f:
                        json.dump({'sequence_id':sid,'sequence':s.sequence,
                                   'n_residues':s.length,'n_h_residues':n_h,
                                   'contacts':int(sol.contacts),
                                   'coords':[list(c) for c in sol.coords]}, f)

    log(f'DONE  tried={n_tried} feasible={n_feasible} new={n_new_added} best={best_seen}')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--sid', required=True)
    ap.add_argument('--strategy', required=True, choices=list(STRATEGIES.keys()))
    ap.add_argument('--scout-id', type=int, required=True)
    ap.add_argument('--time-budget', type=float, default=14400)
    args = ap.parse_args()
    scout_loop(args.sid, args.strategy, args.scout_id, args.time_budget)

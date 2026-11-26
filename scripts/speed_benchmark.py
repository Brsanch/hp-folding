"""Head-to-head speed benchmark vs WL2012 claimed times.

WL2012 Table II claimed times (in hours, from Wuste-Landau 2012):
  2D64 -42: <10s     (or 30 hr nPERMis, <0.33 hr FRESS)
  2D85 -53: 0.03 hr  (= 108s)
  2D100a -48: 0.07 hr (= 252s)
  2D100b -50: 0.17 hr (= 612s)

For our results, we measure time-to-FIRST-hit of the modern reproducible best
(as reached by Davis 2016, Shi 2021, and our prior runs):
  2D50 -19, 2D60 -36, 2D64 -42, 2D85 -52, 2D100b -49 (= our S10).

Each sequence: run N parallel SA chains, record time when ANY chain
first hits target. Repeat 5 times for statistical robustness.
"""
from __future__ import annotations
import sys, math, random, time, json, multiprocessing as mp
from pathlib import Path
sys.path.insert(0, "src")
from hp_folding import BENCHMARKS
from hp_folding.moves import (
    corner_flip, crankshaft, pivot, segment_rotate, end_flip,
    local_rebridge, bond_rebridge, end_rebridge, lmw_pull,
    fragment_random_walk,
)
from hp_folding.verifier import count_contacts

NB = ((1, 0), (-1, 0), (0, 1), (0, -1))


# Targets are the modern-reproducible bests, NOT the unverified WL2012 records
TARGETS = {
    "S5":  ("S5",  23),  # canonical record (verified by all)
    "S6":  ("S6",  19),  # modern reproducible (cited record -21 unverified by anyone with public coords)
    "S7":  ("S7",  36),  # canonical record
    "S8":  ("S8",  42),  # canonical record
    "S9":  ("S9",  52),  # modern reproducible (cited record -53 unverified)
    "S10": ("S10", 49),  # modern reproducible (cited record -50 unverified)
}

WL2012_CLAIMED_TIMES = {
    "S8":  10.0,    # WL2012 claimed <10s for -42
    "S9":  108.0,   # WL2012 claimed 108s for -53 (record we don't reproduce)
    "S10": 612.0,   # WL2012 claimed 612s for -50 (record we don't reproduce, this is 2D100b)
}


def random_saw(seq, rng):
    n = len(seq)
    for _ in range(500):
        coords = [(0, 0)]
        occ = {(0, 0)}
        ok = True
        for _ in range(n - 1):
            last = coords[-1]
            avail = [(last[0]+dx, last[1]+dy) for dx, dy in NB
                     if (last[0]+dx, last[1]+dy) not in occ]
            if not avail:
                ok = False
                break
            ch = rng.choice(avail)
            coords.append(ch)
            occ.add(ch)
        if ok and len(coords) == n:
            return coords
    return [(i, 0) for i in range(n)]


def davis_propose(c, rng):
    """Davis 2016 move-set with FRW added."""
    u = rng.random()
    if u < 0.55: return lmw_pull(c, rng)
    if u < 0.68: return bond_rebridge(c, rng)
    if u < 0.74: return end_rebridge(c, rng)
    if u < 0.84: return fragment_random_walk(c, rng)
    if u < 0.88: return pivot(c, rng)
    if u < 0.90: return corner_flip(c, rng)
    if u < 0.93: return crankshaft(c, rng)
    if u < 0.96: return segment_rotate(c, rng)
    if u < 0.99: return local_rebridge(c, rng, seg_len=rng.choice([4, 6, 8]))
    return end_flip(c, rng)


def chain_run_with_time(args):
    """Run SA, record (time, contacts) every time we hit a new best."""
    seq, target, max_steps, T_start, T_end, seed = args
    rng = random.Random(seed)
    cur = random_saw(seq, rng)
    cur_c = count_contacts(seq, cur)
    best = cur_c
    best_time = -1.0
    t0 = time.monotonic()
    for step in range(max_steps):
        T = T_start * (1 - step/max_steps) + T_end * (step/max_steps)
        new = davis_propose(cur, rng)
        if new is None: continue
        c = count_contacts(seq, new)
        dE = -(c - cur_c)
        if dE <= 0 or rng.random() < math.exp(-dE/max(T, 1e-6)):
            cur = new
            cur_c = c
            if c > best:
                best = c
                if c >= target and best_time < 0:
                    best_time = time.monotonic() - t0
                    return best, best_time, time.monotonic() - t0
    return best, best_time, time.monotonic() - t0


def run_one_sequence(sid, target, n_chains=14, max_steps=10_000_000,
                     n_replicates=3):
    seq = BENCHMARKS[sid].sequence
    n = len(seq)
    # Tune temperature based on Davis Tc
    DAVIS_TC = {"S5": 0.5, "S6": 0.5, "S7": 0.42, "S8": 0.39,
                "S9": 0.545, "S10": 0.535}
    tc = DAVIS_TC.get(sid, 0.5)
    T_start = max(2.0, 4 * tc)
    T_end = max(0.05, 0.1 * tc)

    print(f"\n=== {sid} (n={n}, H={seq.count('H')}, target≥{target}) ===",
          flush=True)
    print(f"  T_start={T_start:.2f} → T_end={T_end:.2f}", flush=True)

    times_to_target = []
    bests = []
    for rep in range(n_replicates):
        wall_start = time.monotonic()
        with mp.Pool(n_chains) as pool:
            args_list = [(seq, target, max_steps, T_start, T_end,
                          rep * 100000 + seed) for seed in range(n_chains)]
            # Use imap so we can capture earliest hit
            first_hit = -1
            best_overall = 0
            for best, hit_time, _wall in pool.imap_unordered(chain_run_with_time, args_list):
                if best > best_overall:
                    best_overall = best
                if best >= target and hit_time > 0:
                    if first_hit < 0 or hit_time < first_hit:
                        first_hit = hit_time
                    pool.terminate()
                    break
        wall = time.monotonic() - wall_start
        times_to_target.append(first_hit)
        bests.append(best_overall)
        if first_hit > 0:
            print(f"  rep {rep+1}: HIT {target} in {first_hit:.1f}s (wall={wall:.1f}s)",
                  flush=True)
        else:
            print(f"  rep {rep+1}: best={best_overall} (didn't reach {target}), wall={wall:.1f}s",
                  flush=True)

    successful = [t for t in times_to_target if t > 0]
    if successful:
        med = sorted(successful)[len(successful) // 2]
        return {
            "sid": sid, "target": target, "best_seen": max(bests),
            "n_replicates": n_replicates, "successful": len(successful),
            "median_time_to_target_s": med,
            "min_time_to_target_s": min(successful),
            "max_time_to_target_s": max(successful),
        }
    else:
        return {
            "sid": sid, "target": target, "best_seen": max(bests),
            "n_replicates": n_replicates, "successful": 0,
            "median_time_to_target_s": None,
        }


def main():
    print("=== Speed benchmark: time-to-modern-reproducible-best ===")
    print(f"  using Davis 2016 move proportions + FRW + LMW pull")
    print(f"  14 parallel SA chains, 10M-step cap, target = modern best")
    sys.stdout.flush()

    results = {}
    # Run smaller sequences first (fast); skip S6 (we know it doesn't hit 21 reliably)
    # Plus: test S9 and S10 which are the gap-claimed sequences
    for sid in ["S5", "S7", "S8", "S9", "S10"]:
        target = TARGETS[sid][1]
        r = run_one_sequence(sid, target, n_chains=14, max_steps=10_000_000,
                             n_replicates=3)
        results[sid] = r
        # Quick print
        if r["median_time_to_target_s"]:
            print(f"  → {sid}: median {r['median_time_to_target_s']:.1f}s",
                  flush=True)
        else:
            print(f"  → {sid}: did not hit target in any replicate", flush=True)

    # Save and print summary
    out = Path("docs/speed_benchmark_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print("\n\n=== SUMMARY ===")
    print(f"{'Seq':<4} | {'Target':<7} | {'Best':<5} | {'Hits':<6} | {'Median time':<14} | WL2012 claim")
    print("-" * 75)
    for sid, r in results.items():
        wl_time = WL2012_CLAIMED_TIMES.get(sid, "—")
        hit = f"{r['successful']}/{r['n_replicates']}"
        med = f"{r['median_time_to_target_s']:.1f}s" if r['median_time_to_target_s'] else "—"
        wl = f"{wl_time}s for record" if isinstance(wl_time, (int, float)) else "—"
        print(f"{sid:<4} | {r['target']:<7} | {r['best_seen']:<5} | {hit:<6} | "
              f"{med:<14} | {wl}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

"""MLX-batched HP-folding sampler.

Runs K independent simulated-annealing chains in parallel on the M-series
GPU. Each step proposes a pivot (or end-flip) per chain, validates SAW
constraints in parallel, and applies a per-chain Metropolis accept rule.
The CPU pull-move polisher is invoked separately on the best K' chains.

Pivot moves alone are NOT ergodic on the 2D HP lattice (some configurations
are unreachable). We rely on the CPU polish phase using pull moves to cover
that gap. The GPU phase's job is to cheaply explore many basins.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import mlx.core as mx


@dataclass
class BatchSAResult:
    best_contacts_per_chain: list[int]
    best_coords_per_chain: list[list[tuple[int, int]]]
    final_contacts_per_chain: list[int]
    steps: int
    wall_seconds: float
    accept_rate: float
    valid_rate: float


def _seq_to_h_mask(sequence: str) -> mx.array:
    return mx.array([1 if c == "H" else 0 for c in sequence], dtype=mx.int32)


def _initial_chains(K: int, n: int) -> mx.array:
    """K straight-line horizontal chains, shape (K, n, 2) int32."""
    xs = mx.broadcast_to(mx.arange(n, dtype=mx.int32)[None, :], (K, n))
    ys = mx.zeros((K, n), dtype=mx.int32)
    return mx.stack([xs, ys], axis=-1)  # (K, n, 2)


def _has_collision(coords: mx.array) -> mx.array:
    """coords: (K, n, 2) int32 -> (K,) bool, True if any duplicate position."""
    M = 200_000  # large enough that x*M + y is unique for x, y in [-100k, 100k]
    encoded = coords[..., 0] * M + coords[..., 1]  # (K, n)
    sorted_enc = mx.sort(encoded, axis=-1)
    diffs = sorted_enc[..., 1:] - sorted_enc[..., :-1]
    return mx.any(diffs == 0, axis=-1)


def _count_contacts(coords: mx.array, h_mask: mx.array, n: int) -> mx.array:
    """Vectorized H-H contact count per chain. Returns (K,) int32."""
    # pairwise manhattan distance: (K, n, n)
    diff = coords[:, :, None, :] - coords[:, None, :, :]
    dist = mx.sum(mx.abs(diff), axis=-1)
    # static pair mask (n, n): non-sequential, i<j, both H
    idx = mx.arange(n)
    sep = mx.abs(idx[:, None] - idx[None, :])
    h_pair = (h_mask[:, None] * h_mask[None, :]).astype(mx.bool_)
    pair_mask = (sep > 1) & (idx[:, None] < idx[None, :]) & h_pair
    contacts = mx.sum(((dist == 1) & pair_mask[None, ...]).astype(mx.int32),
                      axis=(-2, -1))
    return contacts


def _propose_pivot(
    coords: mx.array, n: int, key_seed: int
) -> mx.array:
    """Per-chain pivot: pick interior index k uniform in [1, n-2],
    rotate suffix coords[k+1:] by 90/180/270 about coords[k]."""
    K = coords.shape[0]
    # pivot_idx ∈ [1, n-2]
    pivot_idx = mx.random.randint(1, n - 1, shape=(K,), key=mx.random.key(key_seed))
    rotation = mx.random.randint(0, 3, shape=(K,), key=mx.random.key(key_seed + 1))
    # Pivot point per chain
    chain_idx = mx.arange(K)
    pivot_pt = coords[chain_idx, pivot_idx]  # (K, 2)
    # Suffix mask: positions > pivot_idx
    pos = mx.arange(n)[None, :]  # (1, n)
    suffix_mask = pos > pivot_idx[:, None]  # (K, n)
    # Relative coords
    rel = coords - pivot_pt[:, None, :]  # (K, n, 2)
    rx, ry = rel[..., 0], rel[..., 1]
    rot_b = rotation[:, None]  # (K, 1)
    # 0 -> 90 ccw: (x,y) -> (-y, x)
    # 1 -> 180:    (x,y) -> (-x, -y)
    # 2 -> 270:    (x,y) -> ( y, -x)
    new_rx = mx.where(rot_b == 0, -ry,
              mx.where(rot_b == 1, -rx, ry))
    new_ry = mx.where(rot_b == 0, rx,
              mx.where(rot_b == 1, -ry, -rx))
    new_rel = mx.stack([new_rx, new_ry], axis=-1)  # (K, n, 2)
    rotated = new_rel + pivot_pt[:, None, :]
    return mx.where(suffix_mask[..., None], rotated, coords)


def _propose_end_flip(coords: mx.array, n: int, key_seed: int) -> mx.array:
    """Per-chain end flip: rotate residue 0 or n-1 to a free neighbour
    of its interior neighbour. Implemented as a 90/180/270 rotation of
    the single end residue around its anchor — collision check is unified
    elsewhere via _has_collision."""
    K = coords.shape[0]
    pick_tail = mx.random.randint(0, 2, shape=(K,), key=mx.random.key(key_seed))
    rotation = mx.random.randint(0, 3, shape=(K,), key=mx.random.key(key_seed + 1))
    # end_idx = 0 if pick_tail==0 else n-1; anchor = 1 or n-2
    end_idx = mx.where(pick_tail == 0, mx.zeros((K,), dtype=mx.int32),
                       mx.full((K,), n - 1, dtype=mx.int32))
    anchor_idx = mx.where(pick_tail == 0, mx.ones((K,), dtype=mx.int32),
                          mx.full((K,), n - 2, dtype=mx.int32))
    chain_idx = mx.arange(K)
    end_pt = coords[chain_idx, end_idx]
    anchor_pt = coords[chain_idx, anchor_idx]
    rel = end_pt - anchor_pt  # (K, 2) — unit vector from anchor to end
    rx, ry = rel[..., 0], rel[..., 1]
    rot = rotation
    new_rx = mx.where(rot == 0, -ry, mx.where(rot == 1, -rx, ry))
    new_ry = mx.where(rot == 0, rx, mx.where(rot == 1, -ry, -rx))
    new_end = mx.stack([anchor_pt[..., 0] + new_rx,
                        anchor_pt[..., 1] + new_ry], axis=-1)  # (K, 2)
    # Update only the end residue per chain.
    pos = mx.arange(n)[None, :]  # (1, n)
    is_end = pos == end_idx[:, None]  # (K, n)
    return mx.where(is_end[..., None], new_end[:, None, :], coords)


def batch_simulated_annealing(
    sequence: str,
    *,
    K: int = 4096,
    n_steps: int = 200_000,
    T_start: float = 4.0,
    T_end: float = 0.05,
    pivot_weight: float = 0.85,
    seed: int = 0,
    initial_coords: mx.array | None = None,
    log_interval: int = 5_000,
) -> BatchSAResult:
    """Run K parallel SA chains on GPU. Returns the best fold per chain."""
    n = len(sequence)
    h_mask = _seq_to_h_mask(sequence)

    if initial_coords is not None:
        coords = initial_coords.astype(mx.int32)
    else:
        coords = _initial_chains(K, n)

    contacts = _count_contacts(coords, h_mask, n)
    best_contacts = contacts
    best_coords = coords

    # Geometric temperature schedule
    log_T0 = mx.array(T_start).log()
    log_T1 = mx.array(T_end).log()

    accept_total = 0
    valid_total = 0
    t0 = time.monotonic()

    # Eval every `eval_interval` steps to keep Metal allocator happy.
    eval_interval = 200

    for step in range(n_steps):
        # Temperature for this step
        frac = step / max(1, n_steps - 1)
        T = mx.exp(log_T0 + (log_T1 - log_T0) * frac)

        # Choose move type per chain
        u_move = mx.random.uniform(shape=(K,), key=mx.random.key(seed * 100003 + step))
        do_pivot = u_move < pivot_weight  # (K,) bool

        proposed_pivot = _propose_pivot(coords, n, key_seed=seed * 7919 + step * 2)
        proposed_end = _propose_end_flip(coords, n, key_seed=seed * 7919 + step * 2 + 1)
        proposed = mx.where(do_pivot[:, None, None], proposed_pivot, proposed_end)

        valid = ~_has_collision(proposed)  # (K,)
        new_contacts = _count_contacts(proposed, h_mask, n)
        delta_E = -(new_contacts.astype(mx.float32) - contacts.astype(mx.float32))
        # Metropolis: accept if delta_E <= 0 (i.e., contacts increase) or with exp(-dE/T)
        u_acc = mx.random.uniform(shape=(K,), key=mx.random.key(seed * 13127 + step * 3))
        accept = valid & ((delta_E <= 0) | (u_acc < mx.exp(-delta_E / T)))

        coords = mx.where(accept[:, None, None], proposed, coords)
        contacts = mx.where(accept, new_contacts, contacts)

        improved = contacts > best_contacts
        best_contacts = mx.where(improved, contacts, best_contacts)
        best_coords = mx.where(improved[:, None, None], coords, best_coords)

        if step % eval_interval == 0:
            mx.eval(coords, contacts, best_contacts, best_coords)

        if step % log_interval == 0:
            mx.eval(coords, contacts, best_contacts, best_coords, accept, valid)
            accept_total += int(mx.sum(accept).item())
            valid_total += int(mx.sum(valid).item())

    mx.eval(best_coords, best_contacts, contacts)

    wall = time.monotonic() - t0
    n_logs = max(1, n_steps // log_interval)
    return BatchSAResult(
        best_contacts_per_chain=[int(c) for c in best_contacts.tolist()],
        best_coords_per_chain=[
            [tuple(map(int, p)) for p in chain]
            for chain in best_coords.tolist()
        ],
        final_contacts_per_chain=[int(c) for c in contacts.tolist()],
        steps=n_steps,
        wall_seconds=wall,
        accept_rate=accept_total / (n_logs * K),
        valid_rate=valid_total / (n_logs * K),
    )

"""Continuous molecular-dynamics-style HP folding.

Residues are points in R² (not lattice cells). The chain is a string of
beads with spring bonds between consecutive residues. Non-sequential
H-H pairs feel an attractive potential when close. All pairs feel a
soft repulsive core to prevent overlap. Gradient descent (or noisy MD)
flows the system to local minima of this continuous energy.

After convergence, the configuration is projected to the integer lattice
and validated as a SAW. The final lattice contact count is the score.

Implemented in MLX — K parallel chains run on GPU simultaneously.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import mlx.core as mx


@dataclass
class MDResult:
    coords: mx.array        # (K, n, 2) float
    contact_counts: list[int]  # raw continuous-side contact counts
    final_energies: list[float]
    wall_seconds: float


def _seq_to_h_mask(sequence: str) -> mx.array:
    return mx.array([1.0 if c == "H" else 0.0 for c in sequence], dtype=mx.float32)


def _initial_continuous(K: int, n: int) -> mx.array:
    """Random scrunched chains: each chain is a small SAW perturbed with
    Gaussian noise to break symmetries."""
    # Start with horizontal lines + small perturbation
    xs = mx.broadcast_to(mx.arange(n, dtype=mx.float32)[None, :], (K, n))
    ys = mx.zeros((K, n), dtype=mx.float32)
    coords = mx.stack([xs, ys], axis=-1)  # (K, n, 2)
    noise = mx.random.normal((K, n, 2)) * 0.3
    return coords + noise


def _energy_components(coords: mx.array, h_mask: mx.array, n: int,
                       lattice_weight: float = 0.0):
    """Return (bond_energy, attract_energy, repel_energy, lattice_energy)."""
    # Bond: sum of (|c[i+1] - c[i]| - 1)^2
    diff_seq = coords[:, 1:, :] - coords[:, :-1, :]
    dist_seq = mx.sqrt(mx.sum(diff_seq * diff_seq, axis=-1) + 1e-9)
    bond_E = mx.sum((dist_seq - 1.0) ** 2, axis=-1) * 5.0  # stronger bond

    # Pairwise distances (K, n, n)
    diff = coords[:, :, None, :] - coords[:, None, :, :]
    dist2 = mx.sum(diff * diff, axis=-1) + 1e-9
    dist = mx.sqrt(dist2)

    idx = mx.arange(n)
    sep = mx.abs(idx[:, None] - idx[None, :])
    h_pair = (h_mask[:, None] * h_mask[None, :])
    nonseq = (sep > 1).astype(mx.float32)
    pair_mask = nonseq * h_pair  # (n, n)
    attract = -mx.exp(-((dist - 1.0) ** 2) / 0.3) * pair_mask[None, ...]
    attract_E = mx.sum(attract, axis=(-2, -1)) * 0.5

    not_self = (sep > 0).astype(mx.float32)
    cutoff = 0.9
    repel_mag = mx.maximum(cutoff - dist, mx.zeros_like(dist)) ** 2 * 80.0
    repel = repel_mag * not_self[None, ...]
    repel_E = mx.sum(repel, axis=(-2, -1)) * 0.5

    # Lattice attractor: cos(2π x) + cos(2π y) potential pulls each coord
    # toward integer values. Energy minima at integer (x, y).
    if lattice_weight > 0:
        lat = -mx.cos(2 * 3.14159265 * coords[..., 0]) - mx.cos(2 * 3.14159265 * coords[..., 1])
        lattice_E = mx.sum(lat, axis=-1) * lattice_weight
    else:
        lattice_E = mx.zeros(coords.shape[0])

    return bond_E, attract_E, repel_E, lattice_E


def _total_energy(coords: mx.array, h_mask: mx.array, n: int,
                  lattice_weight: float = 0.0) -> mx.array:
    b, a, r, l = _energy_components(coords, h_mask, n, lattice_weight)
    return b + a + r + l


def md_step(
    coords: mx.array,
    h_mask: mx.array,
    n: int,
    *,
    lr: float,
    noise_sigma: float,
    seed: int,
    lattice_weight: float = 0.0,
) -> mx.array:
    """One MD step: gradient descent with Gaussian noise (Langevin-like)."""
    def loss(coords):
        return mx.sum(_total_energy(coords, h_mask, n, lattice_weight))

    grad = mx.grad(loss)(coords)
    noise = mx.random.normal(coords.shape, key=mx.random.key(seed)) * noise_sigma
    new_coords = coords - lr * grad + noise
    return new_coords


def run_continuous_md(
    sequence: str,
    *,
    K: int = 1024,
    n_steps: int = 5000,
    lr_start: float = 0.05,
    lr_end: float = 0.005,
    noise_start: float = 0.05,
    noise_end: float = 0.0,
    seed: int = 0,
    initial_coords: mx.array | None = None,
    eval_interval: int = 200,
) -> MDResult:
    n = len(sequence)
    h_mask = _seq_to_h_mask(sequence)
    if initial_coords is None:
        coords = _initial_continuous(K, n)
    else:
        coords = initial_coords.astype(mx.float32)

    t0 = time.monotonic()

    for step in range(n_steps):
        frac = step / max(1, n_steps - 1)
        lr = lr_start * (1 - frac) + lr_end * frac
        ns = noise_start * (1 - frac) + noise_end * frac
        # Lattice attractor: weak in early MD, strong toward end (snap to grid)
        lattice_w = 0.0 if frac < 0.4 else (frac - 0.4) / 0.6 * 0.8
        coords = md_step(coords, h_mask, n, lr=lr, noise_sigma=ns,
                          seed=seed * 100003 + step, lattice_weight=lattice_w)
        if step % eval_interval == 0:
            mx.eval(coords)

    mx.eval(coords)
    energies = _total_energy(coords, h_mask, n, lattice_weight=1.0)
    mx.eval(energies)

    # Crude continuous contact estimate: count pairs with d in [0.8, 1.3] for H-H non-sequential
    diff = coords[:, :, None, :] - coords[:, None, :, :]
    dist = mx.sqrt(mx.sum(diff * diff, axis=-1) + 1e-9)
    idx = mx.arange(n)
    sep = mx.abs(idx[:, None] - idx[None, :])
    h_pair = (h_mask[:, None] * h_mask[None, :]).astype(mx.float32)
    pair_mask = (sep > 1).astype(mx.float32) * h_pair
    near = ((dist > 0.8) & (dist < 1.3)).astype(mx.int32)
    contact_estimates = mx.sum(near * pair_mask[None, ...].astype(mx.int32),
                               axis=(-2, -1)) // 2
    mx.eval(contact_estimates)
    return MDResult(
        coords=coords,
        contact_counts=[int(c) for c in contact_estimates.tolist()],
        final_energies=[float(e) for e in energies.tolist()],
        wall_seconds=time.monotonic() - t0,
    )


def project_to_lattice_and_polish(
    sequence: str, continuous_coords, max_polish_attempts: int = 1000
):
    """Project a single chain (n, 2) to integer lattice via greedy chain
    walk: place residue 0 at origin, then for each subsequent residue
    pick the lattice neighbor closest to the continuous target.
    """
    import math
    n = len(sequence)
    cont = [tuple(map(float, p)) for p in continuous_coords.tolist()]
    # Center on integer near continuous start
    cx, cy = round(cont[0][0]), round(cont[0][1])
    out: list[tuple[int, int]] = [(cx, cy)]
    occupied = {out[0]}
    for i in range(1, n):
        target = cont[i]
        # Try each lattice neighbor of out[i-1], pick closest to target
        prev = out[-1]
        best = None
        best_dist = None
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cand = (prev[0] + dx, prev[1] + dy)
            if cand in occupied:
                continue
            d = (cand[0] - target[0]) ** 2 + (cand[1] - target[1]) ** 2
            if best_dist is None or d < best_dist:
                best_dist = d
                best = cand
        if best is None:
            return None  # stuck
        out.append(best)
        occupied.add(best)
    return out

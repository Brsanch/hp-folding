"""Independent CaDiCaL 3.1.0 + lrat-check formal LRAT proof-certificate pipeline
(ZERO OR-Tools!) for unconditional Z^2 global optimality C*(S6) = 19.

For each branch:
  1. Exports the exact pure-CNF DIMACS encoding (including Cut 1 unary bounding-box
     sum-of-extents encoding and Cut 3 deg_H(H_2) <= 2) via verify_hp_sat.build_s6_cnf.
  2. Runs CaDiCaL (`--lrat --no-binary`) to solve and emit an LRAT proof trace.
  3. Runs Marijn Heule's independent `lrat-check` verifier on (CNF, LRAT) and
     asserts `c VERIFIED` with exit code 0.
  4. Records exact CNF/LRAT SHA-256 hashes, conflict counts, clause counts, and
     timings into `results/s6_lrat_verification.json`, then deletes the multi-GB
     temporary `.lrat` file (unless `--keep-lrat` is passed).

Pre-flight sanity check: verifies that the exact same CaDiCaL binary solves the
identical CNF encoding at C = 19 (slack = {31, 37}, D = 19) as SATISFIABLE and
returns a directly verified 19-contact self-avoiding walk of S6[2..49].
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prove_s6_unconditional_z2 import ODD_H_FULL, verify_exact_z2_diameter_bound
from verify_hp_sat import build_s6_cnf

from hp_folding.grid_cpsat import NBRS
from hp_folding.sequences import BENCHMARKS

REPO = Path(__file__).resolve().parents[1]
OUT_JSON = REPO / "results" / "s6_lrat_verification.json"


def _find_bin(name: str) -> str:
    for candidate in (Path.home() / ".local" / "bin" / name, shutil.which(name)):
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise FileNotFoundError(f"Required binary {name!r} not found on PATH or in ~/.local/bin")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_sanity_c19(cadical_bin: str) -> dict:
    """Verify the exact same CaDiCaL binary finds a valid 19-contact SAW on the identical CNF encoding."""
    d21 = verify_exact_z2_diameter_bound(21)
    cnf, p_var, vc = build_s6_cnf(slack_set={31, 37}, max_wh_minus_2=d21)
    with tempfile.TemporaryDirectory(prefix="hp_s6_c19_") as td:
        cnf_path = Path(td) / "s6_c19_sanity.cnf"
        cnf.write_dimacs(cnf_path)
        t0 = time.time()
        proc = subprocess.run([cadical_bin, str(cnf_path)], capture_output=True, text=True)
        dt = time.time() - t0
        assert proc.returncode == 10, f"Expected SAT (10) for C=19 sanity check, got {proc.returncode}"
        pos: set[int] = set()
        for line in proc.stdout.splitlines():
            if line.startswith("v "):
                for tok in line[2:].split():
                    v = int(tok)
                    if v > 0:
                        pos.add(v)
        seq = BENCHMARKS["S6"].sequence[2:]
        coords = [next(c for c in vc[i] if p_var[(i, c)] in pos) for i in range(len(seq))]
        assert len(set(coords)) == len(seq)
        assert all(
            abs(coords[i][0] - coords[i + 1][0]) + abs(coords[i][1] - coords[i + 1][1]) == 1
            for i in range(len(seq) - 1)
        )
        h_set = {coords[i] for i, ch in enumerate(seq) if ch == "H"}
        ie = sum(
            1
            for u in h_set
            if (u[0] + u[1]) % 2 == 1
            for dx, dy in NBRS
            if (u[0] + dx, u[1] + dy) in h_set
        )
        assert ie >= 26, f"Expected >= 26 H-edges (19 contacts), got {ie}"
        return {
            "status": "SATISFIABLE",
            "vars": cnf.nvars,
            "clauses": len(cnf.clauses),
            "cadical_s": round(dt, 3),
            "verified_contacts": ie - 7,
        }


def solve_and_verify_lrat(
    label: str,
    slack_set: set[int],
    max_wh_minus_2: int,
    cadical_bin: str,
    lrat_check_bin: str,
    keep_lrat_dir: Path | None = None,
) -> tuple[str, dict]:
    cnf, _, _ = build_s6_cnf(slack_set=slack_set, max_wh_minus_2=max_wh_minus_2)
    work_dir = keep_lrat_dir if keep_lrat_dir is not None else Path(tempfile.mkdtemp(prefix=f"hp_lrat_{label}_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    cnf_path = work_dir / f"{label}.cnf"
    lrat_path = work_dir / f"{label}.lrat"
    try:
        cnf.write_dimacs(cnf_path)
        cnf_sha = _sha256_file(cnf_path)
        t0 = time.time()
        c_proc = subprocess.run(
            [cadical_bin, "--lrat", "--no-binary", str(cnf_path), str(lrat_path)],
            capture_output=True,
            text=True,
        )
        cadical_dt = time.time() - t0
        assert c_proc.returncode == 20, (
            f"[{label}] Expected UNSAT (20), got {c_proc.returncode}\n{c_proc.stdout[-500:]}"
        )
        m_conf = re.search(r"c conflicts:\s+(\d+)", c_proc.stdout)
        conflicts = int(m_conf.group(1)) if m_conf else -1

        lrat_bytes = lrat_path.stat().st_size
        lrat_sha = _sha256_file(lrat_path)

        t1 = time.time()
        v_proc = subprocess.run(
            [lrat_check_bin, str(cnf_path), str(lrat_path)],
            capture_output=True,
            text=True,
        )
        check_dt = time.time() - t1
        assert v_proc.returncode == 0 and "c VERIFIED" in v_proc.stdout, (
            f"[{label}] lrat-check FAILED (rc={v_proc.returncode}):\n{v_proc.stdout}\n{v_proc.stderr}"
        )
        m_add = re.search(r"Added clauses = (\d+)", v_proc.stdout)
        added_clauses = int(m_add.group(1)) if m_add else -1

        rec = {
            "status": "UNSAT",
            "lrat_verified": True,
            "vars": cnf.nvars,
            "clauses": len(cnf.clauses),
            "cnf_sha256": cnf_sha,
            "lrat_sha256": lrat_sha,
            "lrat_bytes": lrat_bytes,
            "cadical_conflicts": conflicts,
            "cadical_s": round(cadical_dt, 2),
            "lrat_added_clauses": added_clauses,
            "lrat_check_s": round(check_dt, 2),
        }
        return label, rec
    finally:
        if keep_lrat_dir is None:
            shutil.rmtree(work_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independent CaDiCaL + lrat-check formal LRAT proof pipeline for S6 on Z^2."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing results/s6_lrat_verification.json entries.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of parallel CaDiCaL/lrat-check workers (default: 3).",
    )
    parser.add_argument(
        "--keep-lrat",
        nargs="?",
        const=str(REPO / "results" / "lrat_traces"),
        default=None,
        metavar="DIR",
        help="Keep generated CNF and .lrat files in DIR (default: results/lrat_traces).",
    )
    args = parser.parse_args()
    resume = args.resume
    workers = args.workers
    keep_lrat_dir = Path(args.keep_lrat) if args.keep_lrat is not None else None

    cadical_bin = _find_bin("cadical")
    lrat_check_bin = _find_bin("lrat-check")
    ver = subprocess.check_output([cadical_bin, "--version"], text=True).strip()
    print(f"=== Independent CaDiCaL {ver} + lrat-check LRAT Pipeline (workers={workers}) ===", flush=True)

    d21 = verify_exact_z2_diameter_bound(21)
    d20 = verify_exact_z2_diameter_bound(20)
    assert d21 == 19 and d20 == 22

    if resume and OUT_JSON.exists():
        report = json.loads(OUT_JSON.read_text())
    else:
        report = {"cadical_version": ver, "d21": d21, "d20": d20, "c20_cases": {}}

    # Pre-flight C=19 sanity check in CaDiCaL
    if not (resume and report.get("sanity_c19", {}).get("status") == "SATISFIABLE"):
        s19 = run_sanity_c19(cadical_bin)
        report["sanity_c19"] = s19
        OUT_JSON.write_text(json.dumps(report, indent=2))
        print(
            f"[Sanity check C=19 CaDiCaL] SATISFIABLE in {s19['cadical_s']}s "
            f"(vars={s19['vars']}, clauses={s19['clauses']}, verified_contacts={s19['verified_contacts']})",
            flush=True,
        )
    else:
        print(f"[Sanity check C=19 CaDiCaL] RESUMED SATISFIABLE ({report['sanity_c19']['cadical_s']}s)", flush=True)

    # Build task queue: C=21 + all 10 C=20 cases
    jobs: list[tuple[str, set[int], int]] = []
    if not (resume and report.get("c21", {}).get("lrat_verified") is True):
        jobs.append(("c21", set(), d21))
    else:
        print(
            f"[C=21 unconditional Z^2 LRAT] RESUMED VERIFIED "
            f"(cadical={report['c21']['cadical_s']}s, lrat-check={report['c21']['lrat_check_s']}s)",
            flush=True,
        )

    for k in ODD_H_FULL:
        key = f"H{k}"
        if resume and report["c20_cases"].get(key, {}).get("lrat_verified") is True:
            rec = report["c20_cases"][key]
            print(
                f"  [C=20 LRAT, slack={key:>3s}] RESUMED VERIFIED "
                f"(cadical={rec['cadical_s']}s, lrat-check={rec['lrat_check_s']}s)",
                flush=True,
            )
        else:
            jobs.append((key, {k}, d20))

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(
                solve_and_verify_lrat, lbl, sset, diam, cadical_bin, lrat_check_bin, keep_lrat_dir
            ): lbl
            for lbl, sset, diam in jobs
        }
        for fut in concurrent.futures.as_completed(futs):
            lbl, rec = fut.result()
            if lbl == "c21":
                report["c21"] = rec
                print(
                    f"[C=21 unconditional Z^2 LRAT] c VERIFIED! "
                    f"(cadical={rec['cadical_s']}s, conflicts={rec['cadical_conflicts']}, "
                    f"lrat={rec['lrat_bytes']/1e9:.2f}GB, lrat-check={rec['lrat_check_s']}s)",
                    flush=True,
                )
            else:
                report["c20_cases"][lbl] = rec
                print(
                    f"  [C=20 LRAT, slack={lbl:>3s}] c VERIFIED! "
                    f"(cadical={rec['cadical_s']}s, conflicts={rec['cadical_conflicts']}, "
                    f"lrat={rec['lrat_bytes']/1e9:.2f}GB, lrat-check={rec['lrat_check_s']}s)",
                    flush=True,
                )
            OUT_JSON.write_text(json.dumps(report, indent=2))

    all_ok = (
        report.get("c21", {}).get("lrat_verified") is True
        and len(report["c20_cases"]) == 10
        and all(v.get("lrat_verified") is True for v in report["c20_cases"].values())
    )
    report["all_11_lrat_verified"] = all_ok
    report["c20_total_cadical_s"] = round(sum(v["cadical_s"] for v in report["c20_cases"].values()), 2)
    report["c20_total_lrat_check_s"] = round(sum(v["lrat_check_s"] for v in report["c20_cases"].values()), 2)
    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(f"\nSUMMARY: ALL 11/11 S6 UNCONDITIONAL Z^2 LRAT CERTIFICATES VERIFIED = {all_ok}", flush=True)
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

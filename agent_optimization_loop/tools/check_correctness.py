#!/usr/bin/env python3
"""
check_correctness.py - the hard correctness gate for every iteration.

Compares a candidate score matrix with the pgsc_calc reference, matching rows by
sample ID and columns by model ID (order does not matter). Passes only if every
reference cell is present in the candidate and

    numpy.allclose(candidate, reference, atol=1e-6, rtol=1e-5, equal_nan=True)

holds. Prints a short report and exits 0 on PASS, 1 on FAIL.

Usage:
    python tools/check_correctness.py optim_workspace/v_3/pgs_output.tsv
    python tools/check_correctness.py CANDIDATE.tsv --reference reference/reference_scores.tsv
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ATOL, RTOL = 1e-6, 1e-5
DEFAULT_REF = Path(__file__).resolve().parent.parent / "reference" / "reference_scores.tsv"


def read_matrix(path):
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        rows, vals = [], []
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts or parts == [""]:
                continue
            rows.append(parts[0])
            vals.append([float(x) if x not in ("", "NA", "nan", "NaN") else np.nan
                         for x in parts[1:]])
    return rows, header[1:], np.asarray(vals, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("--reference", default=str(DEFAULT_REF))
    a = ap.parse_args()

    r_rows, r_cols, R = read_matrix(a.reference)
    c_rows, c_cols, C = read_matrix(a.candidate)
    ri = {s: i for i, s in enumerate(c_rows)}
    ci = {m: j for j, m in enumerate(c_cols)}
    miss_rows = [s for s in r_rows if s not in ri]
    miss_cols = [m for m in r_cols if m not in ci]
    print(f"reference : {a.reference}  ({len(r_rows)} samples x {len(r_cols)} models)")
    print(f"candidate : {a.candidate}  ({len(c_rows)} samples x {len(c_cols)} models)")
    if miss_rows or miss_cols:
        print(f"missing samples: {len(miss_rows)} {miss_rows[:5]}")
        print(f"missing models : {len(miss_cols)} {miss_cols[:5]}")
        print("RESULT: FAIL (candidate does not cover the reference matrix)")
        sys.exit(1)

    X = C[np.ix_([ri[s] for s in r_rows], [ci[m] for m in r_cols])]
    diff = np.abs(X - R)
    both = ~np.isnan(X) & ~np.isnan(R)
    nan_mismatch = int((np.isnan(X) != np.isnan(R)).sum())
    ok = np.isclose(X, R, atol=ATOL, rtol=RTOL, equal_nan=True)
    r = np.corrcoef(X[both], R[both])[0, 1] if both.sum() > 1 else float("nan")
    print(f"cells compared: {R.size}   NA mismatches: {nan_mismatch}")
    print(f"max |diff| = {np.nanmax(np.where(both, diff, np.nan)):.3e}   "
          f"median |diff| = {np.nanmedian(np.where(both, diff, np.nan)):.3e}   Pearson r = {r:.8f}")
    bad = (~ok).sum(axis=0)
    worst = np.argsort(-bad)[:5]
    print("models with most failing cells: "
          + ", ".join(f"{r_cols[j]}={int(bad[j])}" for j in worst if bad[j] > 0) or "none")
    passed = bool(ok.all())
    print(f"allclose(atol={ATOL}, rtol={RTOL}): {'PASS' if passed else 'FAIL'} "
          f"({int((~ok).sum())} of {R.size} cells outside tolerance)")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

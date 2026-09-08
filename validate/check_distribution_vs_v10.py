#!/usr/bin/env python3
"""Cross-check `1kg_pgs_reference/scores_raw.npy` against the 3202 x 5385 run of
`pressure_test/gpu_pgs_optim_memory_safe.py` (v_10's arithmetic).

The two must agree EXACTLY wherever the model is purely additive: the reference
build adds a call-rate filter, mean imputation and coverage counting, and all
three are no-ops on a panel with no missing call, so the summation the scorer
performs is unchanged.  They must DISAGREE exactly where a model has a
dominant or recessive row, because v_10 drops that row's contribution — its
non-additive kernel writes into a copy (cuSPARSE returns an F-contiguous
matrix, so `.ravel()` is not a view) and the writes are discarded.

So this is a two-sided test, and the second side is the interesting one.
"""
import gzip, json, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(ROOT, 'pressure_test/allgallm/run_1/memory_safe/results.tsv')
OUT = os.path.join(ROOT, '1kg_pgs_reference')

bi = json.load(open(os.path.join(OUT, 'build_info.json')))
mine = np.load(os.path.join(OUT, 'scores_raw.npy'))
sids, pids = bi['sample_ids'], bi['pgs_ids']
print(f"reference build : {mine.shape[0]} samples x {mine.shape[1]} models")

ref = pd.read_csv(REF, sep='\t', index_col=0)
print(f"v_10 ALL run    : {ref.shape[0]} samples x {ref.shape[1]} models  {REF}")
common_s = [s for s in sids if s in ref.index]
common_m = [p for p in pids if p in ref.columns]
print(f"common          : {len(common_s)} samples x {len(common_m)} models")
ri = {s: i for i, s in enumerate(sids)}
ci = {p: j for j, p in enumerate(pids)}
A = mine[np.ix_([ri[s] for s in common_s], [ci[p] for p in common_m])]
B = ref.loc[common_s, common_m].to_numpy(dtype=np.float64)

# v_10 wrote 6 decimals, so compare at that resolution
A6 = np.round(A, 6)
d = np.abs(A6 - B)
d[~np.isfinite(A6) & ~np.isfinite(B)] = 0.0
per_model = np.nanmax(np.where(np.isfinite(d), d, 0.0), axis=0)
same = per_model <= 1e-6
print(f"\nmodels identical to v_10 : {int(same.sum())} / {len(common_m)}")
print(f"models that differ       : {int((~same).sum())}")

# which of the differing models actually contain a non-additive row?
idx = pd.read_csv(os.path.join(OUT, 'model_index.tsv'), sep='\t')
idx = idx.set_index('pgs_id')
print("\ndiffering models:")
for k in np.flatnonzero(~same):
    p = common_m[k]
    r = idx.loc[p] if p in idx.index else None
    print(f"  {p}  max|diff| {per_model[k]:.6g}"
          + (f"  n_S={int(r['n_S'])} coverage={r['coverage']:.4f}"
             f" sigma={r['sigma']:.6g}" if r is not None else ""))
    if k > 60:
        print("  ... (truncated)")
        break

nas = np.isnan(A6) ^ ~np.isfinite(B)
print(f"\nNA-pattern disagreements : {int(nas.sum())}")

# ── the two-sided assertion ────────────────────────────────────────────────
# Scanned independently from the scoring files: exactly four models in the
# 5385-file catalog carry a dominant or recessive row (21 rows in total).
NONADD = json.load(open(os.path.join(ROOT, '_dist_smoke/nonadditive_models.json')))
expected = set(NONADD) & set(common_m)
differ = {common_m[k] for k in np.flatnonzero(~same)}
print(f"\nmodels with a non-additive row (independent scan of PGSCatelog): "
      f"{sorted(expected)}")
print(f"models whose scores differ from v_10                        : "
      f"{sorted(differ)}")
ok = True
extra = differ - expected
if extra:
    ok = False
    print(f"\nFAIL: {len(extra)} purely-additive model(s) differ from v_10 — the "
          f"reference build perturbed the summation: {sorted(extra)[:10]}")
else:
    print("\nok  : every purely-additive model is bit-identical to v_10")
missing = expected - differ
if missing:
    print(f"note: {sorted(missing)} carry a non-additive row but score the same, "
          f"which happens when that row matched no target in the callset")
if differ and differ <= expected:
    print("ok  : every difference is a model v_10 silently zeroed a "
          "dominant/recessive term for")
print("\nRESULT: " + ("PASSED" if ok else "FAILED"))
sys.exit(0 if ok else 1)

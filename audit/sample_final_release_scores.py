#!/usr/bin/env python3
"""
sample_final_release_scores.py — draw the 20 audit cells of the final release.

Task `13_final_release_and_score_sampled.md`: pull 20 (genotype x PGS model)
cells out of the reference distribution that the final released software
(`calculate_distribution_pgs_1kg.py`, see `11_run_1kg_distribution.md`) wrote
to `1kg_pgs_reference/`, and write them to
`final_release_version_sample_20.csv` so a pgsc_calc run can be pointed at the
same 20 cells and compared cell by cell.

  * 10 cells with a real score, spread over LARGE / MEDIUM / SMALL models;
  * 10 cells the release did NOT calculate, written as NULL.

Nothing here is random (`13_...md` item 3 forbids it).  Every choice is a rank
inside a sorted list, so re-running this on the same bundle reproduces the
same 20 rows byte for byte.

MODEL SIZE is `n_rows` of `model_index.tsv` — the number of rows in the PGS
Catalog scoring file, i.e. the size of the model as published, before any
matching against the panel.  Tiers:

    LARGE   n_rows >= 1e6          (1254 models,   1.0M .. 10.3M rows)
    MEDIUM  1e4 <= n_rows < 1e6    (1693 models,  10.1k .. 997k  rows)
    SMALL   n_rows <  1e4          (2192 models,      1 ..  10k  rows)

Within a tier the models are sorted by (n_rows, pgs_id) and picked at fixed
fractional ranks, so the draw spans each tier rather than clustering at one
end.

WHICH SCORE.  The `score` column is the raw summed score, `scores_raw.tsv` —
sum_j w_j * dosage_j over the model's scoring set S.  That is the quantity
pgsc_calc's `pgs_cal_results.tsv` reports (plink2 `--score ... cols=+scoresums`,
the SUM column), so the two are directly comparable.  z / z_pc are reference
standardisations that live only in this bundle and have no pgsc_calc analogue.

WHAT "NULL" MEANS.  A model whose scoring set is empty (`n_S == 0`) has no
score: the release stores a structural 0.0 in the matrix, which is not a
measurement, and `model_index.tsv` gives it `mu = sigma = NaN`, `coverage =
0.0`, `deployable = 0`, reason "no locus of this model is in S".  Exactly 21 of
the 5385 models are in that state, for three distinct causes, and the draw
below covers all three:

  (a) the scoring file has no usable additive/dosage row at all (`n_rows == 0`)
      — HLA haplotype models (PGS000343: `is_haplotype=True`, effect allele
      `HLA-C*06:02`) and dosage-triplet models that publish
      `dosage_0/1/2_weight` instead of `effect_weight` (the PGS0042xx family);
  (b) the file parses but none of its loci is in the 1000 Genomes callset
      (PGS003451: 2 proxy SNPs in the MHC);
  (c) the file parses, is enormous, and still matches nothing (PGS005228:
      2,379,381 rows whose `effect_allele` equals `other_allele`, so no row can
      be oriented against a target).

GENOTYPES.  Ten 1000 Genomes samples, two per super-population (AFR, AMR, EAS,
EUR, SAS), taken at rank 0 and rank n//2 of each super-population's sorted
sample list.  The same ten carry the scored half and the NULL half, so any
difference between the two halves is a property of the model, not the sample.

Usage:  python3 sample_final_release_scores.py [--reference 1kg_pgs_reference]
                                               [--out final_release_version_sample_20.csv]
"""

import argparse
import csv
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))

SUPERPOPS = ["AFR", "AMR", "EAS", "EUR", "SAS"]

# (label, lower bound, upper bound, fractional ranks inside the tier)
TIERS = [
    ("large",  1_000_000, float("inf"), (1.0, 0.5, 0.0)),
    ("medium",    10_000,   1_000_000,  (1.0, 2 / 3, 1 / 3, 0.0)),
    ("small",          0,      10_000,  (1.0, 0.5, 0.0)),
]


def pick_at(df, fracs):
    """Rows of `df` (already sorted) at fixed fractional ranks, no repeats."""
    out, seen = [], set()
    for f in fracs:
        i = int(round(f * (len(df) - 1)))
        while i in seen:                      # only bites on a tiny tier
            i = (i + 1) % len(df)
        seen.add(i)
        out.append(df.iloc[i])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=os.path.join(ROOT, "1kg_pgs_reference"))
    ap.add_argument("--out", default=os.path.join(ROOT, "final_release_version_sample_20.csv"))
    a = ap.parse_args()

    info = json.load(open(os.path.join(a.reference, "build_info.json")))
    sample_ids = list(info["sample_ids"])
    pgs_ids = list(info["pgs_ids"])
    raw = np.load(os.path.join(a.reference, "scores_raw.npy"), mmap_mode="r")
    assert raw.shape == (len(sample_ids), len(pgs_ids)), raw.shape

    mi = pd.read_csv(os.path.join(a.reference, "model_index.tsv"), sep="\t")
    ps = pd.read_csv(os.path.join(a.reference, "panel_samples.tsv"), sep="\t")

    # ── the ten genotypes: 2 per super-population, by rank in a sorted list ──
    genotypes = []
    for sp in SUPERPOPS:
        ids = sorted(ps.loc[ps.super_population == sp, "sample_id"])
        genotypes += [ids[0], ids[len(ids) // 2]]

    # ── the ten scored models ───────────────────────────────────────────────
    pool = mi[(mi.n_S > 0) & (mi.deployable == 1)]
    scored = []
    for label, lo, hi, fracs in TIERS:
        tier = pool[(pool.n_rows >= lo) & (pool.n_rows < hi)]
        tier = tier.sort_values(["n_rows", "pgs_id"]).reset_index(drop=True)
        for row in pick_at(tier, fracs):
            scored.append((label, row))
    assert len(scored) == 10

    # ── the ten uncalculated models ─────────────────────────────────────────
    # Cause (b) and (c) are one model each, so both are taken outright; the
    # remaining eight come from cause (a) at even ranks of its sorted list.
    null_all = mi[mi.n_S == 0].sort_values("pgs_id").reset_index(drop=True)
    cause_bc = null_all[null_all.n_rows > 0]
    cause_a = null_all[null_all.n_rows == 0].reset_index(drop=True)
    n_a = 10 - len(cause_bc)
    step = (len(cause_a) - 1) / (n_a - 1)
    nulls = list(cause_bc.itertuples(index=False)) + [
        cause_a.iloc[int(round(k * step))] for k in range(n_a)
    ]
    assert len({n.pgs_id for n in nulls}) == 10

    rows, notes = [], []
    for (label, m), sid in zip(scored, genotypes):
        score = float(raw[sample_ids.index(sid), pgs_ids.index(m.pgs_id)])
        assert np.isfinite(score), (sid, m.pgs_id)
        rows.append((sid, m.pgs_id, repr(score)))
        notes.append((sid, m.pgs_id, label, int(m.n_rows), int(m.n_S),
                      float(m.coverage), "scored"))
    for m, sid in zip(nulls, genotypes):
        rows.append((sid, m.pgs_id, "NULL"))
        notes.append((sid, m.pgs_id, "null", int(m.n_rows), 0, 0.0,
                      str(m.reason)))

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["genotype_id", "pgs_model_id", "score"])
        w.writerows(rows)
    print(f"wrote {a.out}: {len(rows)} rows")

    side = os.path.splitext(a.out)[0] + "_provenance.tsv"
    with open(side, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["genotype_id", "pgs_model_id", "tier", "n_rows", "n_S",
                    "coverage", "note"])
        w.writerows(notes)
    print(f"wrote {side}")

    for r, n in zip(rows, notes):
        print(f"  {r[0]:<9} {r[1]:<10} {n[2]:<6} n_rows={n[3]:>9} "
              f"cov={n[5]:.3f}  score={r[2]}")


if __name__ == "__main__":
    main()

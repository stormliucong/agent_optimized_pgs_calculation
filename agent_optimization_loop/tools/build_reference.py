#!/usr/bin/env python3
"""
build_reference.py - the two Python halves of tools/make_reference.sh.

  prepare   benchmark/ -> pgsc_calc inputs
            * merges the 100 single-sample VCFs into one multi-sample VCF
              (they share an identical site list, so genotype columns are
              simply placed side by side; no call is changed)
            * links the scoring files under pgsc_calc's harmonized file name
              (<PGS_ID>_hmPOS_GRCh38.txt.gz)
            * writes the pgsc_calc samplesheet

  finalize  pgsc_calc results -> reference/
            * reference/pgs_cal_results.tsv   pgsc_calc's own SUM matrix
            * reference/reference_scores.tsv  THE GATE'S REFERENCE: pgsc_calc's
              matching decisions (its matched scorefiles) with the sum redone
              in float64. plink2's own sums carry float32-level rounding
              (up to ~1e-4 here) which is larger than the gate tolerance.
            * reference/pgsc_calc_match_summary.csv  per-model match summary
            * reference/agreement.txt        how the two matrices compare

Rules used to recompute a matched row: the effect-allele dosage is the ALT
count if the effect allele is ALT, otherwise 2 - ALT count; a no-call
contributes 0 (no mean imputation); dominant rows use min(d, 1), recessive
rows use d == 2.
"""
import argparse
import csv
import glob
import gzip
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def vcf_paths(bench):
    p = sorted(glob.glob(str(Path(bench) / "genotypes" / "*.vcf.gz")))
    if not p:
        sys.exit(f"no VCFs in {bench}/genotypes - run tools/make_fake_benchmark.py first")
    return p


# ── prepare ──────────────────────────────────────────────────────────────────
def prepare(a):
    bench, work, ss = Path(a.benchmark), Path(a.work), a.sampleset
    inputs = work / "inputs"
    shutil.rmtree(inputs, ignore_errors=True)
    (inputs / "scorefiles").mkdir(parents=True)
    (inputs / "target").mkdir(parents=True)

    for m in sorted(glob.glob(str(bench / "models" / "*.txt.gz"))):
        pid = os.path.basename(m).split(".")[0]
        os.symlink(os.path.abspath(m), inputs / "scorefiles" / f"{pid}_hmPOS_GRCh38.txt.gz")

    paths = vcf_paths(bench)
    meta, sites, cols, ids = [], [], [], []
    for k, p in enumerate(paths):                  # bytes throughout: ~0.3 GB for 100 x 1.1 M calls
        gts = []
        with gzip.open(p, "rb") as f:
            for line in f:
                if line.startswith(b"##"):
                    if k == 0:
                        meta.append(line)
                    continue
                c = line.rstrip(b"\n").split(b"\t")
                if k == 0:
                    sites.append(b"\t".join(c[:9]))
                if line.startswith(b"#CHROM"):
                    ids.append(c[9])
                else:
                    gts.append(c[9])
        if len(gts) != len(sites) - 1:
            sys.exit(f"{p}: site list differs from the first VCF; cannot merge by columns")
        cols.append(np.array(gts, dtype="S3"))
    G = np.stack(cols, axis=1)
    out = inputs / "target" / f"{ss}.vcf.gz"
    with gzip.open(out, "wb", compresslevel=1) as f:
        f.writelines(meta)
        f.write(sites[0] + b"\t" + b"\t".join(ids) + b"\n")
        for i in range(G.shape[0]):
            f.write(sites[i + 1] + b"\t" + b"\t".join(G[i].tolist()) + b"\n")
    with open(inputs / "samplesheet.csv", "w") as f:
        f.write("sampleset,path_prefix,chrom,format\n")
        f.write(f"{ss},{(inputs / 'target' / ss).resolve()},,vcf\n")
    print(f"[prepare] {len(paths)} samples x {G.shape[0]:,} sites -> {out}")
    print(f"[prepare] {len(os.listdir(inputs / 'scorefiles'))} scoring files, samplesheet written")


# ── finalize ─────────────────────────────────────────────────────────────────
def read_vcfs(paths):
    ids, index, cols = [], None, []
    for p in paths:
        keys, d = [], []
        with gzip.open(p, "rt") as f:
            for line in f:
                if line.startswith("##"):
                    continue
                if line.startswith("#CHROM"):
                    ids.append(line.rstrip("\n").split("\t")[9])
                    continue
                c = line.rstrip("\n").split("\t", 10)
                if index is None:
                    keys.append(f"{c[0]}:{c[1]}:{c[3]}:{c[4]}")
                d.append(-1 if "." in c[9] else c[9].count("1"))
        if index is None:
            index = {k: i for i, k in enumerate(keys)}
        cols.append(np.array(d, dtype=np.int8))
    return ids, index, np.stack(cols, axis=1)


def read_aggregated(path):
    with gzip.open(path, "rt") as f:
        rd = csv.DictReader(f, delimiter="\t")
        iid = next(c for c in rd.fieldnames if c.lstrip("#") == "IID")
        out = {}
        for r in rd:
            out[(r[iid], re.sub(r"_hmPOS_GRCh3[78]$", "", r["PGS"]))] = r["SUM"]
    return out


def write_matrix(path, samples, models, get):
    with open(path, "w") as f:
        f.write("sample_id\t" + "\t".join(models) + "\n")
        for i, s in enumerate(samples):
            f.write(s + "\t" + "\t".join(get(i, s, m) for m in models) + "\n")


def finalize(a):
    bench, results, ss, ref = Path(a.benchmark), Path(a.results), a.sampleset, Path(a.out)
    match_dir = results / ss / "match"
    agg = sorted(glob.glob(str(results / ss / "score" / "aggregated_scores.txt*")))
    if not agg or not glob.glob(str(match_dir / "*.scorefile.gz")):
        sys.exit(f"pgsc_calc results not found under {results}/{ss}/(score|match)")
    ref.mkdir(parents=True, exist_ok=True)

    raw = read_aggregated(agg[0])
    models = sorted({m for _, m in raw})
    samples, index, G = read_vcfs(vcf_paths(bench))
    obs = G >= 0
    alt = np.where(obs, G, 0).astype(np.float64)

    totals = {}
    for sf in sorted(glob.glob(str(match_dir / "*.scorefile.gz"))):
        etype = os.path.basename(sf).split("_")[-2]            # additive / dominant / recessive
        reader = pd.read_csv(sf, sep="\t", dtype={"ID": str, "effect_allele": str},
                             float_precision="round_trip", chunksize=200_000)
        for chunk in reader:                                   # bounded memory
            ms = [h.split("_hmPOS")[0] for h in chunk.columns[2:]]
            ids = chunk["ID"].to_numpy()
            rows = np.fromiter((index[k] for k in ids), dtype=np.int64, count=len(ids))
            alt_allele = np.array([k.rsplit(":", 1)[1] for k in ids])
            flip = chunk["effect_allele"].to_numpy() != alt_allele
            d = alt[rows]
            d[flip] = 2.0 - d[flip]
            if etype == "dominant":
                d = np.minimum(d, 1.0)
            elif etype == "recessive":
                d = (d == 2.0).astype(np.float64)
            d[~obs[rows]] = 0.0
            part = chunk.iloc[:, 2:].to_numpy(dtype=np.float64).T @ d   # models x samples
            for j, m in enumerate(ms):
                totals[m] = totals.get(m, 0.0) + part[j]

    write_matrix(ref / "pgs_cal_results.tsv", samples, models,
                 lambda i, s, m: raw.get((s, m), "NA") or "NA")
    write_matrix(ref / "reference_scores.tsv", samples, models,
                 lambda i, s, m: repr(float(totals[m][i])) if m in totals else "NA")
    summ = glob.glob(str(match_dir / f"{ss}_summary.csv"))
    if summ:
        shutil.copy(summ[0], ref / "pgsc_calc_match_summary.csv")

    diffs = [abs(totals[m][i] - float(raw[(s, m)])) for i, s in enumerate(samples)
             for m in models if m in totals and raw.get((s, m)) not in (None, "", "NA")]
    msg = (f"reference_scores.tsv: {len(samples)} samples x {len(models)} models\n"
           f"vs pgsc_calc's own SUM (pgs_cal_results.tsv): max |diff| {max(diffs):.3e}, "
           f"median |diff| {np.median(diffs):.3e}  (plink2 rounding; expected <~1e-4)\n")
    (ref / "agreement.txt").write_text(msg)
    print(msg, end="")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--benchmark", default=str(ROOT / "benchmark"))
    p.add_argument("--work", default=str(ROOT / ".pgsc_calc"))
    p.add_argument("--sampleset", default="AOU")
    f = sub.add_parser("finalize")
    f.add_argument("--benchmark", default=str(ROOT / "benchmark"))
    f.add_argument("--results", default=str(ROOT / ".pgsc_calc" / "results"))
    f.add_argument("--out", default=str(ROOT / "reference"))
    f.add_argument("--sampleset", default="AOU")
    a = ap.parse_args()
    prepare(a) if a.cmd == "prepare" else finalize(a)


if __name__ == "__main__":
    main()

# Provenance of this release

## The released artefact

`calculate_distribution_pgs_1kg.py` here is a byte-identical copy of the file
that built the 1000 Genomes reference bundle `1kg_pgs_reference/`:

```
sha256  e2b3595abff8a91d5a9b383c5b67b5725a5c1ce7710f3e2665ff65a25b57100e
```

`build_info.json` inside that bundle records the run:

| field | value |
|---|---|
| built | 2026-08-31 19:13:09 |
| backend | GPU (cupy / cuSPARSE) |
| samples x models | 3202 x 5385 |
| target loci | 27,683,082 (25,460,374 SNP) |
| model loci read | 34,921,784 |
| reference set | `unrelated` — 2590 of 3202 samples |
| thresholds | min call rate 0.95, min sample call rate 0.99, min coverage 0.75 |
| PCs | 10, from 72,437 thinned loci (MAF >= 0.05, >= 20,000 bp apart) |
| memory budget | 36.0 GB (`PGS_MEM_BUDGET_GB`) |
| deployable models | 5139 of 5385 |
| imputation path active | false — this panel has no missing genotype |

Command line:

```
python3 calculate_distribution_pgs_1kg.py \
    --vcf-dir 1kg_data --pgs-dir PGSCatelog --out-dir 1kg_pgs_reference \
    --panel-meta 1kg_data/manifest.csv \
    --dosage-cache /home/congliu/AllofUsPGS/_dist_cache --resume
```

## Stage timings of that run

| stage | seconds | |
|---|---:|---|
| 1 model loading | 363.0 | read 5385 scoring files |
| 3 VCF parse -> dosage matrix | 15,331.9 | 3a target pass 7,755.8 + 3b dosage pass 7,576.1 |
| S panel target stats | 178.7 | call rates and p_j |
| P genotype PCA | 2.2 | |
| O panel artefacts | 43.4 | |
| 4a model reload | 1,049.1 | per model block |
| 4 variant matching | 203.1 | |
| 5 weight matrix build | 40.0 | |
| 6 score calculation | 6,468.6 | |
| 7 frozen manifest | 315.1 | |
| 8 model statistics | 2.5 | |
| 9 write reference | 31.5 | |
| **total wall clock** | **24,052.1** | 6 h 40 m 52 s |

## Lineage

1. `optim_1kg_workspace/v_1 .. v_10` — the optimisation sequence, benchmarked
   against pgsc_calc on real 1000 Genomes draws. Frozen benchmark record.
2. `pressure_test/gpu_pgs_optim_memory_safe.py` — v_10's arithmetic replanned
   to fit a stated memory budget (two-pass VCF, blocked models, memory-mapped
   dosage matrix), byte-identical output.
3. **this release** — (2) plus the frozen scoring manifest, the call-rate
   scoring set, the mean imputation, per-sample coverage, the stratified
   standardisation, and the non-additive kernel fix.

(1) and (2) are deliberately not shipped here. They still contain the
non-additive bug described in `docs/BUILD_REPORT.md` §2, and they are the
benchmark record rather than a deployable scorer.

## Other files

| file | copied from | unchanged |
|---|---|---|
| `safe_run.sh` | repo root | yes |
| `validate/check_distribution_vs_v10.py` | repo root | yes — its `REF` path points at the pressure-test result it compares against |
| `audit/*` | repo root | yes; the shell driver takes `REPO_ROOT` to find the analysis repo |
| `docs/SPEC.md` | `11_run_1kg_distribution.md` | yes |
| `docs/BUILD_REPORT.md` | `11_run_1kg_distribution_REPORT.md` | yes |
| `docs/REFERENCE_BUNDLE_README.md` | `1kg_pgs_reference/README.md` (generated) | yes |
| `docs/COMPUTE_ENV.md` | `00_computer_env_table.md` | yes |

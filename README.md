# agent_gpu_optimized_final_release

GPU-accelerated, memory-bounded cohort PGS calculation, plus the frozen scoring
manifest that makes the resulting scores transferable.

This is the **exact software that produced `1kg_pgs_reference/`** — the
3202-sample x 5385-model 1000 Genomes reference bundle — packaged so the same
run can be repeated on any cohort of per-sample VCFs. The scorer file here is a
byte-identical copy of the one recorded in that bundle's `build_info.json`
(`sha256 e2b3595abff8a91d5a9b383c5b67b5725a5c1ce7710f3e2665ff65a25b57100e`, see
`PROVENANCE.md`).

One command scores a whole cohort against the whole PGS Catalog and writes a
bundle that another cohort can be put on the scale of.

---

## What is in here

| path | what it is |
|---|---|
| `calculate_distribution_pgs_1kg.py` | **the release.** The whole scorer, in one file: VCF ingest, variant matching, GPU scoring, the frozen manifest, the PCs, the standardisation and the output bundle |
| `safe_run.sh` | memory guardrail: runs a command in a cgroup-capped scope under a machine-wide `MemAvailable` watchdog. Use it — see "Guardrails" |
| `validate/check_distribution_vs_v10.py` | two-sided cross-check of a finished bundle against the optimisation-benchmark scorer it descends from |
| `audit/` | 20 sampled cells of the released 1000 Genomes bundle and a pgsc_calc driver that recomputes exactly those cells. See `audit/README.md` |
| `docs/SPEC.md` | the six requirements the release implements, as originally specified |
| `docs/BUILD_REPORT.md` | how each requirement was implemented, and the non-additive bug this work found in the predecessor |
| `docs/REFERENCE_BUNDLE_README.md` | the README generated into `1kg_pgs_reference/`: the released bundle's file-by-file description and caveats |
| `docs/COMPUTE_ENV.md` | the machine and software versions every timing here was measured on |

The scorer has **no local imports**. Copying `calculate_distribution_pgs_1kg.py`
somewhere else and running it is a supported way to use this repo.

---

## Requirements

* Python 3.13 (3.11+ works), `numpy`, `pandas`, `scipy`, `isal` — `pip install
  -r requirements.txt`.
* `cupy` (`cupy-cuda12x`) and an NVIDIA GPU for the fast path. **Optional**: if
  cupy does not import, the scorer runs the identical arithmetic on
  numpy/scipy and records `backend = CPU` in `build_info.json`. It is much
  slower; it is a fallback, not the intended path.
* `isal` matters. It is the gzip decompressor on the hot read path, and the VCF
  passes are two thirds of the wall clock. Without it the code falls back to
  stdlib `gzip` and the run takes materially longer.
* Disk: scratch for the dosage matrix, which is memory-mapped rather than
  resident. Budget one byte per (sample x target locus) — 82 GiB for the
  1000 Genomes run, 86 GiB for the whole `--dosage-cache` directory — plus the
  output bundle (41 GiB, of which 40 GiB is the per-locus manifest).

Reference machine: NVIDIA GB10, 128 GB unified memory, 20 aarch64 cores, local
NVMe (`docs/COMPUTE_ENV.md`). Unified memory is the reason this release exists
in a memory-bounded form: a host allocation and a device allocation come out of
the same pool.

---

## Input layout

```
cohort_vcfs/                     # --vcf-dir
  SAMPLE_A.vcf.gz                # ONE VCF PER SAMPLE. The sample id is the
  SAMPLE_B.vcf.gz                # filename up to the first '.'
  ...
  manifest.csv                   # optional, --panel-meta
scorefiles/                      # --pgs-dir
  PGS000001.txt.gz               # PGS Catalog harmonised scoring files,
  PGS000002.txt.gz               # GRCh38 (hm_chr / hm_pos columns)
  ...
```

* **Genotypes**: bgzipped single-sample VCFs. Diploid `GT` is read; a
  hemizygous male chrX call is read as the homozygous diploid call, which is
  what pgsc_calc does. Files may be whole-genome or per-chromosome.
* `manifest.csv` is optional and only supplies metadata, with a `sample_id`
  column plus any of `sex`, `population`, `super_population`, `relationship`.
  It is what `--reference-set unrelated` uses to decide who defines the
  reference distribution. Without it, everybody is in the reference set —
  stated in `build_info.json`, not silent.
* **Scoring files**: PGS Catalog `*_hmPOS_GRCh38` harmonised files, gzipped.
  Matching is on `hm_chr`/`hm_pos` with the effect/other alleles; unharmonisable
  rows do not match and are counted, not dropped silently.

---

## Quick start — cohort-level PGS calculation

```bash
./safe_run.sh --mem 72G --gpu 48G --label cohort_pgs -- \
  env PGS_MEM_BUDGET_GB=36 PGS_SPILL_DIR=/scratch/pgs \
  python3 calculate_distribution_pgs_1kg.py \
      --vcf-dir  /data/cohort_vcfs \
      --pgs-dir  /data/scorefiles \
      --out-dir  /data/cohort_pgs_reference \
      --dosage-cache /scratch/pgs_dosage \
      --resume
```

That is the whole thing. It prints a stage-by-stage log and leaves a complete
bundle in `--out-dir`, including a generated `README.md` describing itself.

Start smaller the first time:

```bash
python3 calculate_distribution_pgs_1kg.py --vcf-dir ... --pgs-dir ... \
    --out-dir /tmp/smoke --limit-samples 20 --limit-models 50 --no-manifest
```

### Options

| flag | default | what it does |
|---|---|---|
| `--vcf-dir` | `./1kg_data` | directory of per-sample VCFs |
| `--pgs-dir` | `./PGSCatelog` | directory of scoring files |
| `--out-dir` | `./1kg_pgs_reference` | output bundle |
| `--panel-meta` | `<vcf-dir>/manifest.csv` | cohort metadata CSV |
| `--dosage-cache DIR` | off | persist the two VCF passes here so a re-run skips them. **Set this.** They are two thirds of the run |
| `--resume` | off | reuse `--dosage-cache` and the per-block score checkpoints |
| `--min-call-rate` | `0.95` | a locus enters a model's scoring set S only if the reference panel called it in at least this fraction of samples |
| `--min-sample-call-rate` | `0.99` | a sample below this fraction of \|S\| is flagged and excluded from that model's reference mu/sigma |
| `--min-coverage` | `0.75` | \|S\|/N below this marks the model `deployable=0`. Flagged, never dropped |
| `--reference-set` | `unrelated` | who defines p_j, the call rates, the PCs and mu/sigma. `all` uses everyone |
| `--n-pcs` | `10` | genotype PCs for the stratified standardisation |
| `--pca-variants` | `150000` | candidate PCA loci (common, well-called, distance-thinned) |
| `--pca-maf` | `0.05` | MAF floor for PCA loci |
| `--pca-spacing` | `20000` | minimum bp between PCA loci |
| `--limit-samples`, `--limit-models` | `0` (all) | truncate for a smoke test |
| `--no-manifest` | off | skip writing the per-locus frozen manifest. It is the biggest output; without it the bundle cannot score a future individual |
| `--workers` | auto | parse workers |
| `--reference FILE.tsv` | off | cross-check the scores against a pgsc_calc result matrix as the run finishes |

| env var | what it does |
|---|---|
| `PGS_MEM_BUDGET_GB` | **the one number that matters.** Absolute working-set budget in GB; every block width, tile height and worker count is derived from it rather than from a constant. The released bundle was built at 36 on a 128 GB machine |
| `PGS_SPILL_DIR` | where the memory-mapped dosage matrix is written (default: next to the output) |
| `PGS_MEMBER_CACHE` | directory for the gzip member index cache |

### Guardrails

Use `safe_run.sh`. On this hardware the GPU has no separate memory, so a
runaway device allocation is a runaway *host* allocation, and a cgroup limit
alone does not see it (the driver's pages are not charged to the process
cgroup). `safe_run.sh` pairs the cgroup cap with a machine-wide `MemAvailable`
watchdog, which does catch it, and exports `CUPY_GPU_MEMORY_LIMIT` so cupy
raises `OutOfMemoryError` instead of eating the machine. This is not
theoretical: an unguarded predecessor run took out the host's session manager.

### Scale and runtime

3202 samples x 5385 models x 27.7 M target loci, GB10, `PGS_MEM_BUDGET_GB=36`:
**6 h 41 m** end to end (24,052 s). VCF ingest 4 h 16 m — two streamed passes
over 3202 whole-genome VCFs, 64 % of the run; scoring 1 h 48 m; reading the
5385 scoring files 6 min, plus 17 min reloading them per model block; matching
3 min; frozen manifest 5 min. With `--dosage-cache` populated, a re-run skips
the 4 h 16 m.

---

## What you get

```
out-dir/
  README.md                # generated, describes the bundle it sits in
  build_info.json          # every parameter, shape, threshold and stage timing
  scores_raw.tsv/.npy      # samples x models, the summed score
  scores_z.tsv/.npy        # (raw - mu) / sigma, single mu/sigma
  scores_z_pc.tsv/.npy     # PC-residual standardised — see below
  scores_nobs.tsv/.npy     # samples x models, loci of S each sample called
  model_index.tsv          # per model: coverage, deployable, mu, sigma, quantiles
  model_stats.json.gz      # the same, complete: all quantiles, PC coefficients,
                           # per-super-population mu/sigma
  panel_samples.tsv        # per sample: metadata, reference-set membership, PCs
  panel_targets.tsv.gz     # per target locus: key, contig, pos, REF/ALT, counts,
                           # call rate, p_alt
  pca_loadings.npz         # the frozen PCA axes
  manifest/blk_*.store     # the per-locus frozen manifest
  model_manifest.index.tsv # where each model lives inside manifest/
```

**Which score to use.** `scores_raw` is the summed score and is the one that
compares against pgsc_calc's `SUM`. `scores_z` and `scores_z_pc` are
interpretations of it and exist only inside a bundle. On a multi-ancestry
cohort the two are not interchangeable: a single mu/sigma is dominated by
ancestry-driven allele-frequency differences, and `model_index.tsv`'s
`superpop_mu_spread_sigma` — how many single-mu sigmas separate the most and
least extreme super-population means — is the number that tells you which one
is honest for a given model.

**Read `deployable` before you read a score.** A model whose scoring set covers
less than `--min-coverage` of its published loci is not producing a noisy
version of the published score, it is producing a different quantity. It is
flagged, its scores are still in the matrices, and the decision is yours. In
the released 1000 Genomes bundle 246 of 5385 models are flagged and 21 have an
empty scoring set (no score at all — `mu = sigma = NaN`, and the 0.0 in the
matrix is structural, not a measurement).

**Read `scores_nobs` next to every score.** It is the per-sample, per-model
count of loci actually called, which is what a denominator should have been. A
shrunken score and a real score look identical without it.

---

## Putting a new cohort on an existing reference scale

The point of freezing the manifest is that a later individual is scored with
the panel's locus set, orientation and allele frequencies, never with
recomputed ones. Two routes:

1. **Re-run this script on the new cohort** to get its own bundle. That gives
   you raw scores computed the same way, but a *different* reference
   distribution — appropriate when the new cohort is itself the reference (an
   All of Us-scale panel), not when you want scores on the 1000 Genomes scale.

2. **Apply the frozen manifest.** `calculate_distribution_pgs_1kg.py` exposes
   `load_model_manifest(store_path, offset, length)`; the per-locus recipe,
   including the PC projection and the mean imputation with the frozen `p_eff`,
   is written out in `docs/REFERENCE_BUNDLE_README.md` under "Scoring a new
   individual on this reference". Note honestly: the release ships the manifest,
   the loader and the recipe, but **not** a packaged apply-to-a-new-cohort CLI.
   Writing one on top of `load_model_manifest` is small; pretending it is
   already here would not be.

---

## What the release does that a plain scorer does not

Summarised from `docs/SPEC.md`; `docs/BUILD_REPORT.md` has the implementation.

1. **The scoring set S is defined by call rate, not by intersection.** A locus
   one sample happens to miss stays in S and is imputed for that sample, so
   every individual stays on the same scale. Coverage `|S|/N` is reported per
   model and a low one is flagged.
2. **Residual missingness inside S is mean-imputed** to `2 * p_eff`, plink2's
   default, which keeps `E[score]` unbiased. One extra sparse matmul on the
   existing kernel.
3. **`p_eff` is frozen per locus**, in effect-allele orientation, so a future
   individual is imputed with the panel's number.
4. **Per-sample coverage travels with every score** (`n_obs = A @ obs`), and
   samples below the threshold are excluded from the reference mu/sigma rather
   than carrying a shrunken score into it.
5. **Standardisation comes last and is stratified**, against the panel's own
   genotype PCs, whose loadings are frozen so a new individual can be projected
   onto the same axes.
6. **Absent vs no-call is resolved before any of the above**, and the answer is
   recorded (`imputation_path_active` in `build_info.json`) rather than assumed.

---

## Correctness

* **The reference bundle was cross-checked against its predecessor at full
  scale.** `validate/check_distribution_vs_v10.py` is a two-sided test: the
  scores must agree *exactly* wherever a model is purely additive (the call-rate
  filter, the imputation and the coverage counting are all no-ops on a panel
  with no missing call, so the summation is unchanged and bit-identical), and
  must *disagree* exactly where a model has a dominant or recessive row.
* **The disagreement is a bug this work found in the predecessor.** cuSPARSE's
  SpMM returns an F-contiguous matrix, so `scores.ravel()` returns a copy and
  every `atomicAdd` the non-additive kernel makes was landing in a discarded
  temporary. Every version before this one silently scored dominant/recessive
  rows as zero. It survived because 16 of 3.17 billion matched entries in the
  PGS Catalog are non-additive — but PGS000802 is 14 of 19 rows, so for that
  model the old score was mostly missing. Fixed here
  (`xp.ascontiguousarray(scores)` before the kernel); additive results are
  untouched and stay bit-identical.
* **20 cells of the released bundle are pinned in `audit/`**, 10 scored across
  large/medium/small models and 10 the release could not score, with a
  hard-coded pgsc_calc driver that recomputes exactly those cells for an
  independent comparison.

---

## Caveats

* **Per-sample VCFs must resolve absent vs no-call.** These inputs are slices
  of a joint callset, so `0/0` is explicit and "absent from this sample's VCF"
  means "not in the callset". Against *variant-only* per-sample VCFs, absent is
  ambiguous, and mean imputation would impute loci that were really hom-ref.
  There, you need a per-sample callable-region definition first.
* **Do not reuse a bundle's `p_eff` on a different genotyping platform** without
  re-reading the coverage numbers. `|S|/N` is a property of the panel *and* the
  platform.
* **chrY is not scored** if it is not in the input; chrY model rows then never
  match, which shows up as reduced coverage, not as an error.
* **The PCs are thinned by physical distance, not LD-pruned.** Adequate for the
  global-ancestry axes used here, and nothing else.
* **The manifest is the big output** (40 GB for 5385 models x 3.2 billion
  entries). `--no-manifest` removes it, at the cost of the bundle's ability to
  score a future individual.

---

## Provenance

See `PROVENANCE.md` for the checksum, the exact command line and the stage
timings of the released 1000 Genomes build. The lineage is: the optimisation
sequence `v_1..v_10` -> the memory-safe rewrite that survives 1000 genotypes x
1000 models inside a stated budget -> this release, which adds the frozen manifest, the coverage accounting, the
standardisation and the non-additive fix. The `v_*` workspaces and the
pressure-test scorer are deliberately *not* included: they are the frozen
benchmark record, they still carry the non-additive bug, and none of them
should be used in production.

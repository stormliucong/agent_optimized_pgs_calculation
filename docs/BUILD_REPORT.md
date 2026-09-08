# PGS reference distributions from 1000 Genomes

`calculate_distribution_pgs_1kg.py` builds `1kg_pgs_reference/`: a **frozen
scoring manifest** plus the reference distribution of every PGS Catalog model
over the 1000 Genomes panel, applying the six criteria in
`11_run_1kg_distribution.md`.

It is `pressure_test/gpu_pgs_optim_memory_safe.py` — v_10's arithmetic inside a
memory budget — with the scoring stage extended and an output stage added.
Nothing upstream of scoring changed: the matching rules, the ordinal tie-break,
the isal/shared-memory inflate ring, the batched device VCF parse, the two
streamed VCF passes, the memory-mapped dosage matrix and the budget-sized block
loops are all inherited unedited. `pressure_test/` and the `v_1..v_10`
workspaces are untouched.

---

## 1. What the six criteria became, concretely

### §1 — S by call rate, not by intersection

A locus enters model *j*'s scoring set **S** if it matched the model **and** the
reference panel called it in ≥ `--min-call-rate` (default 0.95) of samples. The
candidate pool is the target *union*, which the dosage matrix already provides
as its per-column observed count, so this costs one sequential pass over `D`
(stage `S_panel_target_stats`) and a device comparison per matched entry.

`coverage = |S| / N` is reported for every model in `model_index.tsv`, together
with `n_matched` (before the filter) and `n_dropped_low_call_rate`. A model
below `--min-coverage` (default 0.75) is marked `deployable=0` with a reason.
It is **flagged, not dropped** — its scores are still in the matrices, so the
decision stays with the reader.

### §2 — mean imputation of residual missingness inside S

An uncalled locus in S contributes `w_j · 2·p_j^eff` rather than zero, which is
plink2's default and what keeps `E[score]` unbiased. Implemented as one extra
sparse matmul on the existing kernel:

```
scores  = Wd @ dose        # signed weight on the ALT dosage   (inherited)
scores += Wo @ obs         # the +2w a flipped locus owes       (inherited)
scores += Wfill @ (1-obs)  # NEW: the mean-imputed effect dosage
n_obs   = A @ obs          # NEW: §4
```

`Wfill` carries `w_j · 2·p_j^eff` on the same sparsity pattern as `Wd`. That is
the task's `sign_j·w_j·2p_j` *plus* `Wo`'s `2w` over the missing entries, merged
into one matrix: for a flipped locus `p^eff = 1 − p^alt`, so `w·2(1−p^alt)` is
exactly `(−w·2p^alt) + 2w`. One matrix instead of two, algebraically identical.

Non-additive loci are imputed with the expectation of the same transform under
Hardy–Weinberg at `p^eff`: `q(2−q)` for dominant, `q²` for recessive.

**On the 1000 Genomes panel both new matmuls are skipped**, because the panel has
no missing call at all (see §6) — `Wfill @ (1−obs)` is a matmul against a zero
matrix and `A @ obs` is the constant row count. Skipping them is exact, not an
approximation, and it is what keeps the raw scores bit-identical to the
inherited scorer's (§4 of this report).

### §3 — p_j frozen with the manifest

`manifest/blk_*.store` + `model_manifest.index.tsv` hold, per locus of S:

| column | type | meaning |
|---|---|---|
| `var` | int32 | index into `panel_targets.tsv.gz` — the key, contig, position, REF, ALT and panel call rate |
| `weight` | float64 | the model's effect weight, bit-exact from the scoring file |
| `flags` | int8 | bit 0 = flip (effect allele is the target's REF); bits 1–2 = additive / dominant / recessive |
| `ea` | int8 | effect-allele base code |
| `p_eff` | float32 | **the frozen panel effect-allele frequency** |

`p_eff` is stored in *effect-allele orientation*, so a future individual is
imputed with the panel's number and never with one recomputed from their own
data. float32 is a deliberate 4-byte convenience: at 3.2 billion entries every
byte is 3.2 GB, and the exact value is always reconstructible from
`panel_targets`' integer columns as `n_alt_reference / (2 · n_obs_reference)`,
complemented when the flip bit is set.

The panel call rate is **not** duplicated per locus. It is a property of the
target, it is in `panel_targets.tsv.gz` under the same `var` index, and 3.2
billion copies of it would be 13 GB of pure redundancy.

### §4 — per-sample coverage next to every score

`scores_nobs.tsv` / `.npy` is the full samples × models matrix of `n_obs[i,j]`
— the loci of S that sample *i* actually called — computed as `A @ obs` against
the 0/1 pattern of the weight matrix. A sample below
`--min-sample-call-rate` (default 0.99) of `|S|` is counted in
`n_sample_flagged` and **excluded from that model's reference μ/σ** rather than
carrying a shrunken score into it.

### §5 — standardise last, and stratify

Two standardisations are emitted:

* `scores_z` — a single μ/σ over the QC-passing reference samples;
* `scores_z_pc` — the raw score regressed on the panel's top `--n-pcs` genotype
  PCs, with the **residual** standardised. `pc_beta` and `sigma_resid` are in
  `model_stats.json.gz`, so the same adjustment applies to a new individual.

The PCs are computed here from the dosage matrix (common, well-called,
distance-thinned SNPs; GRM → eigendecomposition), and what is frozen is the
per-variant **loadings**, in `pca_loadings.npz`, so a future individual is
projected onto the same axes:

```
x_j = (dosage_j − 2·p_j) / sqrt(2·p_j·(1−p_j))      (missing → 0)
PC  = (x · V) / sqrt(K)
```

`model_index.tsv`'s `superpop_mu_spread_sigma` — how many single-μ σ separate
the most and least extreme super-population means — is the number that says
whether `z` or `z_pc` is the honest one for a given model.

### §6 — absent vs. no-call, resolved before any of the above

This is the criterion the input data settles rather than the code. The 1000
Genomes VCFs in `1kg_data/` are per-sample slices of the NYGC joint callset
(`download_1kg_all.py` splits one multi-sample VCF per chromosome, writing every
record for every sample). Consequently:

* all 3202 samples carry the **identical** 73,554,796 records;
* `1kg_data/manifest.csv` reports `n_missing = 0` for every sample — `0/0` is
  explicit and there is not one `./.` in the cohort;
* therefore "absent from this sample's VCF" means "absent from the callset",
  i.e. genuinely not assayed. **No callable-region definition is needed or
  assumed.**

The build measures this rather than trusting it: `build_info.json` records
`imputation_path_active`, and the README states the minimum observed per-target
call rate. The caveat that matters for reuse is in the generated README:
against *variant-only* per-sample VCFs "absent" is ambiguous, and this bundle's
`p_eff` must not be applied there without a callable-region definition, or §2
would impute loci that were really hom-ref.

---

## 2. A bug this work found in v_10

`score_gpu`'s non-additive kernel has never worked, in any version.

cuSPARSE's SpMM returns an **F-contiguous** matrix. `scores.ravel()` is C-order,
so on an F-contiguous array it returns a **copy** — every `atomicAdd` the
dominant/recessive kernel makes lands in a temporary that is then discarded (and
its `row·tile_S + s` index arithmetic assumes row-major besides). Measured
directly:

```
W @ d  ->  C-contig False   F-contig True
scores.ravel() shares memory: False
```

Confirmed end to end on a 40-sample fixture whose model is entirely
dominant/recessive: `pressure_test/gpu_pgs_optim_memory_safe.py` returns
`0.000000` for every sample, and so did the new script before the fix. One
`xp.ascontiguousarray(scores)` before the kernel repairs it; the additive result
it copies is untouched, so additive models stay bit-identical.

**Blast radius.** An independent scan of all 5385 scoring files finds exactly
**four** models with a non-additive row — PGS000318 (2 recessive), PGS000319 (2
recessive), PGS000802 (7 dominant + 7 recessive of 19 rows), PGS004700 (3
recessive of 12 rows) — 21 rows in total, of which 16 matched a target in the
ALL run. That is why it survived: 16 of 3.17 billion matched entries. But
PGS000802 is 14/19 non-additive, so for that model v_10's score is mostly
missing, not slightly wrong.

Fixed in `calculate_distribution_pgs_1kg.py` only. `pressure_test/` and
`optim_1kg_workspace/v_*` are deliberately left alone — they are the frozen
benchmark record — but the same one-line fix should be ported before any of them
is used in production. `check_distribution_vs_v10.py` turns this into a
two-sided test at full scale.

---

## 3. Validation


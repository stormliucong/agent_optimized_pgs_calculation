# 1000 Genomes PGS reference distributions

Built by `calculate_distribution_pgs_1kg.py` from the per-individual 1000
Genomes VCFs in `/home/congliu/AllofUsPGS/1kg_data` and the PGS Catalog scoring files in `/home/congliu/AllofUsPGS/PGSCatelog`.
Shape: **3202 samples x 5385 models**, 27,683,082 target loci.

Everything here is a *frozen scoring manifest*: the same locus set, the same
orientation and the same allele frequencies are applied to the panel and must
be applied unchanged to every future individual, or that individual's score is
not on this scale.

## Files

| file | what it is |
|---|---|
| `build_info.json` | every parameter, shape, timing and threshold of the build |
| `panel_samples.tsv` | one row per panel sample: sex, population, super-population, relatedness, whether it is in the reference set, and its genotype PCs |
| `panel_targets.tsv.gz` | one row per target locus: `var` index, key, contig, position, REF, ALT, the integer counts `n_obs_reference` / `n_alt_reference` / `n_obs_all`, and the derived `call_rate` and `p_alt`. `p_alt` is exactly `n_alt_reference / (2 * n_obs_reference)` — reconstruct it from the integers if you need more than the 10 digits printed |
| `pca_loadings.npz` | the frozen PCA: variant keys, `p_j`, `sd_j`, loadings `V`, eigenvalues |
| `model_index.tsv` | one row per model: coverage, deployability, mu, sigma, PC-residual sigma, and the quantiles of the reference distribution |
| `model_stats.json.gz` | the same per model but complete: all quantiles, PC regression coefficients, per-super-population mu/sigma |
| `manifest/blk_*.store` + `model_manifest.index.tsv` | the per-locus frozen manifest of every model |
| `scores_raw.tsv`, `scores_z.tsv`, `scores_z_pc.tsv`, `scores_nobs.tsv` | samples x models: raw score, z against a single mu/sigma, z against the PC-adjusted mean, and the number of loci of S the sample called |
| `scores_*.npy` | the same matrices in binary, in the native (unsorted) order of `build_info.json`'s `sample_ids` / `pgs_ids` |

## The five decisions this bundle freezes

1. **S is defined by call rate, not by intersection.** A locus is in model *j*'s
   scoring set S if it matched the model AND the reference panel called it in at
   least **0.95** of samples. `coverage = |S| / N` is in
   `model_index.tsv`; a model below **0.75** is marked
   `deployable=False` — it is not that its score is noisy, it is that the score
   is a different quantity from the published one.
2. **Residual missingness inside S is mean-imputed**, dosage -> `2 * p_eff`, so
   E[score] stays unbiased. This is what plink2 does by default.
3. **`p_eff` is frozen per locus** in the manifest, in effect-allele
   orientation. Impute a new individual with the stored number, never a
   recomputed one.
4. **Per-sample coverage travels with every score.** `scores_nobs.tsv` holds
   `n_obs[i,j]`, the loci of S sample *i* actually called; a sample below
   **0.99** of `|S|` is excluded from the reference mu/sigma
   and counted in `model_index.tsv`'s `n_sample_flagged`.
5. **Standardisation comes last, and it is stratified.** `z` uses a single
   mu/sigma; `z_pc` regresses the raw score on the top 10 genotype PCs and
   standardises the residual. On a five-continent panel the two are not
   interchangeable: `model_index.tsv`'s `superpop_mu_spread_sigma` is how many
   single-mu sigmas separate the most and least extreme super-population means.

## Scoring a new individual on this reference

```python
m = load_model_manifest('manifest/blk_003.store', offset, length)  # from the index
# dosage[k] = ALT-allele dosage of target m['var'][k] in the new sample,
#             or -1 where the new sample did not call that locus
flip = (m['flags'] & 1) == 1
eff  = np.where(flip, 2.0 - dosage, dosage)          # effect-allele dosage
eff  = np.where(dosage < 0, 2.0 * m['p_eff'], eff)   # frozen mean imputation
raw  = float((m['weight'] * eff).sum())
n_obs = int((dosage >= 0).sum())                     # QC this against |S|
z_pc = (raw - (beta[0] + beta[1:] @ pcs)) / sigma_resid   # from model_stats
```
`load_model_manifest` is defined in `calculate_distribution_pgs_1kg.py`.
Non-additive loci (`flags >> 1` of 1 or 2) apply `min(eff,1)` / `1(eff==2)`
instead, and impute `q(2-q)` / `q^2`.

## Caveats worth reading before using this

* **This panel has no missing genotypes at all.** The 1000 Genomes NYGC release
  is a joint callset and every sample carries every site: measured here, the
  minimum per-target call rate over the reference set is
  1.0000 and 27,683,082 of 27,683,082 targets clear the
  0.95 threshold, so the mean-imputation path never fires. That is a property of THIS panel, not of the method:
  the imputation and the per-sample coverage counters are implemented, tested
  and frozen precisely because an array or an exome callset will need them.
  `coverage = |S|/N` therefore reduces to the model's match rate on this
  panel — which is the number that actually varies, from 0 to 1.
* **Absent means absent, not uncalled (§6).** These are per-sample slices of a
  joint callset, so 0/0 is explicit in every file and a locus missing from a
  sample's VCF is a locus that is not in the callset at all. No callable-region
  definition is needed or assumed. **Do not reuse this bundle's `p_eff` against
  variant-only per-sample VCFs** without a callable-region definition: there,
  absent is ambiguous and step 2 would impute loci that were really hom-ref.
* **Male chrX is hemizygous and is read as the homozygous diploid call**, which
  is what pgsc_calc does; chrY is not in this release at all, so chrY model
  rows never match.
* 612 of 3202 samples are trio children and are excluded from
  the reference set (`--reference-set all` overrides). They are still scored,
  and are a free check that the frozen manifest transfers.
* 246 of 5385 models are marked NOT deployable at coverage < 0.75
  (median coverage 0.943). Their scores are in the matrices and
  their statistics are in `model_index.tsv`; they are flagged, not dropped.
* The PCs are thinned by physical distance (20,000 bp), not LD-pruned.
  That is adequate for the global-ancestry axes used here and nothing else.


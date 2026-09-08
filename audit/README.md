# 20 pinned cells of the released 1000 Genomes bundle

Twenty (genotype x PGS model) cells drawn out of `1kg_pgs_reference/` — the
bundle this release produced — so that the release's output can be checked
cell by cell against an independent implementation.

| file | what it is |
|---|---|
| `final_release_version_sample_20.csv` | the deliverable: `genotype_id, pgs_model_id, score`, with `NULL` where the release did not calculate a score |
| `final_release_version_sample_20_provenance.tsv` | why each cell was picked: size tier, `n_rows`, `\|S\|`, coverage, and the release's own reason for a NULL |
| `sample_final_release_scores.py` | regenerates both from the bundle. Deterministic — no random sampling |
| `sampled_pgs_calc_final_release.sh` | recomputes exactly these cells with pgsc_calc v2.3.0 and writes a side-by-side comparison |

## How the 20 were chosen

Nothing is random. Every pick is a fixed rank in a sorted list, so the draw
reproduces byte for byte on the same bundle.

**Ten scored cells**, spanning model size (`n_rows` = rows in the published
scoring file):

| tier | range | picked |
|---|---|---|
| large | `n_rows >= 1e6` (1254 models) | PGS003162 (10,316,178 rows), PGS000554 (1,114,993), PGS004720 (1,000,300) |
| medium | `1e4 <= n_rows < 1e6` (1693) | PGS002988 (996,763), PGS005130 (708,315), PGS004102 (61,651), PGS001232 (10,053) |
| small | `n_rows < 1e4` (2192) | PGS001007 (9,993), PGS001632 (302), PGS001033 (1) |

**Ten NULL cells.** A model with an empty scoring set (`n_S = 0`) has no score:
the matrix holds a structural `0.0` that is not a measurement, and
`model_index.tsv` gives it `mu = sigma = NaN`, `coverage = 0`, `deployable = 0`.
Exactly 21 of 5385 models are in that state, for three distinct causes, and the
draw covers all three:

| cause | models picked |
|---|---|
| the scoring file has no additive row to score at all (`n_rows = 0`) — HLA haplotype models, and models that publish `dosage_0/1/2_weight` instead of `effect_weight` | PGS000343, PGS003757, PGS004256, PGS004260, PGS004262, PGS004272, PGS004280, PGS004304 |
| the file parses but none of its loci is in the callset | PGS003451 (2 proxy SNPs in the MHC) |
| the file parses, is enormous, and still matches nothing | PGS005228 (2,379,381 rows whose `effect_allele` equals `other_allele`, so no row can be oriented against a target) |

**Ten genotypes**, two per super-population (AFR, AMR, EAS, EUR, SAS), at rank 0
and rank n//2 of each super-population's sorted sample list: HG01879, HG03342,
HG00551, HG01462, HG00403, HG02116, HG00096, HG01779, HG01583, HG03867. The
same ten carry both halves, so any difference between the halves is a property
of the model and not of the sample.

**Which score.** `score` is the raw summed score (`scores_raw`), the quantity
pgsc_calc reports as `SUM`. `z` and `z_pc` are reference standardisations that
exist only inside a bundle and have no pgsc_calc analogue.

## Reproducing the CSV

```bash
python3 sample_final_release_scores.py --reference /path/to/1kg_pgs_reference \
                                       --out final_release_version_sample_20.csv
```

## Recomputing the same cells with pgsc_calc

```bash
REPO_ROOT=/path/to/AllofUsPGS ./sampled_pgs_calc_final_release.sh --dry-run
REPO_ROOT=/path/to/AllofUsPGS ./sampled_pgs_calc_final_release.sh
```

`REPO_ROOT` is the analysis repo holding `1kg_data/`, `PGSCatelog/`, `pgs_cal/`
and `pgs_cal_run.sh`; it defaults to the script's own directory, which is right
when the script is run from that repo's root. The driver symlinks the 10 VCFs
and 20 scoring files into a workspace, runs the unmodified `pgs_cal_run.sh`
wrapper with `MIN_OVERLAP=0` (so a model that matches nothing comes out as `NA`
rather than vanishing), `SEX_FEMALE=1` (plink2 will not import chrX without a
sex assignment), `CPUS20=1` and a cold `RESUME`, and then joins the 20 sampled
cells onto the resulting matrix into `final_release_vs_pgs_cal_20.csv`
(`final_release_score, pgs_cal_score, delta, agree`).

**It has not been run.** pgsc_calc wants one multi-sample VCF per sampleset, so
stage 0 merges ten whole-genome 1000 Genomes VCFs (73,554,796 records each) into
a single ~11 GB file, which plink2 then imports into a pfile. Hours of wall
clock and hundreds of GB of scratch, for 200 cells. `--dry-run` stages the
inputs and prints the exact command without launching it.

A cell agrees when both sides are missing, or when both are finite and within
`1e-6` relative tolerance. Note before reading a disagreement: pgsc_calc and
this release do not define the scoring set identically — the release drops loci
below a 0.95 panel call rate and mean-imputes the rest, pgsc_calc scores what
plink2 matched — so on a cohort with missing genotypes the two are expected to
differ. On this panel there are no missing genotypes, which is what makes the
comparison clean.

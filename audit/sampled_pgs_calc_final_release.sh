#!/bin/bash
# =============================================================================
# sampled_pgs_calc_final_release.sh
#
# Re-compute, with pgs_cal (PGScatalog/pgsc_calc v2.3.0), the 20 (genotype x PGS
# model) cells that `sample_final_release_scores.py` drew out of the final
# release's 1000 Genomes reference bundle into
# `final_release_version_sample_20.csv`, and put the two side by side.
#
# The 20 cells are HARD-CODED below (item 6 of 13_final_release_and_score_
# sampled.md allows it).  They are 10 genotypes x 20 models: pgsc_calc computes
# a full matrix, so it produces 200 cells, of which the 20 sampled ones are
# extracted for the comparison.  The extra 180 are free and are written out
# too — they cost nothing beyond the score step, which is not the bottleneck.
#
#   scored half   10 models with a real score, spanning model size:
#                 LARGE  PGS003162 (10,316,178 rows), PGS000554, PGS004720
#                 MEDIUM PGS002988, PGS005130, PGS004102, PGS001232
#                 SMALL  PGS001007, PGS001632, PGS001033 (1 row)
#   NULL half     10 models the release could not score at all (n_S = 0):
#                 PGS003451 (2 rows, neither locus in the callset)
#                 PGS005228 (2,379,381 rows, effect_allele == other_allele)
#                 PGS000343 (HLA haplotype model, no SNP row)
#                 PGS003757, PGS004256, PGS004260, PGS004262, PGS004272,
#                 PGS004280, PGS004304 (dosage_0/1/2_weight format, i.e. no
#                 effect_weight column, so no additive row to score)
#                 pgsc_calc is expected to emit NA / drop these columns as
#                 well; that is the point of including them.
#
# ---------------------------------------------------------------------------
# THIS SCRIPT IS DELIBERATELY NOT RUN AS PART OF THE TASK (item 6: "do not try
# to actually run the pgs_cal").  It is slow by construction: the 10 target
# VCFs are whole-genome per-sample slices of the 1000 Genomes NYGC callset
# (73,554,796 records each), pgsc_calc wants ONE multi-sample VCF per
# sampleset, so stage 0 merges ~10 x 1.1 GB of gzip into a single ~11 GB VCF
# and plink2 then imports it into a pfile.  Budget hours and ~200-300 GB of
# scratch, and note that PGS003162 alone carries 10.3 M scoring rows.
# ---------------------------------------------------------------------------
#
# Configuration reuse.  Nothing about pgs_cal is re-invented here: the script
# builds the two input directories pgs_cal_run.sh expects and hands it the same
# environment run_pgs_cal_1kg.py uses for its 1000 Genomes reference runs.
#
#   MIN_OVERLAP=0.0       keep every model's column no matter how little of it
#                         matched, so a model that matches nothing comes out as
#                         NA instead of vanishing.  This is what makes the NULL
#                         half comparable at all.
#   NO_MEAN_IMPUTATION    default 0 = the release's policy (§2: residual
#                         missingness inside S is mean-imputed).  It is a no-op
#                         either way here: plink2 only mean-imputes at >= 50
#                         samples and this run has 10, and the 1000 Genomes
#                         panel has no missing call to impute (see the bundle
#                         README).  Set to 1 to state "missing contributes
#                         nothing" explicitly.
#   SEX_FEMALE=1          these VCFs carry chrX and plink2 refuses to import it
#                         without a sex assignment; the wrapper writes an
#                         all-female --update-sex table plus --split-par b38,
#                         which is how every other scorer in this repo reads
#                         chrX (male hemizygous call read as the homozygous
#                         diploid call, as pgsc_calc does).
#   MAX_CPUS=nproc,
#   CPUS20=1              use the whole machine; --max_cpus alone is only a
#                         ceiling, CPUS20 raises the per-process cpus request.
#   RESUME=" "            a single space, so pgs_cal_run.sh's ${RESUME:--resume}
#                         does not fall back to -resume: a cold, honest run.
#   ancestry adjustment   not run (pgsc_calc skips it unless --run_ancestry).
#
# Outputs (under sampled_pgs_cal_workspace/):
#   pgs_cal_results.tsv                10 samples x 20 models, SUM per PGS
#   pgs_cal_time.tsv                   per-stage timing
#   final_release_vs_pgs_cal_20.csv    the 20 sampled cells, both scores, delta
#   inputs/ results/ work/ logs/       pgs_cal's own scratch and reports
#
# Usage:
#   ./sampled_pgs_calc_final_release.sh              # run it
#   ./sampled_pgs_calc_final_release.sh --dry-run    # stage inputs, print the
#                                                    # pgs_cal command, stop
#   ./sampled_pgs_calc_final_release.sh --compare-only
#                                                    # only rebuild the
#                                                    # comparison from an
#                                                    # existing results file
# =============================================================================
set -euo pipefail

# REPO_ROOT is the analysis repo that holds 1kg_data/, PGSCatelog/, pgs_cal/
# and pgs_cal_run.sh.  It defaults to this script's directory, which is where
# it lives in that repo; the copy shipped inside
# agent_gpu_optimized_final_release/audit/ needs REPO_ROOT pointed back at it.
ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
VCF_SRC="${VCF_SRC:-${ROOT}/1kg_data}"
PGS_SRC="${PGS_SRC:-${ROOT}/PGSCatelog}"
SAMPLE_CSV="${SAMPLE_CSV:-${ROOT}/final_release_version_sample_20.csv}"
WS="${WS:-${ROOT}/sampled_pgs_cal_workspace}"
RUNNER="${RUNNER:-${ROOT}/pgs_cal_run.sh}"

MODE="run"
case "${1:-}" in
    --dry-run)      MODE="dry" ;;
    --compare-only) MODE="compare" ;;
    "")             ;;
    *) echo "usage: $0 [--dry-run|--compare-only]" >&2; exit 2 ;;
esac

# ─── the sampled cells, hard-coded ──────────────────────────────────────────
GENOTYPES=(HG01879 HG03342 HG00551 HG01462 HG00403 HG02116 HG00096 HG01779 HG01583 HG03867)

# 10 scored models, largest first within each size tier
MODELS_SCORED=(PGS003162 PGS000554 PGS004720 PGS002988 PGS005130 PGS004102 PGS001232 PGS001007 PGS001632 PGS001033)
# 10 models the release returned NULL for
MODELS_NULL=(PGS003451 PGS005228 PGS000343 PGS003757 PGS004256 PGS004260 PGS004262 PGS004272 PGS004280 PGS004304)
MODELS=("${MODELS_SCORED[@]}" "${MODELS_NULL[@]}")

GENO_DIR="${WS}/sampled_input/genotype"
MODEL_DIR="${WS}/sampled_input/models"

echo "============================================================"
echo "  pgs_cal on the final release's 20 sampled cells"
echo "  genotypes: ${#GENOTYPES[@]}   models: ${#MODELS[@]}   mode: ${MODE}"
echo "  workspace: ${WS}"
echo "============================================================"

if [[ "${MODE}" != "compare" ]]; then
    # ─── stage the inputs as symlinks (no copy: these VCFs are ~1.1 GB each) ─
    rm -rf "${GENO_DIR}" "${MODEL_DIR}"
    mkdir -p "${GENO_DIR}" "${MODEL_DIR}" "${WS}/logs"

    for s in "${GENOTYPES[@]}"; do
        src="${VCF_SRC}/${s}.vcf.gz"
        [[ -f "${src}" ]] || { echo "missing genotype: ${src}" >&2; exit 1; }
        ln -sf "${src}" "${GENO_DIR}/${s}.vcf.gz"
        [[ -f "${src}.tbi" ]] && ln -sf "${src}.tbi" "${GENO_DIR}/${s}.vcf.gz.tbi"
    done
    for m in "${MODELS[@]}"; do
        src="${PGS_SRC}/${m}.txt.gz"
        [[ -f "${src}" ]] || { echo "missing model: ${src}" >&2; exit 1; }
        ln -sf "${src}" "${MODEL_DIR}/${m}.txt.gz"
    done
    echo "[stage] ${GENO_DIR}: $(ls "${GENO_DIR}"/*.vcf.gz | wc -l) VCFs"
    echo "[stage] ${MODEL_DIR}: $(ls "${MODEL_DIR}"/*.txt.gz | wc -l) scoring files"
fi

# ─── run pgs_cal through the repo's unmodified wrapper ──────────────────────
if [[ "${MODE}" == "run" ]]; then
    [[ -x "${RUNNER}" ]] || { echo "missing ${RUNNER}" >&2; exit 1; }
    env \
        GENO_DIR="${GENO_DIR}" \
        MODEL_DIR="${MODEL_DIR}" \
        WS="${WS}" \
        WORK_DIR="${WS}/work" \
        SAMPLESET="SAMPLED20" \
        MIN_OVERLAP="0.0" \
        NO_MEAN_IMPUTATION="${NO_MEAN_IMPUTATION:-0}" \
        MAX_CPUS="$(nproc)" \
        MAX_MEM="${MAX_MEM:-100.GB}" \
        CPUS20="1" \
        SEX_FEMALE="1" \
        RESUME=" " \
        bash "${RUNNER}" 2>&1 | tee "${WS}/logs/sampled_run.log"
elif [[ "${MODE}" == "dry" ]]; then
    cat <<EOF

[dry-run] inputs are staged.  The run would be:

  GENO_DIR=${GENO_DIR} \\
  MODEL_DIR=${MODEL_DIR} \\
  WS=${WS} WORK_DIR=${WS}/work SAMPLESET=SAMPLED20 \\
  MIN_OVERLAP=0.0 NO_MEAN_IMPUTATION=0 MAX_CPUS=$(nproc) MAX_MEM=100.GB \\
  CPUS20=1 SEX_FEMALE=1 RESUME=" " \\
  bash ${RUNNER}

which is pgsc_calc v2.3.0 with --target_build GRCh38 --only_score
--min_overlap 0.0, no ancestry adjustment and no imputation stage.
Not launched: see the header of this script for why.
EOF
    exit 0
fi

# ─── side-by-side comparison of the 20 sampled cells ────────────────────────
RESULTS="${WS}/pgs_cal_results.tsv"
if [[ ! -s "${RESULTS}" ]]; then
    echo "[compare] ${RESULTS} not present — nothing to compare." >&2
    exit 1
fi

python3 - "${SAMPLE_CSV}" "${RESULTS}" "${WS}/final_release_vs_pgs_cal_20.csv" <<'PYEOF'
"""Join the 20 sampled release cells onto the pgs_cal matrix.

A cell is NULL on the release side when the model's scoring set was empty, and
NA/absent on the pgs_cal side when no variant of the model matched the target.
Those two are the same statement, so `agree` is true when both are missing or
when both are finite and within tolerance.
"""
import csv, math, sys

sample_csv, results_tsv, out_csv = sys.argv[1:4]

with open(results_tsv) as f:
    r = csv.DictReader(f, delimiter='\t')
    mat = {row[r.fieldnames[0]]: row for row in r}

def as_float(v):
    if v is None:
        return None
    v = v.strip()
    if v in ('', 'NA', 'NaN', 'nan', 'NULL', '.'):
        return None
    return float(v)

rows, n_agree, n_cmp = [], 0, 0
with open(sample_csv) as f:
    for rec in csv.DictReader(f):
        g, m = rec['genotype_id'], rec['pgs_model_id']
        rel = as_float(rec['score'])
        cal = as_float(mat.get(g, {}).get(m))
        if rel is None and cal is None:
            agree, delta = True, ''
        elif rel is None or cal is None:
            agree, delta = False, ''
        else:
            delta = cal - rel
            agree = abs(delta) <= 1e-6 * max(1.0, abs(rel))
        n_cmp += 1
        n_agree += bool(agree)
        rows.append([g, m,
                     'NULL' if rel is None else repr(rel),
                     'NA' if cal is None else repr(cal),
                     '' if delta == '' else repr(delta),
                     int(bool(agree))])

with open(out_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['genotype_id', 'pgs_model_id', 'final_release_score',
                'pgs_cal_score', 'delta', 'agree'])
    w.writerows(rows)
print(f"[compare] {n_agree}/{n_cmp} cells agree -> {out_csv}")
PYEOF

echo "============================================================"
echo "  matrix:     ${RESULTS}"
echo "  timing:     ${WS}/pgs_cal_time.tsv"
echo "  comparison: ${WS}/final_release_vs_pgs_cal_20.csv"
echo "============================================================"

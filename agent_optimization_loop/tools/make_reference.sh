#!/usr/bin/env bash
# make_reference.sh - compute the correctness reference with pgsc_calc v2.3.0.
#
#   benchmark/ --(prepare)--> .pgsc_calc/inputs --(pgsc_calc)--> .pgsc_calc/results
#              --(finalize)--> reference/reference_scores.tsv (+ provenance files)
#
# Settings (the study's): --min_overlap 0, no mean imputation, no ancestry
# adjustment, --target_build GRCh38, --only_score, everything else default.
#
# Needs: nextflow (>= 24.04, with Java 17+) and either conda/mamba (profile
# "conda") or a working docker (profile "docker"). If nextflow is missing and
# conda/mamba is present, it is installed into .pgsc_calc/nxf.
#
# usage:  tools/make_reference.sh [--profile conda|docker|mamba] [--cpus N] [--mem 16.GB]
# env:    PYTHON (.venv/bin/python)  PGSC_CALC_DIR (.pgsc_calc/pgsc_calc)
#         PGSC_PLINK2_ENV / PGSC_UTILS_ENV: conda prefixes to use instead of pgsc_calc's
#         own plink2 / pgscatalog-utils environments (e.g. on linux-aarch64)
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
WORK="$ROOT/.pgsc_calc"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
PGSC_CALC_DIR="${PGSC_CALC_DIR:-$WORK/pgsc_calc}"
PGSC_TAG="v2.3.0"

ncpu() { getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4; }
PROFILE=""; CPUS="$(ncpu)"; MEM="16.GB"
while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --cpus)    CPUS="$2"; shift 2 ;;
    --mem)     MEM="$2"; shift 2 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done
if [ -z "$PROFILE" ]; then
  if command -v conda >/dev/null || command -v mamba >/dev/null; then PROFILE=conda
  elif docker info >/dev/null 2>&1; then PROFILE=docker
  else echo "need conda/mamba or a working docker for pgsc_calc"; exit 1; fi
fi
mkdir -p "$WORK"
t0=$(date +%s)

# ── nextflow ─────────────────────────────────────────────────────────────────
export PATH="$WORK/nxf/bin:$PATH"
if ! command -v nextflow >/dev/null; then
  CONDA=$(command -v mamba || command -v conda || true)
  [ -n "$CONDA" ] || { echo "nextflow not found and no conda/mamba to install it (see https://www.nextflow.io/docs/latest/install.html)"; exit 1; }
  echo "[reference] installing nextflow into $WORK/nxf"
  "$CONDA" create -y -q -p "$WORK/nxf" -c conda-forge -c bioconda "nextflow>=24.04" >/dev/null
fi
echo "[reference] $(nextflow -version 2>/dev/null | grep -o 'version [0-9.]*' | head -1)  profile=$PROFILE  cpus=$CPUS"

# ── pgsc_calc ────────────────────────────────────────────────────────────────
if [ ! -f "$PGSC_CALC_DIR/main.nf" ]; then
  git -c advice.detachedHead=false clone -q --depth 1 --branch "$PGSC_TAG" https://github.com/PGScatalog/pgsc_calc.git "$PGSC_CALC_DIR"
fi

# ── inputs ───────────────────────────────────────────────────────────────────
"$PYTHON" tools/build_reference.py prepare --work "$WORK"

# ── run ──────────────────────────────────────────────────────────────────────
export NXF_ANSI_LOG=false
export NXF_CONDA_CACHEDIR="${NXF_CONDA_CACHEDIR:-$WORK/conda_envs}"
rm -rf "$WORK/results"
EXTRA=()
if [ -n "${PGSC_PLINK2_ENV:-}${PGSC_UTILS_ENV:-}" ]; then        # self-built conda prefixes
  {
    echo "process {"
    if [ -n "${PGSC_PLINK2_ENV:-}" ]; then echo "  withLabel: plink2 { ext.conda = '$PGSC_PLINK2_ENV' }"; fi
    if [ -n "${PGSC_UTILS_ENV:-}" ]; then echo "  withLabel: pgscatalog_utils { ext.conda = '$PGSC_UTILS_ENV' }"; fi
    echo "}"
  } > "$WORK/overrides.config"
  EXTRA=(-c "$WORK/overrides.config")
fi
nextflow -log "$WORK/nextflow.log" run "$PGSC_CALC_DIR/main.nf" \
  -profile "$PROFILE" \
  -c tools/pgsc_calc_kit.config ${EXTRA[@]+"${EXTRA[@]}"} \
  -work-dir "$WORK/work" \
  --input "$WORK/inputs/samplesheet.csv" \
  --scorefile "$WORK/inputs/scorefiles/*.txt.gz" \
  --target_build GRCh38 \
  --min_overlap 0.0 \
  --only_score \
  --max_cpus "$CPUS" \
  --max_memory "$MEM" \
  --outdir "$WORK/results"

# ── reference ────────────────────────────────────────────────────────────────
"$PYTHON" tools/build_reference.py finalize --results "$WORK/results" --out reference
rm -rf "$WORK/work" "$WORK/inputs/target"          # large scratch; results/ and logs are kept
echo "[reference] done in $(( $(date +%s) - t0 )) s -> reference/reference_scores.tsv"

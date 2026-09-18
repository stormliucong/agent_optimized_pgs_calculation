# Step 0: Setup

**Goal:** get this machine ready for the optimization loop. You will record
the hardware, build a Python environment, generate the synthetic benchmark,
compute the pgsc_calc reference scores and confirm that the starting script
and the gate run. Do not optimize anything here.

Work from the repository root. If a step has already been done (for example
`.venv/` exists, `benchmark/` verifies, or `reference/reference_scores.tsv`
exists), check it and move on.

## 1. Record the machine

Write `optim_workspace/ENVIRONMENT.md` (create `optim_workspace/`). Keep it to one table plus notes. Include:

- OS and kernel (`uname -a`; on macOS also `sw_vers`)
- CPU model and core count, including performance and efficiency cores if the platform
  distinguishes them (`lscpu`, `nproc`, or `sysctl -n machdep.cpu.brand_string hw.ncpu hw.perflevel0.physicalcpu`
  on macOS)
- RAM (`free -g`, or `sysctl -n hw.memsize` on macOS)
- Free disk space in the repo's filesystem (`df -h .`)
- Accelerator:
  - NVIDIA: `nvidia-smi` (GPU model, memory, driver, CUDA version)
  - Apple Silicon: chip name, unified memory. The GPU is reachable through Metal
    (PyTorch MPS or MLX).
  - none: say so
- Python version you will use

You need at least 8 GB of RAM and 15 GB of free disk. If the machine has less,
stop and tell the user.

## 2. Python environment

1. Use Python ≥ 3.10. Create `.venv/` with `python3 -m venv .venv` and use
   `.venv/bin/python` for everything from now on.
2. `.venv/bin/pip install -r requirements.txt`
3. If an NVIDIA GPU with a working driver is present, install the CuPy wheel
   that matches the driver's CUDA major version (`cupy-cuda12x` or
   `cupy-cuda13x`). Then confirm that it works:
   `.venv/bin/python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount(), (cp.arange(10)**2).sum())"`.
   If CuPy cannot be made to work, record why in `ENVIRONMENT.md` and continue
   on CPU.
4. Do not install anything else yet. The optimization loop may add packages
   later (numba, PyTorch, MLX, isal, …) when an iteration needs them. It must
   record every one in that iteration's `run.log`.

## 3. Build the synthetic benchmark

```bash
.venv/bin/python tools/make_fake_benchmark.py
.venv/bin/python tools/make_fake_benchmark.py --verify
```

This writes about 0.8 GB: 100 single-sample VCFs in `benchmark/genotypes/` and
48 PGS Catalog-style scoring files in `benchmark/models/`, plus
`benchmark/MANIFEST.tsv` (a checksum of every file). The verify step must report
`148/148`. From here on, `benchmark/` must not change: the reference in step 4
is computed from exactly these files.

Read the docstring of `tools/make_fake_benchmark.py` to learn what the data look
like. Do not read the generator for tricks to exploit. The loop must stay
general, and a later prompt forbids tuning to the benchmark's contents.

## 4. Compute the reference with pgsc_calc

The correctness gate compares every version with scores from **pgsc_calc
v2.3.0**, the community-standard pipeline. Compute them once:

```bash
tools/make_reference.sh            # optional: --profile conda|docker|mamba --cpus N --mem 16.GB
```

The script installs Nextflow into `.pgsc_calc/nxf` through conda/mamba if it is
missing, and clones pgsc_calc v2.3.0 into `.pgsc_calc/`. It merges the VCFs into
the one multi-sample VCF pgsc_calc expects and runs the pipeline with the
study's settings: `--min_overlap 0`, no mean imputation, no ancestry
adjustment, GRCh38. It then writes:

- `reference/reference_scores.tsv`: **the gate's reference**. This is pgsc_calc's
  matching decisions with the sum recomputed in float64, because plink2's own
  sums are rounded more coarsely than the gate tolerance (see
  `tools/build_reference.py`).
- `reference/pgs_cal_results.tsv`: pgsc_calc's own score matrix, kept for provenance.
- `reference/pgsc_calc_match_summary.csv`: pgsc_calc's per-model match summary.
- `reference/agreement.txt`: how the two matrices compare.

The run takes about 10 to 45 minutes and several GB of disk, most of it building
conda environments the first time. The script needs **conda or mamba**
(Miniforge is fine) or a **working docker**. If neither exists, install
Miniforge into your home directory (no sudo) and retry.

Known platform problems (fix them, do not skip the reference):

- **linux-aarch64 (Arm servers, GB10, Graviton):** bioconda has no `plink2`
  build. Build plink2 **v2.00a5.10** from source (github.com/chrchang/plink-ng,
  `2.0/build_dynamic`; `make` with `NO_AVX2=1 NO_SSE42=1 NO_LAPACK=1`), put the
  binary in `<prefix>/bin/plink2` of an empty conda prefix, and export
  `PGSC_PLINK2_ENV=<prefix>`. The pgscatalog-utils environment solved on
  aarch64 can lack the python `zstandard` module. If aggregation then fails to read
  `.sscore.zst` files, create a prefix with the same packages plus
  `zstandard` and export `PGSC_UTILS_ENV=<prefix>`.
- **macOS on Apple Silicon:** if a conda package has no `osx-arm64` build, either
  create the environments for `osx-64` (Rosetta, `CONDA_SUBDIR=osx-64`) or use
  `--profile docker` with Docker Desktop.
- A failed run can simply be re-run. Nextflow's log is `.pgsc_calc/nextflow.log`.

Check the result:
- `reference/reference_scores.tsv` has 100 samples × 48 models and no `NA`;
- `reference/agreement.txt` reports max |diff| below about 1e-3 against
  pgsc_calc's own sums. A larger gap means something went wrong. Stop and report it.

Record the reference run's wall clock and profile in `ENVIRONMENT.md`. If
pgsc_calc cannot be made to run at all, **stop and tell the user**. The loop
has no valid gate without it.

## 5. Check the starting script and the gate

1. `.venv/bin/python optim_src/gpu_pgs.py --selftest` must print `PASS`. It
   falls back to numpy/scipy when CuPy is absent.
2. `.venv/bin/python tools/check_correctness.py reference/reference_scores.tsv`
   must print `RESULT: PASS` (the reference compared with itself). This proves
   the gate tool works.

Do **not** run the starting script on the benchmark here. That is the job of
`v_1` in the next step.

## 6. Finish

Append a short "setup complete" section to `optim_workspace/ENVIRONMENT.md` that
lists the installed package versions (`.venv/bin/pip freeze`) and the
outcome of each check above. Then continue with `prompts/01_optimize.md`.

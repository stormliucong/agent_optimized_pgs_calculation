# Step 1: PGS Optimization Loop (agent instructions)

This is the benchmark optimization prompt from the study (`06_run_optim_src.md`).
Only what a portable, synthetic setup requires has changed: the paths, the
pgsc_calc reference file, a fixed gate tool, a command-line interface for the
script, and the use of whatever accelerator this machine has.

**Audience:** an autonomous coding agent (e.g. Claude Code).
**Goal:** iteratively rewrite `optim_src/gpu_pgs.py` into a faster, accelerated
script that computes polygenic scores (PGS) for every *(genotype × model)*
combination. Reduce the total wall-clock time while producing results that
match the reference.

The target scale is up to ~100 genotypes and ~100 models, so optimizations must
go **beyond** a per-individual, per-model loop and exploit the full batch.

`N_VERSIONS` is 10 unless the user asked for another number.

---

## 1. Ground rules (read carefully)

- `benchmark/`, `reference/`, `.pgsc_calc/`, `tools/` and `optim_src/` are **READ-ONLY**. Never modify,
  move, rename, or write inside them.
- All work goes in `optim_workspace/`.
- Each iteration lives in its own subfolder: `optim_workspace/v_1/`, `optim_workspace/v_2/`, … `optim_workspace/v_<N_VERSIONS>/`.
- During iteration *N* you may **write only** to `optim_workspace/v_<N>/`.
  You may **read** any earlier `v_<N>` folder (to inspect logs, code, and results).
- Exactly **four files** are allowed in each iteration folder, no others:
  `planning.txt`, `gpu_pgs_optim.py`, `pgs_output.tsv`, `run.log`.
- Temporary cache files are allowed *during* an iteration to speed up computation,
  but **must be deleted before the iteration ends**. Caches are never shared
  between iterations, so every iteration must run cold and independently.

---

## 2. Input / output contract (fixed for all iterations)

**Input** (always the same):
- `benchmark/genotypes/`: genotype data (one VCF per individual) for the `N_sample` individuals.
- `benchmark/models/`: the `N_model` PGS model definitions (PGS Catalog harmonized scoring files).

**Output** (always the same):
- `optim_workspace/v_<N>/pgs_output.tsv`: a `[N_sample, N_model]` matrix,
  tab-separated. Rows = individuals, columns = models. First column header `sample_id`,
  row labels = sample IDs, column headers = PGS IDs (e.g. `PGS900001`).
- Keep the row order, column order, and header convention **identical across every
  version** so outputs are directly comparable.

**Command line** (so the final script can be reused on other data):
`gpu_pgs_optim.py` must accept `--vcf-dir`, `--pgs-dir`, `--output` and `--workers`.
With no arguments it reads `benchmark/genotypes`, `benchmark/models` and writes
`pgs_output.tsv` next to itself.

---

## 3. Correctness requirement (non-negotiable)

A speedup only counts if the result is still correct.

- Treat `reference/reference_scores.tsv` as the **reference**. Setup computed it with
  pgsc_calc v2.3.0 on this benchmark, with no mean imputation and a minimum overlap
  of 0. `reference/pgsc_calc_match_summary.csv` holds pgsc_calc's per-model match
  summary.
- Every version's `pgs_output.tsv`, **including v_1**, must pass the gate:
  `.venv/bin/python tools/check_correctness.py optim_workspace/v_<N>/pgs_output.tsv`
  (`allclose`, `atol=1e-6`, `rtol=1e-5`, matched by sample and model ID).
- Record the full gate output in that version's `run.log`.
- If results diverge, the optimization is **invalid**. Fix it before reporting a
  speedup.

---

## 4. Per-iteration deliverables

Every `optim_workspace/v_<N>/` must contain these four files:

| File                | Purpose                                                                                     |
|---------------------|---------------------------------------------------------------------------------------------|
| `planning.txt`      | Reasoning for this iteration: the bottleneck targeted and the optimization strategy.        |
| `gpu_pgs_optim.py`  | Self-contained, runnable script. Reads `benchmark/`, writes `pgs_output.tsv`, measures time.|
| `pgs_output.tsv`    | The produced `[N_sample, N_model]` matrix.                                                   |
| `run.log`           | Full run output: measured wall-clock time (and GPU time if available) + correctness result. |

`gpu_pgs_optim.py` must be **fully standalone**. It must not import from any other
`v_<N>` folder.

---

## 5. Workflow

### Review existing iterations
1. Review the files and folders in `optim_workspace/`.
2. Pick up the workflow where it stopped. For example, if `v_4` is complete, start `v_5`. Treat a
   version folder that lacks any of the four files, or whose `run.log` has no gate result,
   as unfinished and redo it.

### v_1: establish the baseline
1. Start from `optim_src/gpu_pgs.py`.
2. Port it into `optim_workspace/v_1/gpu_pgs_optim.py` so it runs against
   `benchmark/` and measures and records its own timing.
3. Write a short `planning.txt` describing this baseline port.
4. Execute it. Capture full stdout/stderr **and the total wall-clock time for each step** in
   `optim_workspace/v_1/run.log`. Example steps include VCF parsing time, model loading and matching time, GPU kernel calculation time, etc.
5. **Verify** `pgs_output.tsv` against the reference with the gate. The starting script is
   **not validated**. If v_1 fails the gate, find out why, fix the port (change
   as little as correctness requires) and record in `planning.txt` exactly what was wrong.
   v_1 is the first version that passes.
6. The final optimization objective is the *total wall-clock time* from input to output.
7. Save `pgs_output.tsv`. This is the **baseline** for measuring speedup.

### v_2 … v_<N_VERSIONS>: optimize
For each new version *N*:
1. **Inspect** the prior `run.log` and `planning.txt` files to see the current timing
   and what has already been tried.
2. **Plan**: write `planning.txt` reasoning about the next bottleneck and a
   concrete optimization. Focus on kernel-level acceleration and batching
   across the whole problem, for example:
   - batch all genotypes × models into fused / vectorized kernels instead of
     looping per pair;
   - minimize host↔device transfers and keep data resident on the device;
   - use coalesced memory access, operator fusion, and mixed precision where safe;
   - exploit parallelism across the full ~100×100 space.
   Keep optimizations **general**, not tuned to this specific benchmark's contents.
3. **Implement** `gpu_pgs_optim.py` in `v_<N>/`.
4. **Run** it. Write the timing and output to `run.log`.
5. **Verify** `pgs_output.tsv` against the reference with the gate and record the result in
   `run.log`.
6. **Clean up** any temporary cache created during the iteration.
7. Tell the user in one line: version, total wall clock, speed-up vs v_1, gate result, main change.

### Stop
Stop after completing `v_<N_VERSIONS>`. Do **not** create more versions. Then continue with
`prompts/02_report.md`.

---

## 6. Notes for the agent

- Use a **consistent timing method** across versions (measure the same code
  region, from process start to output written) so speedups are comparable. You may warm up the
  device, but the recorded timed run must be cold with respect to any cross-iteration cache.
- If a version turns out slower or incorrect, keep it as a data point but state
  this clearly in `planning.txt` / `run.log`, and adjust the next iteration.
- Aim for a monotonic story: each `planning.txt` should build on lessons from
  previous logs.
- DO NOT try to explore the data. For example, do not design an algorithm specifically for the benchmark genotypes or PGS models.
- If a library is missing, install it into `.venv` and record it in `run.log`.
- Imputation is **not** required for this calculation. A missing genotype contributes nothing.
- Ancestry adjustment is **not** required for this task.
- Set MIN_OVERLAP = 0 to calculate all combinations regardless of the model and genotype overlaps (output NA if overlap is zero).
- Read the current computational environment (`optim_workspace/ENVIRONMENT.md`) and try to leverage
  THIS machine's maximum computation resources (CPU, GPU, RAM, etc.). The optimization should be customized for this machine:
  - NVIDIA GPU: CuPy / CUDA (cuSPARSE, custom `RawKernel`s).
  - Apple Silicon: the GPU through Metal (PyTorch MPS or MLX) where it pays off. Note that MPS has
    no float64, so accumulate in float64 on the CPU if float32 cannot pass the gate.
  - CPU only: vectorized numpy/scipy, multiprocessing, numba.
  Whatever the backend, the output must pass the same gate.
- In real practice, the cohort size will be close to the 1000 Genomes Project (~3,200 genomes) and the PGS models will include all models in the PGS Catalog (~5,000). Take that into consideration when you optimize.

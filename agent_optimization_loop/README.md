# Agent-optimized polygenic scoring: reproduction kit

This kit reproduces the **agent optimization loop** from *Agent-optimized GPU
polygenic scoring at cohort scale* on your own machine. An autonomous coding
agent (Claude Code) receives a slow, partly incorrect cohort-wide polygenic
score (PGS) scorer. Over repeated cycles of profile → plan → implement →
cold run → correctness gate, it rewrites that scorer into a fast one that
agrees with the community-standard pipeline, pgsc_calc.

The repository holds only prompts, the starting script and small tools. During
setup the agent generates everything else locally: the synthetic benchmark
data, and the reference scores from a real pgsc_calc v2.3.0 run on that data.
No real genomes are involved. The only downloads are software: Python packages,
Nextflow, pgsc_calc and its conda environments or containers.

## Quick start

```bash
git clone <this-repo-url> pgs-agent-loop
cd pgs-agent-loop
claude
```

Then type:

```
do the job
```

`CLAUDE.md` tells the agent to run the three prompts in `prompts/` in order.
In step 0 the agent sets up the environment, generates the benchmark and runs
pgsc_calc once to compute the reference. In step 1 it runs ten optimization
iterations, and in step 2 it writes a report. The
optimized script ends up in `optimized/pgs_scorer.py`. The version-by-version
trajectory is in `optim_workspace/SUMMARY.md`.

To run fewer iterations, say `do the job with 5 versions`. If the session stops
part-way, open `claude` again and say `do the job`. The agent resumes from the
last complete version.

### Unattended run

The loop runs for one to three hours. To let it run without approving each
command, use headless mode, **preferably inside a disposable VM or container**:

```bash
claude -p "do the job" --dangerously-skip-permissions --output-format stream-json --verbose > agent.log 2>&1 &
```

`-p` on its own prints nothing until the run ends. The `stream-json` output
lets you watch `agent.log` while the run progresses.

In interactive mode, `.claude/settings.json` pre-approves file edits and common
read-only and Python commands, so you are asked only occasionally. Claude Code
applies those rules only after you have accepted the folder's trust dialog once
in an interactive session. In headless mode, `--dangerously-skip-permissions`
makes this irrelevant.

## Requirements

| | minimum | notes |
|---|---|---|
| Claude Code | current version, logged in | the study used Claude Code with an Opus model |
| Python | 3.10+ | with `venv` (on Debian/Ubuntu: `apt install python3-venv`) |
| conda/mamba **or** docker | any recent | for pgsc_calc; [Miniforge](https://github.com/conda-forge/miniforge) installs without sudo |
| git, internet access | | to clone pgsc_calc and fetch its environments during setup |
| RAM | 8 GB | 16 GB or more recommended |
| Disk | 15 GB free | 0.8 GB benchmark, pgsc_calc environments and scratch, venv, workspace |
| Accelerator | optional | NVIDIA GPU (CUDA), Apple Silicon GPU, or CPU only |

### Where it runs

- **Linux VM with an NVIDIA GPU** (for example an AWS `g5`/`g6`, a GCP `g2`, a
  DigitalOcean GPU droplet or a Lambda instance). This is closest to the
  study. Use an image that already has the NVIDIA driver (a "Deep Learning" or
  "AI/ML ready" image). Setup installs the matching CuPy wheel. Check
  `nvidia-smi` before starting.
- **Linux VM without a GPU** (for example a DigitalOcean CPU droplet or an AWS
  `c7i`/`c7g`). This works. The agent optimizes for the CPU cores it finds.
  Pick 8 or more vCPUs and 16 GB of RAM.
- **MacBook (Apple Silicon)** This works. The agent can use the GPU through
  Metal (PyTorch MPS or MLX). MPS has no float64, so expect it to keep the
  final accumulation on the CPU. Install Python 3.10+ (for example with
  `brew install python`) if the system one is older.
- **Windows** Use WSL2 (Ubuntu) and follow the Linux instructions.

On a fresh Ubuntu VM:

```bash
sudo apt update && sudo apt install -y git python3 python3-venv
curl -fsSL https://claude.ai/install.sh | bash      # or: npm install -g @anthropic-ai/claude-code
curl -fsSLo miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh
bash miniforge.sh -b && ~/miniforge3/bin/conda init bash && exec bash   # conda, for pgsc_calc
claude                                              # log in once
git clone <this-repo-url> pgs-agent-loop && cd pgs-agent-loop && claude
```

## What is in the box

```
CLAUDE.md                     entry point: "do the job" → prompts 00, 01, 02
prompts/00_setup.md           record hardware, build .venv, generate data, run pgsc_calc, check tools
prompts/01_optimize.md        the optimization loop, adapted from the study's 06_run_optim_src.md
prompts/02_report.md          trajectory table, 3× confirmation runs, publish the final script
optim_src/gpu_pgs.py          starting scorer (unmodified from the study; unvalidated on purpose)
tools/make_fake_benchmark.py  deterministic generator of the synthetic benchmark
tools/make_reference.sh       runs pgsc_calc v2.3.0 on the benchmark and builds reference/
tools/build_reference.py      its Python halves: input preparation, exact float64 reference
tools/pgsc_calc_kit.config    Nextflow settings: no mean imputation, publish scores, use all cores
tools/check_correctness.py    the hard gate: allclose(atol=1e-6, rtol=1e-5) vs the reference
.claude/settings.json         pre-approved commands for interactive runs
```

Setup creates (all git-ignored):

```
.venv/                        Python environment
benchmark/                    ~0.8 GB synthetic data + MANIFEST.tsv (checksums)
.pgsc_calc/                   Nextflow, pgsc_calc v2.3.0, its conda envs and run results
reference/                    reference_scores.tsv (the gate), pgs_cal_results.tsv
                              (pgsc_calc's own sums), match summary, agreement.txt
```

The loop and report create:

```
optim_workspace/ENVIRONMENT.md  the machine the numbers were measured on
optim_workspace/v_1 … v_10/     planning.txt, gpu_pgs_optim.py, pgs_output.tsv, run.log
optim_workspace/SUMMARY.md      trajectory table, confirmation runs, discussion
optimized/pgs_scorer.py         the fastest version that passed the gate, with a README
```

### Doing the setup by hand

The agent does all of this in step 0. To do it yourself instead, run the
commands below and then say "do the job". The agent will find the setup
finished and go straight to the loop.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pip install cupy-cuda12x                 # only with an NVIDIA GPU (cupy-cuda13x for CUDA 13)
.venv/bin/python tools/make_fake_benchmark.py      # benchmark/, ~1 min
tools/make_reference.sh                            # reference/, 10–45 min, needs conda/mamba or docker
.venv/bin/python tools/check_correctness.py reference/reference_scores.tsv   # RESULT: PASS
```

### How the reference is made

`tools/make_reference.sh` merges the 100 VCFs into the one multi-sample VCF that
pgsc_calc expects. They share a site list, so the genotype columns are simply
placed side by side. It then runs pgsc_calc v2.3.0 with the study's settings:
`--min_overlap 0`, no mean imputation, no ancestry adjustment, `--target_build
GRCh38`, and `--only_score`. Everything else stays at the pipeline default, so
strand-ambiguous and multi-allelic variants are excluded and strand-flipped
matches are allowed.

plink2 sums in reduced precision, so pgsc_calc's own scores
(`reference/pgs_cal_results.tsv`) are off by up to ~1e-4 on the largest models.
That is more than the gate tolerance. The gate's reference
(`reference/reference_scores.tsv`) therefore keeps pgsc_calc's matching
decisions exactly and redoes only the sum in float64. `reference/agreement.txt`
records how far apart the two are.

When this kit was built, the two agreed to within 9.4e-5 (Pearson r = 1.00000000),
with pgsc_calc taking 349 s on a 20-core host. An independent
brute-force implementation of pgsc_calc's rules reproduced the reference to
5e-13. The unmodified starting script failed the gate in all 4,800 cells
(r = 0.73).

On **linux-aarch64**, bioconda has no plink2, so it must be built from source.
`prompts/00_setup.md` tells the agent how, and `tools/make_reference.sh` accepts
the resulting environment through `PGSC_PLINK2_ENV`.

## The benchmark

The benchmark is a scaled-down, synthetic version of the study's frozen
benchmark:

| | study | this kit |
|---|---|---|
| genotypes | 100 simulated All of Us-like VCFs | 100 simulated VCFs |
| sites per VCF | 7.65 M | 1.10 M |
| models | 100 real PGS Catalog files | 48 simulated PGS Catalog-format files |
| model rows | 34.8 M | 7.2 M (80 to 1.05 M per model) |
| reference | pgsc_calc v2.3.0 | pgsc_calc v2.3.0, run during setup (matching), float64 arithmetic |

The scoring files reproduce the quirks of real PGS Catalog files that a
correct scorer has to handle: effect alleles on either VCF allele,
strand-ambiguous SNPs, opposite-strand rows, missing `other_allele`, indels,
ungenotyped loci, dominant and recessive rows, and a second genome build in
`chr_position`. The starting script mishandles several of these, just as it did
in the study. **The first iteration therefore has to find and fix correctness
defects before any speed work counts.** Its unmodified output fails the gate
(r = 0.73 against the reference).

The generator uses its own integer-only random number generator, so the data
are byte-identical on Linux, macOS, x86 and Arm, and across numpy versions. It
was tested with numpy 2.1 and 2.4. Your numbers are therefore comparable with
anyone else's. The reference is computed from your own copy in any case, so the
gate never depends on that property.

## What to expect

- **The starting point.** On the study's machine (NVIDIA GB10, 20 Arm cores), the
  unmodified starting script takes 43.6 s on this benchmark and fails the gate.
  pgsc_calc takes 348.5 s. The corrected v_1 is slower (84.5 ± 7.6 s in the
  trial run below), because the correctness fixes come first and speed comes
  later. Expect a laptop to be slower.
- **The shape of the result.** In the study the agent cut the end-to-end wall
  clock 10.6-fold (184.9 s → 17.4 s). The largest gains came from VCF parsing
  and variant matching, not from the matrix product. Expect a similar story:
  ingestion dominates at first, and the GPU arithmetic is a small fraction
  throughout.
- **Your trajectory will differ from ours, and from your next run.** The study
  ran the same prompt three times. Agent-to-agent variation explained 91–96% of
  the wall-clock variance, far more than the data or re-execution did. One
  replicate's speed-up was 4.2-fold and another's was 17.2-fold. Treat one run
  as one sample. To measure the spread, run the kit in three fresh clones, one
  after another and never at the same time.
- **Tested end to end.** A clean copy of an earlier version of this kit, which shipped the
  reference instead of computing it, was run headless with
  `claude -p "do the job with 2 versions"` on 2026-09-18 (Opus, NVIDIA GB10). It
  finished unattended in 25 minutes, using 48 turns and about US$6. Setup built
  and verified the data. Computing the reference now adds the pgsc_calc run on top. v_1 found and fixed the starting script's matching
  defects and passed the gate (max |diff| 3.5e-13). v_2 rewrote ingestion as
  byte-level numpy in a process pool with one fused float64 GPU kernel. The
  confirmation runs gave v_1 84.5 ± 7.6 s and v_2 2.35 ± 0.13 s (36×), and every
  output passed the gate. The v_2 script also used a shortcut the study warns
  about: it checks whether a VCF's site list is identical to the first file's.
  That holds for this benchmark but not for real per-sample VCFs.
- **Cost.** A full ten-iteration run takes roughly one to three hours of agent
  time, several times the two-version trial's cost. On a fast GPU machine most
  of the gain can arrive in the first couple of versions, and later versions
  fight over the last second.

## What this kit does not cover

This kit covers the benchmark optimization loop only. The study's other
experiments are out of scope: the second loop on resampled real 1000 Genomes
data, held-out validation, the memory-bounded rebuild, and the full
3,202 × 5,385 reference release. Those need about 2 TB of real sequence data,
the full PGS Catalog, and a working pgsc_calc installation to produce a fresh
reference for every draw.

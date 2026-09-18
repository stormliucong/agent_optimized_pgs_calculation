# Agent-optimized PGS scoring: reproduction kit

This repository reproduces the agent optimization loop from the study
"Agent-optimized GPU polygenic scoring at cohort scale". A coding agent (you)
takes a working but slow and partly incorrect cohort-wide polygenic score (PGS)
scorer (`optim_src/gpu_pgs.py`). It then rewrites that scorer, version by
version, into a fast one that matches a pgsc_calc reference. The benchmark data
are synthetic and are generated locally.

## "do the job"

When the user says **"do the job"** (or "run the study", "start", "go"), carry out
these prompt files **in order**, completely and without asking for confirmation
between steps:

1. `prompts/00_setup.md`: environment, data generation, pgsc_calc reference, sanity checks
2. `prompts/01_optimize.md`: the optimization loop (`optim_workspace/v_1` … `v_N`)
3. `prompts/02_report.md`: summary table, confirmation runs, final script

Read each file in full before starting it. If work already exists (for example
`optim_workspace/v_4/` is complete), resume from where it stopped instead of
starting over. The default number of versions is **10**. If the user asks for a
different number ("do the job with 5 versions"), use that number instead.

Stop and ask the user only if a step cannot be completed. Examples: Python ≥ 3.10
is not available, pgsc_calc cannot be made to run, or the disk is full.

## Ground rules that apply throughout

- `benchmark/`, `reference/` and `.pgsc_calc/` are created by setup. After that they,
  `tools/` and `optim_src/` are **read-only**. Never edit `tools/check_correctness.py` or the reference
  files. The correctness gate is not negotiable.
- Work from the starting script and your own measurements. Do not search for, download
  or copy existing PGS scorer implementations or precomputed scores.
- Run only one timed or heavy job at a time. A concurrent job corrupts the wall
  clock you are optimizing.
- Never touch machine-wide state (no `sudo`, `sysctl` or swap changes, and no
  killing processes you did not start).
- Keep the user informed with a one-line status after each version: its wall
  clock, gate result and what changed.

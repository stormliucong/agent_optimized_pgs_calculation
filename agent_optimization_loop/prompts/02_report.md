# Step 2: Report and final script

**Goal:** summarize the trajectory, confirm the headline numbers with repeated
cold runs, and publish the best version as a reusable script. Do not change any
`v_<N>/` folder in this step.

## 1. Trajectory table

Read every `optim_workspace/v_<N>/run.log` and `planning.txt`. Write
`optim_workspace/SUMMARY.md` with one row per version:

| column | content |
|---|---|
| Version | `v_1` … |
| VCF parse | seconds, as that version's `run.log` recorded them |
| Model load + match | seconds |
| Score product | seconds (the sparse or dense product, including device transfer if the log separates it) |
| Total | end-to-end wall clock, seconds |
| vs v_1 | speed-up factor |
| Gate | PASS / FAIL and max \|diff\| |
| Dominant optimization | one line, from `planning.txt` |

Stage names differ between versions, so map them to the three columns above as
faithfully as you can and footnote anything that does not fit. This table has
the same layout as Table 1 of the paper.

## 2. Confirmation runs

Single runs are noisy. Re-run `v_1` and the **fastest version that passed the gate** three times
each, cold, one after another (never concurrently). Write each run's output to a temporary
directory outside `optim_workspace/v_<N>/`, confirm that each output passes the gate, and delete it.
Report mean ± sample standard deviation of the total wall clock for both, and
the resulting speed-up, in `SUMMARY.md`.

## 3. Publish the final script

Copy the fastest passing version to `optimized/pgs_scorer.py` unchanged (except for a
header comment naming the source version, its measured time and the gate result).
Add `optimized/README.md` with:
- what the script computes (the pgsc_calc-compatible rules it implements, in a few bullets);
- how to run it on other data (`--vcf-dir`, `--pgs-dir`, `--output`, `--workers`),
  and which packages and hardware it needs;
- its limits, stated honestly. Examples: single-sample VCFs only, GRCh38 harmonized scoring files
  only, memory growth with cohort × model size, what it assumes about absent
  variants.

## 4. Discussion

End `SUMMARY.md` with a short section (at most about 300 words) covering:
- where the time went at v_1 and at the final version (ingestion vs matching vs arithmetic);
- which optimizations paid the most, and which were dead ends;
- how this compares with the study, which went from 184.9 s to 17.4 s (10.6×) on a 7.65 M-site
  benchmark with an NVIDIA GB10. This kit's benchmark is smaller and your hardware differs, so compare
  the *shape* of the trajectory and the bottlenecks rather than absolute seconds;
- anything in the final script that is likely to break at the real target scale
  (~3,200 genomes × ~5,000 models).

Finally, tell the user where `SUMMARY.md` and `optimized/pgs_scorer.py` are, and
give the headline numbers in two or three lines.

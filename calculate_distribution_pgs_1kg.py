#!/usr/bin/env python3
"""
calculate_distribution_pgs_1kg.py — PGS REFERENCE DISTRIBUTIONS FROM 1000 GENOMES.

`pressure_test/gpu_pgs_optim_memory_safe.py` answers "what does this cohort
score on these models".  That is not enough to deploy a score.  A raw PGS is
only interpretable against a reference distribution, and a reference
distribution is only transferable if the thing that produced it is FROZEN: the
same locus set, the same allele orientation, the same allele frequencies and
the same standardisation have to be applied to the panel and to every
individual scored afterwards.  This script builds that artefact.

It is the memory-safe scorer with its arithmetic intact — the matching rules,
the ordinal tie-break, the isal/shared-memory inflate ring, the batched device
parse, the two streamed VCF passes, the memory-mapped dosage matrix and the
budget-sized block loops are all v_10's, unedited — plus the six things
`11_run_1kg_distribution.md` asks for, in the order it asks for them:

  §1  S IS DEFINED BY CALL RATE, NOT BY INTERSECTION.  A locus enters model j's
      scoring set if it matched the model and the reference panel called it in
      >= 95 % of samples.  `coverage = |S| / N` is reported for every model and
      a low one is FLAGGED (`deployable=0`), not quietly scored: a model that
      only matches a fifth of its loci is not producing the published score.
      The candidate pool is the target UNION, which the dosage matrix already
      gives for free as its per-column observed count.

  §2  RESIDUAL MISSINGNESS INSIDE S IS MEAN-IMPUTED, dosage -> 2*p_j, which is
      plink2's default and keeps E[score] unbiased.  One extra sparse matmul on
      the existing kernel: `scores += Wfill @ (1 - obs)`, where `Wfill` carries
      w_j * 2*p_j^eff on the same sparsity pattern as `Wd`.

  §3  p_j IS FROZEN WITH THE MANIFEST.  Per locus the bundle stores the target
      key, the effect allele, the weight, the flip flag, p_j in effect-allele
      orientation, and (through the target index) the panel call rate.  A new
      individual must be imputed with THIS p_j, not a recomputed one, or their
      score is not on the reference scale.

  §4  PER-SAMPLE COVERAGE TRAVELS WITH EVERY SCORE.  `n_obs = A @ obs`, one
      more matmul against the 0/1 pattern of `Wd` — which is what DENOM should
      have been — and samples below 99 % of |S| are flagged and excluded from
      the reference mu/sigma instead of carrying a shrunken score into it.

  §5  STANDARDISATION COMES LAST, AND IT IS STRATIFIED.  On a five-continent
      panel a single mu/sigma is dominated by ancestry-driven allele-frequency
      differences, so the raw score is regressed on the panel's top genotype
      PCs and the RESIDUAL is standardised.  The PCs are computed here from the
      dosage matrix, and what is frozen is the per-variant loadings, so a future
      individual can be projected onto the same axes.

  §6  ABSENT vs NO-CALL IS RESOLVED BEFORE ANY OF THIS.  The 1000 Genomes input
      is a per-sample slice of the NYGC joint callset: every sample carries
      every site, 0/0 is explicit, and the cohort manifest reports zero missing
      genotypes.  So "absent from this VCF" means "not in the callset", i.e.
      genuinely not assayed, and no callable-region definition is needed.  That
      is checked and recorded in `build_info.json` rather than assumed, and it
      is why the imputation path measures itself out of this particular run —
      see the README's caveats before reusing the bundle against variant-only
      per-sample VCFs, where absent is ambiguous.

Output bundle: `1kg_pgs_reference/` (see its generated README.md).

Usage:
  python calculate_distribution_pgs_1kg.py \
      --vcf-dir 1kg_data --pgs-dir PGSCatelog --out-dir 1kg_pgs_reference \
      --dosage-cache /path/to/scratch --resume

Knobs: --min-call-rate --min-sample-call-rate --min-coverage --reference-set
       --n-pcs --pca-variants --pca-maf --pca-spacing --limit-models
       --limit-samples --no-manifest --reference <pgs_cal tsv to cross-check>
plus the inherited memory ones: PGS_MEM_BUDGET_GB, PGS_SPILL_DIR,
PGS_MEMBER_CACHE.

Everything below this line is the memory-safe scorer's own docstring.

MEMORY-SAFE — v_10's arithmetic, planned to fit inside a stated memory budget.

v_10 is fast and correct and dies at 1000 genotypes x 1000 models: the kernel
OOM killer takes it ~20 minutes into the VCF stage.  Nothing about the
calculation is wrong; the problem is that five structures in it are sized by
(cohort x catalog) and are all materialised at once:

  1. `f_sk`/`f_sd`/`f_so` — every sample's keys, dosages and ordinals are held
     until the LAST VCF has been parsed, at 17 B per (sample, kept record).  At
     1000 x 1000 that is ~28 M records/sample x 1000 x 17 B ~ 470 GB on a 119 GB
     box.  This is what actually killed it.
  2. `read_models` inflates EVERY scoring file before decoding any of them, then
     keeps 20 B/row for all of them at once.  The whole catalog is 63 GB gzipped
     — ~350 GB of text — so at the top of the ladder this dies before the VCFs
     are even opened.
  3. `member_ranges` reads a whole 628 MB VCF into RAM and builds three
     full-length boolean temporaries over it: ~2.5 GB per worker, x20 workers.
  4. `score_gpu`'s tile is a fixed 512 samples wide, and its `dose`/`obs` are
     float64: 16 B x n_var x 512 = 224 GB at n_var = 27 M.  100 x 100 survived
     only because 100 < 512.
  5. `D` itself, n_var x n_samples int8 — 28 GB at 1000 x 1000, ~190 GB for the
     whole cohort x whole catalog.

The fix is the one v_10's post-mortem called for: fold each sample into the
dosage matrix as it is parsed instead of accumulating, and give every remaining
(cohort x catalog)-sized structure a block loop whose width comes from a memory
budget rather than from a constant.  Concretely:

  * PHASE 1 runs the scoring files in BLOCKS sized from the budget.  The first
    pass over them decodes chromosome and position only (`_LOCI_ONLY`) and keeps
    nothing but the locus union, which is what phase 2 needs; the full decode
    happens later, one block at a time, next to the matching that consumes it.
    When the whole catalog fits in one block the full decode is done once and
    kept, so small shapes pay nothing for this.
  * PHASE 2 is now TWO streamed passes over the VCFs instead of one accumulating
    pass.  Pass A builds the target union and the per-target minimum ordinal
    incrementally — a probe plus a scatter-min per batch, and a real union only
    for the keys that are new — so it retains O(n_var), not O(n_var x samples).
    Pass B re-parses and folds each sample straight into its column of `D`.
    Nothing per-sample survives the sample.
  * `D` is a memory-mapped file laid out (n_samples x n_var), so a sample's
    column is a contiguous write and a sample TILE is a contiguous read.  Page
    cache is reclaimable, so a matrix bigger than RAM costs disk, not an OOM.
  * PHASE 3/4/5 (match, weight matrices, scoring) run per model block, and the
    scoring tile width is computed from the budget.  Blocking over models and
    tiling over samples both leave every output cell's summation order
    untouched, so the scores are bit-identical to v_10's.
  * `member_ranges` scans a 32 MB window at a time through an mmap, and caches
    what it found (keyed by path/size/mtime) so pass B and later runs skip it.

Knobs (all optional):
  PGS_MEM_BUDGET_GB   absolute budget in GB; overrides the fraction below
  PGS_MEM_FRACTION    fraction of MemTotal to plan inside (default 0.55)
  PGS_SPILL_DIR       where the dosage matrix is written (default: next to the
                      output TSV)
  PGS_MEMBER_CACHE    directory for the gzip-member index cache

NOTHING about the calculation changed: the matching rules, the ordinal
tie-break, the kernels, the isal/shared-memory inflate ring, the batched device
parse and the two weight matrices are v_10's, unedited.  Everything below this
line is v_10's own docstring.

v_6 — THE SET ALGEBRA MOVES TO THE GPU, AND THE DOSAGE MATRIX STAYS THERE.

v_5 fixed the last matching rule and took the scoring files off the interpreter,
and the profile that came out of it was not what earlier versions had:

    model locus set   np.unique over 11.66 M keys          6.09 s
    vcf worker pool   inflate + line loop + IPC           10.97 s
    target union      np.unique over 46.20 M keys          7.62 s
    ordinal reduction np.lexsort per sample                1.85 s
    rows              np.searchsorted per sample           0.92 s
    member discovery  gzip signature scan                  1.35 s

Half the stage is not parsing at all.  It is four single-threaded numpy sorts on
one core — and sorting is the operation a GPU is best at.  Measured on this host
(46 M int64): np.unique 32.9 s, cp.unique 0.25 s, cp.searchsorted 0.03 s,
cupyx.scatter_min 0.07 s, and 368 MB crosses to the device in 0.43 s because the
memory is unified.

So v_6 changes no algorithm and no rule.  It moves every set operation onto the
GPU and then leaves the results there:

  * the model-locus union, the target union, the target lookup and the ordinal
    reduction are all device calls;
  * `np.lexsort` disappears entirely — the min-ordinal-per-target reduction it
    was emulating IS an atomic scatter-min, one kernel, no sort;
  * the dosage matrix is built on the device and never comes back: matching,
    the weight matrices and the scoring kernels all read it in place, so the
    only host->device traffic left is the parsed variant keys themselves.

Everything below this line is v_5's, unchanged in behaviour.

v_5 — TAKE THE MODEL SIDE OFF THE INTERPRETER, AND FIX THE MERGED CANDIDATE POOL.

Two things changed relative to v_3 (the fastest correct version) and v_4 (the
vectorised-VCF experiment that was slower and failed its gate).

1. CORRECTNESS — the rule v_4 diagnosed but did not fix.
   In the effect-allele-only path, pgsc_calc pools *all* targets at a locus,
   SNP and indel together, and takes the first in target order within a match
   type.  v_4 searched the SNP table first and only consulted indel targets if
   no SNP matched, so at 1:55603694 (effect A, no other allele; callset holds
   A>G and A>ATG) it matched A>G where the reference matched A>ATG.  Here the
   two pools are merged and ranked by the record ordinal, which fixes it.

2. PERFORMANCE — the scoring files, not the VCFs, are this draw's bottleneck.
   v_3/v_4 read scoring files with a python `for line in f` loop doing a dozen
   dict lookups per row.  That is ~1.4 us/row: 12.1 s for a single 8.9 M-row
   model.  At the stated production scale (the whole PGS Catalog, ~5385 models,
   ~10^9 rows) it is the dominant cost, and it is embarrassingly parallel.

   The model side is now array code end to end:
     * inflate with isal in a THREAD pool — isal releases the GIL, measured
       3.3x on 4 threads — so no 450 MB buffer is ever pickled;
     * split each inflated file into line-aligned chunks and decode them in a
       fork pool (buffers inherited copy-on-write, never serialised);
     * inside a chunk, one pass for newlines, one for tabs, strided views for
       the field boundaries, masked multiply-accumulate for the position, and a
       byte lookup for the alleles.  Only the float weight column goes through
       pandas' C tokeniser, which is the one field that is not cheap to decode
       with array ops.
     * matching is then one batch of `searchsorted` calls over ALL models'
       rows at once, on the GPU: 4 probes for the both-alleles path, 16 for the
       effect-allele-only path, against the sorted target-key array resident on
       device.  The python `resolve()` loop survives only for rows whose alleles
       are not single bases and for rows at a locus that holds an indel target,
       where the merged-pool rule above needs the string form.

   The VCF pass reverts to v_3's line loop (v_4's vectorised parse was measured
   slower), with two changes: the per-chromosome membership set is built inside
   the worker from the sorted model-locus array, so no 9 M-entry python set is
   built serially in the parent; and every target row carries the ordinal of its
   record inside its gzip member, needed by the merged-pool rule.  The ordinal
   is the survivor counter rather than the line number, which costs nothing per
   rejected line and is order-equivalent: a locus lives in exactly one member
   and survivors are appended in file order.
"""

import os
import sys
import time

_CUDA_LIB = '/usr/local/lib/ollama/cuda_v12'
if os.path.isdir(_CUDA_LIB) and _CUDA_LIB not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = _CUDA_LIB + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    os.execv(sys.executable, [sys.executable] + sys.argv)

_T0 = time.time()                      # <- the wall clock starts here
import sys as _s; _s.setswitchinterval(float(os.environ.get('PGS_SWITCH', '0.001')))

import gzip
import glob
import hashlib
import io
import json
import mmap
import multiprocessing as mp
import threading
from bisect import bisect_right
from multiprocessing import resource_tracker, shared_memory
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import scipy.sparse as sp
from isal import isal_zlib

try:
    import cupy as cp
    import cupyx
    import cupyx.scipy.sparse as cusp
    GPU = True
except Exception:
    cp = np
    cupyx = None
    cusp = None
    GPU = False

xp = cp if GPU else np


def sync():
    """Make a device timing honest; a no-op without a GPU."""
    if GPU:
        cp.cuda.Stream.null.synchronize()


def to_host(a):
    return cp.asnumpy(a) if GPU and isinstance(a, cp.ndarray) else np.asarray(a)


def scatter_min(dest, index, src):
    """dest[i] = min(dest[i], src[j] for every j with index[j] == i)."""
    if GPU:
        cupyx.scatter_min(dest, index, src)
    else:
        np.minimum.at(dest, index, src)

HERE = os.path.dirname(os.path.abspath(__file__))
VCF_DIR = os.path.join(HERE, '1kg_data')
PGS_DIR = os.path.join(HERE, 'PGSCatelog')
OUT_DIR = os.path.join(HERE, '1kg_pgs_reference')
MIN_OVERLAP = 0.0
WORKERS = max(1, os.cpu_count() or 1)

_ARGS = dict(zip(sys.argv[1:-1], sys.argv[2:]))
_FLAGS = set(sys.argv[1:])


def _arg(name, default=None, cast=str):
    v = _ARGS.get(name)
    return default if v is None else cast(v)


VCF_DIR = os.path.abspath(_arg('--vcf-dir', VCF_DIR))
PGS_DIR = os.path.abspath(_arg('--pgs-dir', PGS_DIR))
OUT_DIR = os.path.abspath(_arg('--out-dir', OUT_DIR))
REF_TSV = os.path.abspath(_arg('--reference', ''))  if _arg('--reference') else ''
PANEL_META = _arg('--panel-meta', os.path.join(VCF_DIR, 'manifest.csv'))
WORKERS = max(1, _arg('--workers', WORKERS, int))
TILE = 512
CHUNK_BYTES = 24 << 20                 # model-file decode granularity

# `OUT_TSV` keeps the inherited name because SPILL_DIR and write_output()
# below are v_10's, unedited; it is now one file inside the reference bundle.
OUT_TSV = os.path.join(OUT_DIR, 'scores_raw.tsv')

# ── the frozen-manifest knobs (section 2 of 11_run_1kg_distribution.md) ─────
# A locus enters the scoring set S if it matched the model AND the reference
# panel called it in at least this fraction of samples.  S is NOT the
# intersection over samples: a locus one sample happens to miss stays in S and
# is mean-imputed for that sample, which is what keeps every individual on the
# same scale.
MIN_CALL_RATE = _arg('--min-call-rate', 0.95, float)
# A sample is flagged for a model when it observed less than this fraction of S.
MIN_SAMPLE_CALL_RATE = _arg('--min-sample-call-rate', 0.99, float)
# |S| / N below this and the model is not deployable on this platform.
MIN_COVERAGE = _arg('--min-coverage', 0.75, float)
# Which panel samples define p_j, the call rates, the PCs and mu/sigma.
REFERENCE_SET = _arg('--reference-set', 'unrelated')
N_PCS = _arg('--n-pcs', 10, int)
PCA_VARIANTS = _arg('--pca-variants', 150000, int)
PCA_MAF = _arg('--pca-maf', 0.05, float)
PCA_SPACING = _arg('--pca-spacing', 20000, int)          # bp between PCA loci
LIMIT_MODELS = _arg('--limit-models', 0, int)
LIMIT_SAMPLES = _arg('--limit-samples', 0, int)
WRITE_MANIFEST = '--no-manifest' not in _FLAGS
# Persist the two VCF passes.  They are 2/3 of the wall clock at full scale, so
# a downstream bug must not cost them twice.
DOSAGE_CACHE = _arg('--dosage-cache', '')
RESUME = '--resume' in _FLAGS


# ─────────────────────────────────────────────────────────────────────────────
# THE MEMORY BUDGET
#
# Every block width in this file is derived from one number rather than from a
# constant, because the shapes this has to survive span four orders of magnitude
# (100 x 100 to 3202 x 5385) and a constant that fits one end wastes or kills the
# other.  The shares below are deliberately conservative: the stages they cover
# do not all peak at the same time, and the slack is what absorbs the
# temporaries inside cupy calls (a device sort allocates a second copy) that no
# planner can see.
# ─────────────────────────────────────────────────────────────────────────────
def _mem_total():
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    return int(line.split()[1]) << 10
    except OSError:
        pass
    return 64 << 30


def _meminfo():
    """(MemTotal, MemAvailable) in bytes."""
    tot = avail = 0
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    tot = int(line.split()[1]) << 10
                elif line.startswith('MemAvailable:'):
                    avail = int(line.split()[1]) << 10
                    break
    except OSError:
        pass
    return tot, avail


def _budget():
    gb = float(os.environ.get('PGS_MEM_BUDGET_GB', '0') or 0)
    if gb > 0:
        return gb * (1 << 30), f"PGS_MEM_BUDGET_GB={gb:g}"
    frac = float(os.environ.get('PGS_MEM_FRACTION', '0.55'))
    tot = _mem_total()
    return tot * frac, f"{frac:g} x MemTotal ({tot/1e9:.0f} GB)"


MEM_BUDGET, MEM_BUDGET_WHY = _budget()
B_SLOTS = 0.15 * MEM_BUDGET            # the inflate ring's shared-memory slots
B_MODEL = 0.35 * MEM_BUDGET            # one model block: text, arrays, matches
B_SCORE = 0.20 * MEM_BUDGET            # one score tile's dose/obs (float64)
B_FOLD = 0.08 * MEM_BUDGET             # the locus-union fold's working set

SPILL_DIR = os.environ.get('PGS_SPILL_DIR', '') or os.path.dirname(OUT_TSV) or '.'
MEMBER_CACHE = os.environ.get('PGS_MEMBER_CACHE', '')
_GZ_RATIO = 6.0                        # text/gzip estimate, refined as we go
# Peak working set per byte of gzipped scoring file, before the exact row
# counts are known: ~6 B of text, ~2 B of decoded columns and ~7.6 B of matched
# entries and CSR (a scoring row is ~60 B of text, so ~10 rows per KB of gzip).
# The three do not peak together — the text is dropped before the matching —
# so 12 is a ceiling, not a sum.
_GZ_PEAK = 12.0
PROGRESS_EVERY = float(os.environ.get('PGS_PROGRESS_SECONDS', '60'))

_BASE = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
_BASE_B = {b'A': 0, b'C': 1, b'G': 2, b'T': 3}
_COMP = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
_AMBIG = {('A', 'T'), ('T', 'A'), ('C', 'G'), ('G', 'C')}
_CHROM = {**{str(i): i for i in range(1, 23)}, 'X': 23, 'Y': 24}
_CHROM_B = {k.encode(): v for k, v in _CHROM.items()}
_CHROM_B.update({b'chr' + k.encode(): v for k, v in _CHROM.items()})
_CHROM_B.update({b'CHR' + k.encode(): v for k, v in _CHROM.items()})
_PAR = {23: ((10001, 2781479), (155701383, 156030895)),
        24: ((10001, 2781479), (56887903, 57217415))}
_MAX_ALLELE_LEN = 100
_CODE_NAME = {**{i: str(i) for i in range(1, 23)}, 23: 'X', 24: 'Y',
              26: 'PAR1', 27: 'PAR2'}
_NAME_CODE = {v: k for k, v in _CODE_NAME.items()}
_INF_ORD = np.int64(1) << 62

# byte -> base code, uppercase and lowercase, everything else -1
_BASE_LUT = np.full(256, -1, np.int8)
for _c, _v in _BASE.items():
    _BASE_LUT[ord(_c)] = _v
    _BASE_LUT[ord(_c.lower())] = _v


def chrom_code(chrom):
    if chrom.startswith('chr') or chrom.startswith('CHR'):
        chrom = chrom[3:]
    return _CHROM.get(chrom.upper())


def target_code(code, pos):
    """Contig code of a TARGET variant, after the reference's PAR split."""
    par = _PAR.get(code)
    if par is not None:
        if par[0][0] <= pos <= par[0][1]:
            return 26
        if par[1][0] <= pos <= par[1][1]:
            return 27
    return code


def snp_key(code, pos, ref, alt):
    return (code << 38) | (pos << 6) | (_BASE[ref] << 3) | _BASE[alt]


def str_key(code, pos, ref, alt):
    return f"{_CODE_NAME[code]}:{pos}:{ref}:{alt}"


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — READ THE SCORING FILES  [DISK -> RAM, threads] + [CPU, fork pool]
#
# The whole stage is array code.  A scoring file is a well-formed TSV with a
# fixed column count, so after one `== '\n'` pass and one `== '\t'` pass every
# field's byte range is known from two strided views, and the fields decode with
# elementwise ops.  The float weight is the exception — correct decimal-to-
# double rounding is not worth reimplementing — so that one column goes through
# pandas' C tokeniser.
# ─────────────────────────────────────────────────────────────────────────────
_CHR_COLS = ('chr_name', 'chromosome', 'chr')
_POS_COLS = ('chr_position', 'position', 'pos')
_EFF_COLS = ('effect_allele', 'ea')
_OTH_COLS = ('other_allele', 'oa', 'noneffect_allele', 'hm_inferotherallele')
_W_COLS = ('effect_weight', 'weight', 'beta')


def _first_col(names, cands):
    for c in cands:
        if c in names:
            return names.index(c)
    return None


def _model_spec(text):
    """Header layout of one inflated scoring file (byte offsets, column ids)."""
    off = 0
    n = len(text)
    while off < n and text[off:off + 1] == b'#':
        nxt = text.find(b'\n', off)
        if nxt < 0:
            return None
        off = nxt + 1
    end = text.find(b'\n', off)
    if end < 0:
        return None
    names = [q.strip().lower() for q in text[off:end].decode().split('\t')]
    harmonised = 'hm_chr' in names and 'hm_pos' in names
    spec = dict(
        names=names, ncols=len(names), body=end + 1,
        chrom=names.index('hm_chr') if harmonised else _first_col(names, _CHR_COLS),
        pos=names.index('hm_pos') if harmonised else _first_col(names, _POS_COLS),
        eff=_first_col(names, _EFF_COLS),
        oth=[names.index(c) for c in _OTH_COLS if c in names],
        w=_first_col(names, _W_COLS),
        dom=names.index('is_dominant') if 'is_dominant' in names else None,
        rec=names.index('is_recessive') if 'is_recessive' in names else None,
    )
    if spec['chrom'] is None or spec['pos'] is None or spec['eff'] is None \
            or spec['w'] is None:
        return None
    return spec


def _inflate_file(path):
    with open(path, 'rb') as f:
        raw = f.read()
    d = isal_zlib.decompressobj(31)
    parts, data = [], raw
    while data:
        parts.append(d.decompress(data))
        data = d.unused_data
        if data:
            d = isal_zlib.decompressobj(31)
    # one member is one decompress call, and b''.join would copy all 300 MB
    return parts[0] if len(parts) == 1 else b''.join(parts)


_MODEL_TEXT = None                     # set in the parent BEFORE the pool is
_MODEL_SPEC = None                     # forked, so the buffers are inherited
_LOCI_ONLY = False                     # copy-on-write and never pickled


def _as_loci(res):
    """A full chunk result reduced to what the locus pass keeps."""
    loc = (res['code'].astype(np.int64) << 32) | res['pos']
    return dict(mi=res['mi'], lo=res['lo'], n=int(res['pos'].size),
                loci=np.unique(loc))


def _field_bounds(sub, tb, ls, le, ncols, k):
    """[start, end) of column k for every line, as views into `sub`."""
    per = ncols - 1
    s = ls if k == 0 else tb[k - 1::per] + 1
    e = le if k == ncols - 1 else tb[k::per]
    return s, e


def _decode_bases(sub, s, e):
    """Single-byte allele -> base code; -1 empty, -2 anything longer/odd."""
    ln = e - s
    code = np.full(ln.size, -2, np.int64)
    one = ln == 1
    if one.any():
        b = sub[np.where(one, s, 0)]
        code = np.where(one, _BASE_LUT[b].astype(np.int64), code)
    code = np.where(one & (code < 0), -2, code)
    code = np.where(ln == 0, -1, code)
    return code.astype(np.int8), ln


def _decode_chunk(task):
    """One line-aligned byte range of one scoring file -> column arrays.

    In `_LOCI_ONLY` mode the caller wants nothing but the (contig, position)
    set, so the fast path stops after the position decode: no alleles, no
    dominant/recessive flags, no pandas pass over the weight column, and no
    per-row arrays kept.  That is a strict SUPERSET of the rows the full decode
    keeps — it drops only the chromosome/position validity test, and every
    later test can only remove rows — which is what the locus pass needs: a
    locus the full decode will not use costs one target row that no model row
    can ever match, while a MISSING locus would silently drop a target."""
    res = _decode_chunk_slow_ok(task)
    return _as_loci(res) if (_LOCI_ONLY and 'loci' not in res) else res


def _decode_chunk_slow_ok(task):
    mi, lo, hi = task
    text = _MODEL_TEXT[mi]
    spec = _MODEL_SPEC[mi]
    ncols = spec['ncols']
    raw = text[lo:hi]
    sub = np.frombuffer(raw, np.uint8)
    if sub.size == 0:
        return _empty_model_chunk(mi)

    nl = np.flatnonzero(sub == 10)
    ls = np.empty(nl.size + 1, np.int64)
    ls[0] = 0
    ls[1:] = nl + 1
    le = np.empty(nl.size + 1, np.int64)
    le[:nl.size] = nl
    le[nl.size] = sub.size
    if le[-1] == ls[-1]:               # no dangling partial line
        ls, le = ls[:-1], le[:-1]
    keep = le > ls
    if not keep.all():                 # blank lines: pandas drops them too
        ls, le = ls[keep], le[keep]
    m = ls.size
    if m == 0:
        return _empty_model_chunk(mi)
    le = np.where(sub[le - 1] == 13, le - 1, le)      # tolerate CRLF

    tb = np.flatnonzero(sub == 9)
    per = ncols - 1
    if tb.size != m * per:
        return _decode_chunk_python(mi, lo, raw, spec)
    # the aggregate count alone would not notice one short line paired with one
    # long one, so check that every line's tab group really lies inside it
    if per and ((tb[per - 1::per] >= le).any() or (tb[0::per] <= ls).any()):
        return _decode_chunk_python(mi, lo, raw, spec)

    # ---- chromosome ------------------------------------------------------
    cs, ce_ = _field_bounds(sub, tb, ls, le, ncols, spec['chrom'])
    clen = ce_ - cs
    lim = sub.size - 1
    has_chr = ((clen >= 3) & (sub[cs] == 99)
               & (sub[np.minimum(cs + 1, lim)] == 104)
               & (sub[np.minimum(cs + 2, lim)] == 114))
    base_o = cs + np.where(has_chr, 3, 0)
    nlen = clen - np.where(has_chr, 3, 0)
    b0 = sub[np.minimum(base_o, lim)].astype(np.int64)
    b1 = sub[np.minimum(base_o + 1, lim)].astype(np.int64)
    d0 = (b0 >= 48) & (b0 <= 57)
    d1 = (b1 >= 48) & (b1 <= 57)
    one, two = (nlen == 1), (nlen == 2)
    code = np.zeros(m, np.int64)
    code = np.where(one & d0, b0 - 48, code)
    code = np.where(two & d0 & d1, (b0 - 48) * 10 + (b1 - 48), code)
    code = np.where(one & ((b0 == 88) | (b0 == 120)), 23, code)        # X / x
    code = np.where(one & ((b0 == 89) | (b0 == 121)), 24, code)        # Y / y
    ok = (code >= 1) & (code <= 24)

    # ---- position: one masked multiply-accumulate per digit column -------
    ps, pe = _field_bounds(sub, tb, ls, le, ncols, spec['pos'])
    plen = pe - ps
    pos = np.zeros(m, np.int64)
    pok = plen > 0
    for k in range(int(plen.max()) if m else 0):
        byte = sub[np.minimum(ps + k, lim)].astype(np.int64)
        digit = byte - 48
        pok &= ~((k < plen) & ((digit < 0) | (digit > 9)))
        pos = np.where(k < plen, pos * 10 + digit, pos)
    ok &= pok

    if _LOCI_ONLY:
        sel = np.flatnonzero(ok)
        return dict(mi=mi, lo=lo, n=int(sel.size),
                    loci=np.unique((code[sel] << 32) | pos[sel]))

    # ---- alleles ---------------------------------------------------------
    es, ee = _field_bounds(sub, tb, ls, le, ncols, spec['eff'])
    eff, eff_len = _decode_bases(sub, es, ee)
    ok &= eff_len > 0

    oth = np.full(m, -1, np.int8)
    oth_s = np.zeros(m, np.int64)
    oth_e = np.zeros(m, np.int64)
    filled = np.zeros(m, bool)
    for col in spec['oth']:
        s, e = _field_bounds(sub, tb, ls, le, ncols, col)
        c, ln = _decode_bases(sub, s, e)
        take = (~filled) & (ln > 0)
        if take.any():
            oth = np.where(take, c, oth).astype(np.int8)
            oth_s = np.where(take, s, oth_s)
            oth_e = np.where(take, e, oth_e)
            filled |= take

    # ---- dominant / recessive flags --------------------------------------
    mt = np.zeros(m, np.int8)
    for col, val in ((spec['dom'], 1), (spec['rec'], 2)):
        if col is None:
            continue
        s, e = _field_bounds(sub, tb, ls, le, ncols, col)
        ln = e - s
        b = sub[np.minimum(s, lim)]
        flag = ((ln == 1) & (b == 49)) | ((ln == 4) & ((b == 116) | (b == 84)))
        if val == 1:
            mt = np.where(flag, np.int8(1), mt)
        else:
            mt = np.where(flag & (mt == 0), np.int8(2), mt)

    # ---- weight: the one field that is not cheap in array code -----------
    w = _weights(raw, spec, m)
    if w is None:
        return _decode_chunk_python(mi, lo, raw, spec)
    ok &= np.isfinite(w)

    # Multi-character or non-ACGT alleles keep their bytes for the python
    # resolver.  A '/' in an other-allele field means "unknown" (v_1..v_4 drop
    # it to the empty string); only a multi-character field can hold one, and
    # those are exactly the rows extracted here.
    cplx = np.flatnonzero(ok & ((eff < 0) | (oth == -2)))
    strings = []
    for i in cplx.tolist():
        e_s = raw[int(es[i]):int(ee[i])].decode('latin1').upper()
        o_s = raw[int(oth_s[i]):int(oth_e[i])].decode('latin1').upper() \
            if filled[i] else ''
        if '/' in o_s:
            o_s = ''
        if not o_s:
            oth[i] = -1
        if eff[i] >= 0 and oth[i] == -1:
            continue           # "A/G" reduced to "no other allele": ordinary row
        strings.append((i, e_s, o_s))

    sel = np.flatnonzero(ok)
    remap = np.full(m, -1, np.int64)
    remap[sel] = np.arange(sel.size)
    out_strings = [(int(remap[i]), e_s, o_s) for i, e_s, o_s in strings]
    return dict(mi=mi, lo=lo, code=code[sel].astype(np.int8), pos=pos[sel],
                eff=eff[sel], oth=oth[sel], w=w[sel], mt=mt[sel],
                strings=out_strings)


def _weights(raw, spec, m):
    """The weight column through pandas' C tokeniser; None if it disagrees."""
    try:
        col = pd.read_csv(io.BytesIO(raw), sep='\t', header=None,
                          names=spec['names'], usecols=[spec['w']],
                          engine='c', quoting=3, na_filter=True,
                          skip_blank_lines=True, float_precision='round_trip')
        s = col[spec['names'][spec['w']]]
        if len(s) != m:
            return None
        return pd.to_numeric(s, errors='coerce').to_numpy(dtype=np.float64)
    except Exception:
        return None


def _empty_model_chunk(mi, lo=0):
    return dict(mi=mi, lo=lo, code=np.zeros(0, np.int8), pos=np.zeros(0, np.int64),
                eff=np.zeros(0, np.int8), oth=np.zeros(0, np.int8),
                w=np.zeros(0, np.float64), mt=np.zeros(0, np.int8), strings=[])


def _decode_chunk_python(mi, lo, raw, spec):
    """Safety net: a chunk whose field layout is not the declared one is read
    line by line, exactly as v_1..v_4 read every file."""
    names, ncols = spec['names'], spec['ncols']
    codes, poss, effs, oths, ws, mts, strings = [], [], [], [], [], [], []
    for line in io.BytesIO(raw):
        parts = line.rstrip(b'\r\n').decode('latin1').split('\t')
        if len(parts) < ncols:
            continue

        def get(j, default=''):
            return parts[j].strip() if j is not None and j < len(parts) else default

        chrom, pos = get(spec['chrom']), get(spec['pos'])
        eff = get(spec['eff']).upper()
        oth = ''
        for col in spec['oth']:
            v = get(col).upper()
            if v:
                oth = v
                break
        if '/' in oth:
            oth = ''
        w = get(spec['w'])
        if not (chrom and pos and eff and w):
            continue
        code = chrom_code(chrom)
        if code is None:
            continue
        try:
            pos = int(pos)
            w = float(w)
        except ValueError:
            continue
        dom = get(spec['dom']).lower() in ('true', '1') if spec['dom'] is not None else False
        rec = get(spec['rec']).lower() in ('true', '1') if spec['rec'] is not None else False
        i = len(codes)
        codes.append(code)
        poss.append(pos)
        ws.append(w)
        mts.append(1 if dom else (2 if rec else 0))
        e_c = _BASE.get(eff, -2) if len(eff) == 1 else -2
        o_c = -1 if not oth else (_BASE.get(oth, -2) if len(oth) == 1 else -2)
        effs.append(e_c)
        oths.append(o_c)
        if e_c < 0 or o_c == -2:
            strings.append((i, eff, oth))
    return dict(mi=mi, lo=lo, code=np.asarray(codes, np.int8), pos=np.asarray(poss, np.int64),
                eff=np.asarray(effs, np.int8), oth=np.asarray(oths, np.int8),
                w=np.asarray(ws, np.float64), mt=np.asarray(mts, np.int8),
                strings=strings)


def read_models(pgs_paths, workers, loci_only=False):
    """-> list of per-model dicts, in the order of `pgs_paths`.

    With `loci_only`, -> (sorted unique loci, rows per file, text bytes per
    file) and no per-row array is ever assembled.

    The caller is responsible for keeping `pgs_paths` inside the budget: this
    inflates all of them at once on purpose (the fork pool then inherits the
    buffers copy-on-write instead of pickling them, which is v_5's trick and
    most of why the stage is fast), so a block is exactly as big as the text it
    can afford to hold."""
    global _MODEL_TEXT, _MODEL_SPEC, _LOCI_ONLY
    _LOCI_ONLY = loci_only
    with ThreadPoolExecutor(min(workers, max(1, len(pgs_paths)))) as ex:
        texts = list(ex.map(_inflate_file, pgs_paths))
    specs = [_model_spec(t) for t in texts]
    # published to the module BEFORE the pool forks: the workers inherit the
    # buffers copy-on-write instead of receiving them through a pickle.
    _MODEL_TEXT, _MODEL_SPEC = texts, specs

    tasks = []
    for mi, (text, spec) in enumerate(zip(texts, specs)):
        if spec is None:
            continue
        n = len(text)
        lo = spec['body']
        while lo < n:
            hi = min(lo + CHUNK_BYTES, n)
            if hi < n:
                nxt = text.find(b'\n', hi)
                hi = n if nxt < 0 else nxt + 1
            tasks.append((mi, lo, hi))
            lo = hi
    tasks.sort(key=lambda t: t[2] - t[1], reverse=True)

    if loci_only:
        # The union is folded as the chunks arrive — holding every chunk's
        # locus array to the end would be the same (catalog-sized) retention
        # this version exists to remove.  The fold is a device sort, as in
        # main(): 6 s of one core per 11.7 M keys against 0.25 s here.
        rows = [0] * len(pgs_paths)
        acc, acc_bytes = [], 0
        merged = np.zeros(0, np.int64)

        def _fold(acc, merged):
            if not acc:
                return merged
            part = np.concatenate(acc + [merged])
            return to_host(xp.unique(xp.asarray(part)))

        if tasks:
            with mp.Pool(min(workers, len(tasks))) as pool:
                for res in pool.imap_unordered(_decode_chunk, tasks, chunksize=1):
                    rows[res['mi']] += res['n']
                    acc.append(res['loci'])
                    acc_bytes += res['loci'].nbytes
                    if acc_bytes > B_FOLD:
                        merged, acc, acc_bytes = _fold(acc, merged), [], 0
        merged = _fold(acc, merged)
        _MODEL_TEXT = _MODEL_SPEC = None
        return merged, rows, [len(t) for t in texts]

    parts = [[] for _ in pgs_paths]
    if tasks:
        with mp.Pool(min(workers, len(tasks))) as pool:
            for res in pool.imap_unordered(_decode_chunk, tasks, chunksize=1):
                parts[res['mi']].append(res)

    models = []
    for mi, path in enumerate(pgs_paths):
        pid = os.path.basename(path).split('.')[0]
        chunks = parts[mi]
        # chunks come back out of order; restore file order so the per-model
        # row order is the file's, deterministically, run to run.
        chunks.sort(key=lambda c: c['lo'])
        if not chunks:
            models.append(dict(pid=pid, code=np.zeros(0, np.int8),
                               pos=np.zeros(0, np.int64), eff=np.zeros(0, np.int8),
                               oth=np.zeros(0, np.int8), w=np.zeros(0, np.float64),
                               mt=np.zeros(0, np.int8), strings=[]))
            continue
        strings, off = [], 0
        for c in chunks:
            for i, e_s, o_s in c['strings']:
                strings.append((i + off, e_s, o_s))
            off += c['code'].size
        models.append(dict(
            pid=pid,
            code=np.concatenate([c['code'] for c in chunks]),
            pos=np.concatenate([c['pos'] for c in chunks]),
            eff=np.concatenate([c['eff'] for c in chunks]),
            oth=np.concatenate([c['oth'] for c in chunks]),
            w=np.concatenate([c['w'] for c in chunks]),
            mt=np.concatenate([c['mt'] for c in chunks]),
            strings=strings))
        # the chunks and the concatenation of them are the same rows twice;
        # drop each model's as soon as it has been joined
        parts[mi] = None
        del chunks
    # the inflated text is the biggest thing this function touched and the
    # matching that comes next does not need it
    _MODEL_TEXT = _MODEL_SPEC = None
    del texts, specs, parts
    return models


# ─────────────────────────────────────────────────────────────────────────────
# GZIP MEMBER DISCOVERY  [DISK]
# ─────────────────────────────────────────────────────────────────────────────
def _member_starts_here(raw, off, probe=1 << 16):
    if off + 18 > len(raw):
        return False
    if raw[off + 3] & 0xE0:
        return False
    try:
        d = isal_zlib.decompressobj(31)
        out = d.decompress(raw[off:off + probe], 4096)
    except Exception:
        return False
    if not out:
        return False
    line = out.split(b'\n', 1)[0]
    return line.startswith(b'##') or line.count(b'\t') >= 9


_SCAN_WINDOW = 1 << 25                 # 32 MB of file per numpy compare


def _member_cache_file(path):
    if not MEMBER_CACHE:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = f"{os.path.realpath(path)}|{st.st_size}|{st.st_mtime_ns}|v1"
    return os.path.join(MEMBER_CACHE, hashlib.sha1(key.encode()).hexdigest() + '.json')


def member_ranges(path):
    """Byte ranges of a VCF's gzip members, in order.

    Runs in a worker process: the parent never holds the 628 MB of compressed
    input, and never reads it at all.  The scan itself is a three-byte numpy
    compare — `mmap.find` is a byte-by-byte C loop with no memchr and took over
    two minutes on these files.

    v_10 read the whole file into RAM and compared it in one go, which is three
    full-length boolean temporaries on top of the file itself: ~2.5 GB per
    worker, and twenty of these run at once.  Here the file is mapped, not read,
    and the compare walks it a window at a time, so a worker's working set is
    ~100 MB whatever the file's size.  The answer is also cached on disk (keyed
    by path, size and mtime), because the second pass over the VCFs wants
    exactly the same ranges and re-reading 2 TB to rediscover them is the most
    expensive way possible to learn nothing new."""
    cache = _member_cache_file(path)
    if cache:
        try:
            with open(cache) as f:
                return [tuple(r) for r in json.load(f)]
        except (OSError, ValueError):
            pass
    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        if size < 4:
            return [(0, size)]
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            a = np.frombuffer(mm, dtype=np.uint8)      # a view, not a copy
            starts = [0]
            off = 0
            while off < size - 3:
                end = min(off + _SCAN_WINDOW, size - 3)
                seg = a[off:end + 3]
                hit = (seg[:-3] == 0x1F) & (seg[1:-2] == 0x8B) & (seg[2:-1] == 0x08)
                for j in np.flatnonzero(hit).tolist():
                    o = off + j
                    if o and _member_starts_here(mm, o):
                        starts.append(o)
                del seg, hit
                off = end
            ends = starts[1:] + [size]
            # a gzip member's last four bytes are its uncompressed size (mod
            # 2^32), so the parent can size its shared-memory slots without
            # inflating anything
            out = [(s, e, int(np.frombuffer(bytes(a[e - 4:e]), '<u4')[0]))
                   for s, e in zip(starts, ends)]
            del a
        finally:
            mm.close()
    if cache:
        try:
            os.makedirs(MEMBER_CACHE, exist_ok=True)
            tmp = f"{cache}.{os.getpid()}"
            with open(tmp, 'w') as f:
                json.dump(out, f)
            os.replace(tmp, cache)
        except OSError:
            pass
    return out


_SHM_CACHE = {}


def inflate_to_slot(task):
    """Inflate one gzip member into a shared-memory SLOT the parent owns.

    The point is what does NOT happen: the inflated text is not pickled back to
    the parent (v_2..v_6 sent every parsed array through a pipe), and the parent
    runs no inflate threads of its own (v_7/v_8 did, and twenty of them cost the
    one thread driving the GPU ~2.3 s of GIL handoffs).  The slots are created
    once and recycled, because creating and unlinking a 300 MB block per member
    was measured at 0.019 s of munmap and tmpfs teardown in the PARENT — 2.17 s
    over a run, which is more than all the kernels put together."""
    idx, path, start, end, slot, cap = task
    with open(path, 'rb') as f:
        f.seek(start)
        raw = f.read(end - start)
    parts = []
    d = isal_zlib.decompressobj(31)
    data = raw
    while data:
        parts.append(d.decompress(data))
        if not d.eof:
            raise ValueError(f"{path}[{start}:{end}] does not end on a gzip "
                             f"member boundary")
        data = d.unused_data
        if data:
            d = isal_zlib.decompressobj(31)
    text = parts[0] if len(parts) == 1 else b''.join(parts)
    n = len(text)
    if n == 0:
        return idx, slot, 0, False
    own = n > cap                       # a member too big for the slot gets its
    if own:                             # own block, which the parent unlinks
        shm = shared_memory.SharedMemory(create=True, size=n)
    else:
        shm = _SHM_CACHE.get(slot)
        if shm is None:
            # the slot belongs to the parent; merely attaching to it must not
            # make this worker's resource tracker unlink it when the pool exits.
            # 3.13 can say so when attaching; before that the registration has
            # to be undone afterwards, which is noisy across several pools (the
            # tracker raises KeyError on the second unregister of a name).
            try:
                shm = shared_memory.SharedMemory(name=slot, track=False)
            except TypeError:
                shm = shared_memory.SharedMemory(name=slot)
                resource_tracker.unregister(shm._name, 'shared_memory')
            _SHM_CACHE[slot] = shm
    shm.buf[:n] = text
    name = shm.name
    del text, parts
    if own:
        shm.close()
    return idx, name, n, own


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — READ VCFs, KEEPING ONLY MODEL LOCI  [DISK->RAM] + [CPU, 20 cores]
# v_3's line loop (v_4's array version was measured slower).  Two changes: the
# membership set is per-chromosome and is built inside the worker from the
# sorted model-locus array — the parent never materialises a 9 M-entry python
# set — and every emitted target carries its survivor ordinal.
# ─────────────────────────────────────────────────────────────────────────────
_NEEDED = None          # sorted int64 (code<<32)|pos; reaches workers by fork


def _pos_set_for(code):
    """The model positions on one contig, as a python set of ints."""
    lo = np.searchsorted(_NEEDED, np.int64(code) << 32)
    hi = np.searchsorted(_NEEDED, (np.int64(code) + 1) << 32)
    return set((_NEEDED[lo:hi] & 0xFFFFFFFF).tolist())


_D_BASE_LUT = cp.asarray(_BASE_LUT) if GPU else _BASE_LUT

# ── the VCF record parser, as two fused kernels ──────────────────────────────
# Written as CUDA rather than as a chain of cupy expressions on purpose.  The
# expression form was measured first: 0.131 s per 293 MB member, ~2.2 GB/s,
# because every field extraction is its own kernel over its own temporary — a
# `== '\t'` pass materialises 54.8 M tab offsets (438 MB) that are then gathered
# with stride 9, and the contig and position decode take another twenty passes.
# One thread per line reads its own record once instead, straight out of the
# inflated text, and nothing but the answer is written.
#
# Two kernels, not one, because the prefilter in between is a sorted-array probe
# that cuPy already does well, and because the second kernel then touches only
# the ~1% of lines that survive it.
VCF_KERNEL_SRC = r'''
// One thread per CHUNK of the inflated text, not per line: a thread owns every
// record whose first byte falls in its chunk, so no line index has to exist.
// It rejects on the spot — the model-locus array is probed inside the kernel —
// and emits only survivors, through an atomic counter.  Overflowing `cap` is
// reported rather than written, and the caller retries with a bigger buffer.
extern "C" __global__
void vcf_scan(const unsigned char* buf, const long long n, const long long chunk,
              const long long nchunk, const long long* needed, const int n_needed,
              long long* out_start, long long* out_key, int* counter, const int cap)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nchunk) return;
    long long p = c * chunk;
    long long stop = p + chunk; if (stop > n) stop = n;
    if (c > 0) {                                  // a record belongs to the
        while (p < n && buf[p - 1] != 10) ++p;    // chunk holding its FIRST byte
    }
    while (p < stop) {
        long long s = p;
        while (p < n && buf[p] != 10) ++p;        // p = end of this line
        long long end = p;
        ++p;
        if (end <= s || buf[s] == '#') continue;

        long long q = s;
        while (q < end && buf[q] != 9) ++q;       // CHROM
        if (q >= end) continue;
        long long cs = s, clen = q - s, r = q + 1;
        if (clen >= 3) {
            unsigned char a = buf[cs], b = buf[cs + 1], d = buf[cs + 2];
            if ((a == 'c' && b == 'h' && d == 'r') ||
                (a == 'C' && b == 'H' && d == 'R')) { cs += 3; clen -= 3; }
        }
        int code = 0;
        if (clen == 1) {
            unsigned char b0 = buf[cs];
            if (b0 >= '0' && b0 <= '9') code = b0 - '0';
            else if (b0 == 'X') code = 23;
            else if (b0 == 'Y') code = 24;
        } else if (clen == 2) {
            unsigned char b0 = buf[cs], b1 = buf[cs + 1];
            if (b0 >= '0' && b0 <= '9' && b1 >= '0' && b1 <= '9')
                code = (b0 - '0') * 10 + (b1 - '0');
        }
        if (code < 1 || code > 24) continue;      // plink2 --chr 1-22, X, Y, XY
        long long pos = 0; int nd = 0; int okd = 1;
        while (r < end && buf[r] != 9) {
            unsigned char b = buf[r];
            if (b < '0' || b > '9') { okd = 0; break; }
            pos = pos * 10 + (b - '0'); ++nd; ++r;
        }
        if (!okd || nd == 0 || r >= end) continue;

        long long key = (((long long)code) << 32) | pos;
        int lo = 0, hi = n_needed - 1, found = 0;   // the prefilter, inline
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            long long v = needed[mid];
            if (v == key) { found = 1; break; }
            if (v < key) lo = mid + 1; else hi = mid - 1;
        }
        if (!found) continue;
        int slot = atomicAdd(counter, 1);
        if (slot < cap) { out_start[slot] = s; out_key[slot] = key; }
    }
}

extern "C" __global__
void vcf_record(const unsigned char* buf, const long long n,
                const long long* starts, const int nsel, const signed char* lut,
                const int max_allele,
                signed char* flag, signed char* refc, signed char* altc,
                signed char* dose, long long* rs_o, long long* re_o,
                long long* as_o, long long* ae_o)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nsel) return;
    long long p = starts[i], end = p;
    while (end < n && buf[end] != 10) ++end;
    flag[i] = 0; dose[i] = -1;
    int tabs = 0;
    while (p < end && tabs < 3) { if (buf[p] == 9) ++tabs; ++p; }   // CHROM POS ID
    if (tabs < 3) return;
    long long rs = p;
    while (p < end && buf[p] != 9) ++p;
    long long re = p;
    if (p >= end) return;
    long long as_ = ++p;
    while (p < end && buf[p] != 9) ++p;
    long long ae = p;
    if (p >= end) return;
    ++p;
    tabs = 0;
    while (p < end && tabs < 4) { if (buf[p] == 9) ++tabs; ++p; } // QUAL FILTER INFO FORMAT
    if (tabs < 4) return;
    long long rlen = re - rs, alen = ae - as_;
    if (rlen < 1 || alen < 1 || rlen > max_allele || alen > max_allele) return;
    for (long long k = as_; k < ae; ++k) if (buf[k] == ',') return;  // multi-allelic

    int acc = 0, nsep = 0, cur = 0, dot = 0, bad = 0;
    for (long long k = p; k < end; ++k) {        // the GT, up to ':' or the
        unsigned char b = buf[k];                // next sample column
        if (b == ':' || b == 9) break;
        if (b == '/' || b == '|') { acc += cur; cur = 0; ++nsep; }
        else if (b >= '0' && b <= '9') { if (b != '0') cur = 1; }
        else if (b == '.') dot = 1;
        else bad = 1;
    }
    acc += cur;
    int d = (nsep == 0) ? acc * 2 : acc;         // one allele -> homozygous
    dose[i] = (signed char)((dot || bad) ? -1 : d);
    signed char rc = (rlen == 1) ? lut[buf[rs]] : (signed char)-1;
    signed char ac = (alen == 1) ? lut[buf[as_]] : (signed char)-1;
    if (rc >= 0 && ac >= 0) { flag[i] = 1; refc[i] = rc; altc[i] = ac; }
    else { flag[i] = 2; rs_o[i] = rs; re_o[i] = re; as_o[i] = as_; ae_o[i] = ae; }
}
'''
if GPU:
    _K_SCAN = cp.RawKernel(VCF_KERNEL_SRC, 'vcf_scan')
    _K_RECORD = cp.RawKernel(VCF_KERNEL_SRC, 'vcf_record')
else:
    _K_SCAN = _K_RECORD = None

_CHUNK = 2048                      # bytes of text per scanning thread
_BATCH_DEV = None                  # the one reused device text buffer
_PIN = _PIN_ARR = None             # the one reused pinned staging buffer
PREFETCH = int(os.environ.get('PGS_PREFETCH', '20'))
BATCH_BYTES = int(os.environ.get('PGS_BATCH_BYTES', '400000000'))   # members inflated ahead
INFLATE_THREADS = int(os.environ.get('PGS_INFLATE_THREADS', '0'))
_CAP = [1 << 20]                   # survivor buffer size, grown on demand


def _dev_contig(buf, ls, t0, lim):
    """VCF CHROM field -> contig code (0 = not one plink2 would import)."""
    clen = t0 - ls
    has_chr = ((clen >= 3) & (buf[ls] == 99)
               & (buf[xp.minimum(ls + 1, lim)] == 104)
               & (buf[xp.minimum(ls + 2, lim)] == 114))
    base = ls + xp.where(has_chr, 3, 0)
    nlen = clen - xp.where(has_chr, 3, 0)
    b0 = buf[xp.minimum(base, lim)].astype(xp.int64)
    b1 = buf[xp.minimum(base + 1, lim)].astype(xp.int64)
    d0 = (b0 >= 48) & (b0 <= 57)
    d1 = (b1 >= 48) & (b1 <= 57)
    one, two = (nlen == 1), (nlen == 2)
    code = xp.zeros(ls.size, xp.int64)
    code = xp.where(one & d0, b0 - 48, code)
    code = xp.where(two & d0 & d1, (b0 - 48) * 10 + (b1 - 48), code)
    code = xp.where(one & (b0 == 88), 23, code)              # X
    code = xp.where(one & (b0 == 89), 24, code)              # Y
    return code, (code >= 1) & (code <= 24)


def _dev_int(buf, s, e, lim):
    """Decimal field -> int64, one masked multiply-accumulate per digit column."""
    ln = e - s
    out = xp.zeros(s.size, xp.int64)
    top = int(ln.max()) if s.size else 0
    for k in range(top):
        digit = buf[xp.minimum(s + k, lim)].astype(xp.int64) - 48
        out = xp.where(k < ln, out * 10 + digit, out)
    return out


def _dev_genotype(buf, gs, ge, lim):
    """The sample column's GT -> dosage, or -1 for missing/unparseable.

    Mirrors the python rule exactly: a '.' anywhere in the GT is missing; the
    dosage is the number of alleles whose digits are not all zero; a single
    allele (hemizygous, which is how this release stores male non-PAR chrX) is
    read as the homozygous diploid call, so it counts twice."""
    flen = ge - gs
    cap = int(min(int(flen.max()) if gs.size else 0, _GT_CAP))
    gt_len = xp.minimum(flen, cap)
    found = xp.zeros(gs.size, bool)
    for k in range(cap):                       # GT ends at the first ':'
        b = buf[xp.minimum(gs + k, lim)]
        isc = (k < gt_len) & (b == 58) & ~found
        gt_len = xp.where(isc, k, gt_len)
        found |= isc
    truncated = (~found) & (flen > cap)        # a GT longer than the scan window
    nsep = xp.zeros(gs.size, xp.int64)
    acc = xp.zeros(gs.size, xp.int64)
    cur = xp.zeros(gs.size, bool)
    dot = xp.zeros(gs.size, bool)
    bad = xp.zeros(gs.size, bool)
    for k in range(cap):
        inb = k < gt_len
        b = buf[xp.minimum(gs + k, lim)]
        sep = inb & ((b == 47) | (b == 124))   # '/' or '|'
        dig = inb & (b >= 48) & (b <= 57)
        dot |= inb & (b == 46)
        bad |= inb & ~(sep | dig | (b == 46))
        acc += xp.where(sep, cur.astype(xp.int64), 0)
        cur = xp.where(sep, False, cur | (dig & (b != 48)))
        nsep += sep
    acc += cur.astype(xp.int64)
    dose = xp.where(nsep == 0, acc * 2, acc)   # one allele -> homozygous
    dose = xp.where(dot | bad, -1, dose)
    return dose.astype(xp.int8), truncated


_BLOCK = 256


def _upload_batch(texts):
    """Several members' text, contiguous on the device, in file order.

    Concatenating is safe because the sweep is chunk-based and line-oriented: a
    member ends on a newline, so a batch is simply more VCF lines.  It has to be
    ONE SAMPLE'S consecutive members, because the ordinal a batch produces is a
    rank in its own byte order."""
    sizes = [len(t) for t in texts]
    total = sum(sizes)
    # ONE device buffer, reused by every batch.  Allocating a fresh ~1 GB block
    # per batch was measured at 0.65 s per batch — cudaFree of a block that size
    # synchronises the device and, on a unified-memory part, decommits pages.
    global _BATCH_DEV
    if _BATCH_DEV is None or _BATCH_DEV.size < total:
        _BATCH_DEV = None
        _BATCH_DEV = cp.empty(total, cp.uint8)
    buf = _BATCH_DEV[:total]
    # `cp.asarray` per member and not `slice.set`: asarray stages the transfer
    # through cupy's pinned buffer pool and manages ~13 GB/s out of the
    # shared-memory slots, where a plain pageable `set` on the same source
    # manages 1.8 GB/s.  The device-to-device copy into the batch that this
    # costs is ~10 ms per GB.  Gathering into one explicitly pinned staging
    # buffer instead was also measured: it moves less (upload 1.67 s -> 1.42 s
    # on the big draw) but the host copy lands in the parent's critical path and
    # the loop came out slower, so it is not what is used.
    off = 0
    for t, n in zip(texts, sizes):
        if n:
            buf[off:off + n] = cp.asarray(np.frombuffer(t, np.uint8))
            off += n
    return buf


def _gpu_parse_member(text, needed_d):
    """One inflated gzip member -> the targets it contributes, on the device."""
    return _gpu_parse_buffer(cp.asarray(np.frombuffer(text, np.uint8)), needed_d)


def _gpu_parse_buffer(buf, needed_d):
    """The device pipeline, over one buffer of VCF text.

    v_7..v_9 ran it once per gzip member — 115 times a run — and every run of it
    is thirty-odd cupy calls whose python-side cost does not depend on how much
    text they cover.  v_10 hands it several members at a time, so the fixed cost
    is paid ~15 times instead."""
    n = int(buf.size)
    if n == 0 or needed_d.size == 0:
        return _empty_member()

    # ---- pass 1: reject, in one sweep of the text, emitting survivors ----
    nchunk = (n + _CHUNK - 1) // _CHUNK
    grid = ((nchunk + _BLOCK - 1) // _BLOCK,)
    counter = cp.zeros(1, cp.int32)
    while True:
        cap = _CAP[0]
        starts = cp.empty(cap, cp.int64)
        keys = cp.empty(cap, cp.int64)
        counter[0] = 0
        _K_SCAN(grid, (_BLOCK,),
                (buf, np.int64(n), np.int64(_CHUNK), np.int64(nchunk),
                 needed_d, np.int32(needed_d.size), starts, keys, counter,
                 np.int32(cap)))
        nsel = int(counter[0])
        if nsel <= cap:
            break
        _CAP[0] = 1 << int(np.ceil(np.log2(nsel * 1.5)))    # retry, once
    if nsel == 0:
        return _empty_member()

    # the atomic counter hands out slots in arrival order, so put the
    # survivors back into file order — the ordinal below depends on it
    starts, keys = starts[:nsel], keys[:nsel]
    order = cp.argsort(starts)
    starts, skey = starts[order], keys[order]

    # ---- pass 2: the ~1% of records that survive the prefilter ----------
    flag = cp.empty(nsel, cp.int8)
    refc = cp.zeros(nsel, cp.int8)
    altc = cp.zeros(nsel, cp.int8)
    dose = cp.empty(nsel, cp.int8)
    rs = cp.zeros(nsel, cp.int64)
    re_ = cp.zeros(nsel, cp.int64)
    as_ = cp.zeros(nsel, cp.int64)
    ae = cp.zeros(nsel, cp.int64)
    _K_RECORD(((nsel + _BLOCK - 1) // _BLOCK,), (_BLOCK,),
              (buf, np.int64(n), starts, np.int32(nsel), _D_BASE_LUT,
               np.int32(_MAX_ALLELE_LEN), flag, refc, altc, dose,
               rs, re_, as_, ae))

    code = skey >> 32
    pos = skey & 0xFFFFFFFF
    # PAR split: plink2 imports chrX/chrY PAR on their own contigs, so a scoring
    # row addressed to X cannot match there.
    tcode = code
    for c, ((a1, b1), (a2, b2)) in _PAR.items():
        on = code == c
        tcode = cp.where(on & (pos >= a1) & (pos <= b1), 26, tcode)
        tcode = cp.where(on & (pos >= a2) & (pos <= b2), 27, tcode)

    # the ordinal is the rank among the targets this member emits, in file
    # order — what the record ordinal would give, at no per-line cost
    keep = flag > 0
    ordv = cp.cumsum(keep.astype(cp.int64)) - 1

    snp = flag == 1
    out = dict(snp_key=((tcode << 38) | (pos << 6)
                        | (refc.astype(cp.int64) << 3) | altc.astype(cp.int64))[snp],
               snp_dose=dose[snp], snp_ord=ordv[snp], ind=None)
    ind = cp.flatnonzero(flag == 2)
    if int(ind.size):
        out['ind'] = (to_host(tcode[ind]), to_host(pos[ind]), to_host(rs[ind]),
                      to_host(re_[ind]), to_host(as_[ind]), to_host(ae[ind]),
                      to_host(dose[ind]), to_host(ordv[ind]))
    return out


def _empty_member():
    return dict(snp_key=xp.zeros(0, xp.int64), snp_dose=xp.zeros(0, xp.int8),
                snp_ord=xp.zeros(0, xp.int64), ind=None)


def _indel_keys(texts, starts, ind):
    """The few non-SNP survivors, as the string keys the matcher uses.

    `starts` is the byte offset of each member inside the batch, so a survivor's
    offset is resolved back to the member that holds it.

    Same strings as v_10, built without the numpy scalars.  v_10 indexed six
    numpy arrays per record, and each of those is an object allocation before it
    is an integer: at 100 x 100 this loop was ~22 s of a 92 s pass, and it grows
    with (samples x indel targets), so it is one of the two things that decide
    whether the top of the ladder finishes.  `.tolist()` converts each array
    once."""
    tcode, pos, rs, re_, as_, ae, dose, ordv = ind
    tl, pl = tcode.tolist(), pos.tolist()
    rsl, rel, asl, ael = rs.tolist(), re_.tolist(), as_.tolist(), ae.tolist()
    name = _CODE_NAME
    keys = []
    add = keys.append
    for i, r0 in enumerate(rsl):
        j = bisect_right(starts, r0) - 1
        t, off = texts[j], starts[j]
        ref = bytes(t[r0 - off:rel[i] - off]).decode('latin1').upper()
        alt = bytes(t[asl[i] - off:ael[i] - off]).decode('latin1').upper()
        add(f"{name[tl[i]]}:{pl[i]}:{ref}:{alt}")
    return keys, dose.astype(np.int8), ordv.astype(np.int64)


def _parse_text_cpu(text, sets):
    """v_3..v_6's line loop, kept as the fallback for a member whose layout the
    array path will not touch.  Identical output, at the old speed."""
    snp_k, snp_d, snp_o = [], [], []
    oth_k, oth_d, oth_o = [], [], []
    cur_code, cur_set = -1, frozenset()
    ordv = 0
    for line in io.BytesIO(text):
        if line[:1] == b'#':
            continue
        t1 = line.find(b'\t')
        t2 = line.find(b'\t', t1 + 1)
        if t2 < 0:
            continue
        code = _CHROM_B.get(line[:t1])
        if code is None:                       # plink2 --chr 1-22, X, Y, XY
            continue
        pos = int(line[t1 + 1:t2])
        if code != cur_code:
            cur_code = code
            cur_set = sets.get(code)
            if cur_set is None:
                cur_set = sets[code] = _pos_set_for(code)
        if pos not in cur_set:                 # <- the prefilter
            continue
        c = line.rstrip(b'\n').split(b'\t')
        if len(c) < 10:
            continue
        ref, alt, gt = c[3].upper(), c[4].upper(), c[9]
        if b',' in alt:
            continue
        if len(ref) > _MAX_ALLELE_LEN or len(alt) > _MAX_ALLELE_LEN:
            continue
        g = gt.split(b':', 1)[0]
        if b'.' in g:
            d = -1
        else:
            try:
                alleles = g.replace(b'|', b'/').split(b'/')
                d = sum(1 for a in alleles if int(a) > 0)
                if len(alleles) == 1:          # hemizygous -> homozygous
                    d *= 2
            except ValueError:
                d = -1
        tcode = target_code(code, pos)
        if len(ref) == 1 and len(alt) == 1 and ref in _BASE_B and alt in _BASE_B:
            snp_k.append((tcode << 38) | (pos << 6)
                         | (_BASE_B[ref] << 3) | _BASE_B[alt])
            snp_d.append(d)
            snp_o.append(ordv)
        else:
            oth_k.append(f"{_CODE_NAME[tcode]}:{pos}:{ref.decode()}:{alt.decode()}")
            oth_d.append(d)
            oth_o.append(ordv)
        ordv += 1
    out = dict(
        snp_key=xp.asarray(np.fromiter(snp_k, np.int64, len(snp_k))),
        snp_dose=xp.asarray(np.fromiter(snp_d, np.int8, len(snp_d))),
        snp_ord=xp.asarray(np.fromiter(snp_o, np.int64, len(snp_o))),
        ind=None)
    if oth_k:
        out['ind_keys'] = (oth_k,
                           np.fromiter(oth_d, np.int8, len(oth_d)),
                           np.fromiter(oth_o, np.int64, len(oth_o)))
    return out


def _sample_id_of(path):
    try:
        with gzip.open(path, 'rb') as f:
            for line in f:
                if line.startswith(b'#CHROM'):
                    c = line.rstrip(b'\n').split(b'\t')
                    if len(c) > 9:
                        return c[9].decode()
                    break
                if not line.startswith(b'#'):
                    break
    except Exception:
        pass
    return os.path.basename(path).split('.')[0]


def _stream_members(vcf_paths, workers, needed_d, on_batch, on_sample_end,
                    ranges=None, label='VCF '):
    """v_10's inflate ring and batched device parse, with the ACCUMULATION
    taken out.

    v_10 appended every batch's output to a per-sample list and did the set
    algebra at the end, which is why it retained 17 B per (sample, kept record)
    — 470 GB at 1000 x 1000 — and why it died there.  The pipeline itself is
    fine and is kept exactly as it was: the shared-memory slot ring, the
    process-pool inflate, the biggest-member-first order, the one-sample
    batching rule and the reused device text buffer.  What changed is that a
    batch is handed to `on_batch(sample_index, parsed, indel)` and then dropped,
    and `on_sample_end(sample_index)` fires as soon as a sample's last batch has
    been handed over — so what a caller retains is its own business, and both
    callers here retain O(n_var), not O(n_var x samples).

    Returns (member ranges per file, stats).  The ranges are worth keeping: the
    second pass over the same VCFs wants the same answer and the scan is a full
    read of every file."""
    n = len(vcf_paths)
    pool = mp.Pool(workers)
    # The member scan is itself a 628 MB read per file, so it runs in the same
    # pool and its results are consumed lazily: the first file's members are
    # already being inflated and parsed while the last file is still being
    # scanned.
    scanned = [None] * n
    if ranges is None:
        scan = pool.imap(member_ranges, vcf_paths)
    else:
        scan = iter(ranges)

    n_members = [0]

    def gen_tasks():
        for i, rng in enumerate(scan):
            scanned[i] = rng
            n_members[0] += len(rng)
            # Biggest members first, as in v_8/v_9 — they set the critical
            # path.  Batching does not need file order: the ordinal a batch
            # produces is a rank in its own byte order, and two targets are only
            # ever ordered against each other when they share a locus, which
            # means they share a member, inside which byte order is byte order.
            # A batch does have to be ONE SAMPLE's members.
            for (s, e, isize) in sorted(rng, key=lambda r: r[2], reverse=True):
                yield (i, vcf_paths[i], s, e, isize)

    tasks = gen_tasks()

    stats = dict(fallback=0, inflate=0.0, device=0.0, members=0)
    sets = {}

    # The inflate moves back into PROCESSES — but unlike v_2..v_6 nothing is
    # pickled home: each worker leaves its member in a shared-memory block and
    # returns a name.  The parent is then single-threaded while it drives the
    # GPU, which is the point.  v_7/v_8 inflated in threads, and the twenty
    # inflate threads cost the one device thread ~2.3 s of the 6.2 s parse loop
    # in GIL handoffs alone; a process pool has no GIL to hand off.
    t = time.time()
    n_done = 0
    pending = deque()
    slots = {}                      # slot name -> (SharedMemory, capacity)
    free = deque()
    first = next(tasks, None)
    if first is None:
        cap = 1 << 20
    else:
        cap = max(first[4], 1 << 20)
    # v_10 hardcoded a 12 GB ring here.  The ring is the one structure in this
    # stage whose size is chosen rather than implied, so it comes out of the
    # budget: enough slots to keep the inflate pool busy, never more than the
    # budget's share of them.
    n_slot = max(2, min(PREFETCH, int(B_SLOTS // cap)))
    for _ in range(n_slot):
        shm = shared_memory.SharedMemory(create=True, size=cap)
        slots[shm.name] = shm
        free.append(shm.name)
    queued = deque([first] if first is not None else [])

    def fill():
        while free and len(pending) < PREFETCH:   # backpressure: a member of
            if queued:                            # inflated text per slot
                task = queued.popleft()
            else:
                task = next(tasks, None)
                if task is None:
                    return
            slot = free.popleft()
            i, path, s, e, _isz = task
            pending.append((slot, pool.apply_async(inflate_to_slot,
                                                   ((i, path, s, e, slot, cap),))))

    # A batch is consecutive members of ONE sample, up to BATCH_BYTES of text.
    # The device pipeline's python cost is per CALL, not per byte, so paying it
    # once for a gigabyte instead of once per 300 MB member is most of what this
    # version does.
    batch = dict(idx=-1, texts=[], starts=[], shms=[], slots=[], total=0)
    n_batches = [0]

    def flush():
        if not batch['texts']:
            return
        n_batches[0] += 1
        t0 = time.time()
        buf = _upload_batch(batch['texts'])
        sync(); stats['upload'] = stats.get('upload', 0.0) + time.time() - t0
        _t1 = time.time()
        res = _gpu_parse_buffer(buf, needed_d) if GPU else None
        sync(); stats['kern'] = stats.get('kern', 0.0) + time.time() - _t1
        ind = None
        if res is None:
            stats['fallback'] += len(batch['texts'])
            res = _parse_text_cpu(b''.join(bytes(t) for t in batch['texts']), sets)
            ind = res.get('ind_keys')
        elif res['ind'] is not None:
            ind = _indel_keys(batch['texts'], batch['starts'], res['ind'])
        stats['device'] += time.time() - t0
        on_batch(batch['idx'], res, ind)
        del buf, res, ind
        for mv in batch['texts']:                 # a live memoryview would keep
            mv.release()                          # its shared block open
        batch['texts'].clear()
        for shm in batch['shms']:                 # the one-off blocks
            shm.close()
            shm.unlink()
        free.extend(batch['slots'])
        batch.update(idx=-1, texts=[], starts=[], shms=[], slots=[], total=0)

    cur_sample = [-1]
    n_samples_done = [0]
    last_beat = [time.time()]

    def sample_boundary(idx):
        """A sample is finished the moment a member of the next one appears:
        the pool returns members in submission order and the tasks are
        generated file by file, so nothing of sample i can arrive after
        something of sample i+1."""
        if idx == cur_sample[0]:
            return
        if cur_sample[0] >= 0:
            on_sample_end(cur_sample[0])
            n_samples_done[0] += 1
            # A pass over the whole cohort is an hour of silence otherwise, and
            # a run nobody can see the progress of is a run nobody can tell
            # apart from a hang.
            now = time.time()
            if now - last_beat[0] >= PROGRESS_EVERY:
                last_beat[0] = now
                done, el = n_samples_done[0], now - t
                _, avail = _meminfo()
                print(f"[{label}] {done}/{n} samples, {n_done} members, "
                      f"{el:.0f}s elapsed, {el/max(1,done):.2f}s/sample, "
                      f"eta {(n - done) * el / max(1, done) / 60:.1f} min;  "
                      f"MemAvailable {avail/1e9:.0f} GB", flush=True)
        cur_sample[0] = idx

    fill()
    while pending:
        slot, fut = pending.popleft()
        t0 = time.time()
        idx, name, size, own = fut.get()
        stats['inflate'] += time.time() - t0
        if not size:
            free.append(slot)
            fill()
            n_done += 1
            continue
        if batch['texts'] and (idx != batch['idx']
                               or batch['total'] + size > BATCH_BYTES
                               or len(batch['texts']) >= n_slot - 1):
            flush()
        sample_boundary(idx)
        shm = None
        if own:                                   # oversized member: one-off
            shm = shared_memory.SharedMemory(name=name)
            text = shm.buf[:size]
            batch['shms'].append(shm)
        else:
            text = slots[name].buf[:size]
        batch['idx'] = idx
        batch['starts'].append(batch['total'])
        batch['texts'].append(text)
        batch['slots'].append(slot)
        batch['total'] += size
        n_done += 1
        fill()
    flush()
    if cur_sample[0] >= 0:
        on_sample_end(cur_sample[0])
    pool.close()
    pool.join()
    for shm in slots.values():
        shm.close()
        try:
            shm.unlink()
        except FileNotFoundError:
            pass
    sync()
    stats['members'] = n_members[0]
    print(f"[DISK] {n} VCFs -> {n_members[0]} gzip members;  "
          f"{int(needed_d.size):,} model loci to keep", flush=True)
    print(f"[{'GPU ' if GPU else 'CPU '}] members parsed: {n_done} in "
          f"{time.time() - t:.2f}s wall  ({stats['fallback']} on the python "
          f"fallback;  {stats['device']:.2f}s on the device, "
          f"{stats['inflate']:.2f}s waiting for inflated members;  "
          f"{len(slots)} shm slots of {cap/1e6:.0f} MB;  "
          f"{n_batches[0]} device batches; upload {stats.get('upload',0):.2f}s "
          f"kernels {stats.get('kern',0):.2f}s)", flush=True)

    return scanned, stats


def _merge_targets(U, Ord, keys, ords):
    """Fold one batch's SNP keys into the running target union and its
    per-target minimum ordinal.

    v_10 kept every batch and did this once, at the end, over the whole cohort.
    Doing it incrementally would be pointless if it re-sorted the union every
    time — so it does not: the common case is a batch whose keys are ALL already
    in the union (every sample in a joint callset lists the same sites), and
    that case is a probe plus the same atomic scatter-min v_10 used, with no
    allocation at all.  A real union is built only for the keys that are new."""
    if keys.size == 0:
        return U, Ord
    if U.size:
        idx, hit = _probe(xp, U, keys)
        if bool(hit.all()):
            scatter_min(Ord, idx, ords)
            return U, Ord
        newU = xp.unique(xp.concatenate([U, keys[~hit]]))
    else:
        newU = xp.unique(keys)
    newOrd = xp.full(int(newU.size), _INF_ORD, xp.int64)
    if U.size:
        newOrd[xp.searchsorted(newU, U)] = Ord
    scatter_min(newOrd, xp.searchsorted(newU, keys), ords)
    return newU, newOrd


def scan_targets(vcf_paths, needed, workers):
    """PASS A over the VCFs: the target set and the ordinal that breaks its
    ties, and NOTHING per sample.

    -> (snp_tgt, oth_tgt, oth_index, tgt_ord, member ranges)"""
    global _NEEDED
    _NEEDED = needed
    needed_d = xp.asarray(needed)
    state = dict(U=xp.zeros(0, xp.int64), Ord=xp.zeros(0, xp.int64))
    oth_ord = {}                       # indel target -> lowest ordinal seen

    def on_batch(i, res, ind):
        state['U'], state['Ord'] = _merge_targets(
            state['U'], state['Ord'], res['snp_key'], res['snp_ord'])
        if ind is not None:
            keys, _dose, ordv = ind
            for k, o in zip(keys, ordv.tolist()):
                prev = oth_ord.get(k)
                if prev is None or o < prev:
                    oth_ord[k] = o

    def _count_sample(i):
        """Pass A keeps nothing per sample; the callback exists so the driver
        can count samples and report progress."""

    ranges, stats = _stream_members(vcf_paths, workers, needed_d, on_batch,
                                    _count_sample, label='pass A')
    snp_tgt, ordU = state['U'], state['Ord']
    n_snp = int(snp_tgt.size)
    oth_tgt = sorted(oth_ord)
    oth_index = {k: n_snp + j for j, k in enumerate(oth_tgt)}
    # Ordinal of every target row: the rank of its record inside its gzip
    # member.  A locus lives entirely in one member and each sample lists that
    # member's sites in the same order, so the min over samples is that order,
    # and it is what breaks ties the way the reference does.
    tgt_ord = xp.concatenate([ordU, xp.asarray(np.fromiter(
        (oth_ord[k] for k in oth_tgt), np.int64, len(oth_tgt)))])
    sync()
    return snp_tgt, oth_tgt, oth_index, tgt_ord, ranges


def fill_dosage(vcf_paths, needed, workers, snp_tgt, oth_index, n_var, D_T,
                ranges):
    """PASS B over the VCFs: each sample's dosages, folded into its own row of
    the memory-mapped dosage matrix and then forgotten.

    This is the change that matters.  v_10 held every sample's keys, dosages and
    ordinals until the last VCF had been read and only then built `D`; here a
    sample's column exists for exactly as long as it takes to write it, so the
    stage's retention is one column (n_var bytes) whatever the cohort size."""
    global _NEEDED
    _NEEDED = needed
    needed_d = xp.asarray(needed)
    col = xp.empty(n_var, xp.int8)
    state = dict(cur=-1, checked=False)
    done = set()

    def on_batch(i, res, ind):
        if i != state['cur']:
            col.fill(-1)
            state['cur'] = i
        sk = res['snp_key']
        if sk.size:
            rows = xp.searchsorted(snp_tgt, sk)
            if not state['checked']:
                # pass A saw these very bytes, so every key is in the target
                # set; check it once rather than trusting it silently, because
                # a miss here would be a wrong dosage, not a crash
                state['checked'] = True
                if int((snp_tgt[xp.minimum(rows, n_var - 1)] != sk).sum()):
                    raise RuntimeError("dosage pass found a key the target "
                                       "pass did not: the two passes disagree")
            col[rows] = res['snp_dose']
        if ind is not None:
            keys, dose, _ordv = ind
            orows = np.fromiter((oth_index[k] for k in keys), np.int64, len(keys))
            col[xp.asarray(orows)] = xp.asarray(dose)

    def on_sample_end(i):
        D_T[i] = to_host(col)
        done.add(i)

    _, stats = _stream_members(vcf_paths, workers, needed_d, on_batch,
                               on_sample_end, ranges=ranges, label='pass B')
    missing = [i for i in range(len(vcf_paths)) if i not in done]
    if missing:
        # a VCF that contributed no record at all: all-missing, as in v_10,
        # where its column of D was never written and kept its -1 fill
        blank = np.full(n_var, -1, np.int8)
        for i in missing:
            D_T[i] = blank
        print(f"[DISK] {len(missing)} VCF(s) contributed no target record",
              flush=True)
    D_T.flush()
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — MATCH  [GPU for the bulk, python only for the awkward rows]
#
# Rules (all of them read off pgsc_calc's own match log, see v_1/planning.txt):
#   * with an other allele, priority refalt > altref > refalt_flip > altref_flip,
#     where "refalt" means the EFFECT allele is the target's REF; a
#     strand-ambiguous pair is dropped outright;
#   * without one, priority no_oa_ref > no_oa_alt > flipped forms, and inside a
#     priority level the candidates — SNP AND INDEL TARGETS TOGETHER — are
#     ranked by target order.  Ambiguity is not a search filter: the winner is
#     chosen first and excluded afterwards if it turns out ambiguous;
#   * duplicate_ID: if two rows of one model best-match the same target, both go.
# ─────────────────────────────────────────────────────────────────────────────
def _probe(xp, tgt, key):
    """searchsorted + equality, as (index, hit)."""
    n = tgt.size
    if n == 0:
        return xp.zeros(key.size, xp.int64), xp.zeros(key.size, bool)
    idx = xp.searchsorted(tgt, key)
    idxc = xp.minimum(idx, n - 1)
    return idxc, (idx < n) & (tgt[idxc] == key)


# Every probe is a binary search into the sorted target array: ~23 dependent
# random reads per row, and v_6..v_8 ran all four (or sixteen) of them over
# every row whether or not it had already matched.  At 11.7 M model rows that is
# over a billion random reads and it was the whole 1.5 s of the stage.  The
# priority rules only ever look at the FIRST hit, so each level now runs on the
# rows that are still unmatched — and most rows match at level one.
def _match_with_oa(xp, tgt, base, eff, oth):
    """Both alleles known and single-base: 4 probes, first hit wins."""
    n = base.size
    t = xp.full(n, -1, xp.int64)
    flip = xp.zeros(n, xp.int8)
    rem = xp.arange(n, dtype=xp.int64)
    ce, co = 3 - eff, 3 - oth
    for r, a, fl in ((eff, oth, 1), (oth, eff, 0), (ce, co, 1), (co, ce, 0)):
        if rem.size == 0:
            break
        key = base[rem] | (r[rem].astype(xp.int64) << 3) | a[rem]
        idx, hit = _probe(xp, tgt, key)
        h = xp.flatnonzero(hit)
        if h.size:
            g = rem[h]
            t[g] = idx[h]
            flip[g] = xp.int8(fl)
        rem = rem[~hit]
    amb = oth == (3 - eff)
    return t, flip, (t >= 0) & ~amb


def _match_no_oa(xp, tgt, tord, base, eff):
    """Effect allele only: 4 priority levels x 3 candidate other alleles,
    lowest target ordinal wins inside a level."""
    n = base.size
    t = xp.full(n, -1, xp.int64)
    flip = xp.zeros(n, xp.int8)
    amb = xp.zeros(n, bool)
    done = xp.zeros(n, bool)
    rem = xp.arange(n, dtype=xp.int64)
    for e_all, is_ref in ((eff, True), (eff, False), (3 - eff, True),
                          (3 - eff, False)):
        if rem.size == 0:
            break
        b, e = base[rem], e_all[rem]
        e64 = e.astype(xp.int64)
        m = rem.size
        lvl_ord = xp.full(m, _INF_ORD, xp.int64)
        lvl_t = xp.full(m, -1, xp.int64)
        lvl_amb = xp.zeros(m, bool)
        for o in range(4):
            key = b | (((e64 << 3) | o) if is_ref else ((o << 3) | e64))
            idx, hit = _probe(xp, tgt, key)
            hit &= (e != o)
            cand = xp.where(hit, tord[idx], _INF_ORD)
            better = cand < lvl_ord
            lvl_ord = xp.where(better, cand, lvl_ord)
            lvl_t = xp.where(better, idx, lvl_t)
            lvl_amb = xp.where(better, e == (3 - o), lvl_amb)
        found = lvl_ord < _INF_ORD
        h = xp.flatnonzero(found)
        if h.size:
            g = rem[h]
            t[g] = lvl_t[h]
            flip[g] = xp.int8(1 if is_ref else 0)
            amb[g] = lvl_amb[h]
            done[g] = True
        rem = rem[~found]
    return t, flip, done & ~amb


def _fallback_pool(snp_tgt, tgt_ord, oth_tgt, oth_index, loci):
    """(code,pos) -> [(ref, alt, target row, ordinal)] for the loci that need
    the string form: every locus holding an indel target, plus the loci of the
    model rows whose alleles are not single bases.

    The selection runs on the device — it is a probe over the whole target
    array — and only the handful of selected rows crosses back to the host."""
    ord_h = to_host(tgt_ord)
    pool = {}
    for j, k in enumerate(oth_tgt):
        ch, po, r, a = k.split(':', 3)
        row = oth_index[k]
        pool.setdefault((_NAME_CODE[ch], int(po)), []).append(
            (r, a, row, int(ord_h[row])))
    if loci.size and snp_tgt.size:
        loci_d = xp.asarray(loci)
        tloc = snp_tgt >> 6
        j = xp.searchsorted(loci_d, tloc)
        jc = xp.minimum(j, loci_d.size - 1)
        sel = xp.flatnonzero((j < loci_d.size) & (loci_d[jc] == tloc))
        keys = to_host(snp_tgt[sel])
        rows = to_host(sel)
        names = 'ACGT'
        for k, row in zip(keys.tolist(), rows.tolist()):
            pool.setdefault((k >> 38, (k >> 6) & 0xFFFFFFFF), []).append(
                (names[(k >> 3) & 7], names[k & 7], row, int(ord_h[row])))
    return pool


def _resolve_slow(code, pos, eff, oth, pool, oth_index):
    """The awkward rows: multi-character alleles, and every effect-allele-only
    row at a locus that holds an indel target."""
    if oth:
        if (eff, oth) in _AMBIG:
            return None
        if len(eff) == 1 and len(oth) == 1 and eff in _BASE and oth in _BASE:
            for r, a, fl in ((eff, oth, 1), (oth, eff, 0),
                             (_COMP[eff], _COMP[oth], 1), (_COMP[oth], _COMP[eff], 0)):
                for (tr, ta, row, _o) in pool.get((code, pos), ()):
                    if tr == r and ta == a:
                        return row, fl
            return None
        for r, a, fl in ((eff, oth, 1), (oth, eff, 0)):
            t = oth_index.get(str_key(code, pos, r, a))
            if t is not None:
                return t, fl
        return None
    lst = pool.get((code, pos))
    if not lst:
        return None
    alleles = (eff, _COMP[eff]) if (len(eff) == 1 and eff in _BASE) else (eff,)
    for e in alleles:
        for is_ref in (True, False):
            best = None
            for (r, a, row, od) in lst:
                if (r if is_ref else a) == e:
                    if best is None or od < best[1]:
                        best = (row, od, (r, a))
            if best is not None:
                return None if best[2] in _AMBIG else (best[0], 1 if is_ref else 0)
    return None


def match_models(models, snp_tgt, tgt_ord, oth_tgt, oth_index, n_var):
    # `snp_tgt` and `tgt_ord` are already device-resident (v_6): the probes read
    # them in place, and nothing but the matched rows comes back.
    # loci that force the string path: those holding an indel target
    ind_loci = np.unique(np.asarray(
        [(_NAME_CODE[k.split(':', 1)[0]] << 32) | int(k.split(':')[1])
         for k in oth_tgt], dtype=np.int64)) if oth_tgt else np.zeros(0, np.int64)

    # Which rows of each model cannot be done in array form: alleles that are
    # not single bases, and effect-allele-only rows at a locus that holds an
    # indel target, where the candidate pool has to be merged by target order.
    #
    # This pre-pass is where v_6..v_8's matching time actually went — not in the
    # probes.  `np.searchsorted` of 11.7 M model loci against the indel-locus
    # array, on one core, plus the mask arithmetic around it, was ~1.4 s of the
    # 1.5 s stage; the probes themselves are 0.05 s on the device.  Everything
    # below stays on the device, and the model columns are uploaded once here
    # and reused by the probes.
    ind_loci_d = xp.asarray(ind_loci)
    # v_10 uploaded every model's four columns here and held them all — 18 B/row
    # over the whole catalog — so that the probe loop below could reuse them.
    # Each model's columns are read by exactly one iteration of that loop, so
    # they are uploaded there instead and released at the end of it; what
    # survives this pre-pass is one bool per row (the mask) and the loci that
    # need the string path.
    slow_masks, slow_loci = [], []
    for mdl in models:
        code = xp.asarray(mdl['code']).astype(xp.int64)
        loc = (code << 32) | xp.asarray(mdl['pos'])
        slow = xp.zeros(int(loc.size), bool)
        if mdl['strings']:
            slow[xp.asarray(np.fromiter((i for i, _, _ in mdl['strings']),
                                        np.int64, len(mdl['strings'])))] = True
        if ind_loci_d.size and loc.size:
            j = xp.searchsorted(ind_loci_d, loc)
            jc = xp.minimum(j, ind_loci_d.size - 1)
            slow |= ((xp.asarray(mdl['oth']) == -1) & (j < ind_loci_d.size)
                     & (ind_loci_d[jc] == loc))
        slow_masks.append(slow)
        if bool(slow.any()):
            slow_loci.append(to_host(loc[slow]))
        del code, loc

    pool = None
    if slow_loci:
        pool = _fallback_pool(snp_tgt, tgt_ord, oth_tgt, oth_index,
                              np.unique(np.concatenate([ind_loci] + slow_loci)))

    tgt_d, ord_d = snp_tgt, tgt_ord

    all_rows, all_cols, all_vals, all_flips, all_mt = [], [], [], [], []
    all_eff = []                       # effect-allele code, for the manifest
    pgs_ids, matched = [], []
    for p, mdl in enumerate(models):
        pgs_ids.append(mdl['pid'])
        code_d = xp.asarray(mdl['code']).astype(xp.int64)
        pos_d = xp.asarray(mdl['pos'])
        eff_d = xp.asarray(mdl['eff'])
        oth_d = xp.asarray(mdl['oth'])
        eff_h, oth_h = mdl['eff'], mdl['oth']
        n = int(pos_d.size)
        if n == 0:
            matched.append(0)
            continue
        slow = slow_masks[p]
        idx_fast = xp.flatnonzero(~slow)

        # ---- the bulk: 4 probes with an other allele, 16 without --------
        if idx_fast.size:
            base = ((code_d << 38) | (pos_d << 6))[idx_fast]
            eff = eff_d[idx_fast]
            oth = oth_d[idx_fast]
            has_oa = oth >= 0
            t = xp.full(base.size, -1, xp.int64)
            flip = xp.zeros(base.size, xp.int8)
            ok = xp.zeros(base.size, bool)
            for with_oa in (True, False):
                sel = xp.flatnonzero(has_oa if with_oa else ~has_oa)
                if sel.size == 0:
                    continue
                if with_oa:
                    st, sf, so = _match_with_oa(xp, tgt_d, base[sel], eff[sel],
                                                oth[sel])
                else:
                    st, sf, so = _match_no_oa(xp, tgt_d, ord_d, base[sel], eff[sel])
                t[sel], flip[sel], ok[sel] = st, sf, so
            # v_6..v_8 brought t, flip and ok back to the host here — 100 MB per
            # model at this scale — and then indexed on the host.  The compaction
            # is a device op; only the matched rows exist afterwards.
            sel_ok = xp.flatnonzero(ok)
            hit_idx = idx_fast[sel_ok]
            hit_t = t[sel_ok]
            hit_f = flip[sel_ok]
        else:
            hit_idx = xp.zeros(0, xp.int64)
            hit_t = xp.zeros(0, xp.int64)
            hit_f = xp.zeros(0, xp.int8)

        # ---- the awkward rows, in python --------------------------------
        s_idx, s_t, s_f = [], [], []
        slow_ix = to_host(xp.flatnonzero(slow))
        if slow_ix.size:
            code_h = to_host(code_d[xp.asarray(slow_ix)])
            pos_h = to_host(pos_d[xp.asarray(slow_ix)])
            smap = {i: (e, o) for i, e, o in mdl['strings']}
            names = 'ACGT'
            for k_, i in enumerate(slow_ix.tolist()):
                if i in smap:
                    e_s, o_s = smap[i]
                else:
                    e_s = names[eff_h[i]] if eff_h[i] >= 0 else ''
                    o_s = names[oth_h[i]] if oth_h[i] >= 0 else ''
                if not e_s:
                    continue
                hit = _resolve_slow(int(code_h[k_]), int(pos_h[k_]), e_s, o_s,
                                    pool or {}, oth_index)
                if hit is None:
                    continue
                s_idx.append(i)
                s_t.append(hit[0])
                s_f.append(hit[1])

        r_idx, r_t, r_f = hit_idx, hit_t, hit_f
        if s_idx:
            r_idx = xp.concatenate([r_idx, xp.asarray(np.asarray(s_idx, np.int64))])
            r_t = xp.concatenate([r_t, xp.asarray(np.asarray(s_t, np.int64))])
            r_f = xp.concatenate([r_f, xp.asarray(np.asarray(s_f, np.int8))])

        # ---- duplicate_ID: several rows on one target -> drop them all --
        # `bincount` over the whole target space, once per model, was 9.3 M
        # counters of host memory per model in v_8; on the device it is a kernel.
        #
        # v_10 says `xp.bincount(r_t, minlength=n_var)`, and on this box that
        # is a landmine: cupy 14.1.1 hands bincount to CUB, and CUB's histogram
        # takes an illegal address once the bin count passes ~12 M — measured
        # 12,000,000 OK, 16,543,769 dead (pressure_test/_bincount_probe.py).
        # The fault is asynchronous, so it surfaces at the next sync with a
        # traceback pointing at innocent code, which is how 100 x 100 repeats 2
        # and 3 died.  Every shape from here up has more than 12 M targets, and
        # v_10 walks into it too.  Below is bincount's OWN non-CUB body — a
        # zeroed array and an atomic scatter of ones — so the counts, and the
        # duplicate_ID rows they drop, are identical.
        if r_t.size:
            if GPU:
                cnt = xp.zeros(n_var, np.intp)
                cupyx.scatter_add(cnt, r_t, 1)
            else:
                cnt = np.bincount(r_t, minlength=n_var)
            keep = cnt[r_t] == 1
            r_idx, r_t, r_f = r_idx[keep], r_t[keep], r_f[keep]
        matched.append(int(r_t.size))
        if r_t.size:
            all_rows.append(xp.full(int(r_t.size), p, xp.int64))
            all_cols.append(r_t)
            all_vals.append(xp.asarray(mdl['w'])[r_idx])
            all_flips.append(r_f)
            all_mt.append(xp.asarray(mdl['mt'])[r_idx])
            all_eff.append(xp.asarray(mdl['eff'])[r_idx])
        # this model is finished: its uploaded columns, its mask and its probe
        # temporaries are all dead, and the next model's are about to be
        # allocated on top of them
        slow_masks[p] = None
        del code_d, pos_d, eff_d, oth_d, slow, r_idx, r_t, r_f

    def cat(xs, dt):
        return xp.concatenate(xs) if xs else xp.zeros(0, dt)
    return (pgs_ids, np.asarray(matched, np.int64),
            cat(all_rows, xp.int64), cat(all_cols, xp.int64),
            cat(all_vals, xp.float64), cat(all_flips, xp.int8),
            cat(all_mt, xp.int8), cat(all_eff, xp.int8))



def write_output(path, sample_ids, pgs_ids, scores, valid):
    r_order = np.argsort(np.asarray(sample_ids, dtype=object), kind='stable')
    c_order = np.argsort(np.asarray(pgs_ids, dtype=object), kind='stable')
    with open(path, 'w') as f:
        f.write("sample_id\t" + "\t".join(pgs_ids[j] for j in c_order) + "\n")
        for i in r_order:
            f.write(sample_ids[i] + "\t" + "\t".join(
                "NA" if not valid[j] else f"{scores[i, j]:.6f}" for j in c_order) + "\n")


def _load_matrix(path):
    with open(path) as f:
        head = f.readline().rstrip('\n').split('\t')
        cols = head[1:]
        rows, cells = [], {}
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip('\n').split('\t')
            rows.append(parts[0])
            for c, v in zip(cols, parts[1:]):
                cells[(parts[0], c)] = v
    return rows, cols, cells


def verify(out_path, ref_path, atol=1e-6, rtol=1e-5):
    print("\n--- correctness check vs pgs_cal reference ---")
    if not os.path.exists(ref_path):
        print(f"reference missing: {ref_path}")
        print("RESULT: SKIPPED")
        return False
    o_rows, o_cols, o_cells = _load_matrix(out_path)
    r_rows, r_cols, r_cells = _load_matrix(ref_path)
    print(f"output    : {len(o_rows)} samples x {len(o_cols)} models  {out_path}")
    print(f"reference : {len(r_rows)} samples x {len(r_cols)} models  {ref_path}")

    missing_s = sorted(set(r_rows) - set(o_rows))
    missing_m = sorted(set(r_cols) - set(o_cols))
    extra_s = sorted(set(o_rows) - set(r_rows))
    extra_m = sorted(set(o_cols) - set(r_cols))
    for label, xs in (("samples only in reference", missing_s),
                      ("models only in reference", missing_m),
                      ("samples only in output", extra_s),
                      ("models only in output", extra_m)):
        if xs:
            print(f"  {label}: {xs}")

    n, n_bad, n_na = 0, 0, 0
    max_abs, max_rel, worst = 0.0, 0.0, None
    per_model = {}
    a_all, b_all = [], []
    for s in r_rows:
        for m in r_cols:
            if s not in o_rows or m not in o_cols:
                n_bad += 1
                continue
            a, b = o_cells[(s, m)], r_cells[(s, m)]
            if a == "NA" or b == "NA":
                if a != b:
                    n_bad += 1
                    print(f"  NA mismatch {s}/{m}: output={a} reference={b}")
                else:
                    n_na += 1
                continue
            av, bv = float(a), float(b)
            n += 1
            a_all.append(av)
            b_all.append(bv)
            d = abs(av - bv)
            rel = d / max(abs(bv), 1e-30)
            ok = d <= atol + rtol * abs(bv)
            if not ok:
                n_bad += 1
            if d > max_abs:
                max_abs, worst = d, (s, m, av, bv)
            max_rel = max(max_rel, rel)
            pm = per_model.setdefault(m, [0.0, 0])
            pm[0] = max(pm[0], d)
            pm[1] += 0 if ok else 1

    if a_all:
        a_arr, b_arr = np.asarray(a_all), np.asarray(b_all)
        rng = float(b_arr.max() - b_arr.min())
        r = float(np.corrcoef(a_arr, b_arr)[0, 1]) if len(a_arr) > 1 and rng > 0 else 1.0
    else:
        rng, r = 0.0, 1.0
    print(f"cells compared     : {n} numeric + {n_na} NA-agreeing")
    print(f"max |difference|   : {max_abs:.6g}" + (
        f"   at {worst[0]}/{worst[1]}  output={worst[2]:.6f} reference={worst[3]:.6f}"
        if worst else ""))
    print(f"max relative diff  : {max_rel:.6g}")
    print(f"pearson r          : {r:.9f}   (reference value range {rng:.6g})")
    print(f"tolerance          : atol={atol:g} rtol={rtol:g}")
    for m in sorted(per_model):
        d, bad = per_model[m]
        print(f"  {m}: max|diff| {d:.6g}  failing cells {bad}")
    ok = (n_bad == 0) and not (missing_s or missing_m or extra_s or extra_m)
    print(f"failing cells      : {n_bad}")
    print("RESULT: " + ("PASSED" if ok else "FAILED"))
    return ok


def _free_device():
    """Hand cupy's cached blocks back.

    On a GB10 the device pool IS host memory, so a pool holding the previous
    model block's peak is host memory the next block cannot have."""
    if GPU:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()




# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — THE WEIGHT MATRICES
#
# v_10 built two: `Wd` (signed weight on the ALT dosage) and `Wo` (the +2w a
# flipped locus owes on every CALLED sample).  Together they give
#
#     score_i = sum_j  w_j * e_ij ,   e_ij = d_ij if not flipped else 2 - d_ij
#
# with an UNCALLED locus contributing zero — which silently shrinks that
# sample's score and is the bug §2 of the task exists to close.  Two matrices
# are added on exactly the same sparsity pattern:
#
#   `Wfill`  w_j * 2*p_j^eff, applied to (1 - obs): the mean-imputed effect
#            dosage of a locus this sample did not call.  Note this is the same
#            number as the task's `sign_j*w_j*2p_j` plus `Wo`'s 2w on the miss —
#            for a flipped locus p_j^eff = 1 - p_j^alt, so w*2*(1-p^alt) is
#            precisely (-w*2p^alt) + 2w.  Building it directly in effect-allele
#            orientation is one matrix instead of two and is exactly equal.
#   `A`      the 0/1 pattern, applied to obs: n_obs[i,j], the number of loci of
#            model j that sample i actually called.  This is what DENOM should
#            have been, and it is what makes a shrunken score visible.
#
# Both are skipped when the panel has no missingness at all (the 1000 Genomes
# NYGC callset is one such panel: every sample lists every site and calls it),
# in which case `Wfill @ miss` is a matmul against a zero matrix and
# `A @ obs` is the constant row-nnz.  Skipping them is not an approximation —
# it is the same number for free — and it keeps the scores of such a panel
# bit-identical to the memory-safe scorer's.
# ─────────────────────────────────────────────────────────────────────────────
NONADD_KERNEL_SRC = r'''
extern "C" __global__
void apply_nonadditive(
    const signed char* dosage_tile,
    const int tile_S,
    const int* pgs_row, const int* var_row, const double* weight,
    const signed char* model, const signed char* flip, const double* imp,
    const int use_imp, const int n_entry, double* scores)
{
    int e = blockIdx.x;
    if (e >= n_entry) return;
    long long vr = var_row[e];
    double w = weight[e];
    signed char md = model[e], fl = flip[e];
    double im = imp[e];
    const signed char* drow = dosage_tile + vr * (long long)tile_S;
    double* orow = scores + (long long)pgs_row[e] * tile_S;
    for (int s = threadIdx.x; s < tile_S; s += blockDim.x) {
        signed char raw = drow[s];
        double val;
        if (raw < 0) {
            if (!use_imp) continue;
            val = im;                  // E[g(dosage)] under the panel's p_j
        } else {
            double d = (double)raw;
            if (fl == 1) d = 2.0 - d;
            if (md == 1) d = fmin(d, 1.0);
            else if (md == 2) d = (d >= 2.0) ? 1.0 : 0.0;
            val = d;
        }
        atomicAdd(&orow[s], val * w);
    }
}
'''
_NONADD_KERNEL = cp.RawKernel(NONADD_KERNEL_SRC, 'apply_nonadditive') if GPU else None


def build_matrices(n_pgs, n_var, rows, cols, vals, flips, models, p_eff,
                   need_imp):
    """Wd, Wo (v_10's, untouched), plus Wfill and A when the panel can miss.

    v_1..v_5 built these with scipy on the host and let `score_gpu` upload the
    result; at 10 M non-zeros that COO->CSR conversion is 0.45 s of one core.
    Built on the device it is a sort of the same entries, which the GPU does in
    tens of milliseconds, and no CSR ever crosses the bus."""
    add = models == 0
    r, c, v, fl = rows[add], cols[add], vals[add], flips[add]
    sign = xp.where(fl == 1, -1.0, 1.0)
    m = fl == 1
    if GPU:
        mk = lambda d, i, j: cusp.csr_matrix((d, (i, j)), shape=(n_pgs, n_var))
    else:
        mk = lambda d, i, j: sp.csr_matrix((d, (i, j)), shape=(n_pgs, n_var))
    Wd = mk(v * sign, r, c)
    Wo = mk(2.0 * v[m], r[m], c[m])
    Wfill = A = None
    if need_imp:
        Wfill = mk(v * 2.0 * p_eff[add], r, c)
        A = mk(xp.ones(int(rows.size), xp.float64), rows, cols)
    for W in (Wd, Wo, Wfill, A):
        if W is None:
            continue
        if hasattr(W, 'sort_indices') and not getattr(W, 'has_sorted_indices', False):
            W.sort_indices()
    na = ~add
    q = p_eff[na]
    # the imputed value of a NON-additive locus is the expectation of the same
    # transform under Hardy-Weinberg at the panel's effect-allele frequency:
    # dominant  E[min(d,1)] = 1-(1-q)^2 ;  recessive  E[1(d=2)] = q^2.
    mt = models[na]
    imp = xp.where(mt == 1, q * (2.0 - q), xp.where(mt == 2, q * q, 2.0 * q))
    nonadd = dict(pgs=rows[na].astype(xp.int32), var=cols[na].astype(xp.int32),
                  w=vals[na], model=mt, flip=flips[na], imp=imp)
    return Wd, Wo, Wfill, A, nonadd


def score_tile(n_var, need_imp):
    """How many samples one scoring tile may hold.

    `dose` and `obs` are float64 and n_var rows tall, so the tile costs
    16 B x n_var per sample — 224 GB at v_10's fixed 512 with n_var = 27 M, and
    16 GB even at the 100 x 100 rung.  It is the second-biggest allocation in
    the program and the easiest to get wrong, so it is measured, not chosen.
    Imputation adds one more float64 plane (`miss`), hence 26 rather than 18."""
    per = (26 if need_imp else 18) * max(1, n_var)
    return max(1, min(TILE, int(B_SCORE // per)))


def score_gpu(Wd, Wo, Wfill, A, nonadd, D_T, tile_size, out_scores, out_nobs,
              col0, s_size):
    """v_10's scoring loop over a memory-mapped dosage matrix, plus coverage.

    The additive arithmetic is untouched when `Wfill` is None, which is the case
    on a panel with no missing calls: an output cell is one CSR row against one
    sample column, so it sees the same terms in the same order whatever the tile
    width is and whichever block of models it is in, and the result is
    bit-identical to the memory-safe scorer's."""
    n_pgs, n_var = Wd.shape
    n_samp = D_T.shape[0]
    na = {k: cp.asarray(v) for k, v in nonadd.items()} if GPU else nonadd
    use_imp = np.int32(1 if Wfill is not None else 0)

    for c0 in range(0, n_samp, tile_size):
        c1 = min(c0 + tile_size, n_samp)
        tile = xp.ascontiguousarray(xp.asarray(np.asarray(D_T[c0:c1])).T)
        dose = xp.maximum(tile, 0).astype(xp.float64)
        obs = (tile >= 0).astype(xp.float64)
        scores = Wd @ dose
        scores += Wo @ obs
        del dose
        if Wfill is not None:
            miss = 1.0 - obs
            scores += Wfill @ miss
            nobs = to_host(A @ obs).T
            del miss
        else:
            nobs = None
        del obs
        if na['pgs'].size:
            if GPU:
                # cuSPARSE's SpMM returns an F-CONTIGUOUS array, so `.ravel()`
                # — which is C-order — silently returns a COPY, and every
                # atomicAdd the non-additive kernel makes lands in a temporary
                # that is then thrown away.  v_10 and the memory-safe scorer
                # both have this: measured on a 40-sample fixture whose model
                # is entirely dominant/recessive, both return exactly 0.000000
                # for every sample.  It is invisible on the PGS Catalog at
                # large — 16 of 3.17 billion matched entries are non-additive —
                # which is presumably why it survived, but a reference
                # distribution must not carry it.  One copy, before the kernel;
                # the additive result it copies is untouched.
                scores = xp.ascontiguousarray(scores)
                _NONADD_KERNEL((int(na['pgs'].size),), (128,),
                               (tile.ravel(), np.int32(c1 - c0), na['pgs'], na['var'],
                                na['w'], na['model'], na['flip'], na['imp'],
                                use_imp, np.int32(na['pgs'].size), scores.ravel()))
            else:
                for e in range(nonadd['pgs'].size):
                    raw = np.asarray(D_T[c0:c1, nonadd['var'][e]])
                    ok = raw >= 0
                    d = raw.astype(np.float64)
                    if nonadd['flip'][e] == 1:
                        d = 2.0 - d
                    if nonadd['model'][e] == 1:
                        d = np.minimum(d, 1.0)
                    elif nonadd['model'][e] == 2:
                        d = np.where(d >= 2.0, 1.0, 0.0)
                    scores[nonadd['pgs'][e], ok] += (d * nonadd['w'][e])[ok]
                    if use_imp:
                        scores[nonadd['pgs'][e], ~ok] += (
                            nonadd['imp'][e] * nonadd['w'][e])
        out_scores[c0:c1, col0:col0 + n_pgs] = to_host(scores).T
        if nobs is None:
            out_nobs[c0:c1, col0:col0 + n_pgs] = s_size
        else:
            out_nobs[c0:c1, col0:col0 + n_pgs] = np.rint(nobs).astype(np.int32)
        del tile, scores


# ─────────────────────────────────────────────────────────────────────────────
# PHASE S — WHAT THE PANEL SAYS ABOUT EVERY TARGET LOCUS
#
# One sequential pass over the dosage matrix gives, per target column:
#   n_obs_ref   how many REFERENCE samples called it   -> the call rate that
#               defines S, and the denominator of p_j
#   alt_ref     the sum of their ALT dosages           -> p_j
#   n_obs_all   the same over every sample, only to decide whether the run
#               needs the imputation path at all
# The same pass gathers a strided sample of SNP columns for the PCA, because a
# second pass over 89 GB to fetch 600 k columns would cost more than keeping
# them.
# ─────────────────────────────────────────────────────────────────────────────
def panel_stats(D_T, ref_mask, cand_idx):
    n_samp, n_var = D_T.shape
    obs_ref = xp.zeros(n_var, xp.int32)
    alt_ref = xp.zeros(n_var, xp.int32)
    obs_all = xp.zeros(n_var, xp.int32)
    G = (np.empty((n_samp, cand_idx.size), np.int8)
         if cand_idx is not None and cand_idx.size else None)
    cand_d = xp.asarray(cand_idx) if G is not None else None
    ref_d = xp.asarray(ref_mask)
    # int8 tile + bool + two int8 temporaries + slack
    step = max(1, min(256, int(B_SCORE // (8 * max(1, n_var)))))
    t0 = time.time()
    for c0 in range(0, n_samp, step):
        c1 = min(c0 + step, n_samp)
        t = xp.asarray(np.asarray(D_T[c0:c1]))
        o = t >= 0
        obs_all += o.sum(axis=0, dtype=xp.int32)
        rm = ref_d[c0:c1][:, None]
        orf = o & rm
        obs_ref += orf.sum(axis=0, dtype=xp.int32)
        alt_ref += xp.where(orf, t, xp.int8(0)).sum(axis=0, dtype=xp.int32)
        if G is not None:
            G[c0:c1] = to_host(t[:, cand_d])
        del t, o, orf, rm
        _free_device()
        print(f"[STAT] panel target statistics {c1}/{n_samp} samples "
              f"({time.time() - t0:.0f}s)", flush=True)
    n_ref = int(ref_mask.sum())
    call_rate = obs_ref.astype(xp.float64) / max(1, n_ref)
    p_alt = xp.where(obs_ref > 0,
                     alt_ref.astype(xp.float64) / xp.maximum(2 * obs_ref, 1), 0.0)
    return dict(obs_ref=obs_ref, alt_ref=alt_ref, obs_all=obs_all,
                call_rate=call_rate, p_alt=p_alt, n_ref=n_ref, G=G)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE P — GENOTYPE PCs
#
# §5: with a multi-ancestry panel a single mu/sigma is dominated by
# ancestry-driven allele-frequency differences, so the raw score is regressed on
# the panel's top PCs and the RESIDUAL is standardised.  The PCs have to be
# projectable onto a future individual, so what is frozen is not the panel's PC
# scores but the per-variant loadings that produce them:
#
#     x_j = (dosage_j - 2 p_j) / sqrt(2 p_j (1 - p_j))    (missing -> 0)
#     PC  = (x . V) / sqrt(K)
#
# which reproduces the panel's own scores exactly (V is X^T U / sqrt(K*lambda),
# so X V / sqrt(K) = U sqrt(lambda)).
# ─────────────────────────────────────────────────────────────────────────────
def build_pca(G, cand_idx, cand_key, stats, ref_mask, n_pcs):
    p = to_host(stats['p_alt'])[cand_idx]
    cr = to_host(stats['call_rate'])[cand_idx]
    maf = np.minimum(p, 1.0 - p)
    keep = (cr >= 0.99) & (maf >= PCA_MAF)
    # LD is not pruned — it is thinned by physical distance, which is enough for
    # the global-ancestry axes this is used for and costs one pass.
    code = cand_key >> 38
    pos = (cand_key >> 6) & ((1 << 32) - 1)
    sel, last_c, last_p = [], -1, -(1 << 62)
    for i in np.flatnonzero(keep):
        c, q = int(code[i]), int(pos[i])
        if c != last_c or q - last_p >= PCA_SPACING:
            sel.append(i)
            last_c, last_p = c, q
    sel = np.asarray(sel, np.int64)
    if sel.size > PCA_VARIANTS:
        sel = sel[np.linspace(0, sel.size - 1, PCA_VARIANTS).astype(np.int64)]
    K = int(sel.size)
    if K < 100:
        print(f"[PCA ] only {K} usable variants — PCs skipped", flush=True)
        return None
    pk, sd = p[sel], np.sqrt(2.0 * p[sel] * (1.0 - p[sel]))
    g = xp.asarray(G[:, sel])
    X = xp.where(g >= 0,
                 (g.astype(xp.float32) - xp.asarray(2.0 * pk, xp.float32))
                 / xp.asarray(sd, xp.float32), xp.float32(0))
    del g
    Xr = X[xp.asarray(ref_mask)]
    C = (Xr @ Xr.T).astype(xp.float64) / K
    # The GRM is (n_reference x n_reference) — 54 MB at 2590 samples — and this
    # box's cupy has no libcusolver, so the eigendecomposition is numpy's.  The
    # matmuls that build and consume it stay on the device, which is where the
    # (n_ref x K) work actually is.
    lam_h, U_h = np.linalg.eigh(to_host(C))
    lam_h, U_h = lam_h[::-1], U_h[:, ::-1]
    # A GRM of M reference samples has at most M-1 informative axes, and an
    # axis whose eigenvalue is numerical noise would come back as a direction
    # made entirely of the floor below.  Keep only the real ones.
    npc = min(n_pcs, max(0, int(ref_mask.sum()) - 1),
              int((lam_h > max(lam_h[0], 1e-300) * 1e-8).sum()))
    if npc < 1:
        print("[PCA ] the reference GRM has no informative axis — PCs skipped",
              flush=True)
        return None
    if npc < n_pcs:
        print(f"[PCA ] only {npc} of {n_pcs} requested PCs are informative",
              flush=True)
    lam_h = lam_h[:npc]
    U = xp.asarray(np.ascontiguousarray(U_h[:, :npc]))
    lam = xp.asarray(lam_h)
    V = (Xr.T.astype(xp.float64) @ U) / xp.sqrt(K * lam)[None, :]
    # deterministic sign: the largest-magnitude loading of each PC is positive
    sgn = xp.sign(V[xp.argmax(xp.abs(V), axis=0), xp.arange(V.shape[1])])
    sgn = xp.where(sgn == 0, 1.0, sgn)
    V = V * sgn[None, :]
    P = (X.astype(xp.float64) @ V) / np.sqrt(K)
    out = dict(keys=cand_key[sel], var_index=cand_idx[sel], p=pk, sd=sd, n_pcs=npc,
               loadings=to_host(V), eigenvalues=to_host(lam), scores=to_host(P),
               n_variants=K)
    del X, Xr, C, V, P
    _free_device()
    print(f"[PCA ] {K} variants, {npc} PCs, "
          f"eigenvalues {', '.join(f'{v:.3g}' for v in out['eigenvalues'][:5])}",
          flush=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# THE FROZEN MANIFEST  (§3)
#
# One file, `model_manifest.store`, holding one isal-deflated blob per model,
# located by `model_manifest.index.tsv`.  5385 separate files would be 5385
# inodes and 5385 opens; the store is one sequential write.  A blob is a JSON
# header line followed by the raw little-endian bytes of its columns:
#
#   var     int32    index into panel_targets — the LOCUS KEY, and through it
#                    the contig, position, REF, ALT and panel call rate
#   weight  float64  the model's effect weight, bit-exact from the scoring file
#   flags   int8     bit 0 = flip (the effect allele is the target's REF),
#                    bits 1-2 = 0 additive / 1 dominant / 2 recessive
#   ea      int8     effect-allele base code (0=A 1=C 2=G 3=T, -1 = not a
#                    single base; the authoritative string is the target's)
#   p_eff   float32  THE FROZEN PANEL EFFECT-ALLELE FREQUENCY.  A future
#                    individual's missing locus is imputed with THIS number,
#                    never with one recomputed from their own data, or their
#                    score is not on the reference scale.  float32 is a 4-byte
#                    convenience copy: 3.2 billion entries make every byte
#                    3.2 GB, and its ~6e-8 relative error moves an imputed
#                    locus's contribution by ~1e-8 of one weight.  The EXACT
#                    value is always reconstructible from panel_targets'
#                    integer columns, n_alt_reference / (2 * n_obs_reference),
#                    complemented when the flip bit is set.
#
# The panel call rate is deliberately not duplicated per locus: it is a
# property of the TARGET, it is in `panel_targets.tsv.gz` under the same `var`
# index, and 3.2 billion copies of it would be 13 GB of redundancy.
# ─────────────────────────────────────────────────────────────────────────────
_MANIFEST_COLS = [('var', '<i4'), ('weight', '<f8'), ('flags', '<i1'),
                  ('ea', '<i1'), ('p_eff', '<f4')]


def manifest_blob(pgs_id, var, weight, flags, ea, p_eff):
    hdr = json.dumps(dict(pgs_id=pgs_id, n=int(var.size),
                          columns=[[c, d] for c, d in _MANIFEST_COLS],
                          note='raw C-order columns follow this line'))
    body = [hdr.encode() + b'\n']
    for arr, (_, dt) in zip((var, weight, flags, ea, p_eff), _MANIFEST_COLS):
        body.append(np.ascontiguousarray(arr, dt).tobytes())
    return isal_zlib.compress(b''.join(body), 1)


def load_model_manifest(store_path, offset, length):
    """Read one model's frozen manifest back.  The reference bundle's README
    points at this function; it is the only reader the format needs."""
    with open(store_path, 'rb') as f:
        f.seek(offset)
        raw = isal_zlib.decompress(f.read(length))
    nl = raw.index(b'\n')
    hdr = json.loads(raw[:nl])
    out, off = {'pgs_id': hdr['pgs_id'], 'n': hdr['n']}, nl + 1
    for name, dt in hdr['columns']:
        w = np.dtype(dt).itemsize * hdr['n']
        out[name] = np.frombuffer(raw, dt, hdr['n'], off)
        off += w
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PHASE O — THE DISTRIBUTIONS  (§4 sample QC, §5 standardise last and stratify)
# ─────────────────────────────────────────────────────────────────────────────
_QUANTILES = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0,
              75.0, 90.0, 95.0, 97.5, 99.0, 99.5, 99.9]


def _scatter_add(dest, index, val=1):
    if GPU:
        cupyx.scatter_add(dest, index, val)
    else:
        np.add.at(dest, index, val)


def model_statistics(scores, nobs, s_size, n_rows, n_matched, n_dropped,
                     ref_mask, pcs, superpop):
    """Per model: coverage, the QC'd panel distribution, and the PC regression.

    The order matters and is the order of the task: S is already fixed and
    imputed, per-sample coverage is already known, so standardisation comes
    LAST and it is stratified — the raw score is regressed on the panel's top
    PCs and the residual is standardised, because on a five-continent panel a
    single mu/sigma mostly measures ancestry."""
    n_samp, n_mod = scores.shape
    zr = np.full_like(scores, np.nan)
    zp = np.full_like(scores, np.nan)
    P = pcs['scores'] if pcs else np.zeros((n_samp, 0))
    design = np.column_stack([np.ones(n_samp), P])
    pinv_cache, stats = {}, []
    pops = sorted(set(superpop))
    for j in range(n_mod):
        rec = dict(pgs_id=None, n_rows=int(n_rows[j]), n_matched=int(n_matched[j]),
                   n_dropped_low_call_rate=int(n_dropped[j]), n_S=int(s_size[j]))
        rec['coverage'] = (rec['n_S'] / rec['n_rows']) if rec['n_rows'] else 0.0
        rec['match_rate'] = (rec['n_matched'] / rec['n_rows']) if rec['n_rows'] else 0.0
        if rec['n_S'] == 0:
            rec.update(deployable=False, reason='no locus of this model is in S',
                       n_qc_pass=0, n_reference=0)
            stats.append(rec)
            continue
        scr = np.rint(nobs[:, j]).astype(np.float64) / rec['n_S']
        ok = scr >= MIN_SAMPLE_CALL_RATE
        rec['sample_call_rate_min'] = float(scr.min())
        rec['n_sample_flagged'] = int((~ok).sum())
        use = ok & ref_mask
        rec['n_qc_pass'] = int(ok.sum())
        rec['n_reference'] = int(use.sum())
        rec['deployable'] = bool(rec['coverage'] >= MIN_COVERAGE)
        rec['reason'] = '' if rec['deployable'] else (
            f"coverage {rec['coverage']:.3f} < {MIN_COVERAGE}")
        if rec['n_reference'] < 2:
            rec.update(deployable=False,
                       reason=(rec['reason'] + '; ' if rec['reason'] else '')
                       + 'fewer than 2 reference samples pass QC')
            stats.append(rec)
            continue
        y = scores[:, j]
        yr = y[use]
        mu, sd = float(yr.mean()), float(yr.std(ddof=1))
        rec['mu'], rec['sigma'] = mu, sd
        rec['min'], rec['max'] = float(yr.min()), float(yr.max())
        rec['quantiles'] = {str(q): float(v) for q, v in
                            zip(_QUANTILES, np.percentile(yr, _QUANTILES))}
        if sd > 0:
            zr[:, j] = (y - mu) / sd
        # ── the stratified part: mu(PC), and sigma from the residual ───────
        # With fewer residual degrees of freedom than this the regression
        # interpolates its own noise — sigma_resid collapses towards zero and
        # z_pc becomes a division by it.  Better to say so than to emit it.
        dof = rec['n_reference'] - design.shape[1]
        rec['pc_adjusted'] = bool(dof >= 10)
        if not rec['pc_adjusted']:
            rec['reason'] = ((rec['reason'] + '; ') if rec['reason'] else '') + \
                f'PC adjustment skipped ({dof} residual d.o.f.)'
            stats.append(rec)
            continue
        key = use.tobytes()
        pin = pinv_cache.get(key)
        if pin is None:
            pin = np.linalg.pinv(design[use])
            pinv_cache[key] = pin
        beta = pin @ yr
        resid = yr - design[use] @ beta
        sdr = float(resid.std(ddof=1))
        rec['pc_beta'] = [float(b) for b in beta]
        rec['sigma_resid'] = sdr
        rec['r2_pc'] = float(1.0 - (sdr ** 2) / (sd ** 2)) if sd > 0 else 0.0
        if sdr > 0:
            zp[:, j] = (y - design @ beta) / sdr
        rec['quantiles_z_pc'] = {
            str(q): float(v) for q, v in
            zip(_QUANTILES, np.percentile(zp[use, j], _QUANTILES))} if sdr > 0 else {}
        # per-super-population, which is what makes the need for §5 visible
        by = {}
        for pp in pops:
            sel = use & (superpop == pp)
            if sel.sum() >= 2:
                v = y[sel]
                by[pp] = dict(n=int(sel.sum()), mu=float(v.mean()),
                              sigma=float(v.std(ddof=1)))
        rec['by_super_population'] = by
        if by:
            mus = np.array([b['mu'] for b in by.values()])
            rec['superpop_mu_spread_in_sigma'] = (
                float((mus.max() - mus.min()) / sd) if sd > 0 else 0.0)
        stats.append(rec)
    return stats, zr, zp


def write_matrix(path, sample_ids, pgs_ids, mat, valid, fmt="%.6f"):
    """Same row/column ordering as the inherited `write_output` — sorted by id —
    so the four score matrices can be read side by side."""
    r_order = np.argsort(np.asarray(sample_ids, dtype=object), kind='stable')
    c_order = np.argsort(np.asarray(pgs_ids, dtype=object), kind='stable')
    with open(path, 'w') as f:
        f.write("sample_id\t" + "\t".join(pgs_ids[j] for j in c_order) + "\n")
        for i in r_order:
            row = mat[i]
            f.write(sample_ids[i] + "\t" + "\t".join(
                "NA" if (not valid[j] or not np.isfinite(row[j]))
                else (fmt % row[j]) for j in c_order) + "\n")


def _gz_text(path, lines_iter):
    with open(path, 'wb') as fh:
        co = isal_zlib.compressobj(3, isal_zlib.DEFLATED, 31)
        buf = []
        n = 0
        for ln in lines_iter:
            buf.append(ln)
            n += len(ln)
            if n > (4 << 20):
                fh.write(co.compress(''.join(buf).encode()))
                buf, n = [], 0
        if buf:
            fh.write(co.compress(''.join(buf).encode()))
        fh.write(co.flush())


def write_panel_targets(path, snp_tgt, oth_tgt, stats, n_samp):
    """Every target locus the panel offers, with the numbers the manifest and
    any future run are frozen against."""
    n_snp = int(snp_tgt.size)
    key = to_host(snp_tgt)
    code = (key >> 38).astype(np.int64)
    pos = ((key >> 6) & ((np.int64(1) << 32) - 1)).astype(np.int64)
    ref = ((key >> 3) & 7).astype(np.int64)
    alt = (key & 7).astype(np.int64)
    obs_ref = to_host(stats['obs_ref'])
    alt_ref = to_host(stats['alt_ref'])
    obs_all = to_host(stats['obs_all'])
    cr = to_host(stats['call_rate'])
    pa = to_host(stats['p_alt'])
    bases = 'ACGT'
    cname = [_CODE_NAME.get(int(c), str(int(c))) for c in
             (np.unique(code) if n_snp else [])]
    cmap = dict(zip((np.unique(code) if n_snp else []), cname))

    def rows():
        yield ("var\tkey\tchrom\tpos\tref\talt\tn_obs_reference\t"
               "n_alt_reference\tn_obs_all\tcall_rate\tp_alt\n")
        for j in range(n_snp):
            yield (f"{j}\t{key[j]}\t{cmap[code[j]]}\t"
                   f"{pos[j]}\t{bases[ref[j]]}\t{bases[alt[j]]}\t{obs_ref[j]}\t"
                   f"{alt_ref[j]}\t{obs_all[j]}\t{cr[j]:.6f}\t{pa[j]:.10g}\n")
        for k, s in enumerate(oth_tgt):
            j = n_snp + k
            c, p_, r_, a_ = s.split(':', 3)
            yield (f"{j}\t{s}\t{c}\t{p_}\t{r_}\t{a_}\t{obs_ref[j]}\t"
                   f"{alt_ref[j]}\t{obs_all[j]}\t{cr[j]:.6f}\t{pa[j]:.10g}\n")
    _gz_text(path, rows())


# ─────────────────────────────────────────────────────────────────────────────
# THE DOSAGE MATRIX, AND MAKING THE TWO VCF PASSES SURVIVE A RERUN
#
# The two streamed VCF passes are ~2/3 of the wall clock at full scale (16 041 s
# of 23 527 s on the 3202 x 5385 run).  Everything downstream of them is new
# code, so a bug in it must not cost four and a half hours a second time:
# `--dosage-cache DIR` writes the dosage matrix and the target arrays there,
# keyed by the VCF list and the model-locus set, and reuses them when the key
# still matches.
# ─────────────────────────────────────────────────────────────────────────────
def _dosage_cache_key(vcf_paths, needed):
    h = hashlib.sha1()
    for p in vcf_paths:
        st = os.stat(p)
        h.update(f"{os.path.realpath(p)}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    h.update(np.ascontiguousarray(needed, '<i8').tobytes())
    return h.hexdigest()


def _open_dosage(n_samp, n_var, path=None, mode='w+'):
    """The dosage matrix, as a memory-mapped file laid out (n_samples x n_var).

    v_10 held it as a resident (n_var x n_samples) device array: 28 GB at
    1000 x 1000 and 89 GB for the whole cohort x whole catalog, on a 119 GB
    box.  As a mapping it costs page cache, which is reclaimable — a matrix
    bigger than RAM becomes slow instead of fatal — and the transpose is what
    makes both directions sequential: pass B writes one sample's row, the
    scorer reads a tile of consecutive rows."""
    need = n_samp * max(1, n_var)
    if path is None:
        os.makedirs(SPILL_DIR, exist_ok=True)
        path = os.path.join(SPILL_DIR, f"_dosage.{os.getpid()}.i8")
    if mode != 'r':
        try:
            free = os.statvfs(os.path.dirname(path) or '.')
            free_b = free.f_bavail * free.f_frsize
            if free_b < need * 1.05:
                raise RuntimeError(f"dosage matrix needs {need/1e9:.1f} GB in "
                                   f"{os.path.dirname(path)}, {free_b/1e9:.1f} GB free")
        except OSError:
            pass
    D_T = np.memmap(path, dtype=np.int8, mode=mode, shape=(n_samp, max(1, n_var)))
    print(f"[DISK] dosage matrix {n_samp} x {n_var} int8 = {need/1e9:.2f} GB "
          f"-> {path} ({'reused' if mode == 'r' else 'built'})", flush=True)
    return D_T, path


def _model_blocks(paths, sizes, rows=None):
    """Consecutive scoring files, grouped so that one group fits the budget.

    Two costs bound a group: decoding it (its inflated text plus ~20 B/row of
    columns) and matching it.  The matching figure is 100 B/row: 20 for the
    decoded columns that stay live, 26 for the matched entries, another 26 for
    the moment `cat()` holds both the per-model list and its concatenation, and
    ~28 for the CSR and the COO sort that builds it."""
    blocks, cur, text, nrow = [], [], 0.0, 0
    for i, p in enumerate(paths):
        t = sizes[i]
        r = rows[i] if rows else 0
        if cur and max(text + t + 20.0 * (nrow + r), 100.0 * (nrow + r)) > B_MODEL:
            blocks.append(cur)
            cur, text, nrow = [], 0.0, 0
        cur.append(p)
        text += t
        nrow += r
    if cur:
        blocks.append(cur)
    return blocks


def read_panel_meta(path, sample_ids):
    """sex / population / super-population / relatedness, from the cohort's own
    manifest.  Absent or unlisted samples fall back to 'unknown'/'unrelated',
    which makes the reference set everybody — stated, not silent."""
    info = {}
    if path and os.path.exists(path):
        import csv
        with open(path) as f:
            for r in csv.DictReader(f):
                info[r.get('sample_id', '')] = r
    out = {k: [] for k in ('sex', 'population', 'super_population', 'relationship')}
    for s in sample_ids:
        r = info.get(s, {})
        out['sex'].append(r.get('sex', 'unknown'))
        out['population'].append(r.get('population', 'unknown'))
        out['super_population'].append(r.get('super_population', 'unknown'))
        out['relationship'].append(r.get('relationship', 'unrelated'))
    return {k: np.asarray(v, dtype=object) for k, v in out.items()}, bool(info)


README = """\
# 1000 Genomes PGS reference distributions

Built by `calculate_distribution_pgs_1kg.py` from the per-individual 1000
Genomes VCFs in `{vcf_dir}` and the PGS Catalog scoring files in `{pgs_dir}`.
Shape: **{n_samp} samples x {n_models} models**, {n_var:,} target loci.

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
   least **{min_call_rate}** of samples. `coverage = |S| / N` is in
   `model_index.tsv`; a model below **{min_coverage}** is marked
   `deployable=False` — it is not that its score is noisy, it is that the score
   is a different quantity from the published one.
2. **Residual missingness inside S is mean-imputed**, dosage -> `2 * p_eff`, so
   E[score] stays unbiased. This is what plink2 does by default.
3. **`p_eff` is frozen per locus** in the manifest, in effect-allele
   orientation. Impute a new individual with the stored number, never a
   recomputed one.
4. **Per-sample coverage travels with every score.** `scores_nobs.tsv` holds
   `n_obs[i,j]`, the loci of S sample *i* actually called; a sample below
   **{min_sample_call_rate}** of `|S|` is excluded from the reference mu/sigma
   and counted in `model_index.tsv`'s `n_sample_flagged`.
5. **Standardisation comes last, and it is stratified.** `z` uses a single
   mu/sigma; `z_pc` regresses the raw score on the top {n_pcs} genotype PCs and
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

{caveats}
"""


def main():
    exts = ('.txt.gz', '.tsv.gz', '.txt', '.tsv', '.gz')
    vcf_paths = sorted(glob.glob(os.path.join(VCF_DIR, '*.vcf.gz'))) \
        or sorted(glob.glob(os.path.join(VCF_DIR, '*.vcf*')))
    pgs_paths = sorted(p for p in glob.glob(os.path.join(PGS_DIR, '*'))
                       if os.path.isfile(p) and p.endswith(exts)
                       and not os.path.basename(p).startswith('_')
                       and os.path.basename(p) != 'manifest.csv')
    if LIMIT_SAMPLES:
        vcf_paths = vcf_paths[:LIMIT_SAMPLES]
    if LIMIT_MODELS:
        pgs_paths = pgs_paths[:LIMIT_MODELS]
    n_samp = len(vcf_paths)
    if not n_samp or not pgs_paths:
        raise SystemExit(f"nothing to do: {n_samp} VCFs in {VCF_DIR}, "
                         f"{len(pgs_paths)} models in {PGS_DIR}")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'manifest'), exist_ok=True)
    print(f"backend : {'GPU (cupy/cuSPARSE)' if GPU else 'CPU (numpy/scipy)'}")
    print(f"cohort  : {n_samp} VCFs   models: {len(pgs_paths)} PGS files")
    print(f"workers : {WORKERS}   tile: <= {TILE}   out: {OUT_DIR}")
    print(f"budget  : {MEM_BUDGET/1e9:.1f} GB ({MEM_BUDGET_WHY})   "
          f"model block {B_MODEL/1e9:.1f} GB   score tile {B_SCORE/1e9:.1f} GB")
    print(f"criteria: locus call rate >= {MIN_CALL_RATE}, sample call rate >= "
          f"{MIN_SAMPLE_CALL_RATE}, coverage >= {MIN_COVERAGE}, "
          f"reference set = {REFERENCE_SET}, {N_PCS} PCs")
    sys.stdout.flush()

    timings = []
    gz = [os.path.getsize(p) for p in pgs_paths]

    # ── 1/2: the scoring files, and the loci the VCF pass must keep ─────────
    t = time.time()
    single = sum(gz) * _GZ_PEAK <= B_MODEL
    models_all, rows_per_file, text_per_file = None, None, None
    if single:
        models_all = read_models(pgs_paths, WORKERS)
        rows_per_file = [int(m['pos'].size) for m in models_all]
        dt = time.time() - t
        timings.append(("1_model_loading", dt))
        print(f"[CPU ] models: {len(models_all)}  rows: {sum(rows_per_file):,}"
              f"  ({dt:.2f}s)", flush=True)
        t = time.time()
        parts = [((m['code'].astype(np.int64) << 32) | m['pos']) for m in models_all]
        needed = to_host(xp.unique(xp.asarray(np.concatenate(parts)))) if parts \
            else np.zeros(0, np.int64)
        del parts
        sync()
        timings.append(("2_model_locus_set", time.time() - t))
    else:
        ratio = _GZ_RATIO
        needed = np.zeros(0, np.int64)
        rows_per_file, text_per_file = [], []
        i, nb = 0, 0
        while i < len(pgs_paths):
            lim = max(B_MODEL / (ratio + 2.0), gz[i])
            j, tot = i, 0
            while j < len(pgs_paths) and (j == i or tot + gz[j] <= lim):
                tot += gz[j]
                j += 1
            loci, rws, txt = read_models(pgs_paths[i:j], WORKERS, loci_only=True)
            ratio = max(ratio, sum(txt) / max(1, tot))
            rows_per_file += rws
            text_per_file += txt
            needed = to_host(xp.unique(xp.asarray(
                np.concatenate([needed, loci])))) if loci.size else needed
            del loci
            _free_device()
            nb += 1
            print(f"[CPU ] locus pass block {nb}: {j - i} models, "
                  f"{sum(txt)/1e9:.1f} GB text, {sum(rws):,} rows -> "
                  f"{needed.size:,} loci  ({time.time() - t:.1f}s)", flush=True)
            i = j
        timings.append(("1_model_loading", time.time() - t))
        timings.append(("2_model_locus_set", 0.0))
    print(f"[CPU ] model loci: {needed.size:,}", flush=True)

    with ThreadPoolExecutor(min(WORKERS, max(1, n_samp))) as ex:
        sample_ids = list(ex.map(_sample_id_of, vcf_paths))
    meta, had_meta = read_panel_meta(PANEL_META, sample_ids)
    if REFERENCE_SET == 'unrelated' and had_meta:
        ref_mask = meta['relationship'] == 'unrelated'
    else:
        ref_mask = np.ones(n_samp, bool)
    if ref_mask.sum() < 2:
        print("[WARN] fewer than 2 unrelated samples — reference set = all",
              flush=True)
        ref_mask = np.ones(n_samp, bool)
    print(f"[CPU ] reference set: {int(ref_mask.sum())}/{n_samp} samples"
          + ("" if had_meta else f"  (no panel metadata at {PANEL_META})"),
          flush=True)

    # ── 3: the VCFs, in two streamed passes — or reused from the cache ──────
    cache_hit = False
    dosage_path = None
    if DOSAGE_CACHE:
        os.makedirs(DOSAGE_CACHE, exist_ok=True)
        ck = _dosage_cache_key(vcf_paths, needed)
        cmeta = os.path.join(DOSAGE_CACHE, 'meta.json')
        if os.path.exists(cmeta):
            with open(cmeta) as f:
                cj = json.load(f)
            cache_hit = (cj.get('key') == ck and cj.get('complete') and
                         cj.get('sample_ids') == sample_ids)
            if not cache_hit:
                print("[CACHE] dosage cache present but stale — rebuilding",
                      flush=True)
    if cache_hit:
        t = time.time()
        z = np.load(os.path.join(DOSAGE_CACHE, 'targets.npz'), allow_pickle=False)
        snp_tgt, tgt_ord = xp.asarray(z['snp_tgt']), xp.asarray(z['tgt_ord'])
        with open(os.path.join(DOSAGE_CACHE, 'oth_tgt.json')) as f:
            oth_tgt = json.load(f)
        n_snp = int(snp_tgt.size)
        oth_index = {k: n_snp + j for j, k in enumerate(oth_tgt)}
        n_var = n_snp + len(oth_tgt)
        D_T, dosage_path = _open_dosage(
            n_samp, n_var, os.path.join(DOSAGE_CACHE, 'dosage.i8'), mode='r')
        timings.append(("3_vcf_parse_dosage_matrix", 0.0))
        print(f"[CACHE] reused {n_var:,} targets and the dosage matrix "
              f"({time.time() - t:.1f}s)", flush=True)
    else:
        t = time.time()
        snp_tgt, oth_tgt, oth_index, tgt_ord, ranges = scan_targets(
            vcf_paths, needed, WORKERS)
        n_snp = int(snp_tgt.size)
        n_var = n_snp + len(oth_tgt)
        dt_a = time.time() - t
        print(f"[DISK] target variants: {n_var:,} ({n_snp:,} SNP + "
              f"{len(oth_tgt):,} other)  ({dt_a:.2f}s for the target pass)",
              flush=True)
        t = time.time()
        if DOSAGE_CACHE:
            # marked incomplete BEFORE the fill: a run killed halfway through
            # pass B must not leave a cache that the next run trusts
            with open(os.path.join(DOSAGE_CACHE, 'meta.json'), 'w') as f:
                json.dump(dict(key='', complete=False), f)
        D_T, dosage_path = _open_dosage(
            n_samp, n_var,
            os.path.join(DOSAGE_CACHE, 'dosage.i8') if DOSAGE_CACHE else None)
        fill_dosage(vcf_paths, needed, WORKERS, snp_tgt, oth_index, n_var,
                    D_T, ranges)
        dt_b = time.time() - t
        timings += [("3_vcf_parse_dosage_matrix", dt_a + dt_b),
                    ("3a_target_pass", dt_a), ("3b_dosage_pass", dt_b)]
        print(f"[DISK] dosage matrix filled  ({dt_b:.2f}s)", flush=True)
        if DOSAGE_CACHE:
            np.savez(os.path.join(DOSAGE_CACHE, 'targets.npz'),
                     snp_tgt=to_host(snp_tgt), tgt_ord=to_host(tgt_ord))
            with open(os.path.join(DOSAGE_CACHE, 'oth_tgt.json'), 'w') as f:
                json.dump(list(oth_tgt), f)
            with open(os.path.join(DOSAGE_CACHE, 'meta.json'), 'w') as f:
                json.dump(dict(key=_dosage_cache_key(vcf_paths, needed),
                               complete=True, n_samp=n_samp, n_var=n_var,
                               n_snp=n_snp, sample_ids=sample_ids), f)
            print(f"[CACHE] dosage matrix and targets kept in {DOSAGE_CACHE}",
                  flush=True)
    global _BATCH_DEV
    _BATCH_DEV = None
    _free_device()
    return _run_scoring(vcf_paths, pgs_paths, sample_ids, meta, ref_mask,
                        snp_tgt, oth_tgt, oth_index, tgt_ord, n_snp, n_var,
                        D_T, dosage_path, single, models_all, rows_per_file,
                        text_per_file, gz, timings, needed)


def _run_scoring(vcf_paths, pgs_paths, sample_ids, meta, ref_mask,
                 snp_tgt, oth_tgt, oth_index, tgt_ord, n_snp, n_var,
                 D_T, dosage_path, single, models_all, rows_per_file,
                 text_per_file, gz, timings, needed):
    n_samp, n_mod = len(sample_ids), len(pgs_paths)
    ck_dir = os.path.join(OUT_DIR, '_checkpoints')
    if RESUME:
        os.makedirs(ck_dir, exist_ok=True)

    # ── S: what the panel says about every target, and the PCA candidates ───
    t = time.time()
    n_cand = int(min(max(4 * PCA_VARIANTS, 1), max(n_snp, 1)))
    cand_idx = (np.unique(np.linspace(0, n_snp - 1, n_cand).astype(np.int64))
                if n_snp else np.zeros(0, np.int64))
    scache = os.path.join(DOSAGE_CACHE, 'panel_stats.npz') if DOSAGE_CACHE else ''
    if scache and os.path.exists(scache):
        z = np.load(scache)
        if z['obs_ref'].size == n_var and z['cand_idx'].size == cand_idx.size:
            stats = dict(obs_ref=xp.asarray(z['obs_ref']),
                         alt_ref=xp.asarray(z['alt_ref']),
                         obs_all=xp.asarray(z['obs_all']), G=z['G'],
                         n_ref=int(ref_mask.sum()))
            stats['call_rate'] = stats['obs_ref'].astype(xp.float64) / max(
                1, stats['n_ref'])
            stats['p_alt'] = xp.where(
                stats['obs_ref'] > 0, stats['alt_ref'].astype(xp.float64)
                / xp.maximum(2 * stats['obs_ref'], 1), 0.0)
            cand_idx = z['cand_idx']
            print(f"[CACHE] reused panel target statistics", flush=True)
        else:
            print("[CACHE] panel statistics stale — recomputing", flush=True)
            stats = None
    else:
        stats = None
    if stats is None:
        stats = panel_stats(D_T, ref_mask, cand_idx)
        if scache:
            np.savez(scache, obs_ref=to_host(stats['obs_ref']),
                     alt_ref=to_host(stats['alt_ref']),
                     obs_all=to_host(stats['obs_all']),
                     G=stats['G'], cand_idx=cand_idx)
    timings.append(("S_panel_target_stats", time.time() - t))
    call_rate_d, p_alt_d = stats['call_rate'], stats['p_alt']
    elig = call_rate_d >= MIN_CALL_RATE
    n_elig = int(elig.sum())
    # Does anything in the eligible target set actually miss a call?  If not,
    # `Wfill @ miss` is a matmul against zero and `A @ obs` is the constant row
    # count, and skipping both is exact, not an approximation.
    need_imp = bool(int((stats['obs_all'][elig] < n_samp).sum()) > 0) if n_elig else False
    print(f"[STAT] targets {n_var:,};  call rate >= {MIN_CALL_RATE}: {n_elig:,} "
          f"({100.0*n_elig/max(1,n_var):.2f}%);  mean call rate "
          f"{float(call_rate_d.mean()):.6f};  imputation path: "
          f"{'ON' if need_imp else 'OFF (panel has no missing call)'}", flush=True)

    # ── P: the genotype PCs ────────────────────────────────────────────────
    t = time.time()
    cand_key = to_host(snp_tgt)[cand_idx] if n_snp else np.zeros(0, np.int64)
    pcs = build_pca(stats['G'], cand_idx, cand_key, stats, ref_mask, N_PCS) \
        if stats['G'] is not None else None
    stats['G'] = None
    timings.append(("P_genotype_pca", time.time() - t))
    _free_device()

    # ── the panel-level artefacts, written before anything can still fail ──
    t = time.time()
    write_panel_targets(os.path.join(OUT_DIR, 'panel_targets.tsv.gz'),
                        snp_tgt, oth_tgt, stats, n_samp)
    npc = pcs['scores'].shape[1] if pcs else 0
    with open(os.path.join(OUT_DIR, 'panel_samples.tsv'), 'w') as f:
        f.write("sample_id\tsex\tpopulation\tsuper_population\trelationship\t"
                "in_reference_set" + "".join(f"\tPC{k+1}" for k in range(npc)) + "\n")
        for i, s in enumerate(sample_ids):
            f.write(f"{s}\t{meta['sex'][i]}\t{meta['population'][i]}\t"
                    f"{meta['super_population'][i]}\t{meta['relationship'][i]}\t"
                    f"{int(ref_mask[i])}"
                    + "".join(f"\t{pcs['scores'][i, k]:.6f}" for k in range(npc))
                    + "\n")
    if pcs:
        np.savez_compressed(os.path.join(OUT_DIR, 'pca_loadings.npz'),
                            keys=pcs['keys'], var_index=pcs['var_index'],
                            p=pcs['p'], sd=pcs['sd'], loadings=pcs['loadings'],
                            eigenvalues=pcs['eigenvalues'])
    timings.append(("O_panel_artifacts", time.time() - t))

    # ── 4/5/6: match, weight matrices, manifest and scoring, per model block ─
    blocks = [pgs_paths] if single else _model_blocks(
        pgs_paths, text_per_file, rows_per_file)
    tile = score_tile(n_var, need_imp)
    print(f"[PLAN] {len(blocks)} model block(s), scoring tile {tile} sample(s) "
          f"({(26 if need_imp else 18) * n_var * tile / 1e9:.1f} GB per tile)",
          flush=True)
    scores = np.empty((n_samp, n_mod), np.float64)
    nobs = np.zeros((n_samp, n_mod), np.int32)
    s_size = np.zeros(n_mod, np.int64)
    n_dropped = np.zeros(n_mod, np.int64)
    matched = np.zeros(n_mod, np.int64)
    pgs_ids = [None] * n_mod
    man_index = []
    t4 = t5 = t6 = t7 = reload_s = 0.0
    col0 = 0
    for bi, blk in enumerate(blocks):
        nb = len(blk)
        store = os.path.join(OUT_DIR, 'manifest', f'blk_{bi:03d}.store')
        ckf = os.path.join(ck_dir, f'blk_{bi:03d}.npz')
        if RESUME and os.path.exists(ckf):
            z = np.load(ckf, allow_pickle=True)
            if list(z['ids']) == [os.path.basename(p).split('.')[0] for p in blk]:
                sl = slice(col0, col0 + nb)
                scores[:, sl] = z['scores']
                nobs[:, sl] = z['nobs']
                s_size[sl] = z['s_size']
                n_dropped[sl] = z['n_dropped']
                matched[sl] = z['matched']
                pgs_ids[col0:col0 + nb] = list(z['ids'])
                man_index += [tuple(r) for r in z['man_index'].tolist()]
                col0 += nb
                print(f"[CKPT] block {bi+1}/{len(blocks)} reused", flush=True)
                continue

        if single:
            models = models_all
        else:
            tt = time.time()
            models = read_models(blk, WORKERS)
            reload_s += time.time() - tt

        tt = time.time()
        b_ids, b_matched, rows, cols, vals, flips, mtypes, effs = match_models(
            models, snp_tgt, tgt_ord, oth_tgt, oth_index, n_var)
        t4 += time.time() - tt
        n_raw = int(rows.size)

        # ── §1: S is the matched loci the PANEL calls well, and only those ──
        keep = call_rate_d[cols] >= MIN_CALL_RATE
        drop = xp.zeros(nb, xp.int64)
        if n_raw:
            _scatter_add(drop, rows[~keep], 1)
            rows, cols, vals = rows[keep], cols[keep], vals[keep]
            flips, mtypes, effs = flips[keep], mtypes[keep], effs[keep]
        sz = xp.zeros(nb, xp.int64)
        if int(rows.size):
            _scatter_add(sz, rows, 1)
        # §3: the effect-allele frequency, in the orientation the model uses
        p_eff = xp.where(flips == 1, 1.0 - p_alt_d[cols], p_alt_d[cols]) \
            if int(cols.size) else xp.zeros(0, xp.float64)

        # ── §3: freeze it, before the device arrays are dropped ────────────
        tt = time.time()
        ar = xp.arange(nb, dtype=xp.int64)
        starts = to_host(xp.searchsorted(rows, ar, side='left'))
        ends = to_host(xp.searchsorted(rows, ar, side='right'))
        if WRITE_MANIFEST:
            with open(store, 'wb') as fh:
                for k, pid in enumerate(b_ids):
                    a, b = int(starts[k]), int(ends[k])
                    fl = to_host(flips[a:b]).astype(np.int8) | \
                        (to_host(mtypes[a:b]).astype(np.int8) << 1)
                    blob = manifest_blob(
                        pid, to_host(cols[a:b]).astype(np.int32),
                        to_host(vals[a:b]), fl, to_host(effs[a:b]).astype(np.int8),
                        to_host(p_eff[a:b]).astype(np.float32))
                    off = fh.tell()
                    fh.write(blob)
                    man_index.append((pid, f'blk_{bi:03d}.store', off,
                                      len(blob), b - a))
        else:
            for k, pid in enumerate(b_ids):
                man_index.append((pid, '', 0, 0, int(ends[k]) - int(starts[k])))
        t7 += time.time() - tt
        del effs

        tt = time.time()
        Wd, Wo, Wfill, A, nonadd = build_matrices(
            nb, n_var, rows, cols, vals, flips, mtypes, p_eff, need_imp)
        del rows, cols, vals, flips, mtypes, p_eff
        t5 += time.time() - tt

        sz_h = to_host(sz).astype(np.int32)
        tt = time.time()
        score_gpu(Wd, Wo, Wfill, A, nonadd, D_T, tile, scores, nobs, col0, sz_h)
        t6 += time.time() - tt

        sl = slice(col0, col0 + nb)
        pgs_ids[col0:col0 + nb] = b_ids
        matched[sl] = b_matched
        s_size[sl] = to_host(sz)
        n_dropped[sl] = to_host(drop)
        print(f"[{'GPU ' if GPU else 'CPU '}] block {bi+1}/{len(blocks)}: {nb} "
              f"models, {n_raw:,} matched, {int(sz.sum()):,} in S, "
              f"{int(drop.sum()):,} dropped on call rate, Wd nnz={Wd.nnz:,} "
              f"nonadditive={nonadd['pgs'].size}  (match {t4:.1f}s build {t5:.1f}s "
              f"manifest {t7:.1f}s score {t6:.1f}s cumulative)", flush=True)
        del Wd, Wo, Wfill, A, nonadd, drop, sz
        if not single:
            del models
        if RESUME:
            np.savez(ckf, ids=np.asarray(b_ids, dtype=object),
                     scores=scores[:, sl], nobs=nobs[:, sl], s_size=s_size[sl],
                     n_dropped=n_dropped[sl], matched=matched[sl],
                     man_index=np.asarray(man_index[-nb:], dtype=object))
        col0 += nb
        _free_device()

    if not single:
        timings.append(("4a_model_reload", reload_s))
    timings += [("4_variant_matching", t4), ("5_weight_matrix_build", t5),
                ("6_score_calculation", t6), ("7_frozen_manifest", t7)]

    try:
        D_T._mmap.close()
    except Exception:
        pass
    del D_T
    if dosage_path and not DOSAGE_CACHE:
        try:
            os.unlink(dosage_path)
        except OSError:
            pass

    # ── O: the distributions ───────────────────────────────────────────────
    t = time.time()
    n_rows = np.asarray(rows_per_file, np.int64)
    mstats, zr, zp = model_statistics(scores, nobs, s_size, n_rows, matched,
                                      n_dropped, ref_mask, pcs,
                                      meta['super_population'])
    for j, rec in enumerate(mstats):
        rec['pgs_id'] = pgs_ids[j]
    timings.append(("8_model_statistics", time.time() - t))

    t = time.time()
    valid = s_size > 0
    write_output(os.path.join(OUT_DIR, 'scores_raw.tsv'), sample_ids, pgs_ids,
                 scores, valid)
    write_matrix(os.path.join(OUT_DIR, 'scores_z.tsv'), sample_ids, pgs_ids,
                 zr, valid)
    write_matrix(os.path.join(OUT_DIR, 'scores_z_pc.tsv'), sample_ids, pgs_ids,
                 zp, valid)
    write_matrix(os.path.join(OUT_DIR, 'scores_nobs.tsv'), sample_ids, pgs_ids,
                 nobs.astype(np.float64), valid, fmt="%d")
    for nm, arr in (('scores_raw', scores), ('scores_z', zr),
                    ('scores_z_pc', zp), ('scores_nobs', nobs)):
        np.save(os.path.join(OUT_DIR, nm + '.npy'), arr)
    with open(os.path.join(OUT_DIR, 'model_manifest.index.tsv'), 'w') as f:
        f.write("pgs_id\tstore\toffset\tlength\tn_loci\n")
        for r in man_index:
            f.write("\t".join(str(x) for x in r) + "\n")

    qcols = [str(q) for q in _QUANTILES]
    with open(os.path.join(OUT_DIR, 'model_index.tsv'), 'w') as f:
        f.write("pgs_id\tn_rows\tn_matched\tmatch_rate\tn_dropped_low_call_rate\t"
                "n_S\tcoverage\tdeployable\tpc_adjusted\treason\tn_reference\t"
                "n_sample_flagged\t"
                "mu\tsigma\tsigma_resid\tr2_pc\tsuperpop_mu_spread_sigma\tmin\tmax\t"
                + "\t".join("q" + q for q in qcols) + "\n")
        for rec in mstats:
            g = lambda k, d='': rec.get(k, d)
            q = rec.get('quantiles', {})
            f.write("\t".join([
                str(rec['pgs_id']), str(rec['n_rows']), str(rec['n_matched']),
                f"{rec['match_rate']:.6f}", str(rec['n_dropped_low_call_rate']),
                str(rec['n_S']), f"{rec['coverage']:.6f}",
                str(int(bool(g('deployable', False)))),
                str(int(bool(g('pc_adjusted', False)))), str(g('reason')),
                str(g('n_reference', 0)), str(g('n_sample_flagged', 0)),
                *(f"{g(k, float('nan')):.6g}" for k in
                  ('mu', 'sigma', 'sigma_resid', 'r2_pc',
                   'superpop_mu_spread_in_sigma', 'min', 'max')),
                *(f"{q[c]:.6g}" if c in q else "NA" for c in qcols)]) + "\n")
    _gz_text(os.path.join(OUT_DIR, 'model_stats.json.gz'),
             [json.dumps(mstats, allow_nan=True)])
    timings.append(("9_write_reference", time.time() - t))

    dep = sum(1 for r in mstats if r.get('deployable'))
    cov = np.array([r['coverage'] for r in mstats])
    spread = np.array([r.get('superpop_mu_spread_in_sigma', np.nan)
                       for r in mstats], float)
    total = time.time() - _T0
    timings.append(("TOTAL_wall_clock_input_to_output", total))

    cr_min = float(call_rate_d.min()) if n_var else 1.0
    if need_imp:
        first = f"""\
* **This panel does have missing calls, and they are mean-imputed.** The
  minimum per-target call rate over the reference set is {cr_min:.4f};
  {n_var - n_elig:,} of {n_var:,} targets fall below the {MIN_CALL_RATE}
  threshold and are excluded from every model's S, and the residual
  missingness inside S is imputed with the frozen `p_eff`. Per-sample
  coverage in `scores_nobs.tsv` is the number to check before trusting any
  individual score.
"""
    else:
        first = f"""\
* **This panel has no missing genotypes at all.** It is a joint callset and
  every sample carries every site: measured here, the minimum per-target call
  rate over the reference set is {cr_min:.4f} and all {n_elig:,} of
  {n_var:,} targets clear the {MIN_CALL_RATE} threshold, so the
  mean-imputation path never fires. That is a property of THIS panel, not of
  the method: the imputation and the per-sample coverage counters are
  implemented, tested and frozen precisely because an array or an exome
  callset will need them. `coverage = |S|/N` therefore reduces to the model's
  match rate on this panel — which is the number that actually varies, from 0
  to 1.
"""
    caveats = first + f"""\
* **Absent means absent, not uncalled (§6).** These are per-sample slices of a
  joint callset, so 0/0 is explicit in every file and a locus missing from a
  sample's VCF is a locus that is not in the callset at all. No callable-region
  definition is needed or assumed. **Do not reuse this bundle's `p_eff` against
  variant-only per-sample VCFs** without a callable-region definition: there,
  absent is ambiguous and step 2 would impute loci that were really hom-ref.
* **Male chrX is hemizygous and is read as the homozygous diploid call**, which
  is what pgsc_calc does; chrY is not in this release at all, so chrY model
  rows never match.
* {int((~ref_mask).sum())} of {n_samp} samples are outside the reference set
  (related individuals, per the cohort manifest; `--reference-set all`
  overrides). They are still scored, and are a free check that the frozen
  manifest transfers to an individual who did not help define it.
* {n_mod - dep} of {n_mod} models are marked NOT deployable at coverage < {MIN_COVERAGE}
  (median coverage {np.median(cov):.3f}). Their scores are in the matrices and
  their statistics are in `model_index.tsv`; they are flagged, not dropped.
* The PCs are thinned by physical distance ({PCA_SPACING:,} bp), not LD-pruned.
  That is adequate for the global-ancestry axes used here and nothing else.
"""
    with open(os.path.join(OUT_DIR, 'README.md'), 'w') as f:
        f.write(README.format(
            vcf_dir=VCF_DIR, pgs_dir=PGS_DIR, n_samp=n_samp, n_models=n_mod,
            n_var=n_var, min_call_rate=MIN_CALL_RATE, min_coverage=MIN_COVERAGE,
            min_sample_call_rate=MIN_SAMPLE_CALL_RATE, n_pcs=npc,
            caveats=caveats))
    with open(os.path.join(OUT_DIR, 'build_info.json'), 'w') as f:
        json.dump(dict(
            built=time.strftime('%Y-%m-%d %H:%M:%S'), script=os.path.basename(__file__),
            vcf_dir=VCF_DIR, pgs_dir=PGS_DIR, argv=sys.argv[1:],
            n_samples=n_samp, n_models=n_mod, n_targets=n_var, n_snp_targets=n_snp,
            n_model_loci=int(needed.size), n_eligible_targets=n_elig,
            imputation_path_active=need_imp, reference_set=REFERENCE_SET,
            n_reference_samples=int(ref_mask.sum()),
            min_call_rate=MIN_CALL_RATE, min_sample_call_rate=MIN_SAMPLE_CALL_RATE,
            min_coverage=MIN_COVERAGE, n_pcs=npc,
            pca_variants=(pcs['n_variants'] if pcs else 0),
            pca_maf=PCA_MAF, pca_spacing=PCA_SPACING,
            mem_budget_gb=MEM_BUDGET / (1 << 30), score_tile=tile,
            model_blocks=len(blocks), backend='GPU' if GPU else 'CPU',
            sample_ids=sample_ids, pgs_ids=pgs_ids,
            n_deployable=dep, timings={k: v for k, v in timings}), f, indent=1)

    print(f"\n[DONE] {n_mod} models, {dep} deployable, median coverage "
          f"{np.median(cov):.3f};  super-population mean spread "
          f"{np.nanmedian(spread):.2f} sigma (median)", flush=True)
    print("\n--- stage timings (s) ---")
    for stage, secs in timings:
        print(f"{stage}\t{secs:.3f}")
    print(f"\nTOTAL WALL CLOCK: {total:.3f} s")
    if REF_TSV:
        verify(os.path.join(OUT_DIR, 'scores_raw.tsv'), REF_TSV)
    return 0


if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    sys.exit(main())

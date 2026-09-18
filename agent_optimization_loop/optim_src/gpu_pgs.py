#!/usr/bin/env python3
"""
gpgs_cohort.py — cohort-scale polygenic scoring on the GPU.

INPUT   --vcf-dir   : directory of N per-sample VCFs (.vcf / .vcf.gz), AllofUs-like
        --pgs-dir   : directory of M PGS scoring files (.tsv/.txt[.gz])
OUTPUT  --output    : an [N_samples x M_pgs] score matrix (.npy + .tsv header/index)

Core idea (why this is fast):
  score = W @ D
    W : sparse [M x |V|]  weight matrix  (CSR — a 20-variant PGS costs 20 nnz, not |V|)
    D : dense  [|V| x N]  dosage matrix   (only variants used by >=1 PGS -> |V| is small)
  The per-variant binary search that the reference repo repeats per (patient,PGS)
  is done ONCE here, on the host, and cached as CSR column indices. The GPU then
  does nothing but coalesced multiply-accumulate via cuSPARSE csrmm.

Backends:
  * GPU present  -> cupy + cupyx.scipy.sparse (cuSPARSE SpMM) + a custom RawKernel
                    for the rare non-additive (dominant/recessive) variants.
  * No GPU       -> numpy + scipy.sparse fallback (same math), so this file runs
                    and self-tests anywhere. Swap happens automatically below.

Legend used in comments:
  [DISK->RAM]  file read (CPU/IO bound)
  [CPU]        host compute (single- or multi-core, CPU bound)
  [H2D]        host RAM -> GPU VRAM copy (PCIe/NVLink transfer)
  [GPU]        device compute (cuSPARSE / custom kernel)
  [D2H]        GPU VRAM -> host RAM copy
"""

import os, sys, gzip, glob, time, argparse, multiprocessing as mp
import numpy as np
import scipy.sparse as sp

# ─────────────────────────────────────────────────────────────────────────────
# BACKEND SELECTION  (GPU if available, else CPU — mirrors the repo's pattern)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import cupy as cp
    import cupyx.scipy.sparse as cusp
    GPU = True
except Exception:
    cp = np
    cusp = None
    GPU = False

# ── Custom CUDA kernel: non-additive (dominant/recessive) contributions ───────
# cuSPARSE handles the additive bulk (~all of the PGS Catalog). Dominant/recessive
# are nonlinear in dosage, so they can't live in the linear W. This kernel runs
# on the SAME dosage tile already resident in VRAM: one block per non-additive
# entry, threads stride the cohort, atomicAdd into the score tile.  [GPU]
NONADD_KERNEL_SRC = r'''
extern "C" __global__
void apply_nonadditive(
    const float* dosage_tile,   // [n_var * tile_S] variant-major, already on GPU
    const int    tile_S,
    const int*   pgs_row,       // [n_entry]
    const int*   var_row,       // [n_entry]
    const float* weight,        // [n_entry]
    const signed char* model,   // [n_entry] 1=dominant 2=recessive
    const signed char* flip,    // [n_entry] 1=effect allele is the non-designated one
    const int    n_entry,
    float* scores)              // [n_pgs * tile_S] row-major, accumulate
{
    int e = blockIdx.x;                          // one block per non-additive entry
    if (e >= n_entry) return;
    long long vr = var_row[e];
    float w = weight[e];
    signed char md = model[e], fl = flip[e];
    const float* drow = dosage_tile + vr * tile_S;
    float*       orow = scores + (long long)pgs_row[e] * tile_S;
    for (int s = threadIdx.x; s < tile_S; s += blockDim.x) {   // stride cohort
        float d = drow[s];
        if (d != d) continue;                    // per-sample missing (NaN) skip
        if (fl == 1) d = 2.0f - d;               // orient to effect allele
        if (md == 1) d = fminf(d, 1.0f);         // dominant
        else if (md == 2) d = (d >= 2.0f) ? 1.0f : 0.0f;   // recessive
        atomicAdd(&orow[s], d * w);
    }
}
'''
_NONADD_KERNEL = cp.RawKernel(NONADD_KERNEL_SRC, 'apply_nonadditive') if GPU else None


# ─────────────────────────────────────────────────────────────────────────────
# CANONICAL VARIANT KEY  (orientation-independent, biallelic)
# Designated allele = lexicographically larger of the two. D stores the dosage
# of the designated allele, so flip is decided purely from the PGS effect allele
# and never depends on which allele the VCF happened to call REF vs ALT.
# ─────────────────────────────────────────────────────────────────────────────
def canon_key(chrom, pos, a1, a2):
    a1, a2 = a1.upper(), a2.upper()
    lo, hi = (a1, a2) if a1 < a2 else (a2, a1)
    chrom = chrom[3:] if chrom.startswith('chr') else chrom
    return f"{chrom}:{pos}:{lo}:{hi}"          # production: pack into uint64 (repo FNV1a)

def designated_allele(a1, a2):
    a1, a2 = a1.upper(), a2.upper()
    return a1 if a1 > a2 else a2


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1  —  READ PGS FILES  [DISK->RAM] + [CPU]
# Build the UNION variant index (only variants any PGS needs) and, per PGS, the
# list of (variant_row, weight, flip, model). This is where the one-time "search"
# structure is created; its result becomes CSR column indices later.
# ─────────────────────────────────────────────────────────────────────────────
def read_pgs_file(path):
    """Parse one PGS scoring file. Returns list of (chrom,pos,effect,other,weight,model)."""
    opener = gzip.open if path.endswith('.gz') else open
    rows, header, cols = [], None, {}
    with opener(path, 'rt') as f:                         # [DISK->RAM]
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.rstrip('\n').split('\t')
            if header is None:
                header = [p.strip().lower() for p in parts]
                cols = {name: i for i, name in enumerate(header)}
                continue
            def get(*names, default=None):
                for n in names:
                    if n in cols and cols[n] < len(parts):
                        v = parts[cols[n]].strip()
                        if v:                     # treat empty cells as missing -> try next
                            return v
                return default
            # Benchmark VCFs are GRCh38. PGS Catalog files carry harmonized GRCh38
            # coordinates in hm_chr/hm_pos, while chr_name/chr_position may sit on the
            # original (often GRCh37) build. Prefer the harmonized coords so variants
            # line up with the VCF; fall back to the native columns only if absent.
            chrom = get('hm_chr', 'chr_name', 'chromosome', 'chr')
            pos   = get('hm_pos', 'chr_position', 'position', 'pos')
            eff   = get('effect_allele', 'ea')
            # other_allele is optional in PGS Catalog files; harmonization supplies
            # hm_inferOtherAllele when it is missing.
            oth   = get('other_allele', 'oa', 'noneffect_allele',
                        'hm_inferotherallele', default='')
            if '/' in oth:            # ambiguous inferred other allele -> drop (skip below)
                oth = ''
            w     = get('effect_weight', 'weight', 'beta')
            if not (chrom and pos and eff and w):
                continue
            is_dom = (get('is_dominant', default='') or '').lower() in ('true', '1')
            is_rec = (get('is_recessive', default='') or '').lower() in ('true', '1')
            model = 1 if is_dom else (2 if is_rec else 0)
            try:
                rows.append((str(chrom), int(pos), eff.upper(), oth.upper(),
                             float(w), model))
            except ValueError:
                continue
    return rows

def build_pgs_index(pgs_paths):
    """[CPU] Union index + per-PGS entries. Sorts PGS by size for load balance."""
    union = {}                                            # canon_key -> row index in D
    pgs_entries = []                                      # per PGS: list of (row,weight,flip,model)
    pgs_ids = []
    for path in pgs_paths:
        pid = os.path.basename(path).split('.')[0]
        rows = read_pgs_file(path)
        entries = []
        for chrom, pos, eff, oth, w, model in rows:
            if not oth:                                   # infer other allele if missing (rare)
                continue
            k = canon_key(chrom, pos, eff, oth)
            r = union.get(k)
            if r is None:
                r = len(union); union[k] = r
            flip = 0 if eff == designated_allele(eff, oth) else 1
            entries.append((r, w, flip, model))
        if entries:
            pgs_ids.append(pid)
            pgs_entries.append(entries)

    # ── Engineering: order PGS rows by size (nnz) ─────────────────────────────
    # A combined matrix mixes 20-variant and 10^6-variant models. Ordering rows
    # by nnz (a) balances work if a custom row-kernel/multi-GPU row-sharding is
    # used, and (b) groups similar rows for better cache behavior. CSR already
    # makes tiny rows cheap; this just makes the *schedule* even.  [CPU]
    order = sorted(range(len(pgs_ids)), key=lambda i: len(pgs_entries[i]), reverse=True)
    pgs_ids     = [pgs_ids[i]     for i in order]
    pgs_entries = [pgs_entries[i] for i in order]
    return union, pgs_ids, pgs_entries


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2  —  BUILD SPARSE WEIGHT MATRIX W  [CPU]
# Additive + flip fold into one CSR: flipped weight w contributes w*(2-d) =
# 2w - w*d, so the matrix entry on d is -w and 2w goes into a per-PGS constant.
# Non-additive entries are peeled off into COO arrays for the custom kernel.
# ─────────────────────────────────────────────────────────────────────────────
def build_weight_matrix(union, pgs_entries):
    n_pgs, n_var = len(pgs_entries), len(union)
    rows, cols, vals = [], [], []
    const = np.zeros(n_pgs, dtype=np.float64)             # flip constants (additive only)
    na_pgs, na_var, na_w, na_model, na_flip = [], [], [], [], []
    for p, entries in enumerate(pgs_entries):
        for (r, w, flip, model) in entries:
            if model == 0:                                # additive -> linear W
                rows.append(p); cols.append(r)
                vals.append(-w if flip == 1 else w)
                if flip == 1:
                    const[p] += 2.0 * w
            else:                                         # dominant/recessive -> kernel
                na_pgs.append(p); na_var.append(r)
                na_w.append(w); na_model.append(model); na_flip.append(flip)
    W = sp.csr_matrix((np.asarray(vals, np.float32),
                       (np.asarray(rows), np.asarray(cols))),
                      shape=(n_pgs, n_var))
    W.sort_indices()      # cuSPARSE prefers sorted column indices; improves D locality
    nonadd = dict(pgs=np.asarray(na_pgs, np.int32),   var=np.asarray(na_var, np.int32),
                  w=np.asarray(na_w, np.float32),      model=np.asarray(na_model, np.int8),
                  flip=np.asarray(na_flip, np.int8))
    return W, const, nonadd


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3  —  READ VCFs -> DOSAGE MATRIX D  [DISK->RAM] + [CPU, multi-core]
# Only union variants are stored. Absent/missing -> 0 (repo behavior; production
# would mean-impute or use a joint callset to distinguish ref from no-call).
# Each worker returns one column (one sample); parallel across the N files.
# ─────────────────────────────────────────────────────────────────────────────
def _read_one_vcf(args):
    path, union = args
    opener = gzip.open if path.endswith('.gz') else open
    col = np.zeros(len(union), dtype=np.int8)             # dense column, missing->0
    sample_id = os.path.basename(path).split('.')[0]
    with opener(path, 'rt') as f:                         # [DISK->RAM]
        for line in f:
            if line.startswith('##'):
                continue
            if line.startswith('#CHROM'):
                c = line.rstrip('\n').split('\t')
                if len(c) > 9: sample_id = c[9]
                continue
            c = line.rstrip('\n').split('\t')
            if len(c) < 10:
                continue
            chrom, pos, ref, alt, gt = c[0], c[1], c[3].upper(), c[4].upper(), c[9]
            if ',' in alt:                                # skip multiallelic for clarity
                continue
            k = canon_key(chrom, pos, ref, alt)
            r = union.get(k)
            if r is None:
                continue
            g = gt.split(':')[0].replace('|', '/')
            if '.' in g:
                continue
            try:
                alt_dosage = sum(1 for a in g.split('/') if int(a) > 0)
            except ValueError:
                continue
            # canonicalize to designated allele's dosage
            desig = designated_allele(ref, alt)
            col[r] = alt_dosage if alt == desig else (2 - alt_dosage)
    return sample_id, col

def read_dosage_matrix(vcf_paths, union, workers):
    n_var = len(union)
    D = np.zeros((n_var, len(vcf_paths)), dtype=np.int8)  # [|V| x N] host, int8 (4x smaller)
    sample_ids = [None] * len(vcf_paths)
    tasks = [(p, union) for p in vcf_paths]
    # [CPU, multi-core] — VCF parsing is the real bottleneck; parallelize across files.
    with mp.Pool(workers) as pool:
        for i, (sid, col) in enumerate(pool.imap(_read_one_vcf, tasks)):
            D[:, i] = col
            sample_ids[i] = sid
    return D, sample_ids


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4  —  GPU SCORING  (tiled over samples)  [H2D] + [GPU] + [D2H]
# W stays resident on the GPU (small: nnz ≈ total variants across all PGS).
# D is streamed one sample-tile at a time so any cohort/|V| fits in VRAM.
# ─────────────────────────────────────────────────────────────────────────────
def score_gpu(W, const, nonadd, D, tile_size):
    n_pgs, n_var = W.shape
    n_samp = D.shape[1]
    const_v = const.reshape(-1, 1)

    if GPU:
        W_dev = cusp.csr_matrix(W)                        # [H2D] weights -> VRAM (once)
        const_dev = cp.asarray(const_v.astype(np.float32))
        na = {k: cp.asarray(v) for k, v in nonadd.items()}  # [H2D] non-additive lists (once)
        out = cp.empty((n_pgs, n_samp), dtype=cp.float32)
    else:
        W_dev, const_dev, na = W, const_v, nonadd
        out = np.empty((n_pgs, n_samp), dtype=np.float32)

    for c0 in range(0, n_samp, tile_size):                # sample tiling
        c1 = min(c0 + tile_size, n_samp)
        if GPU:
            d_i8 = cp.asarray(D[:, c0:c1])                # [H2D] int8 tile (4x less PCIe)
            d_f  = d_i8.astype(cp.float32)                # [GPU] cast on device
            scores = W_dev @ d_f                          # [GPU] cuSPARSE csrmm  <-- THE kernel
            scores += const_dev                           # [GPU] add flip constants
            if na['pgs'].size:                            # [GPU] custom kernel for dom/rec
                tS = c1 - c0
                _NONADD_KERNEL((int(na['pgs'].size),), (128,),
                    (d_f.ravel(), np.int32(tS), na['pgs'], na['var'],
                     na['w'], na['model'], na['flip'],
                     np.int32(na['pgs'].size), scores.ravel()))
            out[:, c0:c1] = scores                        # stays on GPU until final [D2H]
        else:
            d_f = D[:, c0:c1].astype(np.float32)
            scores = (W_dev @ d_f) + const_dev            # scipy SpMM (CPU mirror)
            for e in range(nonadd['pgs'].size):           # CPU mirror of the kernel
                vr = nonadd['var'][e]; d = d_f[vr].astype(np.float64).copy()
                valid = ~np.isnan(d)
                if nonadd['flip'][e] == 1: d = 2.0 - d
                if   nonadd['model'][e] == 1: d = np.minimum(d, 1.0)
                elif nonadd['model'][e] == 2: d = np.where(d >= 2.0, 1.0, 0.0)
                scores[nonadd['pgs'][e], valid] += (d * nonadd['w'][e])[valid]
            out[:, c0:c1] = scores

    scores_host = cp.asnumpy(out) if GPU else out         # [D2H] final matrix -> host
    return scores_host.T                                  # -> [N_samples x M_pgs]


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────
def run(vcf_dir, pgs_dir, output, tile_size, workers, time_output=None):
    vcf_paths = sorted(glob.glob(os.path.join(vcf_dir, '*.vcf*')))
    pgs_paths = sorted(glob.glob(os.path.join(pgs_dir, '*')))
    backend = 'GPU (cupy/cuSPARSE)' if GPU else 'CPU (numpy/scipy fallback)'
    print(f"backend: {backend}")
    print(f"cohort : {len(vcf_paths)} VCFs   models: {len(pgs_paths)} PGS files")

    timings = []                                                      # (stage, seconds)
    t0 = time.time()

    t = time.time()
    union, pgs_ids, pgs_entries = build_pgs_index(pgs_paths)          # PHASE 1 [DISK+CPU]
    dt = time.time() - t; timings.append(("model_loading_pgs_parse", dt))
    print(f"[CPU ] union variants: {len(union):,}  scored PGS: {len(pgs_ids)}  ({dt:.2f}s)")

    t = time.time()
    W, const, nonadd = build_weight_matrix(union, pgs_entries)        # PHASE 2 [CPU]
    dt = time.time() - t; timings.append(("model_loading_weight_matrix", dt))
    dens = W.nnz / (W.shape[0] * W.shape[1] + 1e-9)
    print(f"[CPU ] W: {W.shape[0]}x{W.shape[1]}  nnz={W.nnz:,}  density={dens:.2e}  "
          f"nonadditive={nonadd['pgs'].size}  ({dt:.2f}s)")

    t = time.time()
    D, sample_ids = read_dosage_matrix(vcf_paths, union, workers)     # PHASE 3 [DISK+CPU]
    dt = time.time() - t; timings.append(("vcf_parsing_dosage_matrix", dt))
    print(f"[DISK] D: {D.shape[0]}x{D.shape[1]} int8 ({D.nbytes/1e6:.1f} MB host)  ({dt:.2f}s)")

    t = time.time()
    scores = score_gpu(W, const, nonadd, D, tile_size)               # PHASE 4 [H2D/GPU/D2H]
    dt = time.time() - t; timings.append(("gpu_score_calculation", dt))
    print(f"[{'GPU ' if GPU else 'CPU '}] scored -> {scores.shape[0]}x{scores.shape[1]}  ({dt:.2f}s)")

    t = time.time()
    np.save(output + '.npy', scores)                                  # [RAM->DISK]
    with open(output + '.tsv', 'w') as f:
        f.write("sample_id\t" + "\t".join(pgs_ids) + "\n")
        for i, sid in enumerate(sample_ids):
            f.write(sid + "\t" + "\t".join(f"{v:.6f}" for v in scores[i]) + "\n")
    dt = time.time() - t; timings.append(("write_output", dt))
    print(f"[DISK] wrote {output}.npy and {output}.tsv")

    timings.append(("total", time.time() - t0))
    if time_output:
        with open(time_output, 'w') as f:
            f.write("stage\tseconds\n")
            for stage, secs in timings:
                f.write(f"{stage}\t{secs:.4f}\n")
        print(f"[DISK] wrote {time_output}")
    return scores, sample_ids, pgs_ids


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST  — generates tiny synthetic cohort+PGS on disk, runs the pipeline,
# checks against a brute-force reference. (Uses small N/M; the script itself is
# size-agnostic via tiling + multiprocessing.)
# ─────────────────────────────────────────────────────────────────────────────
def selftest(tmp='/tmp/gpgs_selftest'):
    rng = np.random.default_rng(7)
    NUC = np.array(list('ACGT'))
    N_SAMP, N_PGS, N_VAR = 40, 25, 300
    vdir, pdir = os.path.join(tmp, 'vcf'), os.path.join(tmp, 'pgs')
    os.makedirs(vdir, exist_ok=True); os.makedirs(pdir, exist_ok=True)

    variants = []
    for i in range(N_VAR):
        a, b = rng.choice(4, 2, replace=False)
        ref, alt = NUC[a], NUC[b]                          # random VCF orientation
        variants.append((str(1 + i % 22), 1000 + i, ref, alt))
    alt_dosage = rng.integers(0, 3, size=(N_VAR, N_SAMP))  # truth genotypes

    for s in range(N_SAMP):                                # write per-sample VCFs
        with open(os.path.join(vdir, f"S{s:03d}.vcf"), 'w') as f:
            f.write("##fileformat=VCFv4.2\n")
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS%03d\n" % s)
            for v in range(N_VAR):
                ch, po, ref, alt = variants[v]
                gt = {0: "0/0", 1: "0/1", 2: "1/1"}[int(alt_dosage[v, s])]
                f.write(f"{ch}\t{po}\t.\t{ref}\t{alt}\t.\t.\t.\tGT\t{gt}\n")

    pgs_defs = []                                          # write PGS files
    for p in range(N_PGS):
        k = rng.integers(10, 60)
        idx = rng.choice(N_VAR, size=k, replace=False)
        with open(os.path.join(pdir, f"PGS{p:04d}.tsv"), 'w') as f:
            f.write("chr_name\tchr_position\teffect_allele\tother_allele\teffect_weight\tis_dominant\tis_recessive\n")
            defs = []
            for v in idx:
                ch, po, ref, alt = variants[v]
                eff = rng.choice([ref, alt])               # random effect orientation
                oth = alt if eff == ref else ref
                w = float(rng.normal())
                m = rng.choice([0, 0, 0, 0, 1, 2])         # mostly additive
                f.write(f"{ch}\t{po}\t{eff}\t{oth}\t{w:.5f}\t"
                        f"{'true' if m==1 else 'false'}\t{'true' if m==2 else 'false'}\n")
                defs.append((v, eff, oth, w, m))
            pgs_defs.append(defs)

    scores, sample_ids, pgs_ids = run(vdir, pdir, os.path.join(tmp, 'out'),
                                      tile_size=16, workers=4)

    # brute-force reference from truth genotypes
    ref = np.zeros((N_SAMP, N_PGS), dtype=np.float64)
    id2p = {pid: j for j, pid in enumerate(pgs_ids)}
    for p in range(N_PGS):
        col = id2p[f"PGS{p:04d}"]
        for (v, eff, oth, w, m) in pgs_defs[p]:
            ch, po, ref_a, alt_a = variants[v]
            desig = designated_allele(ref_a, alt_a)
            counted = alt_dosage[v] if alt_a == desig else 2 - alt_dosage[v]
            d = counted.astype(np.float64).copy()
            if eff != desig: d = 2.0 - d                   # flip to effect allele
            if   m == 1: d = np.minimum(d, 1.0)
            elif m == 2: d = np.where(d >= 2.0, 1.0, 0.0)
            ref[:, col] += d * w
    sid2i = {sid: i for i, sid in enumerate(sample_ids)}
    ref_sorted = ref[[sid2i[f"S{s:03d}"] for s in range(N_SAMP)]]  # align sample order
    got = scores[[sid2i[f"S{s:03d}"] for s in range(N_SAMP)]]
    err = np.max(np.abs(ref_sorted - got))
    print(f"\nself-test max abs error vs brute force: {err:.3e}")
    print("PASS ✓" if err < 1e-4 else "FAIL ✗")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Cohort-scale GPU PGS scoring (W @ D)")
    ap.add_argument('--vcf-dir'); ap.add_argument('--pgs-dir')
    ap.add_argument('--output', default='cohort_scores')
    ap.add_argument('--time-output', default=None, help='per-stage timing TSV path')
    ap.add_argument('--tile-size', type=int, default=512, help='samples per GPU tile')
    ap.add_argument('--workers', type=int, default=max(1, mp.cpu_count()))
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest or not (a.vcf_dir and a.pgs_dir):
        selftest()
    else:
        run(a.vcf_dir, a.pgs_dir, a.output, a.tile_size, a.workers, a.time_output)

#!/usr/bin/env python3
"""
make_fake_benchmark.py - build the synthetic benchmark in benchmark/.

Nothing is downloaded. Everything is simulated and bit-reproducible on any
machine (Linux, macOS, x86, Arm): randomness comes from a counter-based
splitmix64 generator written here in plain integer arithmetic, so the output does
not depend on the numpy version, the platform's RNG or its libm. The first build
writes benchmark/MANIFEST.tsv (sha256 of every file's uncompressed content). A
rebuild or --verify checks against it, so you can tell whether the data the
reference was computed from (tools/make_reference.sh) are still intact:

    python tools/make_fake_benchmark.py            # build (~0.8 GB, under a minute on 8 cores)
    python tools/make_fake_benchmark.py --verify   # compare with MANIFEST.tsv

Output
    benchmark/genotypes/AOU000001.vcf.gz ...  100 single-sample VCFs, GRCh38,
                                              ~1.1 M sites each, every site listed
                                              (hom-ref included), ~1 % no-calls
    benchmark/models/PGS900001.txt.gz ...     48 PGS Catalog style harmonized
                                              scoring files (format_version 2.0),
                                              80 to 1.05 M rows each, 7.3 M in total

This is a scaled-down stand-in for the study's frozen benchmark (100 All of
Us-like VCFs x 100 PGS Catalog models, 7.65 M sites per VCF). Its shape is the
same: per-sample VCFs whose sites are drawn from the model loci. The
scoring files carry the quirks of real PGS Catalog files, which a correct
scorer has to handle and which the starting script (optim_src/gpu_pgs.py) does
not:

  * effect allele is REF in some VCF records and ALT in others
  * strand-ambiguous SNPs (A/T, C/G) occur at their natural rate
  * ~10 % of SNP rows are written on the opposite strand (complemented alleles)
  * ~6 % of rows have no other_allele, only hm_inferOtherAllele (1 % as "A/G")
  * two models have no other_allele column at all
  * ~3 % indels
  * ~8 % of the locus pool is not genotyped
  * one model carries dominant and recessive rows
  * chr_name/chr_position are on another (GRCh37-like) build; only the
    harmonized hm_chr/hm_pos columns line up with the VCFs
"""
import argparse
import gzip
import hashlib
import multiprocessing as mp
import os
import shutil
import sys
import zlib
from pathlib import Path

import numpy as np

SEED = 20260918
N_GENOTYPES = 100
N_LOCI = 1_200_000          # locus pool shared by all models
GENOTYPED_FRAC = 0.92       # fraction of the pool present in the VCFs
MISSING_RATE = 0.01         # per-call no-call rate
INDEL_FRAC = 0.03

MODEL_SIZES = (
    [80, 120, 170, 230, 300, 400, 520, 650, 850, 1000,
     1400, 1800, 2200, 2800, 3300, 3600, 4500, 5600, 6600, 8400]            # small
    + [10500, 14700, 19600, 24500, 29400, 35000, 42700, 51100, 61600,
       70000, 80500, 91000, 105000, 122500, 140000, 161000, 182000, 210000]  # mid
    + [266000, 315000, 364000, 420000, 490000, 560000, 630000, 700000]      # large
    + [945000, 1050000])                                                   # genome-wide
DOMREC_SIZE_INDEX = 25          # the model built from this size gets dominant/recessive rows
NO_OTHER_COL_SIZE_INDEX = {3, 17}

CHROM_LEN = {
    "1": 248956422, "2": 242193529, "3": 198295559, "4": 190214555,
    "5": 181538259, "6": 170805979, "7": 159345973, "8": 145138636,
    "9": 138394717, "10": 133797422, "11": 135086622, "12": 133275309,
    "13": 114364328, "14": 107043718, "15": 101991189, "16": 90338345,
    "17": 83257441, "18": 80373285, "19": 58617616, "20": 64444167,
    "21": 46709983, "22": 50818468,
}
BASES = np.array(list("ACGT"))
SNP_PAIRS = [(a, b) for a in "ACGT" for b in "ACGT" if a != b]      # 12 ordered pairs
COMP = str.maketrans("ACGT", "TGCA")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "benchmark"


# ── portable RNG: splitmix64 over a counter, keyed by a named stream ─────────
def _mix(z):
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def rand_u64(stream, n):
    key = _mix(np.array([(SEED << 32) ^ zlib.crc32(stream.encode())], dtype=np.uint64))
    with np.errstate(over="ignore"):
        x = (np.arange(1, n + 1, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)) ^ key
    return _mix(_mix(x))


def uniform(stream, n):
    """n doubles in [0, 1), identical on every platform."""
    return (rand_u64(stream, n) >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


def randint(stream, n, k):
    return np.minimum((uniform(stream, n) * k).astype(np.int64), k - 1)


# ── locus pool ───────────────────────────────────────────────────────────────
def build_loci():
    chroms = list(CHROM_LEN)
    lens = np.array([CHROM_LEN[c] for c in chroms], dtype=np.int64)
    quota = N_LOCI * lens // lens.sum()
    rem = N_LOCI * lens - quota * lens.sum()
    for i in np.argsort(-rem, kind="stable")[: N_LOCI - int(quota.sum())]:
        quota[i] += 1
    chr_idx = np.repeat(np.arange(len(chroms)), quota)
    pos = np.empty(N_LOCI, dtype=np.int64)
    u = uniform("pos", N_LOCI)
    o = 0
    for c, n in zip(chroms, quota):
        step = (CHROM_LEN[c] - 20_000) // n
        k = np.arange(n, dtype=np.int64)
        pos[o:o + n] = 10_000 + k * step + (u[o:o + n] * (step - 1)).astype(np.int64)
        o += n
    is_indel = uniform("indel", N_LOCI) < INDEL_FRAC
    pair = randint("pair", N_LOCI, 12)
    base = randint("ibase", N_LOCI, 4)
    elen = randint("ilen", N_LOCI, 3) + 1
    e = [randint(f"iext{j}", N_LOCI, 4) for j in range(3)]
    is_del = uniform("idel", N_LOCI) < 0.5
    a1, a2 = [], []
    for i in range(N_LOCI):
        if is_indel[i]:
            b = BASES[base[i]]
            ext = "".join(BASES[e[j][i]] for j in range(elen[i]))
            if is_del[i]:
                a1.append(b + ext); a2.append(b)
            else:
                a1.append(b); a2.append(b + ext)
        else:
            x, y = SNP_PAIRS[pair[i]]
            a1.append(x); a2.append(y)
    u = uniform("af", N_LOCI)
    af = 0.01 + 0.98 * (u * u * u)                                     # frequency of a2
    genotyped = uniform("genotyped", N_LOCI) < GENOTYPED_FRAC
    a2_is_alt = uniform("orient", N_LOCI) < 0.5
    chrom = np.array(chroms)[chr_idx]
    return dict(chrom=chrom, pos=pos, a1=a1, a2=a2, af=af,
                genotyped=genotyped, a2_is_alt=a2_is_alt)


_L = None


def _init(out):
    global _L, OUT
    OUT = Path(out)
    _L = build_loci()
    idx = np.flatnonzero(_L["genotyped"])
    ch, po, a1, a2, alt2 = _L["chrom"], _L["pos"], _L["a1"], _L["a2"], _L["a2_is_alt"]
    _L["site_prefix"] = [
        f"{ch[i]}\t{po[i]}\t.\t{a1[i] if alt2[i] else a2[i]}\t{a2[i] if alt2[i] else a1[i]}\t.\t.\t.\tGT\t"
        for i in idx]
    _L["p_alt"] = np.where(alt2[idx], _L["af"][idx], 1.0 - _L["af"][idx])


def _write(path, text):
    data = text.encode()
    with gzip.open(path, "wb", compresslevel=4) as f:
        f.write(data)
    return hashlib.sha256(data).hexdigest()


def write_vcf(s):
    sid = f"AOU{s + 1:06d}"
    p = _L["p_alt"]
    n = p.size
    dos = (uniform(f"g1:{s}", n) < p).astype(np.int8) + (uniform(f"g2:{s}", n) < p).astype(np.int8)
    gt = np.array(["0/0", "0/1", "1/1", "./."])[np.where(uniform(f"miss:{s}", n) < MISSING_RATE, 3, dos)]
    head = ("##fileformat=VCFv4.2\n##source=make_fake_benchmark.py\n##reference=GRCh38\n"
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
            f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sid}\n")
    body = "\n".join(map(str.__add__, _L["site_prefix"], gt.tolist())) + "\n"
    name = f"genotypes/{sid}.vcf.gz"
    return name, _write(OUT / name, head + body)


def fmt_w(x):
    return repr(float(f"{x:.6g}"))


def write_model(args):
    m, size_i = args
    k = MODEL_SIZES[size_i]
    pid = f"PGS{900001 + m}"
    ch, po, a1, a2 = _L["chrom"], _L["pos"], _L["a1"], _L["a2"]
    keys = rand_u64(f"sel:{m}", N_LOCI)
    sel = np.sort(np.argsort(keys, kind="stable")[:k])
    sd = 0.25 if k < 10_000 else (0.05 if k < 250_000 else 0.01)
    w = (uniform(f"w:{m}", k) * 2.0 - 1.0) * sd
    u_eff, u_strand, u_oth = uniform(f"eff:{m}", k), uniform(f"strand:{m}", k), uniform(f"oth:{m}", k)
    shift37 = 5_000 + randint(f"pos37:{m}", k, 395_000)
    u_dom = uniform(f"dom:{m}", k)
    domrec = size_i == DOMREC_SIZE_INDEX
    no_other_col = size_i in NO_OTHER_COL_SIZE_INDEX
    cols = ["rsID", "chr_name", "chr_position", "effect_allele"]
    if not no_other_col:
        cols.append("other_allele")
    cols.append("effect_weight")
    if domrec:
        cols += ["is_dominant", "is_recessive"]
    cols += ["hm_source", "hm_rsID", "hm_chr", "hm_pos", "hm_inferOtherAllele"]
    lines = []
    for r, i in enumerate(sel):
        eff, oth = (a1[i], a2[i]) if u_eff[r] < 0.5 else (a2[i], a1[i])
        if len(eff) == 1 and len(oth) == 1 and u_strand[r] < 0.10:     # other strand
            eff, oth = eff.translate(COMP), oth.translate(COMP)
        extra = next(b for b in "ACGT" if b not in (eff, oth))
        infer, oth_out = "", oth
        if no_other_col:
            infer, oth_out = (oth if u_oth[r] < 0.9 else f"{oth}/{extra}"), None
        elif u_oth[r] < 0.05:
            infer, oth_out = oth, ""
        elif u_oth[r] < 0.06:
            infer, oth_out = f"{oth}/{extra}", ""
        rs = f"rs{900000000 + int(i)}"
        row = [rs, str(ch[i]), str(max(int(po[i]) - int(shift37[r]), 1)), eff]
        if oth_out is not None:
            row.append(oth_out)
        row.append(fmt_w(w[r]))
        if domrec:
            row += ["True" if u_dom[r] < 0.1 else "False",
                    "True" if 0.1 <= u_dom[r] < 0.2 else "False"]
        row += ["ENSEMBL", rs, str(ch[i]), str(int(po[i])), infer]
        lines.append("\t".join(row))
    head = ("###PGS CATALOG SCORING FILE - see https://www.pgscatalog.org/downloads/#dl_ftp_scoring for additional information\n"
            "#format_version=2.0\n##POLYGENIC SCORE (PGS) INFORMATION\n"
            f"#pgs_id={pid}\n#pgs_name=FAKE_{pid}\n#trait_reported=Simulated trait {m + 1}\n"
            "#trait_mapped=simulated trait\n#trait_efo=EFO_0000000\n#genome_build=GRCh37\n"
            f"#variants_number={k}\n#weight_type=beta\n##SOURCE INFORMATION\n#pgp_id=PGP999999\n"
            "#citation=Simulated for the reproduce_study kit (not a real score)\n"
            "##HARMONIZATION DETAILS\n#HmPOS_build=GRCh38\n#HmPOS_date=2026-09-18\n"
            '#HmPOS_match_chr={"True": null, "False": null}\n'
            '#HmPOS_match_pos={"True": null, "False": null}\n')
    name = f"models/{pid}.txt.gz"
    return name, _write(OUT / name, head + "\t".join(cols) + "\n" + "\n".join(lines) + "\n")


def sha_of(path):
    h = hashlib.sha256()
    with gzip.open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_one(args):
    path, name, want = args
    p = Path(path) / name
    return name, (p.exists() and sha_of(p) == want)


def verify(workers):
    man = OUT / "MANIFEST.tsv"
    if not man.exists():
        print(f"no {man}")
        return False
    rows = [(str(OUT), *l.rstrip("\n").split("\t")) for l in list(open(man))[1:]]
    with mp.Pool(workers) as pool:
        res = pool.map(_verify_one, rows)
    bad = [n for n, ok in res if not ok]
    print(f"verified {len(res) - len(bad)}/{len(res)} files against MANIFEST.tsv")
    for n in bad[:10]:
        print(f"  MISMATCH or missing: {n}")
    return not bad


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--force", action="store_true", help="delete and rebuild genotypes/ and models/")
    ap.add_argument("--verify", action="store_true", help="only check files against MANIFEST.tsv")
    ap.add_argument("--write-manifest", action="store_true",
                    help="(maintainers) overwrite MANIFEST.tsv with the hashes of this build")
    ap.add_argument("--workers", type=int, default=max(1, min(os.cpu_count() or 1, 8)))
    a = ap.parse_args()
    OUT = Path(a.out)
    if a.verify:
        sys.exit(0 if verify(a.workers) else 1)
    if (OUT / "genotypes").is_dir() and any((OUT / "genotypes").iterdir()) and not a.force:
        print(f"{OUT}/genotypes already exists; use --force to rebuild or --verify to check it")
        return
    shutil.rmtree(OUT / "genotypes", ignore_errors=True)
    shutil.rmtree(OUT / "models", ignore_errors=True)
    (OUT / "genotypes").mkdir(parents=True)
    (OUT / "models").mkdir(parents=True)
    order = np.argsort(rand_u64("model_order", len(MODEL_SIZES)), kind="stable")
    print(f"building benchmark in {OUT} with {a.workers} workers ...", flush=True)
    with mp.Pool(a.workers, initializer=_init, initargs=(str(OUT),)) as pool:
        hashes = dict(pool.imap_unordered(write_model, list(enumerate(order.tolist()))))
        print(f"  {len(hashes)} scoring files written", flush=True)
        for n, h in pool.imap_unordered(write_vcf, range(N_GENOTYPES)):
            hashes[n] = h
    print(f"  {N_GENOTYPES} VCFs written", flush=True)
    man = OUT / "MANIFEST.tsv"
    if a.write_manifest or not man.exists():
        with open(man, "w") as f:
            f.write("file\tsha256_of_uncompressed_content\n")
            for k in sorted(hashes):
                f.write(f"{k}\t{hashes[k]}\n")
        print(f"wrote {man}")
        return
    want = dict(l.rstrip("\n").split("\t") for l in list(open(man))[1:])
    bad = [k for k in want if hashes.get(k) != want[k]]
    print(f"checked against MANIFEST.tsv: {len(want) - len(bad)}/{len(want)} files identical")
    if bad:
        print("WARNING: this build differs from the shipped data, so the reference scores do not apply.")
        sys.exit(1)


if __name__ == "__main__":
    main()

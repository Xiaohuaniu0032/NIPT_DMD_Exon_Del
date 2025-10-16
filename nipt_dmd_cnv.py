#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIPT FASTQ -> maternal DMD exon deletion screening pipeline.

Steps:
1) Align FASTQs to reference (auto-detect bwa-mem2 or fallback to bwa-mem) -> BAM
2) Sort + Mark duplicates (samtools)
3) Count read depth over DMD exons (with optional padding)
4) GC extraction from reference; GC correction (bin-wise loess-like)
5) Within-sample normalization (median of autosomes bins) and optional PoN normalization
6) Compute log2 ratio / Z-score; call suspicious deletions (per-exon + merged segments)
7) Plot along exon order

Dependencies (CLI): bwa-mem2 or bwa, samtools
Python libs: pandas, numpy, matplotlib, pysam (for FASTA fetch), tqdm (optional)

Author: ChatGPT (GPT-5 Thinking)
"""

import os
import sys
import argparse
import subprocess as sp
from pathlib import Path
import pandas as pd
import numpy as np
import math
from collections import defaultdict
import tempfile
import shutil

try:
    import pysam
except ImportError:
    sys.stderr.write("[WARN] pysam not found; GC calculation from FASTA will fail. Install via `pip install pysam`.\n")
    pysam = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:
    plt = None

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k):
        return x

def run(cmd, shell=False):
    sys.stderr.write(f"[CMD] {' '.join(cmd) if isinstance(cmd, (list, tuple)) else cmd}\n")
    ret = sp.run(cmd, shell=shell)
    if ret.returncode != 0:
        sys.stderr.write(f"[ERR] Command failed: {cmd}\n")
        sys.exit(ret.returncode)

def which(prog):
    return shutil.which(prog) is not None

def detect_aligner():
    if which("bwa-mem2"):
        return "bwa-mem2"
    if which("bwa"):
        return "bwa"
    sys.stderr.write("[ERR] Neither bwa-mem2 nor bwa found in PATH.\n")
    sys.exit(1)

def ensure_index(ref_fa, aligner):
    # bwa index if missing
    need = False
    if aligner == "bwa-mem2":
        # check one of bwa-mem2 index extensions
        exts = [".0123", ".amb", ".ann", ".bwt.2bit.64", ".pac"]
        need = not any(Path(ref_fa + e).exists() for e in exts)
        if need:
            run(["bwa-mem2", "index", ref_fa])
    else:
        exts = [".amb", ".ann", ".bwt", ".pac", ".sa"]
        need = not all(Path(ref_fa + e).exists() for e in exts)
        if need:
            run(["bwa", "index", ref_fa])
    # fasta .fai
    fai = ref_fa + ".fai"
    if not Path(fai).exists():
        run(["samtools", "faidx", ref_fa])

def align_and_sort(fq1, fq2, ref, outprefix, threads, aligner):
    bam = f"{outprefix}.sorted.bam"
    if Path(bam).exists():
        sys.stderr.write(f"[INFO] Found {bam}, skip alignment.\n")
        return bam
    samtools_view = ["samtools", "view", "-b", "-@", str(threads)]
    samtools_sort = ["samtools", "sort", "-@", str(threads), "-o", bam, "-"]
    if aligner == "bwa-mem2":
        aln = ["bwa-mem2", "mem", "-t", str(threads), ref, fq1, fq2]
    else:
        aln = ["bwa", "mem", "-t", str(threads), ref, fq1, fq2]

    # pipe: align -> bam -> sort
    p1 = sp.Popen(aln, stdout=sp.PIPE)
    p2 = sp.Popen(samtools_view, stdin=p1.stdout, stdout=sp.PIPE)
    p3 = sp.Popen(samtools_sort, stdin=p2.stdout)
    p1.stdout.close()
    p2.stdout.close()
    ret = p3.wait()
    if ret != 0:
        sys.stderr.write("[ERR] Alignment/sort failed.\n")
        sys.exit(1)
    return bam

def mark_duplicates(in_bam, outprefix, threads):
    bam = f"{outprefix}.sorted.mkdup.bam"
    if Path(bam).exists():
        sys.stderr.write(f"[INFO] Found {bam}, skip markdup.\n")
        return bam
    # fixmate + sort by position to improve markdup perf
    tmp_fix = f"{outprefix}.fixmate.bam"
    run(["samtools", "fixmate", "-m", "-@", str(threads), in_bam, tmp_fix])
    tmp_sort2 = f"{outprefix}.pos.sorted.bam"
    run(["samtools", "sort", "-@", str(threads), "-o", tmp_sort2, tmp_fix])
    run(["samtools", "markdup", "-@", str(threads), tmp_sort2, bam])
    # index
    run(["samtools", "index", bam])
    # cleanup
    for f in [tmp_fix, tmp_sort2]:
        try: Path(f).unlink()
        except: pass
    return bam

def read_bed(bed_path, padding=0, genome_fai=None):
    cols = ["chrom", "start", "end", "name"]
    df = pd.read_csv(bed_path, sep="\t", header=None, comment="#")
    if df.shape[1] < 3:
        raise ValueError("BED must have at least 3 columns: chrom, start, end; optional 4th as exon name.")
    if df.shape[1] == 3:
        df["name"] = [f"exon_{i+1}" for i in range(df.shape[0])]
    else:
        df = df.iloc[:, :4]
        df.columns = cols
    df["start"] = df["start"].astype(int) - int(padding)
    df["end"]   = df["end"].astype(int) + int(padding)
    df["start"] = df["start"].clip(lower=0)
    # clip to chrom length if fai provided
    if genome_fai and Path(genome_fai).exists():
        chrom_len = {}
        with open(genome_fai) as f:
            for line in f:
                c, ln, *_ = line.strip().split("\t")
                chrom_len[c] = int(ln)
        df["end"] = df.apply(lambda r: min(r["end"], chrom_len.get(r["chrom"], r["end"])), axis=1)
    return df

def bam_depth_over_intervals(bam, intervals_df, mapq=30):
    # count properly paired reads with MAPQ >= mapq overlapping intervals
    # Use samtools bedcov for speed if available
    out = []
    bed_tmp = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".bed")
    for _, r in intervals_df.iterrows():
        bed_tmp.write(f"{r.chrom}\t{r.start}\t{r.end}\t{r.name}\n")
    bed_tmp.close()
    try:
        cmd = ["samtools", "bedcov", bed_tmp.name, bam]
        p = sp.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError("samtools bedcov failed")
        for line in p.stdout.strip().splitlines():
            chrom, start, end, name, covsum = line.strip().split("\t")
            start, end = int(start), int(end)
            length = max(1, end - start)
            depth = float(covsum) / float(length)
            out.append((chrom, start, end, name, depth, length))
    finally:
        try: os.unlink(bed_tmp.name)
        except: pass
    df = pd.DataFrame(out, columns=["chrom","start","end","name","depth","length"])
    return df

def calc_gc_for_intervals(ref_fa, intervals_df):
    if pysam is None:
        sys.stderr.write("[WARN] pysam not installed; GC% will be NA.\n")
        intervals_df["gc"] = np.nan
        return intervals_df
    fasta = pysam.FastaFile(ref_fa)
    gcs = []
    for _, r in intervals_df.iterrows():
        seq = fasta.fetch(r.chrom, int(r.start), int(r.end)).upper()
        if not seq:
            gcs.append(np.nan); continue
        gc = (seq.count("G") + seq.count("C")) / max(1, len(seq.replace("N","")))
        gcs.append(gc)
    intervals_df = intervals_df.copy()
    intervals_df["gc"] = gcs
    return intervals_df

def loess_like_gc_correction(depth, gc, bins=20):
    """Approximate LOESS: bin by GC%, subtract per-bin median, then local smooth by rolling median."""
    s = pd.Series(depth).astype(float).copy()
    g = pd.Series(gc).astype(float).copy()
    ok = (~s.isna()) & (~g.isna())
    corrected = s.copy()
    # GC binning
    q = pd.qcut(g[ok], q=min(bins, ok.sum()), duplicates="drop")
    med_by_bin = s[ok].groupby(q).median()
    # map each point to its bin median
    bin_med = pd.Series(index=s.index, dtype=float)
    # create mapping
    # need labels for each ok index:
    labels = q.astype(str)
    map_med = {k: v for k, v in med_by_bin.items()}
    # initialize with NaN
    bin_med[:] = np.nan
    for idx, lab in zip(s[ok].index, labels):
        bin_med.loc[idx] = map_med.get(lab, np.nan)
    # subtract bin median
    corrected = s - bin_med
    # rolling median smooth (window=5)
    corrected = corrected.rolling(window=5, min_periods=1, center=True).median()
    # add back global median to keep scale near original
    corrected = corrected - np.nanmedian(corrected) + np.nanmedian(s)
    return corrected.values

def within_sample_normalize(depth, intervals_df):
    """Use global median of autosomes (chr1-22) to scale to ~1.0 copies baseline in female."""
    df = intervals_df.copy()
    d = df["depth"].values.astype(float)
    chrom = df["chrom"].astype(str).values
    autosome_mask = np.array([c.lower().replace("chr","") in [str(i) for i in range(1,23)] for c in chrom])
    baseline = np.nanmedian(d[autosome_mask]) if autosome_mask.any() else np.nanmedian(d)
    norm = d / (baseline + 1e-8)
    return norm, baseline

def load_pon(pon_path):
    """PoN depth matrix: rows=exon(name), cols=samples; values=raw depth (or normalized).
       We compute per-exon median as reference."""
    mat = pd.read_csv(pon_path, sep="\t", index_col=0)
    ref_median = mat.median(axis=1)  # per-exon
    return ref_median

def zscore_series(x):
    x = pd.Series(x).astype(float)
    return (x - x.mean()) / (x.std(ddof=1) + 1e-8)

def call_deletions(df, log2_thr=-0.35, z_thr=3.0, min_run=2):
    """Flag exons with log2<=thr and |Z|>=z_thr; then merge consecutive into segments."""
    df = df.copy()
    df["is_del"] = (df["log2_ratio"] <= log2_thr) & (df["Z"].abs() >= z_thr)
    calls = []
    run = []
    for i, r in df.reset_index().iterrows():
        if r["is_del"]:
            run.append(r)
        else:
            if len(run) >= min_run:
                calls.append((run[0]["name"], run[-1]["name"],
                              run[0]["chrom"], int(run[0]["start"]), int(run[-1]["end"]),
                              len(run),
                              float(np.nanmedian([x["log2_ratio"] for x in run])),
                              float(np.nanmedian([x["Z"] for x in run]))))
            run = []
    if len(run) >= min_run:
        calls.append((run[0]["name"], run[-1]["name"],
                      run[0]["chrom"], int(run[0]["start"]), int(run[-1]["end"]),
                      len(run),
                      float(np.nanmedian([x["log2_ratio"] for x in run])),
                      float(np.nanmedian([x["Z"] for x in run]))))
    calls_df = pd.DataFrame(calls, columns=["start_exon","end_exon","chrom","start","end","n_exons","median_log2","median_Z"])
    return df, calls_df

def plot_profile(df, out_png, title="DMD exon copy profile"):
    if plt is None:
        sys.stderr.write("[WARN] matplotlib not available; skip plot.\n")
        return
    fig = plt.figure(figsize=(12,5))
    x = np.arange(df.shape[0]) + 1
    ax = plt.gca()
    ax.plot(x, df["log2_ratio"].values, marker="o", linewidth=1)
    ax.axhline(-0.5, linestyle="--")
    ax.axhline(0.0, linestyle=":")
    ax.set_xlabel("Exon order")
    ax.set_ylabel("log2 ratio")
    ax.set_title(title)
    ax.set_xticks(x[::max(1, len(x)//20)])
    ax.set_xticklabels(df["name"].values[::max(1, len(x)//20)], rotation=90, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="NIPT FASTQ -> maternal DMD exon deletion pipeline")
    ap.add_argument("--fq1", required=True, help="FASTQ R1 (.gz)")
    ap.add_argument("--fq2", required=True, help="FASTQ R2 (.gz)")
    ap.add_argument("--ref", required=True, help="Reference fasta (GRCh38 or GRCh37), indexed")
    ap.add_argument("--bed", required=True, help="DMD exons BED (chrom start end name)")
    ap.add_argument("--outprefix", required=True, help="Output prefix")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--padding", type=int, default=0, help="Pad each exon by N bp on both sides")
    ap.add_argument("--mapq", type=int, default=30)
    ap.add_argument("--pon", default=None, help="Optional panel-of-normals depth matrix (TSV)")
    ap.add_argument("--log2_thr", type=float, default=-0.35)
    ap.add_argument("--z_thr", type=float, default=3.0)
    ap.add_argument("--min_run", type=int, default=2)
    args = ap.parse_args()

    # 0) checks
    for prog in ["samtools"]:
        if not which(prog):
            sys.stderr.write(f"[ERR] Required tool `{prog}` not found in PATH.\n")
            sys.exit(1)
    aligner = detect_aligner()
    ensure_index(args.ref, aligner)

    # 1) Align -> sorted mkdup BAM
    bam_sorted = align_and_sort(args.fq1, args.fq2, args.ref, args.outprefix, args.threads, aligner)
    bam = mark_duplicates(bam_sorted, args.outprefix, args.threads)

    # 2) Load BED (with padding) & GC
    fai = args.ref + ".fai"
    bed_df = read_bed(args.bed, padding=args.padding, genome_fai=fai)
    bed_df = calc_gc_for_intervals(args.ref, bed_df)

    # 3) Depth over intervals
    depth_df = bam_depth_over_intervals(bam, bed_df, mapq=args.mapq)
    df = bed_df.merge(depth_df[["name","depth","length"]], on="name", how="left")

    # 4) GC correction (loess-like)
    gc_corr = loess_like_gc_correction(df["depth"].values, df["gc"].values, bins=20)
    df["depth_gc_corrected"] = gc_corr

    # 5) Within-sample normalization (median autosomes)
    norm, baseline = within_sample_normalize(df["depth_gc_corrected"].values, df)
    df["norm_depth"] = norm

    # 6) PoN normalization to log2 ratio if provided; else log2 vs ~1.0 baseline
    if args.pon:
        ref_median = load_pon(args.pon)  # index: exon name
        # align index by exon name
        if not set(df["name"]).issubset(set(ref_median.index)):
            missing = sorted(set(df["name"]) - set(ref_median.index))
            sys.stderr.write(f"[WARN] {len(missing)} exons missing in PoN; will ignore PoN for those.\n")
        # construct expected copy ~= ref median / ref global median
        # scale PoN to its own median to be dimensionless
        pon_scale = ref_median.median()
        expected = df["name"].map(ref_median / (pon_scale + 1e-8)).values
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = df["norm_depth"].values / expected
            df["log2_ratio"] = np.log2(ratio)
    else:
        # against within-sample expected 1.0
        with np.errstate(divide='ignore', invalid='ignore'):
            df["log2_ratio"] = np.log2(df["norm_depth"].values + 1e-8)

    # 7) Z-score within DMD region
    df["Z"] = zscore_series(df["log2_ratio"].values)

    # 8) Sort by genomic order (chrom, start)
    df = df.sort_values(["chrom","start","end"]).reset_index(drop=True)

    # 9) Call deletions (per-exon flags + merged runs)
    per_exon, calls = call_deletions(df, log2_thr=args.log2_thr, z_thr=args.z_thr, min_run=args.min_run)

    # 10) Save tables
    depth_out = f"{args.outprefix}_DMD_exon_depth.tsv"
    calls_out = f"{args.outprefix}_DMD_calls.tsv"
    cols = ["chrom","start","end","name","length","gc","depth","depth_gc_corrected","norm_depth","log2_ratio","Z"]
    per_exon[cols].to_csv(depth_out, sep="\t", index=False)
    calls.to_csv(calls_out, sep="\t", index=False)

    # 11) Plot
    plot_png = f"{args.outprefix}_DMD_plot.png"
    plot_profile(per_exon, plot_png, title=f"{args.outprefix} DMD exon profile")

    # 12) Notes
    sys.stderr.write("\n[DONE] Outputs:\n")
    sys.stderr.write(f"  - Per-exon table: {depth_out}\n")
    sys.stderr.write(f"  - Calls (merged segments): {calls_out}\n")
    sys.stderr.write(f"  - Plot: {plot_png}\n")
    sys.stderr.write("\n[HINT] Tune --log2_thr/--z_thr/--min_run based on sequencing depth与PoN质量；建议MLPA/qPCR验证阳性。\n")

if __name__ == "__main__":
    main()

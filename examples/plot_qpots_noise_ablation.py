#!/usr/bin/env python3
# Plot qPOTS noise ablation

"""
Install plotting dependencies with ``pip install qpots[examples]``.

Reads:
  - hv_traces_agg.csv (required)
  - hv_traces_raw.csv (optional for per-rep overlay)

Outputs (to --indir):
  - hv_vs_iteration_latex.png
  - hv_vs_iteration_latex.pdf
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--indir", type=str, required=True)
    p.add_argument("--outfile", type=str, default="hv_vs_iteration_latex")
    p.add_argument("--spaghetti", type=int, default=0)
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    indir = Path(args.indir)
    agg_csv = indir / "hv_traces_agg.csv"
    raw_csv = indir / "hv_traces_raw.csv"

    if not agg_csv.exists():
        print(f"ERROR: {agg_csv} not found", file=sys.stderr)
        sys.exit(1)

    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    agg = pd.read_csv(agg_csv)
    raw = pd.read_csv(raw_csv) if args.spaghetti and raw_csv.exists() else None

    fig = plt.figure(figsize=(5.25, 3.5))
    ax = fig.add_subplot(1,1,1)

    if raw is not None and args.spaghetti:
        for (nv, rep), rdf in raw.groupby(["noise_var", "rep"]):
            ax.plot(rdf["iteration"].values, rdf["hypervolume"].values, alpha=0.12, linewidth=0.8)

    for nv, gdf in agg.groupby("noise_var"):
        it = gdf["iteration"].values
        mu = gdf["mean"].values
        sd = gdf["std"].values
        ax.plot(it, mu, label=rf"$\sigma^2={nv:g}$")
        ax.fill_between(it, mu - sd, mu + sd, alpha=0.18, linewidth=0)

    ax.set_xlabel(r"Iteration")
    ax.set_ylabel(r"Expected hypervolume")
    ax.set_title(r"qPOTS on Branin--Currin: noise ablation")
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.legend(title=r"Obs.\ noise variance", frameon=False)

    fig.tight_layout()

    png_path = indir / f"{args.outfile}.png"
    pdf_path = indir / f"{args.outfile}.pdf"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

if __name__ == "__main__":
    main()

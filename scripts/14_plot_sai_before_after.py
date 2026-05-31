import argparse
from pathlib import Path

import pandas as pd

from coursemap.io import read_csv_smart
from coursemap.plots import save_before_after_dot_plot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation", required=True)
    parser.add_argument("--algorithm", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--before-col", default="SAI_before")
    parser.add_argument("--after-col", default="SAI_after")
    parser.add_argument("--label-col", default="학교명")
    parser.add_argument("--title", default="SAI before vs after")
    args = parser.parse_args()

    sim = read_csv_smart(Path(args.simulation))
    if args.algorithm and "algorithm" in sim.columns:
        sim = sim[sim["algorithm"].eq(args.algorithm)].copy()
    if sim.empty:
        raise SystemExit("no rows to plot")

    save_before_after_dot_plot(
        sim,
        before_col=args.before_col,
        after_col=args.after_col,
        label_col=args.label_col,
        highlight_labels=set(),
        out_path=Path(args.out),
        title=args.title,
        tier_labels=True,
        show_summary_lines=True,
    )
    print(f"SAI before/after plot saved: {args.out}")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

from coursemap.io import read_csv_smart
from coursemap.plots import save_sai_stepped_domain_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sai", default="build/tables/school_sai_result.csv")
    parser.add_argument("--out", default="build/figures/sai_stepped_domain_counts.png")
    parser.add_argument("--summary-out", default="build/tables/sai_stepped_domain_counts.csv")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--exclude-school", action="append", default=["대전예술고등학교"])
    args = parser.parse_args()

    sai = read_csv_smart(Path(args.sai))
    save_sai_stepped_domain_counts(
        sai,
        Path(args.out),
        summary_out=Path(args.summary_out),
        sample_size=args.sample_size,
        exclude_schools=set(args.exclude_school),
    )
    print(f"SAI stepped domain count plot saved: {args.out}")
    print(f"SAI stepped domain count sample saved: {args.summary_out}")


if __name__ == "__main__":
    main()

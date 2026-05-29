import argparse
from pathlib import Path

from coursemap.io import read_csv_smart, write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    joint = read_csv_smart(Path(args.joint))
    if joint.empty:
        write_csv(joint, Path(args.out))
        print("joint network: 0 rows")
        return

    network = (
        joint.groupby(["_norm_school", "계열"])
        .agg(강좌수=("표준과목명", "size"), 고유과목수=("표준과목명", "nunique"))
        .reset_index()
    )
    write_csv(network, Path(args.out))
    print(f"joint network: {len(network)} school-domain rows")


if __name__ == "__main__":
    main()

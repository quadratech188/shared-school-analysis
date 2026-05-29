import argparse
from pathlib import Path

from coursemap.io import read_csv_smart, write_csv
from coursemap.sai import compute_sai, regular_offerings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schools", required=True)
    parser.add_argument("--neis-subjects", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    schools = read_csv_smart(Path(args.schools))
    neis_subjects = read_csv_smart(Path(args.neis_subjects))
    offerings = regular_offerings(neis_subjects)
    result = compute_sai(schools, offerings).sort_values("SAI", ascending=False).reset_index(drop=True)
    write_csv(result, Path(args.out))
    print(f"SAI: {len(result)} schools using {result['SAI_algorithm'].iloc[0]}")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import pandas as pd

from coursemap.geo import haversine_km
from coursemap.io import read_csv_smart, write_csv


RADII_KM = (2, 3, 5, 7)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--school-master", required=True)
    parser.add_argument("--subject-matrix", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    schools = read_csv_smart(Path(args.school_master))
    schools = schools[schools["is_general_high"].astype(bool)].copy()
    schools = schools.dropna(subset=["위도", "경도"])

    matrix = read_csv_smart(Path(args.subject_matrix))
    if "학교명" not in matrix.columns:
        raise SystemExit("subject matrix must contain 학교명")
    matrix = matrix.set_index("학교명")
    subjects = [c for c in matrix.columns if c != "학교명"]

    distance_rows = []
    complement_rows = []
    subject_sets = {
        school: set(matrix.columns[matrix.loc[school].fillna(0).astype(float) > 0])
        for school in matrix.index
    }

    for _, src in schools.iterrows():
        src_name = src["학교명"]
        own = subject_sets.get(src_name, set())
        row = {"학교명": src_name, "자체과목수": len(own)}
        neighbors_by_radius = {r: [] for r in RADII_KM}

        for _, dst in schools.iterrows():
            dst_name = dst["학교명"]
            if src_name == dst_name:
                continue
            d = haversine_km(src["위도"], src["경도"], dst["위도"], dst["경도"])
            distance_rows.append({"from_school": src_name, "to_school": dst_name, "distance_km": round(d, 4)})
            for radius in RADII_KM:
                if d <= radius:
                    neighbors_by_radius[radius].append(dst_name)

        for radius, neighbor_names in neighbors_by_radius.items():
            nearby_subjects = set()
            for name in neighbor_names:
                nearby_subjects |= subject_sets.get(name, set())
            row[f"인근학교수_{radius}km"] = len(neighbor_names)
            row[f"보완가능과목수_{radius}km"] = len(nearby_subjects - own)
        complement_rows.append(row)

    out_dir = Path(args.out_dir)
    write_csv(pd.DataFrame(distance_rows), out_dir / "school_distance_matrix.csv")
    write_csv(pd.DataFrame(complement_rows), out_dir / "nearby_school_accessibility.csv")
    print(f"school accessibility: {len(schools)} schools, {len(subjects)} subject columns")


if __name__ == "__main__":
    main()

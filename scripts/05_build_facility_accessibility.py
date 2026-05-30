import argparse
from pathlib import Path

import pandas as pd

from coursemap.geo import haversine_km
from coursemap.io import read_csv_smart, write_csv


RADII_KM = [2, 3, 5, 7]
LABELS = {
    "public_library": "도서관",
    "youth_facility": "청소년",
    "lifelong_education": "평생",
}


def facility_access(schools: pd.DataFrame, facilities: pd.DataFrame, facility_type: str, label: str) -> pd.DataFrame:
    fac = facilities[facilities["facility_type"].eq(facility_type)].dropna(subset=["위도", "경도"]).copy()
    if fac.empty:
        raise SystemExit(f"no geocoded facilities for type: {facility_type}")
    rows = []
    for _, school in schools.iterrows():
        if pd.isna(school["위도"]) or pd.isna(school["경도"]):
            raise SystemExit(f"school location missing: {school['학교명']}")
        distances = [
            haversine_km(school["위도"], school["경도"], facility["위도"], facility["경도"])
            for _, facility in fac.iterrows()
        ]
        row = {"학교명": school["학교명"], f"{label}_최근접km": round(min(distances), 2)}
        for radius in RADII_KM:
            row[f"{label}_{radius}km개수"] = int(sum(distance <= radius for distance in distances))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schools", required=True)
    parser.add_argument("--facilities", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    schools = read_csv_smart(Path(args.schools))
    facilities = read_csv_smart(Path(args.facilities))
    required = {"facility_type", "name", "위도", "경도"}
    missing = required - set(facilities.columns)
    if missing:
        raise SystemExit(f"geocoded facility table missing columns: {sorted(missing)}")

    result = schools[["학교명"]].copy()
    for facility_type, label in LABELS.items():
        result = result.merge(facility_access(schools, facilities, facility_type, label), on="학교명", how="left")

    write_csv(result, Path(args.out))
    print(f"facility accessibility: {len(result)} schools")


if __name__ == "__main__":
    main()

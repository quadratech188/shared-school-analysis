import argparse
from pathlib import Path

import pandas as pd

from coursemap.columns import find_col, pick_series
from coursemap.io import read_csv_smart, write_csv


def text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def public_libraries(path: Path) -> pd.DataFrame:
    raw = read_csv_smart(path)
    promoted = raw.iloc[1:].copy()
    promoted.columns = [str(x).strip() for x in raw.iloc[0].tolist()]
    out = pd.DataFrame({
        "facility_type": "public_library",
        "name": text(pick_series(promoted, "도서관명")),
        "address": text(pick_series(promoted, "주소")),
        "district": text(pick_series(promoted, "시군구")),
    })
    return out[out["name"].ne("")]


def youth_facilities(path: Path) -> pd.DataFrame:
    raw = read_csv_smart(path)
    out = pd.DataFrame({
        "facility_type": "youth_facility",
        "name": text(pick_series(raw, "시설명")),
        "address": text(pick_series(raw, "주     소")),
        "district": text(pick_series(raw, "시군구")),
    })
    return out[out["name"].ne("")]


def lifelong_facilities(path: Path) -> pd.DataFrame:
    raw = read_csv_smart(path)
    out = pd.DataFrame({
        "facility_type": "lifelong_education",
        "name": text(pick_series(raw, "시설명")),
        "address": text(pick_series(raw, find_col(raw, ["소재지도로명주소", "소재지지번주소"]))),
        "district": text(pick_series(raw, "시군구명")),
        "status": text(pick_series(raw, "등록상태구분명")),
    })
    out = out[out["name"].ne("")]
    return out[out["status"].eq("개원")].drop(columns=["status"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data = Path(args.data_dir)
    frames = [
        public_libraries(data / "public_library_2025_daejeon_filtered.csv"),
        youth_facilities(data / "youth_training_facilities_daejeon_filtered.csv"),
        lifelong_facilities(data / "lifelong_education_facilities_daejeon_20260309.csv"),
    ]
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(["facility_type", "name", "address"])
    write_csv(out, Path(args.out))
    print(f"facilities: {len(out)} rows")


if __name__ == "__main__":
    main()

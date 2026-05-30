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
        "road_address": text(pick_series(promoted, "주소")),
        "jibun_address": "",
        "district": text(pick_series(promoted, "시군구")),
    })
    return out[out["name"].ne("")]


def youth_facilities(path: Path) -> pd.DataFrame:
    raw = read_csv_smart(path)
    out = pd.DataFrame({
        "facility_type": "youth_facility",
        "name": text(pick_series(raw, "시설명")),
        "facility_kind": text(pick_series(raw, find_col(raw, ["시설종류", "시설구분", "유형"]))),
        "road_address": text(pick_series(raw, "주     소")),
        "jibun_address": "",
        "district": text(pick_series(raw, "시군구")),
    })
    out = out[out["name"].ne("")]
    return out[out["facility_kind"].str.contains("수련관|문화의집", na=False)]


def lifelong_facilities(path: Path) -> pd.DataFrame:
    raw = read_csv_smart(path)
    out = pd.DataFrame({
        "facility_type": "lifelong_education",
        "name": text(pick_series(raw, "시설명")),
        "facility_kind": text(pick_series(raw, find_col(raw, ["시설구분명", "시설구분"]))),
        "road_address": text(pick_series(raw, "소재지도로명주소")),
        "jibun_address": text(pick_series(raw, "소재지지번주소")),
        "district": text(pick_series(raw, "시군구명")),
        "status": text(pick_series(raw, "등록상태구분명")),
    })
    out = out[out["name"].ne("")]
    out = out[out["status"].eq("개원")].drop(columns=["status"])
    include = ["평생학습관", "평생교육원", "평생교육센터", "시민대학", "교육문화원", "도서관", "문화원"]
    exclude = ["원격", "사업장부설", "백화점", "마트", "문화센터(주)", "(주)"]
    text_blob = out["name"] + " " + out["facility_kind"]
    is_excluded = text_blob.apply(lambda x: any(k in x for k in exclude))
    is_included = text_blob.apply(lambda x: any(k in x for k in include))
    out["hub_suitability"] = "ambiguous"
    out.loc[is_included, "hub_suitability"] = "rule_include"
    return out[~is_excluded]


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
    out["address"] = out["road_address"].where(out["road_address"].ne(""), out["jibun_address"])
    out = out.drop_duplicates(["facility_type", "name", "address"])
    write_csv(out, Path(args.out))
    print(f"facilities: {len(out)} rows")


if __name__ == "__main__":
    main()

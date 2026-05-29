import argparse
from pathlib import Path

import pandas as pd

from coursemap.io import read_csv_smart, write_csv
from coursemap.text import normalize_school_name


def num(df: pd.DataFrame, col: str, default=0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def general_high_flag(neis: pd.DataFrame) -> pd.Series:
    hs_type = neis.get("HS_SC_NM", pd.Series("", index=neis.index)).fillna("").astype(str)
    business = neis.get("HS_GNRL_BUSNS_SC_NM", pd.Series("", index=neis.index)).fillna("").astype(str)
    special = neis.get("SPCLY_PURPS_HS_ORD_NM", pd.Series("", index=neis.index)).fillna("").astype(str)
    vocational = hs_type.str.contains("특성화|마이스터|산업|전문", regex=True)
    special_purpose = hs_type.str.contains("특목|특수목적", regex=True) | special.ne("")
    general = hs_type.str.contains("일반|자율", regex=True) | business.str.contains("일반", regex=True)
    return general & ~vocational & ~special_purpose


def keep_schoolinfo(df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    out = pd.DataFrame({"_norm_name": df["학교명"].map(normalize_school_name)})
    for src, dst in columns.items():
        out[dst] = num(df, src)
    return out.drop_duplicates("_norm_name")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data = Path(args.data_dir)
    loc = read_csv_smart(data / "school_location_20260320.csv")
    neis = read_csv_smart(data / "outputs/raw/neis_school_info_raw.csv")

    loc = loc[(loc["학교급구분"].astype(str).str.contains("고등학교", na=False)) &
              (loc["운영상태"].astype(str).str.contains("운영", na=False))].copy()
    loc["_norm_name"] = loc["학교명"].map(normalize_school_name)

    neis["_norm_name"] = neis["SCHUL_NM"].map(normalize_school_name)
    neis["is_general_high_neis"] = general_high_flag(neis)
    neis_keep = neis[["_norm_name", "SCHUL_NM", "SD_SCHUL_CODE", "HS_SC_NM", "is_general_high_neis"]]

    master = loc[["_norm_name", "학교명", "위도", "경도", "소재지도로명주소"]].rename(
        columns={"소재지도로명주소": "주소"}
    )
    master = master.merge(neis_keep, on="_norm_name", how="left")

    student = read_csv_smart(data / "schoolinfo_2025_daejeon_high_student_class.csv")
    teacher = read_csv_smart(data / "schoolinfo_2025_daejeon_high_teacher.csv")
    building = read_csv_smart(data / "schoolinfo_2025_daejeon_high_school_building.csv")
    support = read_csv_smart(data / "schoolinfo_2025_daejeon_high_support_facilities.csv")

    frames = [
        keep_schoolinfo(student, {
            "학생수(계)": "학생수계",
            "학급수(계)": "학급수계",
            "2학년 학생수": "2학년학생수",
            "3학년 학생수": "3학년학생수",
            "교사수": "교사수_알리미",
        }),
        keep_schoolinfo(teacher, {
            "총계(계)": "교원총계",
            "일반교사(계)": "일반교사",
        }),
        keep_schoolinfo(building, {
            "일반교실": "일반교실",
            "교과교실": "교과교실",
            "특별교실": "특별교실",
            "컴퓨터실": "컴퓨터실",
            "멀티미디어실": "멀티미디어실",
        }),
        keep_schoolinfo(support, {
            "체육관": "체육관",
            "강당": "강당",
            "진로 상담실": "진로상담실",
        }),
    ]
    for frame in frames:
        master = master.merge(frame, on="_norm_name", how="left")

    master["is_general_high"] = master["is_general_high_neis"].fillna(False).astype(bool)
    master = master.sort_values("학교명").reset_index(drop=True)
    write_csv(master, Path(args.out))
    print(f"school master: {len(master)} rows, general high={int(master['is_general_high'].sum())}")


if __name__ == "__main__":
    main()

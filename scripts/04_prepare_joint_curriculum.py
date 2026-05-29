import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from coursemap.columns import find_col, pick_series
from coursemap.io import write_csv
from coursemap.subjects import classify_subject
from coursemap.text import clean_subject_name, normalize_school_name


def text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def read_joint(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="거점학교 운영db", header=None)
    best_idx, best_score = None, -1
    patterns = ("학교명", "교과목", "강좌", "유형", "학년", "학점")
    for idx, row in raw.iterrows():
        values = [str(x).replace("\n", "").replace(" ", "") for x in row.tolist()]
        score = sum(any(pattern in value for value in values) for pattern in patterns)
        if score > best_score:
            best_idx, best_score = idx, score
    if best_idx is None or best_score < 3:
        raise SystemExit(f"cannot detect joint-curriculum header: {path}")
    df = raw.iloc[best_idx + 1:].copy()
    df.columns = [str(x).strip() for x in raw.iloc[best_idx].tolist()]
    return df.dropna(how="all")


def clean_joint(df: pd.DataFrame, semester: str) -> pd.DataFrame:
    colmap = {
        "학교명": find_col(df, ["학교명", "거점학교", "거점학교명"]),
        "유형": find_col(df, ["유형", "운영유형"]),
        "교과목명": find_col(df, ["교과목명", "강좌명", "과목명"]),
        "교과구분": find_col(df, ["교과구분"]),
        "교과영역": find_col(df, ["교과영역"]),
        "교과군": find_col(df, ["교과군"]),
        "대상학년": find_col(df, ["대상학년", "학년"]),
        "학점": find_col(df, ["학점"]),
        "수업장소": find_col(df, ["수업장소", "장소"]),
        "요일": find_col(df, ["요일"]),
    }
    out = pd.DataFrame({"학기": semester}, index=df.index)
    for target, source in colmap.items():
        out[target] = pick_series(df, source)

    school = text(out["학교명"])
    subject = text(out["교과목명"])
    valid = school.ne("") & subject.ne("") & ~subject.isin({"교과목명", "강좌명", "과목명"})
    out = out.loc[valid].copy()

    type_text = text(out["유형"])
    out["유형표준"] = np.select(
        [type_text.str.contains("온라인", na=False), type_text.str.contains("진로", na=False)],
        ["온라인", "진로선택형"],
        default="오프라인",
    )
    out["온라인여부"] = text(out["수업장소"]).str.contains("온닷|온라인|원격", na=False) | (out["유형표준"] == "온라인")
    out["과목정제"] = text(out["교과목명"]).map(clean_subject_name)
    classified = out["과목정제"].map(classify_subject)
    out["표준과목명"] = classified.map(lambda x: x[0])
    out["계열"] = classified.map(lambda x: x[1])
    out["_norm_school"] = text(out["학교명"]).map(normalize_school_name)
    return out.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data = Path(args.data_dir)
    first = clean_joint(read_joint(data / "daejeon_joint_curriculum_2025_1st.xlsx"), "2025_1st")
    second = clean_joint(read_joint(data / "daejeon_joint_curriculum_2025_2nd.xlsx"), "2025_2nd")
    out = pd.concat([first, second], ignore_index=True)
    write_csv(out, Path(args.out))
    print(f"joint curriculum: {len(out)} rows")


if __name__ == "__main__":
    main()

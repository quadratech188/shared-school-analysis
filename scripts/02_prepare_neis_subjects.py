import argparse
from pathlib import Path

import pandas as pd

from coursemap.io import read_csv_smart, write_csv
from coursemap.subjects import classify_subject, is_excluded_subject
from coursemap.text import clean_subject_name, normalize_school_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    path = Path(args.data_dir) / "outputs/raw/neis_his_timetable_raw.csv"
    df = read_csv_smart(path)
    subject_col = "ITRT_CNTNT"
    if subject_col not in df.columns:
        raise SystemExit(f"missing NEIS subject column: {subject_col}")

    out = df.copy()
    out["학교명"] = out.get("SCHUL_NM", "").astype(str)
    out["_norm_school"] = out["학교명"].map(normalize_school_name)
    out["과목원본"] = out[subject_col]
    out["과목정제"] = out["과목원본"].map(clean_subject_name)
    out["제외대상"] = out["과목원본"].map(is_excluded_subject)
    out = out[~out["제외대상"]].copy()
    classified = out["과목정제"].map(classify_subject)
    out["표준과목명"] = classified.map(lambda x: x[0])
    out["계열"] = classified.map(lambda x: x[1])

    keep = [
        "SD_SCHUL_CODE", "학교명", "_norm_school", "GRADE", "SEM", "ALL_TI_YMD",
        "_period_label", "과목원본", "과목정제", "표준과목명", "계열",
    ]
    out = out[[c for c in keep if c in out.columns]]
    write_csv(out, Path(args.out))
    print(f"neis subjects: {len(out)} rows, {out['표준과목명'].nunique()} subjects")


if __name__ == "__main__":
    main()

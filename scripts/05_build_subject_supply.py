import argparse
from pathlib import Path

import pandas as pd

from coursemap.io import read_csv_smart, write_csv
from coursemap.subjects import classify_subject_with_method


def read_overrides(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["subject", "standard_subject", "domain"])
    df = read_csv_smart(path)
    required = {"subject", "standard_subject", "domain"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{path} missing columns: {sorted(missing)}")
    return df[list(required)].dropna(subset=["subject"]).drop_duplicates("subject", keep="last")


def read_ignores(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["subject", "reason"])
    df = read_csv_smart(path)
    required = {"subject", "reason"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{path} missing columns: {sorted(missing)}")
    return df[["subject", "reason"]].dropna(subset=["subject"]).drop_duplicates("subject", keep="last")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", required=True)
    parser.add_argument("--overrides", required=True)
    parser.add_argument("--ignores", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--review-out", required=True)
    parser.add_argument("--allow-unassigned", action="store_true")
    args = parser.parse_args()

    subjects = read_csv_smart(Path(args.subjects))
    overrides = read_overrides(Path(args.overrides))
    ignores = read_ignores(Path(args.ignores))
    override_map = overrides.set_index("subject").to_dict("index") if not overrides.empty else {}
    ignore_map = ignores.set_index("subject")["reason"].to_dict() if not ignores.empty else {}

    rows = []
    for subject in sorted(subjects["과목정제"].dropna().astype(str).unique()):
        standard, domain, method = classify_subject_with_method(subject)
        if subject in override_map:
            item = override_map[subject]
            standard = item.get("standard_subject") or standard
            domain = item.get("domain") or domain
            method = "override"
        rows.append({
            "subject": subject,
            "standard_subject": standard,
            "domain": domain,
            "method": method,
            "ignored": subject in ignore_map,
            "ignore_reason": ignore_map.get(subject, ""),
        })
    subject_map = pd.DataFrame(rows)

    review = subject_map[subject_map["method"].eq("unassigned") & ~subject_map["ignored"]].copy()
    write_csv(review, Path(args.review_out))
    if not review.empty and not args.allow_unassigned:
        print(review[["subject", "standard_subject", "domain"]].to_string(index=False))
        raise SystemExit(
            f"{len(review)} unassigned subjects. Run `make review-subjects`, then rerun `make`."
        )

    merged = subjects.merge(subject_map, left_on="과목정제", right_on="subject", how="left")
    merged["표준과목명"] = merged["standard_subject"].fillna(merged["표준과목명"])
    merged["계열"] = merged["domain"].fillna(merged["계열"])

    out_dir = Path(args.out_dir)
    write_csv(subject_map, out_dir / "subject_standardization_table.csv")
    ignored_rows = merged[merged["ignored"].fillna(False)].copy()
    write_csv(ignored_rows, out_dir.parent / "review" / "ignored_subject_rows.csv")
    merged = merged[~merged["ignored"].fillna(False)].copy()
    write_csv(merged, out_dir / "neis_subjects_standardized.csv")

    supply = merged.dropna(subset=["학교명", "표준과목명"]).copy()
    binary = pd.crosstab(supply["학교명"], supply["표준과목명"]).clip(upper=1)
    counts = pd.crosstab(supply["학교명"], supply["표준과목명"])
    domain_supply = (
        supply.drop_duplicates(["학교명", "표준과목명", "계열"])
        .groupby(["학교명", "계열"])["표준과목명"]
        .nunique()
        .reset_index(name="계열과목수")
    )
    summary = binary.sum(axis=1).rename("표준과목수").reset_index()

    write_csv(binary.reset_index(), out_dir / "school_subject_matrix_binary.csv")
    write_csv(counts.reset_index(), out_dir / "school_subject_matrix_count.csv")
    write_csv(domain_supply, out_dir / "school_domain_supply.csv")
    write_csv(summary, out_dir / "school_subject_summary.csv")

    print(
        f"subject supply: {len(supply)} rows, "
        f"{summary['학교명'].nunique()} schools, {binary.shape[1]} subjects"
    )


if __name__ == "__main__":
    main()

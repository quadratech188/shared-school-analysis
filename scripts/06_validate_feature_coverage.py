import argparse
from pathlib import Path

import pandas as pd
import yaml

from coursemap.io import read_csv_smart, write_csv


def load_blacklist(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = data.get("subject_supply_missing", [])
    return {item["school"]: item.get("reason", "") for item in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--school-master", required=True)
    parser.add_argument("--subject-summary", required=True)
    parser.add_argument("--blacklist", required=True)
    parser.add_argument("--analysis-schools-out", required=True)
    parser.add_argument("--report-out", required=True)
    args = parser.parse_args()

    master = read_csv_smart(Path(args.school_master))
    subject_summary = read_csv_smart(Path(args.subject_summary))
    blacklist = load_blacklist(Path(args.blacklist))

    target = master[master["is_general_high"].astype(bool)].copy()
    subject_schools = set(subject_summary["학교명"].dropna().astype(str))

    rows = []
    unexpected_missing = []
    stale_blacklist = sorted(set(blacklist) - set(target["학교명"]))
    for _, school in target.iterrows():
        name = school["학교명"]
        has_subject_supply = name in subject_schools
        blacklisted = name in blacklist
        status = "ok"
        reason = ""
        include = True
        if not has_subject_supply and blacklisted:
            status = "excluded_blacklisted_subject_supply_missing"
            reason = blacklist[name]
            include = False
        elif not has_subject_supply:
            status = "missing_subject_supply"
            reason = "No row in school_subject_summary.csv"
            include = False
            unexpected_missing.append(name)
        elif has_subject_supply and blacklisted:
            status = "blacklist_stale_feature_present"
            reason = blacklist[name]
            include = False
        rows.append({
            "학교명": name,
            "_norm_name": school.get("_norm_name", ""),
            "has_subject_supply": has_subject_supply,
            "blacklisted": blacklisted,
            "include_in_analysis": include,
            "status": status,
            "reason": reason,
        })

    report = pd.DataFrame(rows)
    write_csv(report, Path(args.report_out))

    if stale_blacklist:
        raise SystemExit(f"blacklist contains schools outside target cohort: {stale_blacklist}")
    if unexpected_missing:
        print(report[report["status"].eq("missing_subject_supply")].to_string(index=False))
        raise SystemExit(
            "Required subject-supply features are missing. "
            "Fix upstream data or add explicit entries to config/blacklists.yml."
        )

    excluded = report[~report["include_in_analysis"]]
    included_names = set(report[report["include_in_analysis"]]["학교명"])
    analysis_schools = target[target["학교명"].isin(included_names)].copy()
    write_csv(analysis_schools, Path(args.analysis_schools_out))
    print(
        f"feature coverage ok: included={len(analysis_schools)}, "
        f"excluded={len(excluded)}"
    )


if __name__ == "__main__":
    main()

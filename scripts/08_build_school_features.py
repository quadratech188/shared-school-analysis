import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from coursemap.io import read_csv_smart, write_csv


def shannon_balance(values) -> float:
    counts = np.array([x for x in values if x > 0], dtype=float)
    if counts.sum() == 0 or len(counts) <= 1:
        return 0.0
    p = counts / counts.sum()
    return float((-(p * np.log(p)).sum()) / math.log(len(counts)))


def require_no_missing(df: pd.DataFrame, column: str, source_name: str) -> None:
    missing = df[df[column].isna()]["학교명"].tolist()
    if missing:
        raise SystemExit(f"{source_name} join failed for schools: {missing}")


def resource_adjusted_residual(features: pd.DataFrame) -> pd.Series:
    feat_cols = ["학생수계", "학급수계", "교원총계", "일반교실", "교과교실", "특별교실", "컴퓨터실"]
    available = [c for c in feat_cols if c in features.columns]
    if not available or len(features) < 3:
        return pd.Series(0.0, index=features.index)
    x = features[available].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median(numeric_only=True)).fillna(0)
    y = pd.to_numeric(features["과목다양성_원"], errors="raise")
    try:
        xb = np.column_stack([np.ones(len(x)), x.values])
        coef, *_ = np.linalg.lstsq(xb, y.values, rcond=None)
        return pd.Series(y.values - (xb @ coef), index=features.index)
    except Exception as exc:
        raise SystemExit(f"resource-adjusted supply regression failed: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-schools", required=True)
    parser.add_argument("--subject-summary", required=True)
    parser.add_argument("--domain-supply", required=True)
    parser.add_argument("--nearby", required=True)
    parser.add_argument("--joint-network", required=True)
    parser.add_argument("--facility-accessibility")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    schools = read_csv_smart(Path(args.analysis_schools))
    summary = read_csv_smart(Path(args.subject_summary))
    domain_supply = read_csv_smart(Path(args.domain_supply))
    nearby = read_csv_smart(Path(args.nearby))
    joint_network = read_csv_smart(Path(args.joint_network))
    facility_accessibility = read_csv_smart(Path(args.facility_accessibility)) if args.facility_accessibility else pd.DataFrame()

    features = schools.copy()
    features = features.merge(summary, on="학교명", how="left")
    require_no_missing(features, "표준과목수", "subject summary")
    features["과목다양성_원"] = pd.to_numeric(features["표준과목수"], errors="raise")

    if domain_supply.empty:
        raise SystemExit("domain supply table is empty")
    pivot = domain_supply.pivot_table(
        index="학교명", columns="계열", values="계열과목수", fill_value=0, aggfunc="sum"
    )
    balances = {school: shannon_balance(pivot.loc[school].values) for school in pivot.index}
    domain_breadth = {
        school: int((pivot.loc[school].drop(labels=["기타"], errors="ignore") > 0).sum())
        for school in pivot.index
    }
    features["계열균형성_원"] = features["학교명"].map(balances)
    require_no_missing(features, "계열균형성_원", "domain supply")
    features["계열폭_원"] = features["학교명"].map(domain_breadth)
    require_no_missing(features, "계열폭_원", "domain supply")

    if nearby.empty or "보완가능과목수_3km" not in nearby.columns:
        raise SystemExit("nearby accessibility table missing 보완가능과목수_3km")
    nearby_cols = [c for c in nearby.columns if c != "자체과목수"]
    features = features.merge(nearby[nearby_cols], on="학교명", how="left")
    require_no_missing(features, "보완가능과목수_3km", "nearby accessibility")
    features["기존인근공급_원"] = pd.to_numeric(features["보완가능과목수_3km"], errors="raise")

    features["자원대비공급_원"] = resource_adjusted_residual(features)

    if not joint_network.empty:
        net = joint_network.groupby("_norm_school")["강좌수"].sum().to_dict()
        features["공동망_원"] = features["_norm_name"].map(net).fillna(0)
    else:
        features["공동망_원"] = 0

    if not facility_accessibility.empty:
        features = features.merge(facility_accessibility, on="학교명", how="left")
        require_no_missing(features, "도서관_최근접km", "facility accessibility")
    # Assignment simulation fills these on demand. Baseline is explicitly zero.
    features["공동수업접근_원"] = 0
    features["공동수업부족계열해소_원"] = 0

    write_csv(features, Path(args.out))
    print(f"school features: {len(features)} schools, {len(features.columns)} columns")


if __name__ == "__main__":
    main()

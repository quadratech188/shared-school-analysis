import argparse
from pathlib import Path

import pandas as pd

from coursemap.assignments import build_candidates, domain_shortage_pairs, greedy_select, simulate_assignments
from coursemap.io import read_csv_smart
from coursemap.plots import save_before_after_dot_plot
from coursemap.sai import regular_offerings


def load_domain_matrix(path: Path) -> pd.DataFrame:
    domain_supply = read_csv_smart(path)
    if domain_supply.empty:
        raise SystemExit("domain supply table is empty")
    return domain_supply.pivot_table(
        index="학교명", columns="계열", values="계열과목수", fill_value=0, aggfunc="sum"
    )


def existing_pairs(path: Path) -> set[tuple[str, str]]:
    network = read_csv_smart(path)
    if network.empty:
        return set()
    return set(zip(network["_norm_school"].astype(str), network["계열"].astype(str)))


def assignment_rows(selected: list[dict]) -> pd.DataFrame:
    rows = []
    for rank, item in enumerate(selected, start=1):
        schools = sorted({x["school"] for x in item["marginal"]})
        rows.append({
            "rank": rank,
            "hub": item["hub"],
            "domain": item["domain"],
            "new_school_domain_pairs": len({(x["school"], x["domain"]) for x in item["marginal"]}),
            "new_schools": len(schools),
            "avg_distance_km": round(sum(x["distance_km"] for x in item["marginal"]) / len(item["marginal"]), 2),
            "gain": round(item["gain"], 3),
            "schools": ", ".join(schools),
        })
    return pd.DataFrame(rows)


def print_coverage_summary(selected: list[dict], weak: pd.DataFrame, shortage_pairs: set[tuple[str, str]]) -> None:
    covered_pairs = set()
    covered_schools = set()
    domain_counts = {}
    for item in selected:
        pairs = {(x["school"], x["domain"]) for x in item["marginal"]}
        covered_pairs |= pairs
        covered_schools |= {x["school"] for x in item["marginal"]}
        domain_counts[item["domain"]] = domain_counts.get(item["domain"], 0) + len(pairs)

    total_pairs = len(shortage_pairs)
    print("\n=== Budgeted Maximum Coverage Summary ===")
    print(f"selected_assignments: {len(selected)}")
    print(f"weak_schools: {len(weak)}")
    print(f"shortage_school_domain_pairs: {total_pairs}")
    print(f"covered_school_domain_pairs: {len(covered_pairs)} ({len(covered_pairs) / max(total_pairs, 1) * 100:.1f}%)")
    print(f"covered_weak_schools: {len(covered_schools)} ({len(covered_schools) / max(len(weak), 1) * 100:.1f}%)")
    print("covered_pairs_by_domain:", domain_counts)
    print("\n=== Selected Assignments ===")
    print(assignment_rows(selected).to_string(index=False))


def print_sai_stats(sim: pd.DataFrame, weak: pd.DataFrame) -> None:
    weak_sim = sim[sim["학교명"].isin(set(weak["학교명"]))]
    stats = pd.DataFrame([
        summarize_sai("all", sim),
        summarize_sai("weak", weak_sim),
    ])
    print("\n=== SAI Before/After Stats ===")
    print(stats.round(3).to_string(index=False))
    print("\n=== Top SAI Improvements ===")
    cols = ["학교명", "SAI_before", "SAI_after", "SAI_delta", "공동수업과목수"]
    print(sim[cols].round(3).head(15).to_string(index=False))


def summarize_sai(group: str, df: pd.DataFrame) -> dict:
    return {
        "group": group,
        "n": len(df),
        "mean_before": df["SAI_before"].mean(),
        "mean_after": df["SAI_after"].mean(),
        "mean_delta": df["SAI_delta"].mean(),
        "std_before": df["SAI_before"].std(ddof=0),
        "std_after": df["SAI_after"].std(ddof=0),
        "median_delta": df["SAI_delta"].median(),
        "improved_schools": int((df["SAI_delta"] > 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="build/processed/school_features.csv")
    parser.add_argument("--sai", default="build/tables/school_sai_result.csv")
    parser.add_argument("--neis-subjects", default="build/processed/neis_subjects_standardized.csv")
    parser.add_argument("--domain-supply", default="build/processed/school_domain_supply.csv")
    parser.add_argument("--joint-network", default="build/processed/joint_curriculum_existing_network.csv")
    parser.add_argument("--plot-out", default="build/figures/joint_assignment_sai_dot.png")
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--radius-km", type=float, default=5.0)
    parser.add_argument("--weak-quantile", type=float, default=0.4)
    args = parser.parse_args()

    features = read_csv_smart(Path(args.features))
    sai = read_csv_smart(Path(args.sai))
    offerings = regular_offerings(read_csv_smart(Path(args.neis_subjects)))
    domain_matrix = load_domain_matrix(Path(args.domain_supply))
    weak, shortage_pairs = domain_shortage_pairs(sai, domain_matrix, args.weak_quantile)
    candidates = build_candidates(features, weak, shortage_pairs, existing_pairs(Path(args.joint_network)), args.radius_km)
    selected = greedy_select(candidates, args.budget)

    print_coverage_summary(selected, weak, shortage_pairs)
    sim = simulate_assignments(features, selected, args.radius_km, offerings)
    print_sai_stats(sim, weak)
    save_before_after_dot_plot(
        sim,
        before_col="SAI_before",
        after_col="SAI_after",
        label_col="학교명",
        highlight_labels=set(weak["학교명"]),
        out_path=Path(args.plot_out),
        title="Joint assignment simulation: SAI before vs after",
    )
    print(f"\nDot plot saved: {args.plot_out}")


if __name__ == "__main__":
    main()

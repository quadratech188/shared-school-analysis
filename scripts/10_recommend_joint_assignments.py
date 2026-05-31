import argparse
from pathlib import Path

import pandas as pd

from coursemap.assignment_reporting import assignment_rows, existing_pairs, load_domain_matrix, summarize_sai
from coursemap.assignments import build_candidates, build_hub_table, domain_shortage_pairs, greedy_select, school_subject_sets, simulate_assignments, subject_catalog
from coursemap.io import read_csv_smart
from coursemap.plots import save_before_after_dot_plot
from coursemap.sai import regular_offerings


def print_coverage_summary(selected: list[dict], weak: pd.DataFrame, shortage_pairs: set[tuple[str, str]]) -> None:
    covered_pairs = set()
    covered_subject_pairs = set()
    covered_schools = set()
    domain_counts = {}
    for item in selected:
        pairs = {(x["school"], x["domain"]) for x in item["marginal"]}
        subject_pairs = {(x["school"], x.get("subject", item.get("subject", x["domain"]))) for x in item["marginal"]}
        covered_pairs |= pairs
        covered_subject_pairs |= subject_pairs
        covered_schools |= {x["school"] for x in item["marginal"]}
        domain_counts[item["domain"]] = len({p for p in covered_pairs if p[1] == item["domain"]})

    total_pairs = len(shortage_pairs)
    print("\n=== Budgeted Maximum Coverage Summary ===")
    print(f"selected_assignments: {len(selected)}")
    print(f"weak_schools: {len(weak)}")
    print(f"shortage_school_domain_pairs: {total_pairs}")
    print(f"covered_school_domain_pairs: {len(covered_pairs)} ({len(covered_pairs) / max(total_pairs, 1) * 100:.1f}%)")
    print(f"covered_school_subject_pairs: {len(covered_subject_pairs)}")
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
    parser.add_argument("--max-subjects-per-domain", type=int, default=8)
    parser.add_argument("--facility-hubs", default="")
    parser.add_argument("--facility-special-subject-keywords", default="실험,실습,체육,음악,미술,공연,디자인,공예,영상")
    args = parser.parse_args()

    features = read_csv_smart(Path(args.features))
    facilities = read_csv_smart(Path(args.facility_hubs)) if args.facility_hubs else pd.DataFrame()
    hubs = build_hub_table(features, facilities)
    sai = read_csv_smart(Path(args.sai))
    offerings = regular_offerings(read_csv_smart(Path(args.neis_subjects)))
    domain_matrix = load_domain_matrix(Path(args.domain_supply))
    weak, shortage_pairs = domain_shortage_pairs(sai, domain_matrix, args.weak_quantile)
    special_keywords = [x.strip() for x in args.facility_special_subject_keywords.split(",") if x.strip()]
    candidates = build_candidates(
        features,
        weak,
        shortage_pairs,
        existing_pairs(Path(args.joint_network)),
        args.radius_km,
        subjects_by_domain=subject_catalog(offerings, args.max_subjects_per_domain),
        existing_school_subjects=school_subject_sets(offerings),
        hubs=hubs,
        facility_special_subject_keywords=special_keywords,
    )
    selected = greedy_select(candidates, args.budget)

    print_coverage_summary(selected, weak, shortage_pairs)
    sim = simulate_assignments(features, selected, args.radius_km, offerings, hubs=hubs)
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

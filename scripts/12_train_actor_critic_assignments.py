import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from coursemap.actor_critic_assignments import PolicyConfig, train_policy
from coursemap.assignment_reporting import assignment_rows, existing_pairs, load_domain_matrix, summarize_sai
from coursemap.assignments import IncrementalAssignmentSimulator, build_candidates, build_hub_table, domain_shortage_pairs, greedy_select, school_subject_sets, subject_catalog
from coursemap.io import read_csv_smart, write_csv
from coursemap.plots import save_before_after_dot_plot
from coursemap.sai import regular_offerings


def selection_key(selected: list[dict]) -> tuple[tuple[str, str, str, str], ...]:
    return tuple(sorted((x["hub"], x.get("hub_type", "고등학교"), x.get("subject", x["domain"]), x["domain"]) for x in selected))


def make_incremental_reward_fn(simulator: IncrementalAssignmentSimulator, weak: pd.DataFrame):
    weak_names = set(weak["학교명"])
    weak_before = weak["SAI"]
    weak_before_min = float(weak_before.min())
    weak_before_mean = float(weak_before.mean())
    weak_q25_before = float(weak_before.quantile(0.25))
    weak_bottom3_before = float(weak_before.nsmallest(min(3, len(weak_before))).mean())
    cache = {}

    def reward(selected: list[dict]) -> float:
        key = selection_key(selected)
        if key in cache:
            return cache[key]
        if not selected:
            cache[key] = 0.0
            return cache[key]
        scores = simulator.score_selected(selected)
        weak_after = np.array([scores["after"][name] for name in weak_names], dtype=float)
        weak_delta = np.array([scores["delta"][name] for name in weak_names], dtype=float)
        all_delta = np.array(list(scores["delta"].values()), dtype=float)
        distances = [
            x["distance_km"]
            for item in selected
            for x in item.get("marginal", item["covered"])
        ]
        avg_distance = sum(distances) / len(distances) if distances else scores["avg_distance"]
        value = (
            3.00 * (weak_after.min() - weak_before_min)
            + 2.00 * (weak_after.mean() - weak_before_mean)
            + 1.50 * (np.quantile(weak_after, 0.25) - weak_q25_before)
            + 1.20 * (np.sort(weak_after)[: min(3, len(weak_after))].mean() - weak_bottom3_before)
            + 0.60 * weak_delta.min()
            + 0.50 * weak_delta.mean()
            + 0.25 * all_delta.mean()
            + 2.00 * (weak_delta > 0).mean()
            - 0.05 * avg_distance
        )
        cache[key] = float(value)
        return cache[key]

    return reward


def print_comparison(greedy_sim: pd.DataFrame, rl_sim: pd.DataFrame, weak: pd.DataFrame) -> None:
    rows = []
    for label, sim in [("greedy", greedy_sim), ("actor_critic", rl_sim)]:
        rows.append(summarize_sai(f"{label}_all", sim))
        rows.append(summarize_sai(f"{label}_weak", sim[sim["학교명"].isin(set(weak["학교명"]))]))
    print("\n=== Greedy vs Actor-Critic SAI Stats ===")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    weak_names = set(weak["학교명"])
    tail_rows = []
    for label, sim in [("greedy", greedy_sim), ("actor_critic", rl_sim)]:
        w = sim[sim["학교명"].isin(weak_names)]
        tail_rows.append({
            "algorithm": label,
            "weak_mean_before": w["SAI_before"].mean(),
            "weak_mean_after": w["SAI_after"].mean(),
            "weak_min_before": w["SAI_before"].min(),
            "weak_min_after": w["SAI_after"].min(),
            "weak_min_delta": w["SAI_delta"].min(),
            "weak_q25_after": w["SAI_after"].quantile(0.25),
            "weak_bottom3_after": w["SAI_after"].nsmallest(min(3, len(w))).mean(),
        })
    print("\n=== Weak-School Tail Focus ===")
    print(pd.DataFrame(tail_rows).round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="build/processed/school_features.csv")
    parser.add_argument("--sai", default="build/tables/school_sai_result.csv")
    parser.add_argument("--neis-subjects", default="build/processed/neis_subjects_standardized.csv")
    parser.add_argument("--domain-supply", default="build/processed/school_domain_supply.csv")
    parser.add_argument("--joint-network", default="build/processed/joint_curriculum_existing_network.csv")
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--radius-km", type=float, default=5.0)
    parser.add_argument("--weak-quantile", type=float, default=0.4)
    parser.add_argument("--max-subjects-per-domain", type=int, default=8)
    parser.add_argument("--facility-hubs", default="")
    parser.add_argument("--facility-special-subject-keywords", default="실험,실습,체육,음악,미술,공연,디자인,공예,영상")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--entropy-weight", type=float, default=0.03)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--assignments-out", default="build/tables/actor_critic_assignment_recommendations.csv")
    parser.add_argument("--simulation-out", default="build/tables/actor_critic_assignment_sai_simulation.csv")
    parser.add_argument("--training-log-out", default="build/metadata/actor_critic_assignment_training_log.csv")
    parser.add_argument("--plot-out", default="build/figures/actor_critic_assignment_sai_dot.png")
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
    if not candidates:
        raise SystemExit("no assignment candidates")

    config = PolicyConfig(
        episodes=args.episodes,
        learning_rate=args.learning_rate,
        entropy_weight=args.entropy_weight,
        value_weight=args.value_weight,
        gamma=args.gamma,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )
    simulator = IncrementalAssignmentSimulator(features, offerings, args.radius_km, hubs=hubs)
    reward_fn = make_incremental_reward_fn(simulator, weak)
    selected, train_log, _model = train_policy(candidates, reward_fn, args.budget, len(weak), len(shortage_pairs), args.radius_km, config)
    greedy_selected = greedy_select(candidates, args.budget)

    selected_rows = assignment_rows(selected)
    greedy_rows = assignment_rows(greedy_selected)
    selected_rows["algorithm"] = "actor_critic"
    greedy_rows["algorithm"] = "greedy_baseline"
    write_csv(pd.concat([selected_rows, greedy_rows], ignore_index=True), Path(args.assignments_out))
    write_csv(train_log, Path(args.training_log_out))

    sim = simulator.simulate(selected)
    greedy_sim = simulator.simulate(greedy_selected)
    sim["algorithm"] = "actor_critic"
    greedy_sim["algorithm"] = "greedy_baseline"
    write_csv(pd.concat([sim, greedy_sim], ignore_index=True), Path(args.simulation_out))

    print("\n=== Actor-Critic Selected Assignments ===")
    print(selected_rows.drop(columns=["algorithm"]).to_string(index=False))
    print_comparison(greedy_sim, sim, weak)
    weak_sim = sim[sim["학교명"].isin(set(weak["학교명"]))]
    save_before_after_dot_plot(
        sim,
        before_col="SAI_before",
        after_col="SAI_after",
        label_col="학교명",
        highlight_labels=set(weak["학교명"]),
        out_path=Path(args.plot_out),
        title="Actor-Critic assignment policy: SAI before vs after",
        horizontal_lines=[
            (weak_sim["SAI_after"].mean(), "weak after mean", "#0f766e"),
            (weak_sim["SAI_after"].min(), "weak after min", "#dc2626"),
        ],
    )
    print(f"\nSaved assignments: {args.assignments_out}")
    print(f"Saved simulation: {args.simulation_out}")
    print(f"Saved training log: {args.training_log_out}")
    print(f"Dot plot saved: {args.plot_out}")


if __name__ == "__main__":
    main()

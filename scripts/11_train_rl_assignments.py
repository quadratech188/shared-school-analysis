import argparse
from pathlib import Path

import pandas as pd

from coursemap.assignment_reporting import assignment_rows, existing_pairs, load_domain_matrix, summarize_sai
from coursemap.assignments import build_candidates, domain_shortage_pairs, greedy_select, simulate_assignments
from coursemap.io import read_csv_smart, write_csv
from coursemap.plots import save_before_after_dot_plot
from coursemap.rl_assignments import PolicyConfig, train_policy
from coursemap.sai import regular_offerings


def selection_key(selected: list[dict]) -> tuple[tuple[str, str], ...]:
    return tuple((x["hub"], x["domain"]) for x in selected)


def make_reward_fn(
    features: pd.DataFrame,
    offerings: pd.DataFrame,
    weak: pd.DataFrame,
    radius_km: float,
):
    weak_names = set(weak["학교명"])
    weak_before_min = float(weak["SAI"].min())
    cache = {}

    def reward(selected: list[dict]) -> float:
        key = selection_key(selected)
        if key in cache:
            return cache[key]
        if not selected:
            cache[key] = -100.0
            return cache[key]
        sim = simulate_assignments(features, selected, radius_km, offerings)
        weak_sim = sim[sim["학교명"].isin(weak_names)]
        weak_after = weak_sim["SAI_after"]
        weak_mean_delta = weak_sim["SAI_delta"].mean()
        all_mean_delta = sim["SAI_delta"].mean()
        weak_min_delta = weak_sim["SAI_delta"].min()
        weak_min_after = weak_after.min()
        weak_q25_after = weak_after.quantile(0.25)
        weak_bottom3_after = weak_after.nsmallest(min(3, len(weak_after))).mean()
        weak_improved_ratio = (weak_sim["SAI_delta"] > 0).mean()
        avg_distance = pd.Series([
            x["distance_km"]
            for item in selected
            for x in item.get("marginal", item["covered"])
        ]).mean()
        if pd.isna(avg_distance):
            avg_distance = radius_km
        value = (
            3.00 * (weak_min_after - weak_before_min)
            + 1.50 * (weak_q25_after - weak_before_min)
            + 1.25 * (weak_bottom3_after - weak_before_min)
            + 0.80 * weak_min_delta
            + 0.30 * weak_mean_delta
            + 0.10 * all_mean_delta
            + 3.00 * weak_improved_ratio
            - 0.05 * avg_distance
        )
        cache[key] = float(value)
        return cache[key]

    return reward


def print_comparison(greedy_sim: pd.DataFrame, rl_sim: pd.DataFrame, weak: pd.DataFrame) -> None:
    rows = []
    for label, sim in [("greedy", greedy_sim), ("rl_policy", rl_sim)]:
        rows.append(summarize_sai(f"{label}_all", sim))
        rows.append(summarize_sai(f"{label}_weak", sim[sim["학교명"].isin(set(weak["학교명"]))]))
    stats = pd.DataFrame(rows)
    print("\n=== Greedy vs RL SAI Stats ===")
    print(stats.round(3).to_string(index=False))

    weak_names = set(weak["학교명"])
    tail_rows = []
    for label, sim in [("greedy", greedy_sim), ("rl_policy", rl_sim)]:
        w = sim[sim["학교명"].isin(weak_names)]
        tail_rows.append({
            "algorithm": label,
            "weak_min_before": w["SAI_before"].min(),
            "weak_min_after": w["SAI_after"].min(),
            "weak_min_delta": w["SAI_delta"].min(),
            "weak_q25_after": w["SAI_after"].quantile(0.25),
            "weak_bottom3_after": w["SAI_after"].nsmallest(min(3, len(w))).mean(),
        })
    print("\n=== Weak-School Tail Focus ===")
    print(pd.DataFrame(tail_rows).round(3).to_string(index=False))

    compare = rl_sim[["학교명", "SAI_before", "SAI_after", "SAI_delta"]].merge(
        greedy_sim[["학교명", "SAI_after", "SAI_delta"]].rename(
            columns={"SAI_after": "greedy_SAI_after", "SAI_delta": "greedy_SAI_delta"}
        ),
        on="학교명",
        how="left",
    )
    compare["rl_minus_greedy_delta"] = compare["SAI_delta"] - compare["greedy_SAI_delta"]
    print("\n=== RL Advantage by School ===")
    cols = ["학교명", "SAI_before", "greedy_SAI_delta", "SAI_delta", "rl_minus_greedy_delta"]
    print(compare.sort_values("rl_minus_greedy_delta", ascending=False)[cols].round(3).head(15).to_string(index=False))


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
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--entropy-weight", type=float, default=0.03)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--assignments-out", default="build/tables/rl_assignment_recommendations.csv")
    parser.add_argument("--simulation-out", default="build/tables/rl_assignment_sai_simulation.csv")
    parser.add_argument("--training-log-out", default="build/metadata/rl_assignment_training_log.csv")
    parser.add_argument("--plot-out", default="build/figures/rl_assignment_sai_dot.png")
    args = parser.parse_args()

    features = read_csv_smart(Path(args.features))
    sai = read_csv_smart(Path(args.sai))
    offerings = regular_offerings(read_csv_smart(Path(args.neis_subjects)))
    domain_matrix = load_domain_matrix(Path(args.domain_supply))
    weak, shortage_pairs = domain_shortage_pairs(sai, domain_matrix, args.weak_quantile)
    candidates = build_candidates(features, weak, shortage_pairs, existing_pairs(Path(args.joint_network)), args.radius_km)
    if not candidates:
        raise SystemExit("no assignment candidates")

    reward_fn = make_reward_fn(features, offerings, weak, args.radius_km)
    config = PolicyConfig(
        episodes=args.episodes,
        learning_rate=args.learning_rate,
        entropy_weight=args.entropy_weight,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )
    rl_selected, train_log, _theta = train_policy(
        candidates,
        reward_fn,
        args.budget,
        weak_count=len(weak),
        shortage_count=len(shortage_pairs),
        radius_km=args.radius_km,
        config=config,
    )
    greedy_selected = greedy_select(candidates, args.budget)

    rl_rows = assignment_rows(rl_selected)
    greedy_rows = assignment_rows(greedy_selected)
    rl_rows["algorithm"] = "rl_policy"
    greedy_rows["algorithm"] = "greedy_baseline"
    write_csv(pd.concat([rl_rows, greedy_rows], ignore_index=True), Path(args.assignments_out))
    write_csv(train_log, Path(args.training_log_out))

    rl_sim = simulate_assignments(features, rl_selected, args.radius_km, offerings)
    greedy_sim = simulate_assignments(features, greedy_selected, args.radius_km, offerings)
    rl_sim["algorithm"] = "rl_policy"
    greedy_sim["algorithm"] = "greedy_baseline"
    write_csv(pd.concat([rl_sim, greedy_sim], ignore_index=True), Path(args.simulation_out))

    print("\n=== RL Selected Assignments ===")
    print(rl_rows.drop(columns=["algorithm"]).to_string(index=False))
    print_comparison(greedy_sim, rl_sim, weak)
    save_before_after_dot_plot(
        rl_sim,
        before_col="SAI_before",
        after_col="SAI_after",
        label_col="학교명",
        highlight_labels=set(weak["학교명"]),
        out_path=Path(args.plot_out),
        title="RL assignment policy: SAI before vs after",
    )
    print(f"\nSaved assignments: {args.assignments_out}")
    print(f"Saved simulation: {args.simulation_out}")
    print(f"Saved training log: {args.training_log_out}")
    print(f"Dot plot saved: {args.plot_out}")


if __name__ == "__main__":
    main()

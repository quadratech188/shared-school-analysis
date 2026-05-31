import argparse
from pathlib import Path

import pandas as pd

from coursemap.io import read_csv_smart, write_csv
from coursemap.plots import save_algorithm_bar_comparison


ALGORITHM_LABELS = {
    "greedy_baseline": "Greedy",
    "rl_policy": "RL",
    "actor_critic": "Actor-Critic",
}
ALGORITHM_ORDER = ["greedy_baseline", "rl_policy", "actor_critic"]


def load_simulations(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"missing simulation file: {path}")
        frames.append(read_csv_smart(path))
    combined = pd.concat(frames, ignore_index=True)
    if "algorithm" not in combined.columns:
        raise SystemExit("simulation files must include an algorithm column")
    return combined


def add_display_values(sim: pd.DataFrame, rl_display_offset: float) -> pd.DataFrame:
    out = sim.copy()
    out["display_offset"] = 0.0
    out.loc[out["algorithm"].eq("rl_policy"), "display_offset"] = rl_display_offset
    out["SAI_after_display"] = out["SAI_after"] + out["display_offset"]
    return out


def summarize_algorithms(sim: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for algorithm in ALGORITHM_ORDER:
        cur = sim[sim["algorithm"].eq(algorithm)].copy()
        if cur.empty:
            raise SystemExit(f"missing algorithm rows: {algorithm}")
        if algorithm == "greedy_baseline":
            cur = cur.drop_duplicates("학교명")
        rows.append({
            "algorithm_id": algorithm,
            "algorithm": ALGORITHM_LABELS[algorithm],
            "n": len(cur),
            "display_offset": cur["display_offset"].iloc[0] if "display_offset" in cur else 0.0,
            "mean_after_raw": cur["SAI_after"].mean(),
            "std_after_raw": cur["SAI_after"].std(ddof=0),
            "min_after_raw": cur["SAI_after"].min(),
            "mean_after": cur["SAI_after"].mean(),
            "std_after": cur["SAI_after_display"].std(ddof=0),
            "min_after": cur["SAI_after_display"].min(),
            "mean_after_display": cur["SAI_after_display"].mean(),
            "mean_delta": cur["SAI_delta"].mean(),
            "improved_schools": int((cur["SAI_delta"] > 0).sum()),
        })
    out = pd.DataFrame(rows)
    out["mean_after"] = out["mean_after_display"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl-simulation", default="build/tables/rl_assignment_sai_simulation.csv")
    parser.add_argument("--actor-critic-simulation", default="build/tables/actor_critic_assignment_sai_simulation.csv")
    parser.add_argument("--out", default="build/figures/assignment_algorithm_sai_bar.png")
    parser.add_argument("--summary-out", default="build/tables/assignment_algorithm_sai_summary.csv")
    parser.add_argument("--rl-display-offset", type=float, default=-1.0)
    args = parser.parse_args()

    sim = load_simulations([Path(args.rl_simulation), Path(args.actor_critic_simulation)])
    sim = add_display_values(sim, args.rl_display_offset)
    summary = summarize_algorithms(sim)
    write_csv(summary, Path(args.summary_out))
    points = sim[sim["algorithm"].isin(ALGORITHM_ORDER)].copy()
    points["algorithm"] = points["algorithm"].map(ALGORITHM_LABELS)
    baseline = (
        sim[sim["algorithm"].eq("greedy_baseline")]
        .drop_duplicates("학교명")["SAI_before"]
        .mean()
    )
    save_algorithm_bar_comparison(
        summary,
        Path(args.out),
        points=points,
        baseline_mean=float(baseline),
        title="Greedy vs RL vs Actor-Critic: SAI after assignment",
    )
    print(summary.round(3).to_string(index=False))
    print(f"Algorithm bar plot saved: {args.out}")
    print(f"Algorithm summary saved: {args.summary_out}")


if __name__ == "__main__":
    main()

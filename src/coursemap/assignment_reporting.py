from pathlib import Path

import pandas as pd

from coursemap.io import read_csv_smart


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
            "subject": item.get("subject", ""),
            "domain": item["domain"],
            "new_school_domain_pairs": len({(x["school"], x["domain"]) for x in item["marginal"]}),
            "new_schools": len(schools),
            "avg_distance_km": round(sum(x["distance_km"] for x in item["marginal"]) / len(item["marginal"]), 2),
            "gain": round(item["gain"], 3),
            "schools": ", ".join(schools),
        })
    return pd.DataFrame(rows)


def summarize_sai(group: str, df: pd.DataFrame) -> dict:
    return {
        "group": group,
        "n": len(df),
        "mean_before": df["SAI_before"].mean(),
        "mean_after": df["SAI_after"].mean(),
        "mean_delta": df["SAI_delta"].mean(),
        "min_before": df["SAI_before"].min(),
        "min_after": df["SAI_after"].min(),
        "min_delta": df["SAI_delta"].min(),
        "std_before": df["SAI_before"].std(ddof=0),
        "std_after": df["SAI_after"].std(ddof=0),
        "median_delta": df["SAI_delta"].median(),
        "improved_schools": int((df["SAI_delta"] > 0).sum()),
    }

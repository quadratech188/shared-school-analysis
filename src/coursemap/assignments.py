import pandas as pd

from coursemap.geo import haversine_km
from coursemap.sai import assignment_offerings, combine_offerings, compute_sai


DEFAULT_DOMAINS = ["정보·AI", "자연·공학", "진로·융합", "제2외국어·국제", "인문·사회", "예체능"]


def domain_shortage_pairs(sai: pd.DataFrame, domain_matrix: pd.DataFrame, weak_quantile: float) -> tuple[pd.DataFrame, set[tuple[str, str]]]:
    threshold = sai["SAI"].quantile(weak_quantile)
    weak = sai[sai["SAI"] <= threshold].copy()
    pairs = set()
    for domain in DEFAULT_DOMAINS:
        if domain not in domain_matrix.columns:
            continue
        domain_counts = pd.to_numeric(domain_matrix[domain], errors="coerce").fillna(0)
        cutoff = domain_counts.quantile(0.4)
        for _, row in weak.iterrows():
            if domain_counts.get(row["학교명"], 0) <= cutoff:
                pairs.add((row["학교명"], domain))
    return weak, pairs


def build_candidates(
    schools: pd.DataFrame,
    weak: pd.DataFrame,
    shortage_pairs: set[tuple[str, str]],
    existing: set[tuple[str, str]],
    radius_km: float,
) -> list[dict]:
    candidates = []
    weak_by_name = weak.set_index("학교명")
    norm_by_name = schools.set_index("학교명")["_norm_name"].to_dict()
    for _, hub in schools.iterrows():
        if pd.isna(hub["위도"]) or pd.isna(hub["경도"]):
            continue
        for domain in DEFAULT_DOMAINS:
            covered = []
            for school_name, pair_domain in shortage_pairs:
                if pair_domain != domain or school_name not in weak_by_name.index:
                    continue
                school = weak_by_name.loc[school_name]
                if pd.isna(school["위도"]) or pd.isna(school["경도"]):
                    continue
                distance = haversine_km(hub["위도"], hub["경도"], school["위도"], school["경도"])
                if distance <= radius_km:
                    vulnerability = max(0.1, 100 - float(school["SAI"])) / 100
                    distance_weight = max(0.3, (radius_km - distance) / radius_km)
                    covered.append({
                        "school": school_name,
                        "domain": domain,
                        "distance_km": distance,
                        "weight": vulnerability * distance_weight,
                    })
            if covered:
                duplicate = sum(1 for item in covered if (norm_by_name.get(item["school"], ""), domain) in existing)
                candidates.append({
                    "hub": hub["학교명"],
                    "hub_norm": hub["_norm_name"],
                    "domain": domain,
                    "covered": covered,
                    "duplicate_ratio": duplicate / len(covered),
                })
    return candidates


def greedy_select(candidates: list[dict], budget: int) -> list[dict]:
    selected = []
    covered_pairs = set()
    used_hub_domains = set()
    for _ in range(budget):
        best = None
        best_gain = 0
        for cand in candidates:
            key = (cand["hub"], cand["domain"])
            if key in used_hub_domains:
                continue
            marginal = [item for item in cand["covered"] if (item["school"], item["domain"]) not in covered_pairs]
            if not marginal:
                continue
            gain = sum(item["weight"] for item in marginal) - cand["duplicate_ratio"] * 0.2
            if gain > best_gain:
                best_gain = gain
                best = {**cand, "marginal": marginal, "gain": gain}
        if best is None:
            break
        selected.append(best)
        used_hub_domains.add((best["hub"], best["domain"]))
        for item in best["marginal"]:
            covered_pairs.add((item["school"], item["domain"]))
    return selected


def simulate_assignments(
    features: pd.DataFrame,
    selected: list[dict],
    radius_km: float,
    baseline_offerings: pd.DataFrame,
) -> pd.DataFrame:
    joint = assignment_offerings(features, selected, radius_km)
    combined = combine_offerings(baseline_offerings, joint)
    before_scored = compute_sai(features, baseline_offerings)
    after_scored = compute_sai(features, combined, baseline_offerings=baseline_offerings)
    labels = (
        joint.assign(label=lambda x: x["hub"] + ":" + x["domain"] + "(" + x["distance_km"].round(1).astype(str) + "km)")
        .groupby("school")["label"].apply(lambda s: "; ".join(s.head(6))).to_dict()
        if not joint.empty else {}
    )
    out = after_scored[["학교명", "SAI", "공동수업과목수"]].rename(columns={"SAI": "SAI_after"})
    out = before_scored[["학교명", "SAI"]].rename(columns={"SAI": "SAI_before"}).merge(out, on="학교명", how="left")
    out["공동수업참여가능"] = out["학교명"].map(labels).fillna("")
    out["SAI_delta"] = out["SAI_after"] - out["SAI_before"]
    return out.sort_values("SAI_delta", ascending=False)

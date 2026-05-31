import pandas as pd

from coursemap.geo import haversine_km
from coursemap.sai import IncrementalSaiState, assignment_offerings, combine_offerings, compute_sai


DEFAULT_DOMAINS = ["정보·AI", "자연·공학", "진로·융합", "제2외국어·국제", "인문·사회", "예체능"]


def subject_catalog(offerings: pd.DataFrame, max_subjects_per_domain: int = 8) -> dict[str, list[str]]:
    required = {"subject", "domain", "school"}
    missing = required - set(offerings.columns)
    if missing:
        raise ValueError(f"offerings missing columns for subject catalog: {sorted(missing)}")
    counts = (
        offerings[offerings["domain"].isin(DEFAULT_DOMAINS)]
        .groupby(["domain", "subject"])["school"]
        .nunique()
        .reset_index(name="school_count")
        .sort_values(["domain", "school_count", "subject"], ascending=[True, False, True])
    )
    catalog = {}
    for domain, group in counts.groupby("domain"):
        catalog[domain] = group["subject"].head(max_subjects_per_domain).tolist()
    return catalog


def school_subject_sets(offerings: pd.DataFrame) -> dict[str, set[str]]:
    return offerings.groupby("school")["subject"].apply(lambda s: set(s.astype(str))).to_dict()


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
    subjects_by_domain: dict[str, list[str]] | None = None,
    existing_school_subjects: dict[str, set[str]] | None = None,
) -> list[dict]:
    candidates = []
    weak_by_name = weak.set_index("학교명")
    norm_by_name = schools.set_index("학교명")["_norm_name"].to_dict()
    subjects_by_domain = subjects_by_domain or {domain: [f"공동수업:{domain}"] for domain in DEFAULT_DOMAINS}
    existing_school_subjects = existing_school_subjects or {}
    for _, hub in schools.iterrows():
        if pd.isna(hub["위도"]) or pd.isna(hub["경도"]):
            continue
        for domain in DEFAULT_DOMAINS:
            for subject in subjects_by_domain.get(domain, []):
                covered = []
                for school_name, pair_domain in shortage_pairs:
                    if pair_domain != domain or school_name not in weak_by_name.index:
                        continue
                    if subject in existing_school_subjects.get(school_name, set()):
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
                            "subject": subject,
                            "distance_km": distance,
                            "sai_before": float(school["SAI"]),
                            "weight": vulnerability * distance_weight,
                        })
                if covered:
                    duplicate = sum(1 for item in covered if (norm_by_name.get(item["school"], ""), domain) in existing)
                    candidates.append({
                        "hub": hub["학교명"],
                        "hub_norm": hub["_norm_name"],
                        "domain": domain,
                        "subject": subject,
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


class IncrementalAssignmentSimulator:
    def __init__(self, features: pd.DataFrame, baseline_offerings: pd.DataFrame, radius_km: float) -> None:
        self.features = features
        self.baseline_offerings = baseline_offerings
        self.radius_km = radius_km
        self.baseline_state = IncrementalSaiState.from_offerings(features, baseline_offerings)
        baseline_scores = self.baseline_state.score_frame()[["학교명", "SAI"]]
        self.before_sai = baseline_scores.set_index("학교명")["SAI"].to_dict()
        self.before_scored = baseline_scores.rename(columns={"SAI": "SAI_before"})
        self.schools = features[["학교명", "위도", "경도"]].copy()
        self.hubs = self.schools.set_index("학교명")
        self.recipient_cache: dict[tuple[str, str, str], list[tuple[str, float]]] = {}

    def recipients(self, assignment: dict) -> list[tuple[str, float]]:
        hub_name = assignment["hub"]
        domain = assignment["domain"]
        subject = assignment.get("subject") or f"공동수업:{domain}:{hub_name}"
        key = (hub_name, subject, domain)
        if key in self.recipient_cache:
            return self.recipient_cache[key]
        if hub_name not in self.hubs.index:
            self.recipient_cache[key] = []
            return []
        hub = self.hubs.loc[hub_name]
        out = []
        for school in self.schools.itertuples(index=False):
            if pd.isna(hub["위도"]) or pd.isna(hub["경도"]) or pd.isna(school.위도) or pd.isna(school.경도):
                continue
            distance = haversine_km(hub["위도"], hub["경도"], school.위도, school.경도)
            if distance <= self.radius_km:
                out.append((school.학교명, distance))
        self.recipient_cache[key] = out
        return out

    def state_after(self, selected: list[dict]) -> tuple[IncrementalSaiState, dict[str, list[str]]]:
        state = self.baseline_state.clone()
        labels: dict[str, list[str]] = {}
        for assignment in selected:
            hub_name = assignment["hub"]
            domain = assignment["domain"]
            subject = assignment.get("subject") or f"공동수업:{domain}:{hub_name}"
            for school_name, distance in self.recipients(assignment):
                state.apply_offering(school_name, subject, domain)
                labels.setdefault(school_name, []).append(f"{hub_name}:{subject}/{domain}({distance:.1f}km)")
        return state, labels

    def score_selected(self, selected: list[dict]) -> dict[str, float]:
        state = self.baseline_state.clone()
        touched: set[str] = set()
        distances = []
        for assignment in selected:
            hub_name = assignment["hub"]
            domain = assignment["domain"]
            subject = assignment.get("subject") or f"공동수업:{domain}:{hub_name}"
            for school_name, distance in self.recipients(assignment):
                if state.apply_offering(school_name, subject, domain):
                    touched.add(school_name)
                distances.append(distance)
        after = dict(self.before_sai)
        for school_name in touched:
            after[school_name] = state.score_school(school_name)
        return {
            "after": after,
            "delta": {school: after[school] - before for school, before in self.before_sai.items()},
            "avg_distance": sum(distances) / len(distances) if distances else self.radius_km,
        }

    def simulate(self, selected: list[dict]) -> pd.DataFrame:
        state, labels = self.state_after(selected)
        after_scored = state.score_frame()[["학교명", "SAI", "공동수업과목수"]].rename(columns={"SAI": "SAI_after"})
        out = self.before_scored.merge(after_scored, on="학교명", how="left")
        out["공동수업참여가능"] = out["학교명"].map(lambda name: "; ".join(labels.get(name, [])[:6])).fillna("")
        out["SAI_delta"] = out["SAI_after"] - out["SAI_before"]
        return out.sort_values("SAI_delta", ascending=False)


def simulate_assignments(
    features: pd.DataFrame,
    selected: list[dict],
    radius_km: float,
    baseline_offerings: pd.DataFrame,
) -> pd.DataFrame:
    return IncrementalAssignmentSimulator(features, baseline_offerings, radius_km).simulate(selected)

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from coursemap.geo import haversine_km


SAI_VERSION = "sai_v4_incremental_offerings"
DOMAINS = ["인문·사회", "자연·공학", "정보·AI", "예체능", "제2외국어·국제", "진로·융합"]
DEFAULT_TARGET_SUBJECT_COUNT = 35


def regular_offerings(neis_subjects: pd.DataFrame) -> pd.DataFrame:
    required = {"학교명", "표준과목명", "계열"}
    missing = required - set(neis_subjects.columns)
    if missing:
        raise ValueError(f"regular offerings missing columns: {sorted(missing)}")
    out = neis_subjects[["학교명", "표준과목명", "계열"]].dropna().drop_duplicates().copy()
    out = out.rename(columns={"학교명": "school", "표준과목명": "subject", "계열": "domain"})
    out = out[out["domain"].isin(DOMAINS)]
    out["source"] = "regular"
    return out.reset_index(drop=True)


def assignment_offerings(
    schools: pd.DataFrame,
    assignments: list[dict],
    radius_km: float,
) -> pd.DataFrame:
    rows = []
    school_required = {"학교명", "위도", "경도"}
    missing = school_required - set(schools.columns)
    if missing:
        raise ValueError(f"schools missing columns: {sorted(missing)}")

    hubs = schools.set_index("학교명")
    for assignment in assignments:
        hub_name = assignment["hub"]
        domain = assignment["domain"]
        subject = assignment.get("subject") or f"공동수업:{domain}:{hub_name}"
        if hub_name not in hubs.index:
            continue
        hub = hubs.loc[hub_name]
        for _, school in schools.iterrows():
            if pd.isna(hub["위도"]) or pd.isna(hub["경도"]) or pd.isna(school["위도"]) or pd.isna(school["경도"]):
                continue
            distance = haversine_km(hub["위도"], hub["경도"], school["위도"], school["경도"])
            if distance <= radius_km:
                rows.append({
                    "school": school["학교명"],
                    "subject": subject,
                    "domain": domain,
                    "source": "joint_assignment",
                    "hub": hub_name,
                    "distance_km": distance,
                })
    return pd.DataFrame(rows, columns=["school", "subject", "domain", "source", "hub", "distance_km"])


def combine_offerings(*frames: pd.DataFrame) -> pd.DataFrame:
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["school", "subject", "domain", "source"])
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(["school", "subject", "domain"]).reset_index(drop=True)


def shannon_balance(counts: list[float]) -> float:
    values = np.array([x for x in counts if x > 0], dtype=float)
    if values.sum() == 0 or len(values) <= 1:
        return 0.0
    p = values / values.sum()
    return float((-(p * np.log(p)).sum()) / math.log(len(values)))


def subject_diversity_score(subject_count: int, target_subject_count: int) -> float:
    return float(min(subject_count / max(target_subject_count, 1), 1.0) * 100.0)


def domain_width_score(domain_counts: dict[str, int] | pd.Series) -> float:
    return float(sum(int(domain_counts.get(domain, 0)) > 0 for domain in DOMAINS) / len(DOMAINS) * 100.0)


def sai_from_counts(domain_counts: dict[str, int] | pd.Series, subject_count: int, target_subject_count: int) -> float:
    width = domain_width_score(domain_counts)
    balance = shannon_balance([int(domain_counts.get(domain, 0)) for domain in DOMAINS]) * 100.0
    diversity = subject_diversity_score(subject_count, target_subject_count)
    return float(0.55 * width + 0.35 * balance + 0.10 * diversity)


def target_subject_count_from_offerings(offerings: pd.DataFrame, default: int = DEFAULT_TARGET_SUBJECT_COUNT) -> int:
    if offerings is None or offerings.empty:
        return default
    counts = offerings.groupby("school")["subject"].nunique()
    if counts.empty:
        return default
    return max(int(counts.max()), default)


@dataclass
class IncrementalSaiState:
    school_names: list[str]
    subject_sets: dict[str, set[str]]
    domain_counts: dict[str, dict[str, int]]
    joint_subject_sets: dict[str, set[str]]
    target_subject_count: int

    @classmethod
    def from_offerings(
        cls,
        schools: pd.DataFrame,
        offerings: pd.DataFrame,
        target_subject_count: int | None = None,
    ) -> "IncrementalSaiState":
        school_names = schools["학교명"].astype(str).tolist()
        subject_sets = {name: set() for name in school_names}
        joint_subject_sets = {name: set() for name in school_names}
        domain_counts = {name: {domain: 0 for domain in DOMAINS} for name in school_names}
        if target_subject_count is None:
            target_subject_count = target_subject_count_from_offerings(offerings)
        if offerings is not None and not offerings.empty:
            for row in offerings.itertuples(index=False):
                school = str(getattr(row, "school"))
                subject = str(getattr(row, "subject"))
                domain = str(getattr(row, "domain"))
                source = str(getattr(row, "source", "regular"))
                if school in subject_sets and domain in DOMAINS and subject not in subject_sets[school]:
                    subject_sets[school].add(subject)
                    domain_counts[school][domain] += 1
                if school in joint_subject_sets and source == "joint_assignment":
                    joint_subject_sets[school].add(subject)
        return cls(school_names, subject_sets, domain_counts, joint_subject_sets, int(target_subject_count))

    def clone(self) -> "IncrementalSaiState":
        return IncrementalSaiState(
            school_names=list(self.school_names),
            subject_sets={school: set(subjects) for school, subjects in self.subject_sets.items()},
            domain_counts={school: dict(counts) for school, counts in self.domain_counts.items()},
            joint_subject_sets={school: set(subjects) for school, subjects in self.joint_subject_sets.items()},
            target_subject_count=self.target_subject_count,
        )

    def apply_offering(self, school: str, subject: str, domain: str, source: str = "joint_assignment") -> bool:
        school = str(school)
        subject = str(subject)
        domain = str(domain)
        if school not in self.subject_sets or domain not in DOMAINS:
            return False
        changed = False
        if subject not in self.subject_sets[school]:
            self.subject_sets[school].add(subject)
            self.domain_counts[school][domain] += 1
            changed = True
        if source == "joint_assignment":
            self.joint_subject_sets[school].add(subject)
        return changed

    def score_school(self, school: str) -> float:
        return sai_from_counts(
            self.domain_counts[school],
            len(self.subject_sets[school]),
            self.target_subject_count,
        )

    def score_frame(self) -> pd.DataFrame:
        rows = []
        for school in self.school_names:
            counts = self.domain_counts[school]
            subject_count = len(self.subject_sets[school])
            rows.append({
                "학교명": school,
                "과목수": subject_count,
                "계열폭": int(sum(counts[domain] > 0 for domain in DOMAINS)),
                "계열균형": shannon_balance([counts[domain] for domain in DOMAINS]),
                "공동수업과목수": len(self.joint_subject_sets[school]),
                "과목다양성": subject_diversity_score(subject_count, self.target_subject_count),
                "계열폭점수": domain_width_score(counts),
                "계열균형성": shannon_balance([counts[domain] for domain in DOMAINS]) * 100.0,
                "SAI": sai_from_counts(counts, subject_count, self.target_subject_count),
                **{f"계열과목수_{domain}": int(counts[domain]) for domain in DOMAINS},
            })
        out = pd.DataFrame(rows)
        out["SAI_algorithm"] = SAI_VERSION
        return out


def offering_metrics(schools: pd.DataFrame, offerings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = offerings.groupby("school") if not offerings.empty else {}
    for _, school in schools.iterrows():
        name = school["학교명"]
        if name in grouped.groups:
            cur = grouped.get_group(name)
            subject_count = cur["subject"].nunique()
            domain_counts = cur.groupby("domain")["subject"].nunique().reindex(DOMAINS, fill_value=0)
            joint_subject_count = cur[cur["source"].eq("joint_assignment")]["subject"].nunique()
        else:
            subject_count = 0
            domain_counts = pd.Series(0, index=DOMAINS)
            joint_subject_count = 0
        rows.append({
            "학교명": name,
            "과목수": subject_count,
            "계열폭": int((domain_counts > 0).sum()),
            "계열균형": shannon_balance(domain_counts.tolist()),
            "공동수업과목수": joint_subject_count,
            **{f"계열과목수_{domain}": int(domain_counts[domain]) for domain in DOMAINS},
        })
    return pd.DataFrame(rows)


def scale_0_100(series: pd.Series, low=None, high=None) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0)
    lo = values.min() if low is None else low
    hi = values.max() if high is None else high
    if pd.isna(lo) or hi == lo:
        return pd.Series(50.0, index=values.index)
    return ((values - lo) / (hi - lo) * 100).clip(0, 100)


def compute_sai(
    schools: pd.DataFrame,
    offerings: pd.DataFrame,
    baseline_offerings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    target = target_subject_count_from_offerings(baseline_offerings if baseline_offerings is not None else offerings)
    metrics = IncrementalSaiState.from_offerings(schools, offerings, target).score_frame()
    result = schools.merge(metrics, on="학교명", how="left")
    return result

import math

import numpy as np
import pandas as pd

from coursemap.geo import haversine_km


SAI_VERSION = "sai_v3_combined_offerings"
DOMAINS = ["인문·사회", "자연·공학", "정보·AI", "예체능", "제2외국어·국제", "진로·융합"]


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
    metrics = offering_metrics(schools, offerings)
    baseline = offering_metrics(schools, baseline_offerings if baseline_offerings is not None else offerings)
    result = schools.merge(metrics, on="학교명", how="left")

    result["과목다양성"] = scale_0_100(metrics["과목수"], baseline["과목수"].min(), baseline["과목수"].max())
    result["계열폭점수"] = (metrics["계열폭"] / len(DOMAINS) * 100).clip(0, 100)
    result["계열균형성"] = (metrics["계열균형"] * 100).clip(0, 100)

    result["SAI"] = (
        0.55 * result["계열폭점수"]
        + 0.35 * result["계열균형성"]
        + 0.10 * result["과목다양성"]
    )
    result["SAI_algorithm"] = SAI_VERSION
    return result

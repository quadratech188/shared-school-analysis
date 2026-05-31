import re
import time
from pathlib import Path

import pandas as pd
import requests


KAKAO_ADDRESS_URL = "https://dapi.kakao.com/v2/local/search/address.json"
KAKAO_KEYWORD_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"


def norm_key(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def read_key_file(path: Path) -> str:
    for encoding in ["utf-8-sig", "utf-8", "cp949"]:
        try:
            value = path.read_text(encoding=encoding).strip()
            if value:
                return value
        except UnicodeDecodeError:
            continue
    raise ValueError(f"API key file is empty or unreadable: {path}")


def clean_address(addr) -> str:
    if pd.isna(addr):
        return ""
    text = str(addr)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\d+층.*$", "", text)
    text = re.sub(r"[,]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def doc_to_latlon(doc: dict) -> tuple[float | None, float | None]:
    try:
        return float(doc["y"]), float(doc["x"])
    except Exception:
        return None, None


def kakao_get(url: str, key: str, query: str, size: int | None = None) -> list[dict]:
    params = {"query": query}
    if size is not None:
        params["size"] = size
    response = requests.get(
        url,
        headers={"Authorization": f"KakaoAK {key}"},
        params=params,
        timeout=15,
    )
    response.raise_for_status()
    return response.json().get("documents", [])


def geocode_one(name, road_addr, jibun_addr, key: str) -> tuple[dict, dict]:
    attempts = []
    for method, query in [
        ("도로명원본", road_addr),
        ("지번원본", jibun_addr),
        ("도로명정제", clean_address(road_addr)),
        ("지번정제", clean_address(jibun_addr)),
    ]:
        if not query or pd.isna(query):
            continue
        docs = kakao_get(KAKAO_ADDRESS_URL, key, str(query))
        attempts.append(f"{method}:{query} -> {len(docs)}")
        if docs:
            lat, lon = doc_to_latlon(docs[0])
            return (
                {"위도": lat, "경도": lon, "geocode_method": method, "needs_review": len(docs) > 1},
                {"query_used": query, "n_candidates": len(docs), "failure_reason": "", "attempts": " | ".join(attempts)},
            )

    if name and not pd.isna(name):
        query = f"대전광역시 {name}"
        docs = kakao_get(KAKAO_KEYWORD_URL, key, query, size=5)
        attempts.append(f"키워드:{query} -> {len(docs)}")
        if docs:
            lat, lon = doc_to_latlon(docs[0])
            return (
                {"위도": lat, "경도": lon, "geocode_method": "키워드", "needs_review": len(docs) > 1},
                {"query_used": query, "n_candidates": len(docs), "failure_reason": "", "attempts": " | ".join(attempts)},
            )

    reason = "no Kakao address/keyword candidates"
    return (
        {"위도": None, "경도": None, "geocode_method": "실패", "needs_review": True},
        {"query_used": None, "n_candidates": 0, "failure_reason": reason, "attempts": " | ".join(attempts)},
    )


def geocode_facilities(
    facilities: pd.DataFrame,
    key: str,
    sleep_seconds: float = 0.12,
    overrides: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"facility_type", "name"}
    missing = required - set(facilities.columns)
    if missing:
        raise ValueError(f"facility geocoding missing columns: {sorted(missing)}")

    override_map = {}
    if overrides is not None and not overrides.empty:
        required_override = {"name", "위도", "경도"}
        missing_override = required_override - set(overrides.columns)
        if missing_override:
            raise ValueError(f"facility geocode overrides missing columns: {sorted(missing_override)}")
        clean_overrides = overrides.dropna(subset=["위도", "경도"]).copy()
        for _, override in clean_overrides.iterrows():
            override_map[norm_key(override["name"])] = override

    out = facilities.copy().reset_index(drop=True)
    logs = []
    geo_rows = []
    for row_id, row in out.iterrows():
        override = override_map.get(norm_key(row.get("name")))
        if override is not None:
            result = {
                "위도": float(override["위도"]),
                "경도": float(override["경도"]),
                "geocode_method": "manual_override",
                "needs_review": False,
            }
            info = {
                "query_used": override.get("manual_address", ""),
                "n_candidates": 1,
                "failure_reason": "",
                "attempts": f"manual_override:{override.get('manual_address', '')}",
            }
        else:
            result, info = geocode_one(
                row.get("name"),
                row.get("road_address", row.get("address")),
                row.get("jibun_address"),
                key,
            )
        geo_rows.append(result)
        logs.append({
            "source": row.get("facility_type"),
            "row_id": row_id,
            "name": row.get("name"),
            "road": row.get("road_address", row.get("address")),
            "jibun": row.get("jibun_address"),
            "method": result["geocode_method"],
            "lat": result["위도"],
            "lon": result["경도"],
            "n_candidates": info["n_candidates"],
            "query_used": info["query_used"],
            "failure_reason": info.get("failure_reason", ""),
            "attempts": info.get("attempts", ""),
            "needs_review": result["needs_review"],
        })
        time.sleep(sleep_seconds)

    geo = pd.DataFrame(geo_rows)
    return pd.concat([out, geo], axis=1), pd.DataFrame(logs)

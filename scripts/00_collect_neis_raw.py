import argparse
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from coursemap.env import resolve_secret
from coursemap.io import write_csv


NEIS_BASE = "https://open.neis.go.kr/hub"
TIMETABLE_PERIODS = {
    "2025_1st": ("20250317", "20250411"),
    "2025_2nd": ("20250908", "20250926"),
}
TARGET_GRADES = ["2", "3"]


def neis_get(endpoint: str, key: str, params: dict, max_retry: int = 3) -> dict:
    url = f"{NEIS_BASE}/{endpoint}"
    query = {"KEY": key, "Type": "json", "pIndex": 1, "pSize": 1000, **params}
    last_error = None
    for attempt in range(max_retry):
        try:
            response = requests.get(url, params=query, timeout=20)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            time.sleep(0.5 * (attempt + 1))
    safe_url = url + "?" + urlencode({**query, "KEY": "***"})
    raise RuntimeError(f"NEIS request failed: {safe_url}: {last_error}")


def neis_rows(data: dict, endpoint: str) -> tuple[list[dict], str, str]:
    if endpoint not in data:
        result = data.get("RESULT", {})
        return [], result.get("CODE", "NO_DATA"), result.get("MESSAGE", "")
    rows = []
    for block in data[endpoint]:
        if "row" in block:
            rows = block["row"]
    return rows, "OK", f"{len(rows)} rows"


def general_high_flag(neis: pd.DataFrame) -> pd.Series:
    hs_type = neis.get("HS_SC_NM", pd.Series("", index=neis.index)).fillna("").astype(str)
    business = neis.get("HS_GNRL_BUSNS_SC_NM", pd.Series("", index=neis.index)).fillna("").astype(str)
    special = neis.get("SPCLY_PURPS_HS_ORD_NM", pd.Series("", index=neis.index)).fillna("").astype(str)
    vocational = hs_type.str.contains("특성화|마이스터|산업|전문", regex=True)
    special_purpose = hs_type.str.contains("특목|특수목적", regex=True) | special.ne("")
    general = hs_type.str.contains("일반|자율", regex=True) | business.str.contains("일반", regex=True)
    return general & ~vocational & ~special_purpose


def date_range(start: str, end: str) -> list[str]:
    cur = datetime.strptime(start, "%Y%m%d")
    stop = datetime.strptime(end, "%Y%m%d")
    days = []
    while cur <= stop:
        if cur.weekday() < 5:
            days.append(cur.strftime("%Y%m%d"))
        cur = cur.fromordinal(cur.toordinal() + 1)
    return days


def sampled_dates(max_dates: int) -> dict[str, list[str]]:
    out = {}
    for label, (start, end) in TIMETABLE_PERIODS.items():
        days = date_range(start, end)
        if max_dates and len(days) > max_dates:
            indices = np.linspace(0, len(days) - 1, max_dates).round().astype(int)
            days = [days[i] for i in sorted(set(indices))]
        out[label] = days
    return out


def collect_school_info(key: str) -> pd.DataFrame:
    data = neis_get("schoolInfo", key, {
        "ATPT_OFCDC_SC_CODE": "G10",
        "SCHUL_KND_SC_NM": "고등학교",
    })
    rows, code, message = neis_rows(data, "schoolInfo")
    if code != "OK" or not rows:
        raise SystemExit(f"schoolInfo returned no rows: {code} {message}")
    out = pd.DataFrame(rows)
    out["is_general_high"] = general_high_flag(out)
    return out


def collect_timetable(key: str, school_info: pd.DataFrame, max_dates: int, sleep_seconds: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    schools = school_info[school_info["is_general_high"]].copy()
    codes = schools["SD_SCHUL_CODE"].dropna().astype(str).tolist()
    dates = sampled_dates(max_dates)
    rows = []
    logs = []
    tasks = [
        (code, schools.loc[schools["SD_SCHUL_CODE"].astype(str).eq(code), "SCHUL_NM"].iloc[0], grade, period, day)
        for code in codes
        for grade in TARGET_GRADES
        for period, period_dates in dates.items()
        for day in period_dates
    ]
    progress = tqdm(tasks, desc="hisTimetable", unit="call")
    for code, school_name, grade, period, day in progress:
        try:
            data = neis_get("hisTimetable", key, {
                "ATPT_OFCDC_SC_CODE": "G10",
                "SD_SCHUL_CODE": code,
                "GRADE": grade,
                "ALL_TI_YMD": day,
            })
            got, status, message = neis_rows(data, "hisTimetable")
            for row in got:
                row["_period_label"] = period
            rows.extend(got)
            logs.append({"school_code": code, "school": school_name, "grade": grade, "date": day, "period": period, "status": status, "message": message, "n": len(got)})
        except Exception as exc:
            logs.append({"school_code": code, "school": school_name, "grade": grade, "date": day, "period": period, "status": "ERR", "message": str(exc), "n": 0})
        progress.set_postfix(rows=len(rows), school=school_name[:8])
        time.sleep(sleep_seconds)
    return pd.DataFrame(rows), pd.DataFrame(logs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--school-info-out", default="data/raw/outputs/raw/neis_school_info_raw.csv")
    parser.add_argument("--timetable-out", default="data/raw/outputs/raw/neis_his_timetable_raw.csv")
    parser.add_argument("--log-out", default="build/metadata/neis_timetable_collection_log.csv")
    parser.add_argument("--max-dates-per-period", type=int, default=5)
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    key = resolve_secret("NEIS_API_KEY", [data_dir / "NEIS api key.txt"])
    if not key:
        raise SystemExit("NEIS key is required. Set NEIS_API_KEY in .env or provide data/raw/NEIS api key.txt.")

    school_info = collect_school_info(key)
    write_csv(school_info, Path(args.school_info_out))
    print(f"schoolInfo collected: {len(school_info)} rows, general high={int(school_info['is_general_high'].sum())}")

    timetable, log = collect_timetable(key, school_info, args.max_dates_per_period, args.sleep_seconds)
    if timetable.empty:
        write_csv(log, Path(args.log_out))
        raise SystemExit("hisTimetable collected zero rows")
    write_csv(timetable, Path(args.timetable_out))
    write_csv(log, Path(args.log_out))
    print(f"hisTimetable collected: {len(timetable)} rows, schools={timetable['SCHUL_NM'].nunique()}")


if __name__ == "__main__":
    main()

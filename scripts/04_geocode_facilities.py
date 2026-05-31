import argparse
from pathlib import Path

from coursemap.env import resolve_secret
from coursemap.geocode import geocode_facilities, read_key_file
from coursemap.io import read_csv_smart, write_csv


def resolve_key(data_dir: Path, key_file: str | None) -> str:
    candidates = []
    if key_file:
        candidates.append(Path(key_file))
    candidates.append(data_dir / "Kakao api key.txt")
    key = resolve_secret("KAKAO_REST_API_KEY", candidates)
    if key:
        return key
    raise SystemExit(
        "Kakao key is required for facility geocoding. "
        "Set KAKAO_REST_API_KEY, add .env, or provide data/raw/Kakao api key.txt."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facilities", required=True)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--kakao-key-file")
    parser.add_argument("--overrides", default="config/facility_geocode_overrides.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--log-out", required=True)
    parser.add_argument("--sleep-seconds", type=float, default=0.12)
    args = parser.parse_args()

    facilities = read_csv_smart(Path(args.facilities))
    override_path = Path(args.overrides)
    overrides = read_csv_smart(override_path) if override_path.exists() else None
    key = resolve_key(Path(args.data_dir), args.kakao_key_file)
    geocoded, log = geocode_facilities(facilities, key, sleep_seconds=args.sleep_seconds, overrides=overrides)
    failures = geocoded[geocoded["위도"].isna() | geocoded["경도"].isna()]
    if not failures.empty:
        write_csv(log, Path(args.log_out))
        failure_log = log[log["method"].eq("실패")].copy()
        failure_cols = ["source", "name", "road", "jibun", "failure_reason", "attempts"]
        detail = failure_log[failure_cols].to_dict("records")
        raise SystemExit(
            f"facility geocoding failed for {len(failures)} rows. "
            f"See {args.log_out} for full log. Failures: {detail}"
        )
    write_csv(geocoded, Path(args.out))
    write_csv(log, Path(args.log_out))
    print(f"facility geocoding: {len(geocoded)} rows, success rate 100%")


if __name__ == "__main__":
    main()

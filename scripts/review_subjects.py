import argparse
from pathlib import Path

import pandas as pd

from coursemap.io import read_csv_smart, write_csv
from coursemap.subjects import DOMAINS


def load_existing(path: Path) -> pd.DataFrame:
    if path.exists():
        return read_csv_smart(path)
    return pd.DataFrame(columns=["subject", "standard_subject", "domain"])


def choose_domain(subject: str) -> str:
    while True:
        print(f"\n과목: {subject}")
        for idx, domain in enumerate(DOMAINS, start=1):
            print(f"{idx}. {domain}")
        answer = input("> ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(DOMAINS):
            return DOMAINS[int(answer) - 1]
        if answer in DOMAINS:
            return answer
        print("번호 또는 계열명을 다시 입력하세요.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", required=True)
    parser.add_argument("--overrides", required=True)
    args = parser.parse_args()

    review_path = Path(args.review)
    if not review_path.exists():
        raise SystemExit("review file not found. Run `make subject-review-list` first.")
    review = read_csv_smart(review_path)
    if review.empty:
        print("No unassigned subjects.")
        return

    override_path = Path(args.overrides)
    existing = load_existing(override_path)
    existing_subjects = set(existing.get("subject", pd.Series(dtype=str)).dropna().astype(str))

    rows = []
    pending = [s for s in review["subject"].dropna().astype(str).tolist() if s not in existing_subjects]
    if not pending:
        print("All review subjects already have overrides.")
        return

    print(f"{len(pending)} subjects need review. Press Ctrl+C to stop; completed choices are saved at the end.")
    for i, subject in enumerate(pending, start=1):
        print(f"\n[{i}/{len(pending)}]")
        domain = choose_domain(subject)
        standard = input(f"표준과목명 [{subject}]: ").strip() or subject
        rows.append({"subject": subject, "standard_subject": standard, "domain": domain})

    out = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    out = out.drop_duplicates("subject", keep="last")
    write_csv(out, override_path)
    print(f"saved overrides: {override_path}")


if __name__ == "__main__":
    main()

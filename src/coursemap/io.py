from pathlib import Path

import pandas as pd


CSV_ENCODINGS = ("utf-8-sig", "cp949", "utf-8")


def read_csv_smart(path: Path, **kwargs) -> pd.DataFrame:
    last_error = None
    for enc in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")

import pandas as pd


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {str(c).replace("\n", "").replace(" ", ""): c for c in df.columns}
    for cand in candidates:
        key = cand.replace("\n", "").replace(" ", "")
        if key in normalized:
            return normalized[key]
    for cand in candidates:
        key = cand.replace("\n", "").replace(" ", "")
        for norm, original in normalized.items():
            if key in norm:
                return original
    return None


def pick_series(df: pd.DataFrame, col: str | None, default=None) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series(default, index=df.index)
    picked = df.loc[:, col]
    if isinstance(picked, pd.DataFrame):
        picked = picked.iloc[:, 0]
    return picked

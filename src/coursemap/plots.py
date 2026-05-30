from pathlib import Path

import pandas as pd


def save_before_after_dot_plot(
    df: pd.DataFrame,
    before_col: str,
    after_col: str,
    label_col: str,
    highlight_labels: set[str],
    out_path: Path,
    title: str,
    horizontal_lines: list[tuple[float, str, str]] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot = df.sort_values(before_col).reset_index(drop=True)
    colors = ["#d97706" if label in highlight_labels else "#64748b" for label in plot[label_col]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(plot[before_col], plot[after_col], c=colors, alpha=0.85)
    lo = min(plot[before_col].min(), plot[after_col].min()) - 2
    hi = max(plot[before_col].max(), plot[after_col].max()) + 2
    ax.plot([lo, hi], [lo, hi], color="#334155", linewidth=1, linestyle="--")
    for y, label, color in horizontal_lines or []:
        ax.axhline(y, color=color, linewidth=1.4, linestyle="-", alpha=0.9)
        ax.text(lo, y, f" {label}: {y:.1f}", color=color, va="bottom", fontsize=8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(before_col)
    ax.set_ylabel(after_col)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

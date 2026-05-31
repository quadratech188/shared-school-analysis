from pathlib import Path

import pandas as pd


def _configure_korean_font() -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    preferred = ["NanumGothic", "Noto Sans CJK KR", "Noto Sans"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font in preferred:
        if font in available:
            plt.rcParams["font.family"] = font
            plt.rcParams["axes.unicode_minus"] = False
            return


def save_before_after_dot_plot(
    df: pd.DataFrame,
    before_col: str,
    after_col: str,
    label_col: str,
    highlight_labels: set[str],
    out_path: Path,
    title: str,
    horizontal_lines: list[tuple[float, str, str]] | None = None,
    tier_labels: bool = False,
    show_summary_lines: bool = False,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _configure_korean_font()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot = df.sort_values(before_col).reset_index(drop=True)
    before = pd.to_numeric(plot[before_col], errors="coerce")
    after = pd.to_numeric(plot[after_col], errors="coerce")

    styles = []
    legend_handles = []
    if tier_labels:
        bottom_one = set(plot.nsmallest(1, before_col)[label_col])
        bottom_three = set(plot.nsmallest(min(3, len(plot)), before_col)[label_col]) - bottom_one
        bottom_quarter = set(plot[before <= before.quantile(0.25)][label_col]) - bottom_one - bottom_three
        for label in plot[label_col]:
            if label in bottom_one:
                styles.append(("#dc2626", 62, "최하"))
            elif label in bottom_three:
                styles.append(("#f97316", 52, "하 3"))
            elif label in bottom_quarter:
                styles.append(("#facc15", 46, "하 25%"))
            elif label in highlight_labels:
                styles.append(("#38bdf8", 38, "취약 학교"))
            else:
                styles.append(("#64748b", 30, "기타"))
        legend_handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#dc2626", markeredgecolor="#7f1d1d", markersize=8, label="최하"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#f97316", markeredgecolor="#7c2d12", markersize=7, label="하 3"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#facc15", markeredgecolor="#854d0e", markersize=7, label="하 25%"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#64748b", markeredgecolor="#334155", markersize=6, label="기타"),
        ]
    else:
        styles = [("#d97706", 38, "highlight") if label in highlight_labels else ("#64748b", 30, "other") for label in plot[label_col]]

    fig, ax = plt.subplots(figsize=(8, 5))
    for category in dict.fromkeys(item[2] for item in styles):
        mask = [item[2] == category for item in styles]
        color = next(item[0] for item in styles if item[2] == category)
        size = next(item[1] for item in styles if item[2] == category)
        ax.scatter(
            before[mask],
            after[mask],
            c=color,
            s=size,
            edgecolors="#111827",
            linewidths=0.35,
            alpha=0.88,
        )
    lo = min(plot[before_col].min(), plot[after_col].min()) - 2
    hi = max(plot[before_col].max(), plot[after_col].max()) + 2
    ax.plot([lo, hi], [lo, hi], color="#334155", linewidth=1, linestyle="--")
    for y, label, color in horizontal_lines or []:
        ax.axhline(y, color=color, linewidth=1.4, linestyle="-", alpha=0.9)
        ax.text(lo, y, f" {label}: {y:.1f}", color=color, va="bottom", fontsize=8)
    if show_summary_lines:
        before_mean = float(before.mean())
        after_mean = float(after.mean())
        before_min = float(before.min())
        after_min = float(after.min())
        ax.axhline(before_mean, color="#2563eb", linewidth=1.2, linestyle="--", alpha=0.75)
        ax.axhline(after_mean, color="#2563eb", linewidth=1.2, linestyle="-", alpha=0.85)
        ax.axhline(before_min, color="#dc2626", linewidth=1.2, linestyle="--", alpha=0.8)
        ax.axhline(after_min, color="#dc2626", linewidth=1.2, linestyle=":", alpha=0.95)
        legend_handles.extend([
            Line2D([0], [0], color="#2563eb", linewidth=1.2, linestyle="--", label="전 평균"),
            Line2D([0], [0], color="#2563eb", linewidth=1.2, linestyle="-", label="후 평균"),
            Line2D([0], [0], color="#dc2626", linewidth=1.2, linestyle="--", label="전 최저"),
            Line2D([0], [0], color="#dc2626", linewidth=1.2, linestyle=":", label="후 최저"),
        ])
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(before_col)
    ax.set_ylabel(after_col)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower right", frameon=True, fontsize=8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_algorithm_bar_comparison(
    summary: pd.DataFrame,
    out_path: Path,
    points: pd.DataFrame | None = None,
    baseline_mean: float | None = None,
    algorithm_col: str = "algorithm",
    mean_col: str = "mean_after",
    std_col: str = "std_after",
    min_col: str = "min_after",
    point_value_col: str = "SAI_after_display",
    title: str = "Algorithm SAI comparison",
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    _configure_korean_font()
    required = {algorithm_col, mean_col, std_col, min_col}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Algorithm comparison missing columns: {sorted(missing)}")

    plot = summary.copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    colors = {
        "Greedy": "#cbd5e1",
        "RL": "#bae6fd",
        "Actor-Critic": "#bfdbfe",
    }
    bar_colors = [colors.get(label, "#64748b") for label in plot[algorithm_col]]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(plot))
    bars = ax.bar(
        x,
        plot[mean_col],
        yerr=plot[std_col],
        color=bar_colors,
        edgecolor="#1f2937",
        linewidth=0.7,
        capsize=7,
        error_kw={"elinewidth": 1.2, "ecolor": "#111827"},
        zorder=2,
    )
    if points is not None and not points.empty:
        point_required = {algorithm_col, point_value_col}
        point_missing = point_required - set(points.columns)
        if point_missing:
            raise ValueError(f"Algorithm point data missing columns: {sorted(point_missing)}")
        x_by_algorithm = {label: idx for idx, label in enumerate(plot[algorithm_col])}
        for label, group in points.groupby(algorithm_col):
            if label not in x_by_algorithm:
                continue
            idx = x_by_algorithm[label]
            offsets = np.linspace(-0.19, 0.19, len(group)) if len(group) > 1 else np.array([0.0])
            ax.scatter(
                idx + offsets,
                group[point_value_col],
                s=18,
                color="#334155",
                alpha=0.62,
                linewidth=0,
                zorder=4,
            )
    for bar, mean_value, std_value, min_value in zip(bars, plot[mean_col], plot[std_col], plot[min_col]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean_value + std_value + 0.55,
            f"mean {mean_value:.1f}\nmin {min_value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#111827",
        )
    if baseline_mean is not None:
        ax.axhline(
            baseline_mean,
            color="#64748b",
            linewidth=1.4,
            linestyle="--",
            alpha=0.9,
            label=f"Before mean {baseline_mean:.1f}",
            zorder=5,
        )
    ymin = max(0, float((plot[mean_col] - plot[std_col]).min()) - 3)
    ymax = float((plot[mean_col] + plot[std_col]).max()) + 4.5
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels(plot[algorithm_col])
    ax.set_ylabel("SAI_after")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    if baseline_mean is not None:
        ax.legend(loc="lower right", frameon=True, fontsize=8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_sai_stepped_domain_counts(
    df: pd.DataFrame,
    out_path: Path,
    summary_out: Path | None = None,
    sample_size: int = 10,
    exclude_schools: set[str] | None = None,
    sai_col: str = "SAI",
    school_col: str = "학교명",
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    _configure_korean_font()
    domains = ["인문·사회", "자연·공학", "정보·AI", "예체능", "제2외국어·국제", "진로·융합"]
    count_cols = [f"계열과목수_{domain}" for domain in domains]
    required = {school_col, sai_col, *count_cols}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"SAI stepped domain plot missing columns: {sorted(missing)}")

    exclude_schools = exclude_schools or set()
    ranked = df[[school_col, sai_col, *count_cols]].dropna(subset=[school_col, sai_col]).copy()
    ranked = ranked[~ranked[school_col].isin(exclude_schools)].sort_values(sai_col).reset_index(drop=True)
    if ranked.empty:
        raise ValueError("SAI stepped domain plot has no rows")
    sample_count = min(sample_size, len(ranked))
    indices = np.rint(np.linspace(0, len(ranked) - 1, sample_count)).astype(int)
    selected = ranked.iloc[indices].drop_duplicates(school_col).reset_index(drop=True)
    while len(selected) < sample_count and len(selected) < len(ranked):
        remaining = ranked[~ranked[school_col].isin(set(selected[school_col]))]
        selected = pd.concat([selected, remaining.head(sample_count - len(selected))], ignore_index=True)
        selected = selected.sort_values(sai_col).reset_index(drop=True)
    selected = selected.sort_values(sai_col, ascending=False).reset_index(drop=True)

    if summary_out is not None:
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(summary_out, index=False, encoding="utf-8-sig")

    colors = {
        "인문·사회": "#fecaca",
        "자연·공학": "#bfdbfe",
        "정보·AI": "#bbf7d0",
        "예체능": "#fde68a",
        "제2외국어·국제": "#ddd6fe",
        "진로·융합": "#fed7aa",
    }
    y = np.arange(len(selected))
    fig, ax = plt.subplots(figsize=(10, 6))
    left = np.zeros(len(selected))
    for domain, col in zip(domains, count_cols):
        values = pd.to_numeric(selected[col], errors="coerce").fillna(0).to_numpy()
        ax.barh(y, values, left=left, color=colors[domain], edgecolor="#334155", linewidth=0.45, label=domain)
        for idx, value in enumerate(values):
            if value >= 2:
                ax.text(left[idx] + value / 2, idx, f"{int(value)}", ha="center", va="center", fontsize=8, color="#111827")
        left += values

    ax.set_yticks(y)
    ax.set_yticklabels(selected[school_col])
    ax.invert_yaxis()
    ax.set_xlabel("계열별 과목 수")
    ax.set_title("SAI 순 균등 표본 학교의 계열별 과목 수")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    ax_sai = ax.twinx()
    ax_sai.set_ylim(ax.get_ylim())
    ax_sai.set_yticks(y)
    ax_sai.set_yticklabels([f"{value:.1f}" for value in selected[sai_col]])
    ax_sai.set_ylabel("SAI")
    ax.legend(loc="lower right", frameon=True, fontsize=8, ncol=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_sai_voronoi_map(
    df: pd.DataFrame,
    out_path: Path,
    boundary_path: Path | None = None,
    boundary_filter: str | None = None,
    value_col: str = "SAI",
    lon_col: str = "경도",
    lat_col: str = "위도",
    label_col: str = "학교명",
    padding_ratio: float = 0.08,
    label_schools: bool = False,
    basemap: bool = False,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import geopandas as gpd
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon as MplPolygon
    from scipy.spatial import Voronoi
    from shapely.geometry import Point, Polygon, box
    from shapely.ops import unary_union

    required = {value_col, lon_col, lat_col, label_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Voronoi map missing columns: {sorted(missing)}")

    plot = df[[label_col, lon_col, lat_col, value_col]].dropna().copy()
    if len(plot) < 4:
        raise ValueError("Voronoi map requires at least 4 schools with coordinates")

    school_gdf = gpd.GeoDataFrame(
        plot,
        geometry=[Point(xy) for xy in plot[[lon_col, lat_col]].to_numpy(dtype=float)],
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")

    boundary_gdf = None
    if boundary_path is not None:
        boundary_gdf = gpd.read_file(boundary_path)
        if boundary_filter:
            text_cols = [
                col for col in boundary_gdf.columns
                if col != boundary_gdf.geometry.name
            ]
            mask = boundary_gdf[text_cols].astype(str).apply(
                lambda col: col.str.contains(boundary_filter, na=False)
            ).any(axis=1)
            boundary_gdf = boundary_gdf[mask].copy()
        if boundary_gdf.empty:
            raise ValueError(f"No boundary geometry matched filter: {boundary_filter}")
        boundary_gdf = boundary_gdf.to_crs("EPSG:3857")
        clip_geom = unary_union(boundary_gdf.geometry)
    else:
        bounds = school_gdf.total_bounds
        x_pad = (bounds[2] - bounds[0]) * padding_ratio
        y_pad = (bounds[3] - bounds[1]) * padding_ratio
        clip_geom = box(bounds[0] - x_pad, bounds[1] - y_pad, bounds[2] + x_pad, bounds[3] + y_pad)

    points = np.column_stack([school_gdf.geometry.x, school_gdf.geometry.y])
    values = plot[value_col].astype(float).to_numpy()

    regions, vertices = _finite_voronoi_polygons(Voronoi(points))
    patches = []
    patch_values = []
    for idx, region in enumerate(regions):
        poly = Polygon(vertices[region]).intersection(clip_geom)
        if poly.is_empty:
            continue
        geoms = [poly] if poly.geom_type == "Polygon" else list(poly.geoms)
        for geom in geoms:
            patches.append(MplPolygon(np.asarray(geom.exterior.coords), closed=True))
            patch_values.append(values[idx])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    if boundary_gdf is not None:
        boundary_gdf.plot(ax=ax, facecolor="#f8fafc", edgecolor="#64748b", linewidth=1.1, alpha=0.95, zorder=0)

    collection = PatchCollection(
        patches,
        cmap="RdBu",
        edgecolor="#334155",
        linewidth=0.6,
        alpha=0.76 if basemap else 0.88,
        zorder=2,
    )
    collection.set_array(np.asarray(patch_values))
    collection.set_clim(values.min(), values.max())
    ax.add_collection(collection)
    ax.scatter(points[:, 0], points[:, 1], s=14, c="#111827", linewidth=0, zorder=3)

    if label_schools:
        for row, point in zip(plot.itertuples(index=False), school_gdf.geometry):
            ax.text(point.x, point.y, f" {getattr(row, label_col)}", fontsize=6, color="#111827", zorder=4)

    if boundary_gdf is not None:
        boundary_gdf.boundary.plot(ax=ax, color="#0f172a", linewidth=1.4, zorder=5)

    if basemap:
        try:
            import contextily as cx
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, crs="EPSG:3857", attribution_size=6)
        except Exception as exc:
            ax.text(
                0.01,
                0.01,
                f"Basemap unavailable: {exc}",
                transform=ax.transAxes,
                fontsize=7,
                color="#475569",
                va="bottom",
            )

    bounds = clip_geom.bounds
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_title("SAI Voronoi map clipped to Daejeon boundary")
    cbar = fig.colorbar(collection, ax=ax, shrink=0.78)
    cbar.set_label(value_col)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_sai_voronoi_comparison_map(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    out_path: Path,
    assignment_hubs: pd.DataFrame | None = None,
    boundary_path: Path | None = None,
    boundary_filter: str | None = None,
    before_value_col: str = "SAI",
    after_value_col: str = "SAI_after",
    lon_col: str = "경도",
    lat_col: str = "위도",
    label_col: str = "학교명",
    before_title: str = "Before",
    after_title: str = "After Actor-Critic",
    basemap: bool = False,
    assignment_radius_km: float = 5.0,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import geopandas as gpd
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon as MplPolygon
    from scipy.spatial import Voronoi
    from shapely.geometry import Point, Polygon, box
    from shapely.ops import unary_union

    required_before = {before_value_col, lon_col, lat_col, label_col}
    required_after = {after_value_col, lon_col, lat_col, label_col}
    missing = (required_before - set(before_df.columns)) | (required_after - set(after_df.columns))
    if missing:
        raise ValueError(f"Voronoi comparison missing columns: {sorted(missing)}")

    before = before_df[[label_col, lon_col, lat_col, before_value_col]].dropna().copy()
    after = after_df[[label_col, lon_col, lat_col, after_value_col]].dropna().copy()
    before = before.sort_values(label_col).reset_index(drop=True)
    after = after.sort_values(label_col).reset_index(drop=True)
    if before[label_col].tolist() != after[label_col].tolist():
        raise ValueError("Before/after Voronoi maps must contain the same schools")
    if len(before) < 4:
        raise ValueError("Voronoi comparison requires at least 4 schools with coordinates")

    school_gdf = gpd.GeoDataFrame(
        before,
        geometry=[Point(xy) for xy in before[[lon_col, lat_col]].to_numpy(dtype=float)],
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")
    hub_gdf = None
    if assignment_hubs is not None and not assignment_hubs.empty:
        hub_required = {label_col, lon_col, lat_col}
        hub_missing = hub_required - set(assignment_hubs.columns)
        if hub_missing:
            raise ValueError(f"Assignment hubs missing columns: {sorted(hub_missing)}")
        hubs = assignment_hubs[[label_col, lon_col, lat_col]].dropna().drop_duplicates(label_col).copy()
        hub_gdf = gpd.GeoDataFrame(
            hubs,
            geometry=[Point(xy) for xy in hubs[[lon_col, lat_col]].to_numpy(dtype=float)],
            crs="EPSG:4326",
        ).to_crs("EPSG:3857")

    boundary_gdf = None
    if boundary_path is not None:
        boundary_gdf = gpd.read_file(boundary_path)
        if boundary_filter:
            text_cols = [col for col in boundary_gdf.columns if col != boundary_gdf.geometry.name]
            mask = boundary_gdf[text_cols].astype(str).apply(
                lambda col: col.str.contains(boundary_filter, na=False)
            ).any(axis=1)
            boundary_gdf = boundary_gdf[mask].copy()
        if boundary_gdf.empty:
            raise ValueError(f"No boundary geometry matched filter: {boundary_filter}")
        boundary_gdf = boundary_gdf.to_crs("EPSG:3857")
        clip_geom = unary_union(boundary_gdf.geometry)
    else:
        bounds = school_gdf.total_bounds
        x_pad = (bounds[2] - bounds[0]) * 0.08
        y_pad = (bounds[3] - bounds[1]) * 0.08
        clip_geom = box(bounds[0] - x_pad, bounds[1] - y_pad, bounds[2] + x_pad, bounds[3] + y_pad)

    points = np.column_stack([school_gdf.geometry.x, school_gdf.geometry.y])
    regions, vertices = _finite_voronoi_polygons(Voronoi(points))
    polygons_by_school = []
    for region in regions:
        poly = Polygon(vertices[region]).intersection(clip_geom)
        if poly.is_empty:
            polygons_by_school.append([])
        else:
            polygons_by_school.append([poly] if poly.geom_type == "Polygon" else list(poly.geoms))

    before_values = before[before_value_col].astype(float).to_numpy()
    after_values = after[after_value_col].astype(float).to_numpy()
    vmin = float(min(before_values.min(), after_values.min()))
    vmax = float(max(before_values.max(), after_values.max()))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True)
    collections = []
    for ax, values, title in zip(axes, [before_values, after_values], [before_title, after_title]):
        if boundary_gdf is not None:
            boundary_gdf.plot(ax=ax, facecolor="#f8fafc", edgecolor="#64748b", linewidth=1.0, alpha=0.95, zorder=0)
        patches = []
        patch_values = []
        for idx, geoms in enumerate(polygons_by_school):
            for geom in geoms:
                patches.append(MplPolygon(np.asarray(geom.exterior.coords), closed=True))
                patch_values.append(values[idx])
        collection = PatchCollection(
            patches,
            cmap="RdBu",
            edgecolor="#334155",
            linewidth=0.45,
            alpha=0.76 if basemap else 0.88,
            zorder=2,
        )
        collection.set_array(np.asarray(patch_values))
        collection.set_clim(vmin, vmax)
        ax.add_collection(collection)
        collections.append(collection)
        ax.scatter(points[:, 0], points[:, 1], s=11, c="#111827", linewidth=0, zorder=3)
        if boundary_gdf is not None:
            boundary_gdf.boundary.plot(ax=ax, color="#0f172a", linewidth=1.2, zorder=5)
        if title == after_title and hub_gdf is not None and not hub_gdf.empty:
            circles = hub_gdf.to_crs("EPSG:5179")
            circles["geometry"] = circles.geometry.buffer(assignment_radius_km * 1000)
            circles = circles.to_crs("EPSG:3857")
            circles.boundary.plot(ax=ax, color="#facc15", linewidth=1.7, linestyle="--", zorder=6)
            hub_gdf.plot(
                ax=ax,
                marker="*",
                markersize=75,
                color="#facc15",
                edgecolor="#111827",
                linewidth=0.7,
                zorder=7,
            )
        if basemap:
            try:
                import contextily as cx
                cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, crs="EPSG:3857", attribution_size=5)
            except Exception as exc:
                ax.text(0.01, 0.01, f"Basemap unavailable: {exc}", transform=ax.transAxes, fontsize=6, color="#475569")
        bounds = clip_geom.bounds
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        ax.set_title(title)

    cbar = fig.colorbar(collections[-1], ax=axes, shrink=0.78, location="right")
    cbar.set_label("SAI")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _finite_voronoi_polygons(vor, radius: float | None = None):
    import numpy as np

    if vor.points.shape[1] != 2:
        raise ValueError("Voronoi input must be 2D")

    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points, axis=0).max() * 2

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            tangent = vor.points[p2] - vor.points[p1]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vertices_arr = np.asarray([new_vertices[v] for v in new_region])
        centroid = vertices_arr.mean(axis=0)
        angles = np.arctan2(vertices_arr[:, 1] - centroid[1], vertices_arr[:, 0] - centroid[0])
        new_regions.append([v for _, v in sorted(zip(angles, new_region))])

    return new_regions, np.asarray(new_vertices)

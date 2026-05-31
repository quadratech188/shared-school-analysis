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

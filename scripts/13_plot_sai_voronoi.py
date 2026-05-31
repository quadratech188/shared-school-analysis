import argparse
from pathlib import Path

import pandas as pd

from coursemap.io import read_csv_smart
from coursemap.plots import save_sai_voronoi_comparison_map, save_sai_voronoi_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sai", default="build/tables/school_sai_result.csv")
    parser.add_argument("--boundary", default="data/raw/high_school_zone_20260320_shp/고등학교학교군.shp")
    parser.add_argument("--boundary-filter", default="대전")
    parser.add_argument("--out", default="build/figures/sai_voronoi_map.png")
    parser.add_argument("--value-col", default="SAI")
    parser.add_argument("--after-simulation")
    parser.add_argument("--after-algorithm", default="actor_critic")
    parser.add_argument("--after-value-col", default="SAI_after")
    parser.add_argument("--assignments")
    parser.add_argument("--facilities", default="build/interim/facilities_geocoded.csv")
    parser.add_argument("--assignment-radius-km", type=float, default=5.0)
    parser.add_argument("--label-schools", action="store_true")
    parser.add_argument("--basemap", action="store_true")
    args = parser.parse_args()

    sai = read_csv_smart(Path(args.sai))
    if args.after_simulation:
        sim = read_csv_smart(Path(args.after_simulation))
        if "algorithm" in sim.columns:
            sim = sim[sim["algorithm"].eq(args.after_algorithm)].copy()
        after = sai[["학교명", "위도", "경도"]].merge(
            sim[["학교명", args.after_value_col]],
            on="학교명",
            how="inner",
        )
        assignment_hubs = None
        if args.assignments:
            assignments = read_csv_smart(Path(args.assignments))
            if "algorithm" in assignments.columns:
                assignments = assignments[assignments["algorithm"].eq(args.after_algorithm)].copy()
            school_hubs = sai[["학교명", "위도", "경도"]].rename(columns={"학교명": "hub"})
            hub_sources = [school_hubs]
            facilities_path = Path(args.facilities)
            if facilities_path.exists():
                facilities = read_csv_smart(facilities_path)
                hub_sources.append(facilities[["name", "위도", "경도"]].rename(columns={"name": "hub"}))
            hub_coords = (
                assignments[["hub"]].drop_duplicates()
                .merge(
                    pd.concat(hub_sources, ignore_index=True).drop_duplicates("hub"),
                    on="hub",
                    how="left",
                )
                .rename(columns={"hub": "학교명"})
            )
            missing_hubs = hub_coords[hub_coords[["위도", "경도"]].isna().any(axis=1)]["학교명"].tolist()
            if missing_hubs:
                raise ValueError(f"Missing coordinates for assignment hubs: {missing_hubs}")
            assignment_hubs = hub_coords
        save_sai_voronoi_comparison_map(
            sai,
            after,
            Path(args.out),
            assignment_hubs=assignment_hubs,
            boundary_path=Path(args.boundary) if args.boundary else None,
            boundary_filter=args.boundary_filter,
            before_value_col=args.value_col,
            after_value_col=args.after_value_col,
            before_title="Before assignment",
            after_title=f"After {args.after_algorithm}",
            basemap=args.basemap,
            assignment_radius_km=args.assignment_radius_km,
        )
    else:
        save_sai_voronoi_map(
            sai,
            Path(args.out),
            boundary_path=Path(args.boundary) if args.boundary else None,
            boundary_filter=args.boundary_filter,
            value_col=args.value_col,
            label_schools=args.label_schools,
            basemap=args.basemap,
        )
    print(f"SAI Voronoi map saved: {args.out}")


if __name__ == "__main__":
    main()

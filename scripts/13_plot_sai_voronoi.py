import argparse
from pathlib import Path

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
        save_sai_voronoi_comparison_map(
            sai,
            after,
            Path(args.out),
            boundary_path=Path(args.boundary) if args.boundary else None,
            boundary_filter=args.boundary_filter,
            before_value_col=args.value_col,
            after_value_col=args.after_value_col,
            before_title="Before assignment",
            after_title=f"After {args.after_algorithm}",
            basemap=args.basemap,
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

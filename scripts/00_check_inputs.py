import argparse
from pathlib import Path

import pandas as pd
import yaml

from coursemap.io import write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    config_path = Path(__file__).resolve().parents[1] / "config" / "inputs.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    rows = []
    for kind in ("required", "optional"):
        for rel in config.get(kind, []):
            path = data_dir / rel
            rows.append({
                "kind": kind,
                "path": rel,
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
            })
    manifest = pd.DataFrame(rows)
    write_csv(manifest, Path(args.out))

    missing = manifest[(manifest["kind"] == "required") & (~manifest["exists"])]
    if not missing.empty:
        print(missing[["kind", "path"]].to_string(index=False))
        raise SystemExit("required input files are missing")
    print(f"input check ok: {len(manifest)} files")


if __name__ == "__main__":
    main()

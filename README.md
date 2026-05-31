# CourseMap Pipeline

Notebook-free data pipeline for CourseMap AI.

The repository keeps code separate from raw data. By default, scripts read from
`data/raw` and write derived files under `build/`.

## Quick Start

```bash
cd coursemap_pipeline
make
```

Useful targets:

```bash
make check-inputs
make collect-neis
make geocode-facilities
make facility-accessibility
make recommend-rl
make clean
make rebuild
```

Override paths when needed:

```bash
make DATA_DIR="/path/to/input-data" BUILD_DIR="build"
```

## Layout

- `scripts/`: small pipeline steps, one output family per file
- `src/coursemap/`: shared code used by multiple steps only
- `build/interim/`: cleaned intermediate tables
- `build/processed/`: model-ready processed tables
- `build/metadata/`: input/schema audit outputs

## Current Scope

The pipeline currently builds:

- cleaned school, subject, facility, and joint-curriculum tables
- Kakao-geocoded public hub locations and school-level facility accessibility
- subject supply matrices and school-level subject summary
- feature coverage validation with explicit blacklists
- school accessibility features
- `school_features.csv`
- SAI scores

## SAI Algorithm

SAI is intentionally computed from combined course offerings, not from a set of
pre-baked database counters.

`src/coursemap/sai.py` owns the algorithm:

- `regular_offerings()` adapts NEIS rows into the internal offering schema.
- `assignment_offerings()` converts virtual joint-course assignments into the
  same offering schema.
- `combine_offerings()` merges regular and joint offerings.
- `compute_sai()` recomputes subject diversity, domain breadth, domain balance,
  and final SAI from the combined offering set. Joint classes affect SAI only
  by changing that offering set; they are not a separate score bucket.

This means optimization code can propose assignments in memory, combine them
with regular offerings, and call `compute_sai()` without writing intermediate
SAI files.

## RL Assignment Policy

`scripts/11_train_rl_assignments.py` trains a PyTorch reinforcement-learning
policy for the main assignment problem. The agent observes candidate
`(hub, subject, domain)` actions, samples a budgeted sequence of joint-course openings,
and receives a tail-focused reward that prioritizes raising the minimum SAI of
weak schools over improving the average. It is compared against the greedy
baseline. Run it with:

```bash
make recommend-rl
```

Required feature joins must fail loudly. Schools with known missing required
features must be listed in `config/blacklists.yml`; they are excluded from
`analysis_schools.csv` rather than scored as zero.

Facility geocoding follows the original notebook: raw address, cleaned address,
then `대전광역시 {facility name}` keyword search. Put the key in `.env`:

```bash
KAKAO_REST_API_KEY=...
```

`.env` is ignored by Git. Direct environment variables and
`data/raw/Kakao api key.txt` are also supported.

NEIS raw collection also reads `.env`:

```bash
NEIS_API_KEY=...
```

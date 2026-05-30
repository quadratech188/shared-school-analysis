# Repository Guidelines

## Project Structure & Module Organization

This repository is a notebook-free data pipeline for CourseMap AI. Pipeline entry
points live in `scripts/`, with one small step per file and numbered execution
order, for example `01_prepare_school_master.py` and `09_score_sai.py`.
Reusable code belongs in `src/coursemap/` only when it is shared by multiple
steps. Configuration files live in `config/`, including input manifests,
blacklists, subject overrides, and ignored subjects.

Raw data is expected under `data/raw/`. Generated outputs go under `build/`:
`build/interim/` for cleaned intermediate tables, `build/processed/` for
model-ready tables, `build/metadata/` for audit reports, `build/review/` for
manual review queues, and `build/figures/` for plots.

## Build, Test, and Development Commands

Run the full pipeline:

```bash
make
```

Check raw input availability:

```bash
make check-inputs
```

Regenerate subject review files, allowing unassigned subjects:

```bash
make subject-review-list
make review-subjects
```

Build Kakao-geocoded public hub accessibility, after adding
`KAKAO_REST_API_KEY=...` to `.env`:

```bash
make geocode-facilities
make facility-accessibility
```

Train the reinforcement-learning assignment policy and compare it with the
greedy baseline:

```bash
make recommend-rl
```

Clean and rebuild generated artifacts:

```bash
make clean
make rebuild
```

Run the current joint-assignment simulation:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib \
python3 scripts/10_recommend_joint_assignments.py --budget 10 --radius-km 5
```

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation and explicit, small functions. Keep
pipeline scripts narrow: read inputs, call shared logic, write outputs. Prefer
Korean column names when matching source data, but keep internal algorithmic
schemas concise, such as `school`, `subject`, and `domain` in SAI offering
frames. Do not add shared helpers unless at least two steps use them.

## Testing Guidelines

There is no dedicated test suite yet. Before committing, run:

```bash
PYTHONPATH=src python3 -m py_compile src/coursemap/*.py scripts/*.py
make
```

For SAI or assignment changes, also run `scripts/10_recommend_joint_assignments.py`
and inspect the printed before/after statistics plus
`build/figures/joint_assignment_sai_dot.png`.

## Commit & Pull Request Guidelines

No repository-specific Git history is available here. Use concise imperative
commit messages, for example `Refactor SAI offering model` or
`Add subject ignore review flow`. Pull requests should describe the pipeline
steps changed, generated outputs affected, validation commands run, and any
blacklist or subject override updates.

## Data Integrity Rules

Required joins must fail loudly. Do not silently fill missing required features
with zero. If a school or subject must be excluded, document it explicitly in
`config/blacklists.yml` or `config/subject_ignores.csv`.

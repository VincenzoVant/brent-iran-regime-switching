"""Where every file lives, plus the size-budget check (spec §6.7, §8.2).

All paths are built from this file's own location (src/brent/paths.py -> two folders up = repo
root), never from the notebook's working directory. So a notebook finds data/raw/ no matter
which folder Jupyter was started in.
"""

import subprocess
from pathlib import Path

import pandas as pd

from brent import config as C

ROOT = Path(__file__).resolve().parents[2]          # repo root
DATA = ROOT / "data"
RAW = DATA / "raw"                                  # frozen downloads + manifest.csv (notebook 01)
PROCESSED = DATA / "processed"                      # weekly_panel.csv (notebook 02)
RESULTS = ROOT / "results"
FORECASTS = RESULTS / "forecasts"
FITS = RESULTS / "fits"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
FROZEN = RESULTS / "frozen"
CHECKPOINTS = RESULTS / "checkpoints"                # partial out-of-sample runs (git-ignored)
NOTEBOOKS = ROOT / "notebooks"


def save_table(df: pd.DataFrame, name: str) -> Path:
    """Write a report table to results/tables/<name>.csv."""
    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / f"{name}.csv"
    df.to_csv(out, index=False)
    return out


def _files(root: Path):
    """All files under root, skipping dot-folders/files and results/checkpoints."""
    skip = root / "results" / "checkpoints"
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if p == skip or skip in p.parents:
            continue
        if p.is_file():
            yield p, rel


def check_budget(raise_: bool = True, root: Path | None = None) -> pd.DataFrame:
    """Check the repo stays small (our "minimal local data" rule).

    Limits (config.SIZE_LIMIT_*): any file <= 10 MB; data/ <= 5 MB; data/ + results/ <= 20 MB;
    each notebook <= 5 MB, all notebooks <= 15 MB. Dot-folders (.venv, .git) and
    results/checkpoints are skipped. Returns a table of violations (empty = all good) and,
    with raise_=True, stops with an error listing them. Called in the last cell of every notebook.
    """
    root = ROOT if root is None else Path(root)
    rows = []
    data_total = total = nb_total = 0
    for p, rel in _files(root):
        size = p.stat().st_size
        if size > C.SIZE_LIMIT_FILE:
            rows.append(("file", str(rel), size, C.SIZE_LIMIT_FILE))
        top = rel.parts[0]
        if top == "data":
            data_total += size
        if top in ("data", "results"):
            total += size
        if top == "notebooks" and p.suffix == ".ipynb":
            nb_total += size
            if size > C.SIZE_LIMIT_NOTEBOOK:
                rows.append(("notebook", str(rel), size, C.SIZE_LIMIT_NOTEBOOK))
    if data_total > C.SIZE_LIMIT_DATA:
        rows.append(("data total", "data/", data_total, C.SIZE_LIMIT_DATA))
    if total > C.SIZE_LIMIT_TOTAL:
        rows.append(("data+results total", "data/ + results/", total, C.SIZE_LIMIT_TOTAL))
    if nb_total > C.SIZE_LIMIT_NOTEBOOKS:
        rows.append(("notebooks total", "notebooks/", nb_total, C.SIZE_LIMIT_NOTEBOOKS))
    out = pd.DataFrame(rows, columns=["rule", "path", "bytes", "limit"])
    if raise_ and not out.empty:
        raise ValueError("Size budget exceeded:\n" + out.to_string(index=False))
    return out


def git_hash() -> str:
    """Short hash of HEAD."""
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()

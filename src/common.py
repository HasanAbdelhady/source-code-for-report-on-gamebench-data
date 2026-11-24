"""Shared utilities for the OSSGameBench v2 bug-resolution study.

Every pipeline script imports this module. Scripts are executable with
`python -I` (isolated mode): each script inserts this directory into
sys.path explicitly, and nothing here depends on the current working
directory. Dataset paths come from config/config.yaml, overridable with
the OSSGAMEBENCH_DATA_ROOT environment variable.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

# Placeholder written into logs and persisted reports in place of the
# (machine-specific) absolute dataset location, so nothing shipped or
# logged depends on where the data happens to live on one machine.
DATA_ROOT_PLACEHOLDER = "<OSSGAMEBENCH_DATA_ROOT>"


def load_config(require_dataset: bool = True) -> dict:
    """Load config and resolve the dataset location.

    Stages that read only the shipped outputs/ artefacts (figures,
    report build) pass require_dataset=False so they can run without
    the 65.8 GB archive. Stages that touch raw parquet keep the strict
    check and fail early with an actionable message.
    """
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    root = Path(os.environ.get("OSSGAMEBENCH_DATA_ROOT", cfg["dataset"]["root"]))
    root = (PROJECT_ROOT / root).resolve() if not root.is_absolute() else root
    if not root.is_dir():
        if require_dataset:
            raise SystemExit(
                f"Dataset root not found: {root}\n"
                "Set dataset.root in config/config.yaml or export "
                "OSSGAMEBENCH_DATA_ROOT=/path/to/extracted/OSSGameBench"
            )
        logging.getLogger("common").warning(
            "dataset root %s not present; this stage does not read raw "
            "dataset files", root)
    cfg["_dataset_root"] = root
    global _DATASET_ROOT
    _DATASET_ROOT = root
    return cfg


_DATASET_ROOT: Path | None = None  # set by load_config()


def _resolve_dataset_root() -> Path | None:
    """Resolved dataset location (same logic as load_config, no existence check)."""
    raw = os.environ.get("OSSGAMEBENCH_DATA_ROOT")
    if raw is None:
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)["dataset"]["root"]
        except (OSError, KeyError, TypeError):
            return None
    root = Path(raw)
    return (PROJECT_ROOT / root).resolve() if not root.is_absolute() else root


def portable_path(p) -> str:
    """Machine-independent rendering of a path for logs and persisted files.

    Paths under PROJECT_ROOT become relative paths; paths under the
    dataset location are rendered against a fixed placeholder instead of
    the absolute directory on this machine.
    """
    p = p if isinstance(p, Path) else Path(p)
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        pass
    global _DATASET_ROOT
    if _DATASET_ROOT is None:
        _DATASET_ROOT = _resolve_dataset_root()
    if _DATASET_ROOT is not None:
        try:
            rel = p.relative_to(_DATASET_ROOT)
            return str(Path(DATA_ROOT_PLACEHOLDER) / rel)
        except ValueError:
            pass
    return str(p)


def dataset_table(cfg: dict, name: str) -> Path:
    return cfg["_dataset_root"] / cfg["dataset"]["tables"][name]


def out_path(cfg: dict, subdir: str, filename: str) -> Path:
    p = PROJECT_ROOT / "outputs" / subdir / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def project_path(*parts: str) -> Path:
    p = PROJECT_ROOT.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


class _PortablePathsFilter(logging.Filter):
    """Rewrite machine-specific absolute paths in log records.

    Keeps shipped logs (logs/*.log) free of user-specific directories:
    project files appear as relative paths, the dataset location as the
    placeholder from portable_path().
    """

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        record.msg = portable_text(msg)
        record.args = ()
        return True


def portable_text(text: str) -> str:
    """Replace known machine-specific absolute prefixes in free text."""
    root_str = str(PROJECT_ROOT)
    if root_str and root_str != "/" and root_str in text:
        text = text.replace(root_str + "/", "").replace(root_str, ".")
    ds = _DATASET_ROOT or _resolve_dataset_root()
    if ds is not None:
        ds_str = str(ds)
        text = text.replace(ds_str + "/", DATA_ROOT_PLACEHOLDER + "/")
        text = text.replace(ds_str, DATA_ROOT_PLACEHOLDER)
    return text


def get_logger(name: str) -> logging.Logger:
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_dir / f"{name}.log", mode="w")
        sh = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh.setFormatter(fmt)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
        logger.addFilter(_PortablePathsFilter())
    return logger


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    logging.getLogger(path.stem).info("wrote %s", portable_path(path))


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def duckdb_conn(cfg: dict):
    """Fresh in-memory DuckDB connection (parquet is built into duckdb>=1.1)."""
    import duckdb

    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false;")
    return con


def quote_glob(cfg: dict, name: str) -> str:
    """DuckDB-safe absolute path string for an entity parquet file."""
    return str(dataset_table(cfg, name)).replace("'", "''")

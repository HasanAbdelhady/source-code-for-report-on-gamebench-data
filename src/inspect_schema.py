"""Inspect the OSSGameBench v2 parquet schema and basic data quality.

Outputs:
  outputs/inventory/schema.json      — full column types, counts, ranges
  outputs/inventory/schema_report.txt — human-readable summary

Run:  python -I src/inspect_schema.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def main() -> None:
    cfg = common.load_config()
    log = common.get_logger("inspect_schema")
    con = common.duckdb_conn(cfg)
    # Record the location portably: the placeholder stands in for wherever
    # the dataset lives on the machine that ran the pipeline.
    report: dict = {
        "dataset_root": common.DATA_ROOT_PLACEHOLDER,
        "dataset_root_note": "resolved from OSSGAMEBENCH_DATA_ROOT env var "
                             "or config/config.yaml dataset.root",
    }

    for name in cfg["dataset"]["tables"]:
        path = common.quote_glob(cfg, name)
        log.info("inspecting %s", name)
        cols = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{path}')"
        ).fetchall()
        ncols = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{path}')"
        ).fetchone()[0]
        entry = {"columns": {c[0]: c[1] for c in cols}, "rows": ncols}

        # Null counts for every column (cheap: single scan, parquet stats)
        nulls = {}
        for c, t in entry["columns"].items():
            q = con.execute(
                f'SELECT COUNT(*) - COUNT("{c}") FROM read_parquet(\'{path}\')'
            ).fetchone()[0]
            if q:
                nulls[c] = q
        entry["null_counts"] = nulls

        # Ranges for timestamp-like columns
        ranges = {}
        for c, t in entry["columns"].items():
            if "TIMESTAMP" in t.upper():
                lo, hi = con.execute(
                    f'SELECT MIN("{c}"), MAX("{c}") FROM read_parquet(\'{path}\')'
                ).fetchone()
                ranges[c] = {"min": str(lo), "max": str(hi)}
        entry["timestamp_ranges"] = ranges

        # Sample first row (truncated bodies) for structure understanding
        try:
            row = con.execute(
                f"SELECT * FROM read_parquet('{path}') LIMIT 1"
            ).fetchdf().to_dict("records")[0]
            entry["sample_row"] = {
                k: (str(v)[:300]) for k, v in row.items()
            }
        except Exception as e:  # pragma: no cover
            entry["sample_row_error"] = str(e)

        report[name] = entry

    # ---- entity key structure ----
    extra = {}
    ip = common.quote_glob(cfg, "issue_pr")
    extra["issue_pr_columns"] = report["issue_pr"]["columns"]
    extra["issue_pr_pairs_distinct"] = con.execute(
        f"SELECT COUNT(*) FROM (SELECT DISTINCT * FROM read_parquet('{ip}'))"
    ).fetchone()[0]

    pc = common.quote_glob(cfg, "pr_commit")
    extra["pr_commit_columns"] = report["pr_commit"]["columns"]

    # Per-project commit-detail files: file count + total commit rows
    # via parquet footers only (no diff loading).
    commits_dir = cfg["_dataset_root"] / cfg["dataset"]["commits_dir"]
    files = sorted(commits_dir.glob("*.parquet"))
    extra["commit_files_count"] = len(files)
    glob = str(commits_dir / "*.parquet").replace("'", "''")
    tot = con.execute(
        f"SELECT SUM(num_rows) FROM parquet_file_metadata('{glob}')"
    ).fetchone()[0]
    extra["total_commit_rows"] = tot
    one = files[0]
    extra["commit_file_columns"] = {
        c[0]: c[1]
        for c in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{str(one)}')"
        ).fetchall()
    }
    report["_extra"] = extra

    out = common.out_path(cfg, "inventory", "schema.json")
    common.save_json(report, out)

    lines = [
        "OSSGameBench v2 schema report — root: "
        f"{common.DATA_ROOT_PLACEHOLDER} (see schema.json note)", "",
    ]
    for name in cfg["dataset"]["tables"]:
        e = report[name]
        lines.append(f"== {name}: {e['rows']} rows ==")
        for c, t in e["columns"].items():
            n = e["null_counts"].get(c, 0)
            lines.append(f"  {c}: {t}" + (f"  [nulls={n}]" if n else ""))
        for c, r in e["timestamp_ranges"].items():
            lines.append(f"  range {c}: {r['min']} .. {r['max']}")
        lines.append("")
    lines.append(f"commit detail files: {extra['commit_files_count']}")
    lines.append(f"total commit rows:   {extra['total_commit_rows']}")
    lines.append(f"issue_pr columns:    {extra['issue_pr_columns']}")
    lines.append(f"pr_commit columns:   {extra['pr_commit_columns']}")
    txt = common.out_path(cfg, "inventory", "schema_report.txt")
    txt.write_text("\n".join(lines), encoding="utf-8")
    log.info("schema report written to %s", txt)
    print("\n".join(lines[:80]))


if __name__ == "__main__":
    main()

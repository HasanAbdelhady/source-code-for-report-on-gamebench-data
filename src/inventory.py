"""Dataset inventory and data-quality audit for OSSGameBench v2.

Outputs:
  outputs/inventory/dataset_inventory.json — machine-readable inventory
  outputs/inventory/dataset_inventory.csv  — flat key/value inventory table
  outputs/inventory/label_distribution.csv — top issue labels (all frequencies)

Run:  python -I src/inventory.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

TS_RE = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$"


def scalar(con, sql):
    return con.execute(sql).fetchone()[0]


def main() -> None:
    cfg = common.load_config()
    log = common.get_logger("inventory")
    con = common.duckdb_conn(cfg)
    T = {k: common.quote_glob(cfg, k) for k in cfg["dataset"]["tables"]}
    inv: dict[str, object] = {}

    # ---------- entity sizes ----------
    counts = {}
    for name in T:
        counts[name] = scalar(con, f"SELECT COUNT(*) FROM read_parquet('{T[name]}')")
    inv["entity_rows"] = counts

    # ---------- composite-key uniqueness ----------
    inv["issue_unique_keys"] = scalar(
        con, f"SELECT COUNT(DISTINCT (project, number)) FROM read_parquet('{T['issue']}')"
    )
    inv["pr_unique_keys"] = scalar(
        con, f"SELECT COUNT(DISTINCT (project, number)) FROM read_parquet('{T['pr']}')"
    )
    inv["issue_pr_raw_rows"] = counts["issue_pr"]
    inv["issue_pr_unique_links"] = scalar(
        con,
        "SELECT COUNT(DISTINCT (project, issue_number, pr_number)) "
        f"FROM read_parquet('{T['issue_pr']}')",
    )
    inv["issue_pr_duplicate_rows"] = (
        inv["issue_pr_raw_rows"] - inv["issue_pr_unique_links"]
    )
    inv["pr_commit_rows"] = counts["pr_commit"]
    inv["pr_commit_unique"] = scalar(
        con,
        "SELECT COUNT(DISTINCT (project, pr_number, commit_sha)) "
        f"FROM read_parquet('{T['pr_commit']}')",
    )

    # ---------- project coverage ----------
    inv["projects_in_project_table"] = scalar(
        con, f"SELECT COUNT(DISTINCT project) FROM read_parquet('{T['project']}')"
    )
    for t in ("issue", "pr", "issue_pr", "pr_comment", "issue_comment"):
        inv[f"projects_in_{t}"] = scalar(
            con, f"SELECT COUNT(DISTINCT project) FROM read_parquet('{T[t]}')"
        )

    # ---------- issue / PR states ----------
    for t in ("issue", "pr"):
        rows = con.execute(
            f"SELECT state, COUNT(*) FROM read_parquet('{T[t]}') GROUP BY 1 ORDER BY 2 DESC"
        ).fetchall()
        inv[f"{t}_state_counts"] = {r[0]: r[1] for r in rows}

    # ---------- timestamp quality ----------
    for t, cols in (
        ("issue", ["created_at", "closed_at"]),
        ("pr", ["created_at", "closed_at", "merged_at"]),
    ):
        for c in cols:
            expr = f'COUNT(*) FILTER (WHERE "{c}" IS NOT NULL AND NOT regexp_full_match("{c}", \'{TS_RE}\'))'
            bad = scalar(
                con, f"SELECT {expr} FROM read_parquet('{T[t]}')"
            )
            inv[f"{t}_{c}_malformed"] = bad
        # negative durations among non-null pairs
        pairs = ("created_at", "closed_at") if t == "issue" else ("created_at", "closed_at")
        neg = scalar(
            con,
            f'SELECT COUNT(*) FROM read_parquet(\'{T[t]}\') WHERE "{pairs[1]}" IS NOT NULL '
            f'AND "{pairs[1]}" < "{pairs[0]}"',
        )
        inv[f"{t}_negative_{pairs[1]}_before_created"] = neg
    neg_m = scalar(
        con,
        f"SELECT COUNT(*) FROM read_parquet('{T['pr']}') "
        "WHERE merged_at IS NOT NULL AND merged_at < created_at",
    )
    inv["pr_negative_merged_before_created"] = neg_m

    # ---------- link referential integrity (on unique links) ----------
    links = (
        "SELECT DISTINCT project, issue_number, pr_number "
        f"FROM read_parquet('{T['issue_pr']}')"
    )
    inv["links_unmatched_to_issue_table"] = scalar(
        con,
        f"SELECT COUNT(*) FROM ({links}) l LEFT JOIN read_parquet('{T['issue']}') i "
        "ON l.project=i.project AND l.issue_number=i.number WHERE i.number IS NULL",
    )
    inv["links_unmatched_to_pr_table"] = scalar(
        con,
        f"SELECT COUNT(*) FROM ({links}) l LEFT JOIN read_parquet('{T['pr']}') p "
        "ON l.project=p.project AND l.pr_number=p.number WHERE p.number IS NULL",
    )

    # ---------- per-project commit files ----------
    commits_dir = cfg["_dataset_root"] / cfg["dataset"]["commits_dir"]
    glob = str(commits_dir / "*.parquet").replace("'", "''")
    inv["commit_detail_files"] = len(list(commits_dir.glob("*.parquet")))
    inv["total_commits"] = scalar(
        con, f"SELECT SUM(num_rows) FROM parquet_file_metadata('{glob}')"
    )

    # ---------- comments timestamps (paper says they exist; verify) ----------
    inv["issue_comment_has_created_at"] = "created_at" in {
        c[0] for c in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{T['issue_comment']}')"
        ).fetchall()
    }
    inv["pr_comment_has_created_at"] = "created_at" in {
        c[0] for c in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{T['pr_comment']}')"
        ).fetchall()
    }

    # ---------- label distribution ----------
    lab = con.execute(
        f"""SELECT unnest(labels) AS label, COUNT(*) AS n
            FROM read_parquet('{T['issue']}')
            WHERE labels IS NOT NULL AND len(labels) > 0
            GROUP BY 1 ORDER BY n DESC"""
    ).fetchdf()
    out = common.out_path(cfg, "inventory", "label_distribution.csv")
    lab.to_csv(out, index=False)
    inv["issues_with_no_labels"] = scalar(
        con,
        f"SELECT COUNT(*) FROM read_parquet('{T['issue']}') "
        "WHERE labels IS NULL OR len(labels)=0",
    )
    inv["distinct_issue_labels"] = len(lab)

    # ---------- observation window ----------
    lo, hi = con.execute(
        f"SELECT MIN(created_at), MAX(created_at) FROM read_parquet('{T['issue']}')"
    ).fetchone()
    inv["issue_created_range"] = {"min": str(lo), "max": str(hi)}
    lo, hi = con.execute(
        f"SELECT MIN(closed_at), MAX(closed_at) FROM read_parquet('{T['issue']}') "
        "WHERE closed_at IS NOT NULL"
    ).fetchone()
    inv["issue_closed_range"] = {"min": str(lo), "max": str(hi)}
    lo, hi = con.execute(
        f"SELECT MIN(merged_at), MAX(merged_at) FROM read_parquet('{T['pr']}') "
        "WHERE merged_at IS NOT NULL"
    ).fetchone()
    inv["pr_merged_range"] = {"min": str(lo), "max": str(hi)}

    common.save_json(inv, common.out_path(cfg, "inventory", "dataset_inventory.json"))

    # flat CSV
    import pandas as pd

    flat = []
    for k, v in inv.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                flat.append((f"{k}.{k2}", v2))
        else:
            flat.append((k, v))
    pd.DataFrame(flat, columns=["item", "value"]).to_csv(
        common.out_path(cfg, "inventory", "dataset_inventory.csv"), index=False
    )
    log.info("inventory written (%d items)", len(inv))
    for k, v in list(inv.items()):
        log.info("%s: %s", k, v)


if __name__ == "__main__":
    main()

"""Robustness checks R1-R7 for the primary comparison.

  R1 broader bug definition (label set: strict + symptom/qualified labels)
  R2 groups from ALL unique link references (incl. unmatched) [expected:
     identical within cohort - reported, not hidden]
  R3 issues with >=1 merged linked PR only
  R4 PR multiplicity as continuous variable (Spearman + per-category)
  R5 within-project-only analysis (projects with >=5 issues per group)
  R6 extreme-duration treatments: cap at 180 / 365 days; drop duration<1h
  R7 exclude the single largest contributing project of each group

Each check reports n, medians, Mann-Whitney U, p, Cliff's delta and whether
the interpretation (direction & qualitative magnitude) is preserved.
Run:  python -I src/robustness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from analyze import cliffs_delta, cliffs_interp  # noqa: E402


def build_flags(con, cfg, decision_filter):
    """issue -> is_bug flag under a given decision set."""
    mapping = pl.read_csv(common.out_path(cfg, "audit", "bug_label_mapping.csv"))
    keep = mapping.filter(pl.col("decision").is_in(decision_filter))["label"].to_list()
    keep_expr = ", ".join("'" + k.replace("'", "''") + "'" for k in keep)
    I = common.quote_glob(cfg, "issue")
    con.execute(f"""
      CREATE OR REPLACE TABLE flags AS
      SELECT project, number, COUNT(*) > 0 AS is_bug
      FROM (SELECT project, number, unnest(labels) lbl
            FROM read_parquet('{I}')
            WHERE labels IS NOT NULL AND len(labels) > 0)
      WHERE lbl IN ({keep_expr})
      GROUP BY 1,2""")


def cohort(con, cfg, extra_where="TRUE", dur_cap=None):
    dur = "(epoch(i.x_ts) - epoch(i.c_ts)) / 86400.0"
    if dur_cap:
        dur = f"LEAST({dur}, {dur_cap})"
    I = common.quote_glob(cfg, "issue")
    con.execute(f"""
      CREATE OR REPLACE TABLE coh AS
      WITH i AS (
        SELECT l.project, l.number AS issue_number,
               try_cast(l.created_at AS TIMESTAMPTZ) c_ts,
               try_cast(l.closed_at AS TIMESTAMPTZ) x_ts
        FROM read_parquet('{I}') l
        JOIN flags f USING (project, number)
        WHERE f.is_bug AND l.state='CLOSED'
          AND try_cast(l.created_at AS TIMESTAMPTZ) IS NOT NULL
          AND try_cast(l.closed_at AS TIMESTAMPTZ) IS NOT NULL
          AND try_cast(l.closed_at AS TIMESTAMPTZ)
              >= try_cast(l.created_at AS TIMESTAMPTZ)),
      lm AS (
        SELECT l.project, l.issue_number, l.pr_number,
               p.pr_number IS NOT NULL AS matched
        FROM (SELECT DISTINCT project, issue_number, pr_number
              FROM read_parquet('{common.quote_glob(cfg, "issue_pr")}')) l
        LEFT JOIN (SELECT DISTINCT project, number AS pr_number
                   FROM read_parquet('{common.quote_glob(cfg, "pr")}')) p
          USING (project, pr_number)),
      k AS (SELECT project, issue_number,
                   COUNT(*) FILTER (matched) k_m,
                   COUNT(*) k_all
            FROM lm GROUP BY 1,2)
      SELECT i.project, i.issue_number, {dur} AS d, k_m, k_all,
             CASE WHEN k_m=0 THEN 'C' WHEN k_m=1 THEN 'A' ELSE 'B' END gA,
             CASE WHEN k_all=0 THEN 'C' WHEN k_all=1 THEN 'A' ELSE 'B' END gB
      FROM i JOIN k USING (project, issue_number) WHERE {extra_where}""")
    return con.execute("SELECT gA, gB, d FROM coh").fetchdf()


def mwu(a, b):
    u, p = stats.mannwhitneyu(b, a, alternative="two-sided")
    return u, p, cliffs_delta(b, a)


def row(check, a, b, extra=""):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return {"check": check, "note": "empty group", "n_A": a.size, "n_B": b.size}
    u, p, d = mwu(a, b)
    return {"check": check, "n_A": int(a.size), "n_B": int(b.size),
            "median_A_days": float(np.median(a)), "median_B_days": float(np.median(b)),
            "median_diff_days": float(np.median(b) - np.median(a)),
            "mwu_U": float(u), "p": float(p), "cliffs_delta": float(d),
            "magnitude": cliffs_interp(d),
            "direction_preserved": bool(np.median(b) > np.median(a)),
            "notes": extra}


def main() -> None:
    cfg = common.load_config()
    log = common.get_logger("robustness")
    con = common.duckdb_conn(cfg)
    rows = []

    # baseline (strict, primary grouping)
    build_flags(con, cfg, ["strict"])
    base = cohort(con, cfg)
    rows.append(row("baseline_strict_primary",
                    base.loc[base.gA == "A", "d"], base.loc[base.gA == "B", "d"]))

    # R1 broad definition
    build_flags(con, cfg, ["strict", "broad"])
    rb = cohort(con, cfg)
    rows.append(row("R1_broad_bug_definition",
                    rb.loc[rb.gA == "A", "d"], rb.loc[rb.gA == "B", "d"],
                    "same grouping rule, broader label set"))

    # R2 all-reference grouping (within strict cohort)
    build_flags(con, cfg, ["strict"])
    base2 = cohort(con, cfg)
    rows.append(row("R2_grouping_on_all_references",
                    base2.loc[base2.gB == "A", "d"], base2.loc[base2.gB == "B", "d"],
                    "cohort has zero unmatched link references, so this is "
                    "identical to baseline by construction"))

    # R3 at least one merged linked PR
    df = pl.read_parquet(common.out_path(cfg, "intermediate", "analysis_dataset.parquet"))
    a3 = df.filter((pl.col("group_matched") == "A_single") & (pl.col("n_merged") >= 1))["duration_days"].to_numpy()
    b3 = df.filter((pl.col("group_matched") == "B_multi") & (pl.col("n_merged") >= 1))["duration_days"].to_numpy()
    rows.append(row("R3_at_least_one_merged_PR", a3, b3))

    # R4 continuous multiplicity
    m = df.filter(pl.col("n_links_matched") >= 1)
    rho, p = stats.spearmanr(m["n_links_matched"], m["duration_days"])
    rows.append({"check": "R4_continuous_multiplicity", "spearman_rho": float(rho),
                 "p": float(p), "n": int(m.height),
                 "direction_preserved": bool(rho > 0),
                 "notes": "rank correlation over issues with >=1 linked PR"})

    # R5 within-project only (>=5 per group)
    both = df.filter(pl.col("group_matched").is_in(["A_single", "B_multi"]))
    cnt = both.group_by("project").agg(
        (pl.col("group_matched") == "A_single").sum().alias("nA"),
        (pl.col("group_matched") == "B_multi").sum().alias("nB"))
    keep = cnt.filter((pl.col("nA") >= 5) & (pl.col("nB") >= 5))["project"].to_list()
    sub = both.filter(pl.col("project").is_in(keep))
    a5 = sub.filter(pl.col("group_matched") == "A_single")["duration_days"].to_numpy()
    b5 = sub.filter(pl.col("group_matched") == "B_multi")["duration_days"].to_numpy()
    rows.append(row("R5_projects_with_ge5_each_group", a5, b5,
                    f"{len(keep)} projects contribute"))

    # R6 extreme durations
    for cap in (180.0, 365.0):
        a6 = np.minimum(df.filter(pl.col("group_matched") == "A_single")["duration_days"].to_numpy(), cap)
        b6 = np.minimum(df.filter(pl.col("group_matched") == "B_multi")["duration_days"].to_numpy(), cap)
        rows.append(row(f"R6_winsorized_at_{int(cap)}_days", a6, b6))
    a6b = df.filter((pl.col("group_matched") == "A_single") & (pl.col("duration_days") >= 1/24))["duration_days"].to_numpy()
    b6b = df.filter((pl.col("group_matched") == "B_multi") & (pl.col("duration_days") >= 1/24))["duration_days"].to_numpy()
    rows.append(row("R6b_drop_durations_under_1h", a6b, b6b))

    # R7 drop largest contributing project of each group
    top_a = df.filter(pl.col("group_matched") == "A_single")["project"].value_counts().sort("count", descending=True).head(1)["project"].to_list()
    top_b = df.filter(pl.col("group_matched") == "B_multi")["project"].value_counts().sort("count", descending=True).head(1)["project"].to_list()
    a7 = df.filter((pl.col("group_matched") == "A_single") & (~pl.col("project").is_in(top_a)))["duration_days"].to_numpy()
    b7 = df.filter((pl.col("group_matched") == "B_multi") & (~pl.col("project").is_in(top_b)))["duration_days"].to_numpy()
    rows.append(row("R7_drop_top_project_per_group", a7, b7,
                    f"dropped {top_a[0]} (A) and {top_b[0]} (B)"))

    out = pd.DataFrame(rows)
    out.to_csv(common.out_path(cfg, "statistics", "robustness_table.csv"), index=False)
    common.save_json(rows, common.out_path(cfg, "statistics", "robustness_results.json"))
    log.info("robustness checks: %d", len(rows))
    for r in rows:
        log.info("%s | nA=%s nB=%s diff=%s d=%s p=%s",
                 r.get("check"), r.get("n_A", r.get("n")), r.get("n_B", "-"),
                 round(r.get("median_diff_days", r.get("spearman_rho", float("nan"))), 3),
                 round(r.get("cliffs_delta", float("nan")), 3),
                 f"{r.get('p', float('nan')):.3g}")


if __name__ == "__main__":
    main()

"""Build the analysis dataset for the PR-multiplicity / closure-duration study.

Stages (row counts logged and persisted):
  S0 all issues                     -> per-issue strict/broad label flags
  S1 strict bug label               (primary; broad kept as a column)
  S2 state CLOSED with non-null closed_at
  S3 timestamps parse + non-negative duration
  S4 eligible analysis cohort

Per issue we record explicitly linked PR references:
  n_links_all      unique (project, pr_number) references in issue_pr
  n_links_matched  those resolvable to a record in pr.parquet
  n_unmatched      references not present in the PR table (kept, not dropped)

Groups (primary, on matched links): A = exactly 1, B = 2+, C = 0.
A parallel classification on ALL references is stored for sensitivity.

Outputs:
  outputs/intermediate/analysis_dataset.parquet
  outputs/intermediate/excluded_issues.csv
  outputs/statistics/exclusion_counts.json
  outputs/statistics/snapshot_estimate.json
  outputs/tables/data_dictionary.csv

Run:  python -I src/prepare_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def main() -> None:
    cfg = common.load_config()
    log = common.get_logger("prepare_data")
    con = common.duckdb_conn(cfg)
    T = {k: common.quote_glob(cfg, k) for k in cfg["dataset"]["tables"]}
    seed = cfg["analysis"]["seed"]

    # ---- 1. issue-level bug flags via the audited label mapping ----------
    mapping = pl.read_csv(common.out_path(cfg, "audit", "bug_label_mapping.csv"))
    issue_file = str(common.dataset_table(cfg, "issue"))
    iss = pl.read_parquet(
        issue_file,
        columns=["project", "number", "state", "created_at", "closed_at", "labels"],
    )
    log.info("S0 issues loaded: %d", len(iss))
    exploded = (
        iss.select(["project", "number", "labels"])
        .explode("labels")
        .rename({"labels": "lbl"})
        .drop_nulls("lbl")
    )
    flags = (
        exploded.join(mapping.select("label", "decision"), left_on="lbl",
                      right_on="label", how="left")
        .group_by(["project", "number"])
        .agg(
            pl.col("decision").eq("strict").any().alias("is_strict"),
            pl.col("decision").is_in(["strict", "broad"]).any().alias("is_broad"),
        )
    )
    log.info("strict issues: %d | broad issues: %d",
             flags["is_strict"].sum(), flags["is_broad"].sum())
    scratch = str(common.out_path(cfg, "intermediate", "_label_flags.parquet"))
    flags.write_parquet(scratch)
    con.execute(f"CREATE TABLE flags AS SELECT * FROM read_parquet('{scratch}')")

    # ---- 2. cohort with stages ---------------------------------------------
    con.execute(f"""
    CREATE TABLE issues AS
    SELECT project,
           number,
           state,
           try_cast(created_at AS TIMESTAMPTZ) AS created_ts,
           try_cast(closed_at AS TIMESTAMPTZ)  AS closed_ts,
           coalesce(is_strict, FALSE) AS is_strict,
           coalesce(is_broad, FALSE)  AS is_broad
    FROM read_parquet('{T["issue"]}') l
    LEFT JOIN flags f USING (project, number)""")
    n0 = con.execute("SELECT COUNT(*) FROM issues").fetchone()[0]
    n1 = con.execute("SELECT COUNT(*) FROM issues WHERE is_strict").fetchone()[0]
    n2 = con.execute(
        "SELECT COUNT(*) FROM issues WHERE is_strict AND state='CLOSED' "
        "AND closed_ts IS NOT NULL").fetchone()[0]
    n3 = con.execute(
        "SELECT COUNT(*) FROM issues WHERE is_strict AND state='CLOSED' "
        "AND closed_ts IS NOT NULL AND created_ts IS NOT NULL "
        "AND closed_ts >= created_ts").fetchone()[0]
    con.execute("""
    CREATE TABLE cohort AS
    SELECT *,
           epoch(closed_ts) - epoch(created_ts) AS duration_s
    FROM issues
    WHERE is_strict AND state='CLOSED'
      AND closed_ts IS NOT NULL AND created_ts IS NOT NULL
      AND closed_ts >= created_ts""")
    log.info("stages: S0=%d S1_strict=%d S2_closed=%d S3_valid_dur=%d", n0, n1, n2, n3)

    # excluded strict bugs (small enough to list)
    con.execute(f"""
    COPY (
      SELECT i.project, i.number, i.state,
             CASE
               WHEN i.state <> 'CLOSED' THEN 'state_not_closed'
               WHEN i.closed_ts IS NULL THEN 'missing_closed_at'
               WHEN i.created_ts IS NULL THEN 'unparseable_created_at'
               WHEN i.closed_ts < i.created_ts THEN 'negative_duration'
             END AS exclusion_reason
      FROM issues i
      WHERE i.is_strict AND NOT (
              i.state='CLOSED' AND i.closed_ts IS NOT NULL
              AND i.created_ts IS NOT NULL AND i.closed_ts >= i.created_ts)
    ) TO '{common.out_path(cfg, "intermediate", "excluded_issues.csv")}' (HEADER)""")

    # ---- 3. linked PR references -------------------------------------------
    con.execute(f"""
    CREATE TABLE links AS
    SELECT DISTINCT project, issue_number, pr_number
    FROM read_parquet('{T["issue_pr"]}')""")
    con.execute(f"""
    CREATE TABLE prs AS
    SELECT project, number AS pr_number, state, contributor,
           try_cast(created_at AS TIMESTAMPTZ) AS pr_created,
           try_cast(closed_at  AS TIMESTAMPTZ) AS pr_closed,
           try_cast(merged_at  AS TIMESTAMPTZ) AS pr_merged
    FROM read_parquet('{T["pr"]}')""")
    con.execute(f"""
    CREATE TABLE links_m AS
    SELECT l.project, l.issue_number, l.pr_number,
           p.pr_number IS NOT NULL AS matched
    FROM links l LEFT JOIN prs p USING (project, pr_number)""")

    # ---- 4. per-issue link + PR attributes ---------------------------------
    con.execute(f"""
    CREATE TABLE link_counts AS
    SELECT project, issue_number,
           COUNT(*) AS n_links_all,
           COUNT(*) FILTER (matched) AS n_links_matched,
           COUNT(*) FILTER (NOT matched) AS n_unmatched
    FROM links_m GROUP BY 1,2""")

    con.execute(f"""
    CREATE TABLE pr_attrs AS
    SELECT lm.project, lm.issue_number,
           COUNT(DISTINCT p.contributor) AS n_contrib,
           COUNT(*) FILTER (p.state='MERGED') AS n_merged,
           COUNT(lm.pr_number) AS n_matched_prs,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY
             CASE WHEN p.pr_merged IS NOT NULL
                  THEN (epoch(p.pr_merged)-epoch(p.pr_created))/86400.0
                  WHEN p.pr_closed IS NOT NULL
                  THEN (epoch(p.pr_closed)-epoch(p.pr_created))/86400.0 END
           ) AS median_pr_time_days,
           MIN((epoch(p.pr_created)-epoch(c.created_ts))/86400.0) AS first_pr_days_after_issue,
           MAX((epoch(p.pr_created)-epoch(c.created_ts))/86400.0) AS last_pr_days_after_issue,
           SUM(CASE WHEN epoch(p.pr_created) <= epoch(c.closed_ts) THEN 1 ELSE 0 END)
             AS n_pr_created_before_issue_close
    FROM links_m lm
    JOIN prs p USING (project, pr_number)
    JOIN cohort c ON c.project=lm.project AND c.number=lm.issue_number
    WHERE lm.matched
    GROUP BY 1,2""")

    con.execute(f"""
    CREATE TABLE commit_counts AS
    SELECT lm.project, lm.issue_number, COUNT(DISTINCT pc.commit_sha) AS n_commits
    FROM links_m lm
    JOIN read_parquet('{T["pr_commit"]}') pc
      ON pc.project=lm.project AND pc.pr_number=lm.pr_number
    WHERE lm.matched
    GROUP BY 1,2""")

    con.execute(f"""
    CREATE TABLE pr_comments AS
    SELECT project, pr_number, COUNT(*) AS n
    FROM read_parquet('{T["pr_comment"]}') GROUP BY 1,2""")
    con.execute(f"""
    CREATE TABLE issue_comments AS
    SELECT project, issue_number, COUNT(*) AS n
    FROM read_parquet('{T["issue_comment"]}') GROUP BY 1,2""")
    con.execute("""
    CREATE TABLE pr_comment_sums AS
    SELECT lm.project, lm.issue_number, SUM(coalesce(pcm.n, 0)) AS n
    FROM links_m lm
    LEFT JOIN pr_comments pcm USING (project, pr_number)
    WHERE lm.matched
    GROUP BY 1,2""")

    # ---- 5. final analysis table --------------------------------------------
    con.execute(f"""
    COPY (
      SELECT c.project, c.number AS issue_number,
             c.created_ts, c.closed_ts,
             c.duration_s / 86400.0 AS duration_days,
             year(c.created_ts) AS created_year,
             c.is_broad,
             coalesce(lc.n_links_all, 0)     AS n_links_all,
             coalesce(lc.n_links_matched, 0) AS n_links_matched,
             coalesce(lc.n_unmatched, 0)     AS n_unmatched,
             CASE WHEN coalesce(lc.n_links_matched,0)=0 THEN 'C_zero'
                  WHEN coalesce(lc.n_links_matched,0)=1 THEN 'A_single'
                  ELSE 'B_multi' END AS group_matched,
             CASE WHEN coalesce(lc.n_links_all,0)=0 THEN 'C_zero'
                  WHEN coalesce(lc.n_links_all,0)=1 THEN 'A_single'
                  ELSE 'B_multi' END AS group_all,
             coalesce(pa.n_contrib, 0)  AS n_contrib,
             coalesce(pa.n_merged, 0)   AS n_merged,
             coalesce(pa.n_matched_prs, 0) AS n_matched_prs,
             pa.median_pr_time_days,
             pa.first_pr_days_after_issue,
             pa.last_pr_days_after_issue,
             coalesce(pa.n_pr_created_before_issue_close, 0)
               AS n_pr_created_before_issue_close,
             coalesce(cc.n_commits, 0)  AS n_commits,
             coalesce(pcs.n, 0)         AS n_pr_comments,
             coalesce(icm.n, 0)         AS n_issue_comments
      FROM cohort c
      LEFT JOIN link_counts lc ON lc.project=c.project AND lc.issue_number=c.number
      LEFT JOIN pr_attrs   pa ON pa.project=c.project AND pa.issue_number=c.number
      LEFT JOIN commit_counts cc ON cc.project=c.project AND cc.issue_number=c.number
      LEFT JOIN pr_comment_sums pcs ON pcs.project=c.project AND pcs.issue_number=c.number
      LEFT JOIN issue_comments icm ON icm.project=c.project AND icm.issue_number=c.number
    ) TO '{common.out_path(cfg, "intermediate", "analysis_dataset.parquet")}' (FORMAT PARQUET)""")

    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{common.out_path(cfg,'intermediate','analysis_dataset.parquet')}')").fetchone()[0]
    log.info("S4 analysis cohort written: %d rows", n)
    for g, cnt in con.execute(f"""SELECT group_matched, COUNT(*) FROM read_parquet(
        '{common.out_path(cfg,"intermediate","analysis_dataset.parquet")}')
        GROUP BY 1 ORDER BY 1""").fetchall():
        log.info("  group %-9s %d", g, cnt)

    # snapshot estimate = max observed event timestamp
    snap = con.execute(f"""
      SELECT max(ts) FROM (
        SELECT max(try_cast(created_at AS TIMESTAMPTZ)) ts FROM read_parquet('{T["issue"]}')
        UNION ALL SELECT max(try_cast(closed_at AS TIMESTAMPTZ)) FROM read_parquet('{T["issue"]}')
        UNION ALL SELECT max(try_cast(created_at AS TIMESTAMPTZ)) FROM read_parquet('{T["pr"]}')
        UNION ALL SELECT max(try_cast(closed_at AS TIMESTAMPTZ)) FROM read_parquet('{T["pr"]}')
        UNION ALL SELECT max(try_cast(merged_at AS TIMESTAMPTZ)) FROM read_parquet('{T["pr"]}'))""").fetchone()[0]
    common.save_json(
        {"snapshot_estimate_utc": str(snap),
         "method": "max of all event timestamps (issue/pr created, closed, merged); "
                   "a lower bound on the true collection date",
         "seed": seed},
        common.out_path(cfg, "statistics", "snapshot_estimate.json"))

    common.save_json(
        {"S0_total_issues": n0, "S1_strict_bug": n1, "S2_closed_nonnull_closed_at": n2,
         "S3_valid_nonneg_duration": n3, "S4_analysis_cohort": int(n)},
        common.out_path(cfg, "statistics", "exclusion_counts.json"))

    # data dictionary
    dd = [
        ("project", "str", "repository full name 'owner/repo' (composite key part)"),
        ("issue_number", "int", "GitHub issue number; unique only within project"),
        ("created_ts/closed_ts", "timestamptz UTC", "issue creation/closure timestamps"),
        ("duration_days", "float days", "closed_ts - created_ts; OBSERVED issue lifetime, NOT debugging effort"),
        ("created_year", "int", "calendar year of issue creation (FE covariate)"),
        ("is_broad", "bool", "issue matches the broader bug definition"),
        ("n_links_all", "int", "unique (project,pr_number) references in issue_pr"),
        ("n_links_matched", "int", "unique references resolved to pr.parquet records"),
        ("n_unmatched", "int", "references with no matching PR record"),
        ("group_matched", "cat", "PRIMARY group: A_single=1 matched PR, B_multi>=2, C_zero=0"),
        ("group_all", "cat", "sensitivity group using all unique references"),
        ("n_contrib", "int", "distinct linked-PR authors (matched PRs)"),
        ("n_merged", "int", "linked PRs with state MERGED"),
        ("n_matched_prs", "int", "matched linked PRs with timestamps"),
        ("median_pr_time_days", "float", "median per-PR creation->merge (or close) duration"),
        ("first/last_pr_days_after_issue", "float", "PR creation offset from issue creation (days)"),
        ("n_pr_created_before_issue_close", "int", "matched linked PRs created at/before issue closure"),
        ("n_commits", "int", "distinct commits reachable via pr_commit for matched PRs"),
        ("n_pr_comments", "int", "pr_comment rows across the matched linked PRs"),
        ("n_issue_comments", "int", "issue_comment table rows for this issue"),
    ]
    import pandas as pd
    pd.DataFrame(dd, columns=["field", "type", "definition"]).to_csv(
        common.out_path(cfg, "tables", "data_dictionary.csv"), index=False)
    log.info("prepare_data complete")


if __name__ == "__main__":
    main()

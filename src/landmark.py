"""Fixed-landmark analysis for the PR-multiplicity / closure-duration study.

Motivation: in the primary analysis the PR group (1 vs >=2 linked PRs) is
defined using links accumulated over the issue's ENTIRE observed lifetime,
so an issue open longer has more opportunity to be in the multi-PR group.
A fixed landmark breaks that dependence by defining group membership at a
FIXED calendar time after issue creation, and comparing only the time from
the landmark to closure.

Design
------
  * Cohort: closed strict-bug issues with >=1 matched linked PR
    (the primary A+B cohort from analysis_dataset.parquet).
  * Landmark L days after issue creation (primary L = 30; 7 and 90 as
    sensitivity).  Only issues still open at L (duration_days >= L) are
    "at risk"; issues closed earlier cannot contribute the outcome and are
    excluded with their counts reported.
  * Landmark group = number of MATCHED linked PRs CREATED by L:
      L_zero (0), L_one (1), L_multi (>=2).
  * Outcome = additional days from the landmark to closure:
      extra_days = duration_days - L.
  * Primary comparison: L_one vs L_multi (mirrors A vs B).
    Secondary: L_zero vs L_multi (PRs appeared only after the landmark).
  * Mechanism statistic: among original Group B issues at risk at L,
    the share whose multi-PR status only developed AFTER the landmark.

Honest limitations (also discussed in the report):
  * PR group at the landmark uses PR CREATION time; the dataset does not
    record WHEN a PR was linked to the issue, so a PR created before L but
    linked afterwards still counts toward the landmark group.
  * Only closed issues are analysed (the outcome needs closed_at); issues
    still open at the dataset snapshot are not represented.
  * Restricting to issues open at L selects a subgroup that is already
    slower than the cohort average; estimates are conditional on that.

Outputs:
  outputs/statistics/landmark_results.json
Run:  python -I src/landmark.py
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
from analyze import (  # noqa: E402
    cliffs_delta, cliffs_interp, hodges_lehmann, bootstrap_median_diff_cluster,
)

PRIMARY_L = 30
SENSITIVITY_L = [7, 30, 90]


def build_landmark_table(cfg, con, ad: pl.DataFrame) -> pl.DataFrame:
    """Per-issue: number of matched linked PRs created by each landmark L."""
    T = {k: common.quote_glob(cfg, k) for k in cfg["dataset"]["tables"]}
    scratch = str(common.out_path(cfg, "intermediate",
                                  "_landmark_cohort.parquet"))
    ad.write_parquet(scratch)
    con.execute("CREATE TABLE ab AS SELECT project, issue_number, created_ts, "
                "closed_ts, duration_days, group_matched, project "
                f"FROM read_parquet('{scratch}') "
                "WHERE group_matched <> 'C_zero'")
    con.execute(f"""
    CREATE TABLE links AS
    SELECT DISTINCT project, issue_number, pr_number
    FROM read_parquet('{T["issue_pr"]}')""")
    con.execute(f"""
    CREATE TABLE prs AS
    SELECT project, number AS pr_number,
           try_cast(created_at AS TIMESTAMPTZ) AS pr_created
    FROM read_parquet('{T["pr"]}')""")
    con.execute(f"""
    CREATE TABLE pr_offsets AS
    SELECT ab.project, ab.issue_number,
           (epoch(p.pr_created) - epoch(ab.created_ts)) / 86400.0 AS offset_days
    FROM ab
    JOIN links l ON l.project = ab.project AND l.issue_number = ab.issue_number
    JOIN prs p ON p.project = l.project AND p.pr_number = l.pr_number""")
    counts = {"L7": "offset_days <= 7", "L30": "offset_days <= 30",
              "L90": "offset_days <= 90"}
    con.execute(f"""
    CREATE TABLE by_landmark AS
    SELECT project, issue_number,
           {', '.join(f"COUNT(*) FILTER ({cond}) AS {name}"
                      for name, cond in counts.items())}
    FROM pr_offsets GROUP BY 1, 2""")
    res = con.execute("""
    SELECT ab.project, ab.issue_number, ab.duration_days, ab.group_matched,
           coalesce(b.L7, 0)  AS n_pr_by_7,
           coalesce(b.L30, 0) AS n_pr_by_30,
           coalesce(b.L90, 0) AS n_pr_by_90
    FROM ab LEFT JOIN by_landmark b USING (project, issue_number)""").df()
    return pl.from_pandas(res)


def group_at_landmark(n: int) -> str:
    return "L_zero" if n == 0 else ("L_one" if n == 1 else "L_multi")


def compare(a: np.ndarray, b: np.ndarray, projects_a, projects_b,
            all_projects, rng, iters, label: str) -> dict:
    """b vs a: Mann-Whitney, Cliff's delta, HL shift, clustered CIs."""
    u = stats.mannwhitneyu(b, a, alternative="two-sided")
    d = cliffs_delta(b, a)
    a_by_p = pd.Series(a).groupby(np.asarray(projects_a)).apply(
        np.asarray).to_dict()
    b_by_p = pd.Series(b).groupby(np.asarray(projects_b)).apply(
        np.asarray).to_dict()
    ci = bootstrap_median_diff_cluster(a_by_p, b_by_p,
                                       sorted(set(projects_a) | set(projects_b)),
                                       rng, iters)
    return {
        "comparison": label,
        "n_ref": int(len(a)), "n_target": int(len(b)),
        "median_ref_days": float(np.median(a)),
        "median_target_days": float(np.median(b)),
        "median_diff_days": float(np.median(b) - np.median(a)),
        "mw_u": float(u.statistic), "mw_p": float(u.pvalue),
        "cliffs_delta": float(d), "cliffs_magnitude": cliffs_interp(d),
        "hodges_lehmann_shift_days": hodges_lehmann(b, a, rng),
        "clustered_ci_median_diff_days": [round(ci[0], 2), round(ci[1], 2)],
        "q1_ref": float(np.quantile(a, .25)), "q3_ref": float(np.quantile(a, .75)),
        "q1_target": float(np.quantile(b, .25)), "q3_target": float(np.quantile(b, .75)),
    }


def main() -> None:
    cfg = common.load_config()
    log = common.get_logger("landmark")
    con = common.duckdb_conn(cfg)
    rng = np.random.default_rng(cfg["analysis"]["seed"])
    iters = cfg["analysis"]["bootstrap_iterations"]

    ad = pl.read_parquet(
        common.out_path(cfg, "intermediate", "analysis_dataset.parquet"))
    tbl = build_landmark_table(cfg, con, ad)
    log.info("A+B cohort rows with landmark counts: %d", len(tbl))

    out: dict = {
        "seed": cfg["analysis"]["seed"],
        "bootstrap_iterations": iters,
        "definition": {
            "cohort": "closed strict-bug issues with >=1 matched linked PR",
            "at_risk": f"issue still open L days after creation "
                       f"(duration_days >= L); primary L = {PRIMARY_L}",
            "landmark_group": "count of matched linked PRs CREATED within "
                              "L days of issue creation (0 / 1 / >=2)",
            "outcome": "additional days from landmark to issue closure",
            "caveats": [
                "PR creation time is used as a proxy for link time; the "
                "dataset has no link-creation timestamps",
                "issues still open at the snapshot are not in the cohort",
                "restricting to issues open at L conditions on slow closure",
            ],
        },
        "by_landmark": {},
    }

    for L in SENSITIVITY_L:
        ncol = f"n_pr_by_{L}"
        t = tbl.filter(pl.col("duration_days") >= L).with_columns(
            pl.col(ncol).map_elements(group_at_landmark,
                                      return_dtype=pl.Utf8).alias("lg"),
            (pl.col("duration_days") - float(L)).alias("extra_days"),
        )
        entry: dict = {
            "L_days": L,
            "n_ab_cohort": int(len(tbl)),
            "n_at_risk": int(len(t)),
            "n_closed_before_landmark": int(len(tbl) - len(t)),
            "group_sizes": t["lg"].value_counts().to_dicts(),
        }
        a = t.filter(pl.col("lg") == "L_one")
        b = t.filter(pl.col("lg") == "L_multi")
        z = t.filter(pl.col("lg") == "L_zero")
        entry["primary_one_vs_multi"] = compare(
            a["extra_days"].to_numpy(), b["extra_days"].to_numpy(),
            a["project"].to_list(), b["project"].to_list(), None,
            rng, iters, "L_one vs L_multi (extra days after landmark)")
        entry["secondary_zero_vs_multi"] = compare(
            z["extra_days"].to_numpy(), b["extra_days"].to_numpy(),
            z["project"].to_list(), b["project"].to_list(), None,
            rng, iters, "L_zero vs L_multi (extra days after landmark)")
        # mechanism: how much of Group B's multiplicity postdates the landmark
        gb = t.filter(pl.col("group_matched") == "B_multi")
        ga = t.filter(pl.col("group_matched") == "A_single")
        entry["original_B_multiplicity_by_landmark"] = (
            {str(k): int(v) for k, v in zip(*np.unique(
                np.clip(gb[ncol].to_numpy(), 0, 3), return_counts=True))}
            if len(gb) else {})
        entry["original_B_share_multi_only_after_landmark"] = (
            float((gb[ncol] < 2).mean()) if len(gb) else None)
        entry["original_A_share_zero_at_landmark"] = (
            float((ga[ncol] == 0).mean()) if len(ga) else None)
        # project-level sign test for one vs multi at this landmark
        proj = (t.filter(pl.col("lg").is_in(["L_one", "L_multi"]))
                .group_by("project")
                .agg(pl.col("extra_days").filter(pl.col("lg") == "L_one")
                     .median().alias("m1"),
                     pl.col("extra_days").filter(pl.col("lg") == "L_multi")
                     .median().alias("m2"))
                .drop_nulls()
                .with_columns((pl.col("m2") > pl.col("m1")).alias("b_longer")))
        wins = int(proj["b_longer"].sum())
        nproj = len(proj)
        entry["project_sign_test"] = {
            "n_projects_both_groups_at_landmark": nproj,
            "projects_where_multi_longer": wins,
            "binomial_p": float(stats.binomtest(wins, nproj, 0.5).pvalue)
            if nproj else None,
        }
        out["by_landmark"][str(L)] = entry
        log.info("L=%d: at risk %d, groups %s, one-vs-multi meddiff %.1f d, delta %.3f",
                 L, len(t), entry["group_sizes"],
                 entry["primary_one_vs_multi"]["median_diff_days"],
                 entry["primary_one_vs_multi"]["cliffs_delta"])

    common.save_json(out, common.out_path(cfg, "statistics",
                                          "landmark_results.json"))
    log.info("landmark analysis complete")


if __name__ == "__main__":
    main()

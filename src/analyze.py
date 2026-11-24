"""Primary, project-level, secondary, timing-bias and exploratory analyses.

Statistical plan (Section 1 is confirmatory; later sections are secondary
or exploratory and are labelled as such everywhere):

 1. PRIMARY: closed strict-bug issues, Group A (exactly 1 matched linked PR)
    vs Group B (>=2 matched linked PRs).
      - Mann-Whitney U (two-sided; one-sided p for directional H1 too)
      - Cliff's delta with Romano et al. magnitude thresholds
      - Hodges-Lehmann shift estimate
      - Bootstrap CIs for the difference in medians:
          (a) issue-level resampling (independence assumed)
          (b) project-cluster resampling (accounts for within-project
              dependence; preferred, reported as the primary CI)
 2. PROJECT-LEVEL: per-project median differences + sign test;
    leave-most-contributing-projects-out sensitivity; project & year
    fixed-effects OLS on log1p(duration) with project-clustered SEs.
 3. SECONDARY: Spearman correlation duration ~ linked-PR count; medians by
    PR-count category; Kruskal-Wallis across categories 1..4.
 4. TIMING / EXPOSURE BIAS: position of linked-PR creation within the issue
    lifetime; 'determined-exposure' subset (all linked PRs created within
    the first 50% of the issue's observed lifetime) re-test.
 5. EXPLORATORY resolution-path profile by group (contributors, merge rate,
    PR durations, commits, comments) - Holm-adjusted within this family.
 6. EXPLORATORY right-censored supplementary (open+closed strict-bug issues
    with >=1 link, KM + log-rank); link counts for open issues are
    snapshot-truncated; interpretation heavily caveated.

Outputs: outputs/statistics/primary_results.json (+ *_table.csv files)
Run:  python -I src/analyze.py
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


def cliffs_delta(x, y):
    nx, ny = len(x), len(y)
    gt = np.sum(x[:, None] > y[None, :])
    lt = np.sum(x[:, None] < y[None, :])
    return (gt - lt) / (nx * ny)


def cliffs_interp(d):
    a = abs(d)
    if a < 0.147:
        return "negligible"
    if a < 0.33:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"


def hodges_lehmann(x, y, rng):
    m = min(len(x), 4000)
    xs = rng.choice(x, size=m, replace=False)
    ys = rng.choice(y, size=min(m, len(y)), replace=False)
    return float(np.median((xs[:, None] - ys[None, :]).ravel()))


def desc(x, name):
    x = np.asarray(x, dtype=float)
    return {
        "group": name, "n": int(len(x)),
        "mean_days": float(x.mean()), "sd_days": float(x.std(ddof=1)),
        "var_days2": float(x.var(ddof=1)),
        "min_days": float(x.min()), "q1_days": float(np.quantile(x, .25)),
        "median_days": float(np.median(x)), "q3_days": float(np.quantile(x, .75)),
        "iqr_days": float(np.quantile(x, .75) - np.quantile(x, .25)),
        "p90_days": float(np.quantile(x, .90)), "p95_days": float(np.quantile(x, .95)),
        "max_days": float(x.max()),
    }


def bootstrap_median_diff(a, b, rng, iters):
    """Percentile bootstrap CI of median(B) - median(A), issue-level."""
    a = np.asarray(a); b = np.asarray(b)
    ds = np.empty(iters)
    for i in range(iters):
        ds[i] = (np.median(rng.choice(b, b.size, replace=True))
                 - np.median(rng.choice(a, a.size, replace=True)))
    return (float(np.quantile(ds, .025)), float(np.quantile(ds, .975)))


def bootstrap_median_diff_cluster(a_by_p, b_by_p, projects, rng, iters):
    """Cluster bootstrap: resample PROJECTS with replacement.

    a_by_p / b_by_p: dict project -> durations array (may be missing).
    Both groups' issues within a selected project are pooled.
    """
    pl_arr = list(projects)
    n = len(pl_arr)
    ds = np.empty(iters)
    for i in range(iters):
        idx = rng.integers(0, n, n)
        sel = [pl_arr[j] for j in idx]
        apool = np.concatenate([a_by_p[p] for p in sel if p in a_by_p]) \
            if any(p in a_by_p for p in sel) else np.array([])
        bpool = np.concatenate([b_by_p[p] for p in sel if p in b_by_p]) \
            if any(p in b_by_p for p in sel) else np.array([])
        if apool.size == 0 or bpool.size == 0:
            ds[i] = np.nan
            continue
        ds[i] = np.median(bpool) - np.median(apool)
    ds = ds[~np.isnan(ds)]
    return (float(np.quantile(ds, .025)), float(np.quantile(ds, .975)))


def main() -> None:
    cfg = common.load_config()
    log = common.get_logger("analyze")
    seed = int(cfg["analysis"]["seed"])
    iters = int(cfg["analysis"]["bootstrap_iterations"])
    rng = np.random.default_rng(seed)
    df = pl.read_parquet(
        common.out_path(cfg, "intermediate", "analysis_dataset.parquet"))
    res: dict = {"seed": seed, "bootstrap_iterations": iters}

    a = (df.filter(pl.col("group_matched") == "A_single")
         ["duration_days"].to_numpy())
    b = (df.filter(pl.col("group_matched") == "B_multi")
         ["duration_days"].to_numpy())
    c = (df.filter(pl.col("group_matched") == "C_zero")
         ["duration_days"].to_numpy())
    res["descriptives"] = [desc(a, "A_single"), desc(b, "B_multi"),
                           desc(c, "C_zero_excluded")]

    # ---- 1. primary Mann-Whitney --------------------------------------------
    u, p2 = stats.mannwhitneyu(b, a, alternative="two-sided")
    _, p1 = stats.mannwhitneyu(b, a, alternative="greater")
    d = cliffs_delta(b, a)
    hl = hodges_lehmann(b, a, np.random.default_rng(seed + 1))
    ci_issue = bootstrap_median_diff(a, b, np.random.default_rng(seed + 2), iters)

    ab = df.filter(pl.col("group_matched").is_in(["A_single", "B_multi"]))
    a_by_p = {p: g["duration_days"].to_numpy() for p, g in
              ab.filter(pl.col("group_matched") == "A_single").group_by("project")}
    b_by_p = {p: g["duration_days"].to_numpy() for p, g in
              ab.filter(pl.col("group_matched") == "B_multi").group_by("project")}
    projects = sorted(set(a_by_p) | set(b_by_p))
    ci_cluster = bootstrap_median_diff_cluster(
        a_by_p, b_by_p, projects, np.random.default_rng(seed + 3), iters)

    res["primary"] = {
        "n_A": int(a.size), "n_B": int(b.size),
        "median_A_days": float(np.median(a)), "median_B_days": float(np.median(b)),
        "median_diff_days": float(np.median(b) - np.median(a)),
        "mannwhitney_U": float(u), "p_two_sided": float(p2),
        "p_one_sided_B_greater": float(p1),
        "cliffs_delta": float(d), "cliffs_magnitude": cliffs_interp(d),
        "hodges_lehmann_shift_days": hl,
        "median_diff_ci95_issue_level": ci_issue,
        "median_diff_ci95_project_cluster": ci_cluster,
        "n_projects_in_AB": len(projects),
    }
    log.info("PRIMARY: median A=%.2fd B=%.2fd U=%.3g p=%.3g d=%.3f (%s) "
             "CIcluster=[%.1f,%.1f]", np.median(a), np.median(b), u, p2, d,
             cliffs_interp(d), ci_cluster[0], ci_cluster[1])

    # ---- 2. project-level ----------------------------------------------------
    both_projects = sorted(set(a_by_p) & set(b_by_p))
    per_proj = pd.DataFrame([{
        "project": p, "n_A": a_by_p[p].size, "n_B": b_by_p[p].size,
        "median_A_days": float(np.median(a_by_p[p])),
        "median_B_days": float(np.median(b_by_p[p])),
        "diff_days": float(np.median(b_by_p[p]) - np.median(a_by_p[p])),
    } for p in both_projects]).sort_values("n_B", ascending=False)
    per_proj.to_csv(common.out_path(cfg, "statistics", "project_comparison_table.csv"),
                    index=False)
    k_pos = int((per_proj["diff_days"] > 0).sum())
    n_proj_both = len(per_proj)
    sign_p = stats.binomtest(k_pos, n_proj_both, 0.5, alternative="two-sided").pvalue
    res["project_level"] = {
        "n_projects_both_groups": n_proj_both,
        "n_projects_B_higher": k_pos, "n_projects_B_lower": n_proj_both - k_pos,
        "sign_test_p": float(sign_p),
        "median_of_project_diffs_days": float(per_proj["diff_days"].median()),
    }
    # leave-most-contributing-out sensitivity (drop top-k projects by n_B)
    loo = {}
    ranked = per_proj.sort_values("n_B", ascending=False)["project"].tolist()
    for k in (1, 5, 10, 20):
        drop = set(ranked[:k])
        aa = np.concatenate([v for p, v in a_by_p.items() if p not in drop])
        bb = np.concatenate([v for p, v in b_by_p.items() if p not in drop])
        loo[f"drop_top{k}_by_nB"] = {
            "n_A": int(aa.size), "n_B": int(bb.size),
            "median_A": float(np.median(aa)), "median_B": float(np.median(bb)),
            "median_diff": float(np.median(bb) - np.median(aa)),
            "cliffs_delta": float(cliffs_delta(bb, aa)),
        }
    res["project_level"]["leave_projects_out"] = loo
    log.info("project-level: %d projects with both groups, B higher in %d "
             "(sign p=%.3g)", n_proj_both, k_pos, sign_p)

    # ---- 2b. fixed-effects regression ------------------------------------------
    model_df = ab.to_pandas().copy()
    model_df["log1p_dur"] = np.log1p(model_df["duration_days"])
    model_df["b_multi"] = (model_df["group_matched"] == "B_multi").astype(int)
    model_df["yr"] = model_df["created_year"].astype(str)
    import statsmodels.formula.api as smf
    ols = smf.ols("log1p_dur ~ b_multi + C(project) + C(yr)", data=model_df)
    fit = ols.fit(cov_type="cluster",
                  cov_kwds={"groups": model_df["project"]})
    coef = float(fit.params["b_multi"])
    lo, hi = (float(x) for x in fit.conf_int().loc["b_multi"])
    res["regression"] = {
        "n": int(fit.nobs), "n_projects": int(model_df["project"].nunique()),
        "coef_b_multi_log1p": coef,
        "se_cluster_project": float(fit.bse["b_multi"]),
        "ci95": [lo, hi], "p": float(fit.pvalues["b_multi"]),
        "multiplicative_interpretation_days_gt1": float(np.exp(coef) - 1),
        "note": "log1p scale; multiplicative reading is meaningful for "
                "durations >> 1 day",
    }
    log.info("FE regression: coef=%.4f (clustered CI %.4f..%.4f), ~%.0f%% "
             "multiplicative for long durations", coef, lo, hi,
             100 * (np.exp(coef) - 1))
    _stage = True

    # ---- 3. secondary: monotonicity / rank correlation ----------------------
    m = df.filter(pl.col("n_links_matched") >= 1)
    rho, rho_p = stats.spearmanr(m["n_links_matched"], m["duration_days"])
    rho_all, rho_all_p = stats.spearmanr(df["n_links_matched"],
                                         df["duration_days"])
    cats = m.with_columns(
        pl.when(pl.col("n_links_matched") >= 5)
        .then(pl.lit("5+")).otherwise(
            pl.col("n_links_matched").cast(pl.Utf8)).alias("kcat"))
    ktab = (cats.group_by("kcat").agg(
        pl.len().alias("n"),
        pl.col("duration_days").median().alias("median_days"),
        pl.col("duration_days").quantile(0.9).alias("p90_days"))
        .sort("kcat"))
    ktab.to_pandas().to_csv(
        common.out_path(cfg, "statistics", "pr_count_categories_table.csv"),
        index=False)
    groups = [g["duration_days"].to_numpy()
              for _, g in cats.filter(pl.col("kcat").is_in(
                  ["1", "2", "3", "4", "5+"])).group_by("kcat", maintain_order=True)]
    kh, kh_p = stats.kruskal(*groups)
    res["secondary"] = {
        "spearman_rho_AB": float(rho), "spearman_p_AB": float(rho_p),
        "spearman_rho_with_zero": float(rho_all),
        "spearman_p_with_zero": float(rho_all_p),
        "kruskal_H": float(kh), "kruskal_p": float(kh_p),
        "categories": ktab.to_dicts(),
    }

    # ---- 4. timing / exposure-bias investigation -----------------------------
    bb = df.filter(pl.col("group_matched") == "B_multi")
    aa = df.filter(pl.col("group_matched") == "A_single")
    bfrac_last = (bb["last_pr_days_after_issue"]
                  / bb["duration_days"]).drop_nulls().to_numpy()
    afrac = (aa["first_pr_days_after_issue"]
             / aa["duration_days"]).drop_nulls().to_numpy()
    bfrac_first = (bb["first_pr_days_after_issue"]
                   / bb["duration_days"]).drop_nulls().to_numpy()
    res["timing"] = {
        "B_last_PR_lifetime_fraction_median": float(np.median(bfrac_last)),
        "B_last_PR_after_90pct_share": float((bfrac_last > 0.9).mean()),
        "B_last_PR_after_75pct_share": float((bfrac_last > 0.75).mean()),
        "B_first_PR_lifetime_fraction_median": float(np.median(bfrac_first)),
        "A_PR_lifetime_fraction_median": float(np.median(afrac)),
    }
    # 'determined-exposure' re-test: ALL linked PRs created within the first
    # 50% of the issue's observed lifetime (multiplicity settled early).
    adf = df.filter(pl.col("group_matched") == "A_single")
    bdf = df.filter(pl.col("group_matched") == "B_multi")

    def determined(f):
        return f.filter(
            (pl.col("duration_days") > 0)
            & (pl.col("first_pr_days_after_issue") >= 0)
            & (pl.col("first_pr_days_after_issue")
               <= 0.5 * pl.col("duration_days"))
            & (pl.col("last_pr_days_after_issue")
               <= 0.5 * pl.col("duration_days"))
        )

    a_det = determined(adf)["duration_days"].to_numpy()
    b_det = determined(bdf)["duration_days"].to_numpy()
    u_d, p_d = stats.mannwhitneyu(b_det, a_det, alternative="two-sided")
    res["timing"]["determined_exposure_subset"] = {
        "definition": "issues with >0 duration whose every matched linked PR "
                      "was created within the first 50% of the issue lifetime",
        "n_A": int(a_det.size), "n_B": int(b_det.size),
        "median_A": float(np.median(a_det)), "median_B": float(np.median(b_det)),
        "median_diff": float(np.median(b_det) - np.median(a_det)),
        "mwu_U": float(u_d), "p": float(p_d),
        "cliffs_delta": float(cliffs_delta(b_det, a_det)),
    }
    log.info("timing: B last-PR at %.1f%% lifetime median; determined-exposure "
             "n_A=%d n_B=%d diff=%.1fd p=%.3g",
             100 * np.median(bfrac_last), a_det.size, b_det.size,
             np.median(b_det) - np.median(a_det), p_d)

    # ---- 5. exploratory resolution-path profile (Holm-adjusted) --------------
    fam = []
    for var in ["n_contrib", "median_pr_time_days", "n_commits",
                "n_pr_comments", "n_issue_comments"]:
        av = adf[var].drop_nulls().to_numpy().astype(float)
        bv = bdf[var].drop_nulls().to_numpy().astype(float)
        u_v, p_v = stats.mannwhitneyu(bv, av, alternative="two-sided")
        fam.append({"measure": var, "n_A": int(av.size), "n_B": int(bv.size),
                    "median_A": float(np.median(av)), "median_B": float(np.median(bv)),
                    "U": float(u_v), "p_raw": float(p_v),
                    "cliffs_delta": float(cliffs_delta(bv, av))})
    # merge incidence: share of issues with >=1 merged linked PR
    ar = adf.with_columns((pl.col("n_merged") >= 1).alias("has_merged"))
    br = bdf.with_columns((pl.col("n_merged") >= 1).alias("has_merged"))
    tab = np.array([[int(ar["has_merged"].sum()), int((~ar["has_merged"]).sum())],
                    [int(br["has_merged"].sum()), int((~br["has_merged"]).sum())]])
    chi2, p_chi, _, _ = stats.chi2_contingency(tab)
    merge = {"measure": "share_with_merged_PR",
             "rate_A": float(ar["has_merged"].mean()),
             "rate_B": float(br["has_merged"].mean()),
             "chi2": float(chi2), "p_raw": float(p_chi)}
    # Holm across the exploratory family
    ps = sorted([(e["p_raw"], i, e) for i, e in enumerate(fam + [merge])])
    mlen = len(ps)
    running = 0.0
    for rank, (_, _, e) in enumerate(ps):
        adj = min(1.0, (mlen - rank) * e["p_raw"])
        running = max(running, adj) if rank else adj
        e["p_holm"] = running if rank == 0 else max(adj, ps[rank - 1][2]["p_holm"])
    res["resolution_path_exploratory"] = {"tests": fam, "merge_rate": merge,
        "note": "exploratory; comment/commit counts are NOT effort measures"}

    # ---- 6. exploratory right-censored supplementary (KM + log-rank) --------
    snap = common.load_json(
        common.out_path(cfg, "statistics", "snapshot_estimate.json")
    )["snapshot_estimate_utc"]
    con = common.duckdb_conn(cfg)
    I = common.quote_glob(cfg, "issue")
    sup = con.execute(f"""
      WITH lab AS (SELECT project, number, state, unnest(labels) lbl,
                         try_cast(created_at AS TIMESTAMPTZ) c_ts,
                         try_cast(closed_at  AS TIMESTAMPTZ) x_ts
                  FROM read_parquet('{I}')
                  WHERE labels IS NOT NULL AND len(labels) > 0),
           map AS (SELECT label, decision FROM read_csv_auto(
                     '{str(common.out_path(cfg,"audit","bug_label_mapping.csv")).replace("'", "''")}')),
           f AS (SELECT lab.project, lab.number,
                 bool_or(map.decision='strict') AS is_strict,
                 max(lab.c_ts) AS created_ts, max(lab.x_ts) AS closed_ts,
                 max(lab.state) AS state
                 FROM lab JOIN map ON lab.lbl=map.label GROUP BY 1,2),
           l AS (SELECT DISTINCT project, issue_number, COUNT(*) OVER (PARTITION BY project, issue_number) AS k
                 FROM read_parquet('{common.quote_glob(cfg, "issue_pr")}'))
      SELECT f.state,
             date_diff('day', created_ts, coalesce(closed_ts,
                    try_cast('{snap}' AS TIMESTAMPTZ))) AS dur_days,
             CASE WHEN k=1 THEN 'A' ELSE 'B' END AS grp
      FROM f JOIN l ON l.project=f.project AND l.issue_number=f.number
      WHERE f.is_strict AND coalesce(closed_ts,
            try_cast('{snap}' AS TIMESTAMPTZ)) >= created_ts
    """).fetchdf()
    km = {}
    for g in ("A", "B"):
        s = sup[sup.grp == g]
        km[g] = {"n": int(len(s)), "n_open": int((s.state == "OPEN").sum()),
                 "median_obs_days": float(s.dur_days.median())}
    # log-rank test
    sup["event"] = (sup.state == "CLOSED").astype(int)
    sup = sup[sup.dur_days >= 0]
    t_events = np.sort(sup.loc[sup.event == 1, "dur_days"].unique())
    o_e, var_sum = 0.0, 0.0
    sarr = sup.to_numpy()
    dur_col, ev_col, grp_col = 0, 5, 4
    dfa = sup.sort_values("dur_days")
    for t in t_events:
        at_risk = dfa[dfa.dur_days >= t]
        n = len(at_risk)
        n_b = int((at_risk.grp == "B").sum())
        ev = dfa[(dfa.dur_days == t) & (dfa.event == 1)]
        d = len(ev)
        d_b = int((ev.grp == "B").sum())
        if n <= 1 or n_b == 0 or n_b == n:
            continue
        e_b = d * n_b / n
        v = d * (n_b / n) * (1 - n_b / n) * (n - d) / (n - 1) if n > 1 else 0
        o_e += d_b - e_b
        var_sum += v
    chi_lr = o_e ** 2 / var_sum if var_sum > 0 else 0.0
    res["censored_supplementary"] = {
        "snapshot_estimate": snap,
        "groups": km,
        "logrank_chi2": float(chi_lr),
        "logrank_p": float(stats.chi2.sf(chi_lr, 1)),
        "caveats": [
            "final link counts for still-open issues are snapshot-truncated "
            "(they can only grow), biasing group assignment",
            "issue-to-PR LINK addition times are not recorded; PR creation "
            "times only upper-bound nothing about when a link appeared",
            "excludes issues with zero links entirely",
            "EXPLORATORY: not part of confirmatory inference",
        ],
    }
    log.info("supplementary KM/log-rank: chi2=%.2f p=%.3g", chi_lr,
             stats.chi2.sf(chi_lr, 1))

    common.save_json(res, common.out_path(cfg, "statistics", "primary_results.json"))
    pd.DataFrame(res["descriptives"]).to_csv(
        common.out_path(cfg, "statistics", "descriptive_statistics_table.csv"),
        index=False)
    log.info("analyze complete")


if __name__ == "__main__":
    main()

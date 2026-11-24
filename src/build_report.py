"""Generate LaTeX numbers + tables from pipeline outputs and compile report.pdf.

numbers.tex defines one \newcommand macro per reported statistic, read
straight from outputs/statistics/*.json|csv, so the prose in
report_source/sections can never drift from the computed results.
Tables are emitted as booktabs files into report_source/tables/.

Compile step: tectonic report_source/main.tex -> report.pdf at project root.
Run:  .venv/bin/python -I src/build_report.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def fint(x):
    return f"{int(x):,}"


def f(x, d=2):
    return f"{x:,.{d}f}"


def sci(x, sig=2):
    """LaTeX-safe scientific notation, e.g. 3.31e-35 -> 3.3 x 10^{-35}."""
    from decimal import Decimal
    dv = Decimal(repr(float(x)))
    exp = dv.adjusted()
    mant = dv.scaleb(-exp)
    return rf"{mant:.{sig}f}\times 10^{{{exp}}}"


def pstr(x):
    """Compact p-value for prose: p<.001 or the exact third-decimal value."""
    return "$p<.001$" if x < 0.001 else f"$p={x:.3f}$"


def pexact(x):
    # float64 underflow: p computed as exactly 0.0 -> report an honest floor
    if x <= 0:
        return r"<10^{-300}"
    return f"{x:.3g}" if x >= 1e-4 else sci(x)


def interval(lo, hi, d=1):
    return f"[{f(lo,d)}, {f(hi,d)}] days"


def esc(s):
    return (str(s).replace("\\", r"\textbackslash{}")
            .replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
            .replace("#", r"\#").replace("$", r"\$")
            .replace("{", r"\{").replace("}", r"\}"))


def load_all():
    cfg = common.load_config(require_dataset=False)
    prim = common.load_json(
        common.out_path(cfg, "statistics", "primary_results.json"))
    excl = common.load_json(
        common.out_path(cfg, "statistics", "exclusion_counts.json"))
    snap = common.load_json(
        common.out_path(cfg, "statistics", "snapshot_estimate.json"))
    rob = pd.read_csv(
        common.out_path(cfg, "statistics", "robustness_table.csv"))
    cats = pd.read_csv(
        common.out_path(cfg, "statistics", "pr_count_categories_table.csv"))
    proj = pd.read_csv(
        common.out_path(cfg, "statistics", "project_comparison_table.csv"))
    proj["project"] = (proj["project"].astype(str)
                       .str.replace(r"^\('", "", regex=True)
                       .str.replace(r"',?\)$", "", regex=True))
    audit = common.load_json(
        common.out_path(cfg, "audit", "bug_label_mapping_summary.json"))
    inv = common.load_json(
        common.out_path(cfg, "inventory", "dataset_inventory.json"))
    lm = common.load_json(
        common.out_path(cfg, "statistics", "landmark_results.json"))
    va = common.load_json(
        common.out_path(cfg, "statistics", "validation_results.json"))
    df = __import__("polars").read_parquet(
        common.out_path(cfg, "intermediate", "analysis_dataset.parquet"))
    return cfg, prim, excl, snap, rob, cats, proj, audit, inv, lm, va, df


def pct(x, d=1):
    return rf"{100 * x:.{d}f}\%"


def write_numbers(prim, excl, snap, audit, proj, lm, va, df):
    d = {a["group"]: a for a in prim["descriptives"]}
    A, B = d["A_single"], d["B_multi"]
    C = d["C_zero_excluded"]
    pr = prim["primary"]
    pl_ = prim["project_level"]
    rg = prim["regression"]
    sc = prim["secondary"]
    tm = prim["timing"]
    det = tm["determined_exposure_subset"]
    km = prim["censored_supplementary"]

    M = {}
    M["nA"] = fint(A["n"]); M["nB"] = fint(B["n"]); M["nC"] = fint(C["n"])
    M["medA"] = f(A["median_days"], 1); M["medB"] = f(B["median_days"], 1)
    M["meanA"] = f(A["mean_days"], 1); M["meanB"] = f(B["mean_days"], 1)
    M["qOneA"] = f(A["q1_days"], 2); M["qThreeA"] = f(A["q3_days"], 1)
    M["qOneB"] = f(B["q1_days"], 2); M["qThreeB"] = f(B["q3_days"], 1)
    M["pNinetyA"] = f(A["p90_days"], 0); M["pNinetyB"] = f(B["p90_days"], 0)
    M["medDiff"] = f(pr["median_diff_days"], 1)
    M["Ustat"] = fint(pr["mannwhitney_U"]).replace(",", r"\,")
    M["pPrim"] = pexact(pr["p_two_sided"])
    M["pPrimOne"] = pexact(pr["p_one_sided_B_greater"])
    M["cliffsD"] = f(pr["cliffs_delta"], 3)
    M["cliffsMag"] = pr["cliffs_magnitude"]
    M["hlShift"] = f(pr["hodges_lehmann_shift_days"], 1)
    M["ciIssue"] = interval(*pr["median_diff_ci95_issue_level"])
    M["ciCluster"] = interval(*pr["median_diff_ci95_project_cluster"])
    M["nProjAB"] = fint(pr["n_projects_in_AB"])
    M["nProjBoth"] = fint(pl_["n_projects_both_groups"])
    M["nBhigher"] = fint(pl_["n_projects_B_higher"])
    M["nBlower"] = fint(pl_["n_projects_B_lower"])
    M["signP"] = pexact(pl_["sign_test_p"])
    M["medProjDiff"] = f(pl_["median_of_project_diffs_days"], 1)
    M["feCoef"] = f(rg["coef_b_multi_log1p"], 3)
    M["feCI"] = f"[{f(rg['ci95'][0], 3)}, {f(rg['ci95'][1], 3)}]"
    M["feP"] = pexact(rg["p"])
    M["fePct"] = f(100 * (rg["multiplicative_interpretation_days_gt1"]), 0)
    M["feN"] = fint(rg["n"]); M["feProj"] = fint(rg["n_projects"])
    M["rho"] = f(sc["spearman_rho_AB"], 3)
    M["rhoP"] = pexact(sc["spearman_p_AB"])
    M["krustH"] = f(sc["kruskal_H"], 1)
    M["krustP"] = pexact(sc["kruskal_p"])
    M["tMedLast"] = f(100 * tm["B_last_PR_lifetime_fraction_median"], 1)
    M["tNinety"] = f(100 * tm["B_last_PR_after_90pct_share"], 1)
    M["tSeventyFive"] = f(100 * tm["B_last_PR_after_75pct_share"], 1)
    M["tMedFirst"] = f(100 * tm["B_first_PR_lifetime_fraction_median"], 1)
    M["aFracMed"] = f(100 * tm["A_PR_lifetime_fraction_median"], 1)
    M["detNA"] = fint(det["n_A"]); M["detNB"] = fint(det["n_B"])
    M["detMedA"] = f(det["median_A"], 1); M["detMedB"] = f(det["median_B"], 1)
    M["detDiff"] = f(det["median_diff"], 1)
    M["detP"] = pexact(det["p"])
    M["detD"] = f(det["cliffs_delta"], 3)
    M["kmAn"] = fint(km["groups"]["A"]["n"])
    M["kmAopen"] = fint(km["groups"]["A"]["n_open"])
    M["kmAmed"] = f(km["groups"]["A"]["median_obs_days"], 0)
    M["kmBn"] = fint(km["groups"]["B"]["n"])
    M["kmBopen"] = fint(km["groups"]["B"]["n_open"])
    M["kmBmed"] = f(km["groups"]["B"]["median_obs_days"], 0)
    M["lrChi"] = f(km["logrank_chi2"], 1)
    M["lrP"] = pexact(km["logrank_p"])
    M["Szero"] = fint(excl["S0_total_issues"])
    M["Sone"] = fint(excl["S1_strict_bug"])
    M["Stwo"] = fint(excl["S2_closed_nonnull_closed_at"])
    M["Sfour"] = fint(excl["S4_analysis_cohort"])
    M["labelsTotal"] = fint(audit["distinct_labels_total"])
    M["labelsStrict"] = fint(audit["labels_strict"])
    M["labelsBroad"] = fint(audit["labels_broad_only"])
    M["labelsExcl"] = fint(audit["labels_excluded"])
    M["issBroad"] = fint(audit["issues_broad"])
    M["snapshot"] = esc(snap["snapshot_estimate_utc"])
    M["nProjCohort"] = fint(df["project"].n_unique())
    M["nLinksCohort"] = fint(df["n_links_matched"].sum())
    M["maxK"] = fint(df["n_links_matched"].max())
    M["nExclStrict"] = fint(excl["S1_strict_bug"] - excl["S2_closed_nonnull_closed_at"])

    # --- fixed-landmark analysis (primary landmark L = 30 days) ---
    lm30 = lm["by_landmark"]["30"]
    pv = {g["lg"]: g["count"] for g in lm30["group_sizes"]}
    om = lm30["primary_one_vs_multi"]
    zm = lm30["secondary_zero_vs_multi"]
    st = lm30["project_sign_test"]
    M["lmCohort"] = fint(lm30["n_ab_cohort"])
    M["lmAtRisk"] = fint(lm30["n_at_risk"])
    M["lmClosed"] = fint(lm30["n_closed_before_landmark"])
    M["lmNZero"] = fint(pv["L_zero"]); M["lmNOne"] = fint(pv["L_one"])
    M["lmNM"] = fint(pv["L_multi"])
    M["lmMedOne"] = f(om["median_ref_days"], 1)
    M["lmMedM"] = f(om["median_target_days"], 1)
    M["lmDiff"] = f(om["median_diff_days"], 1)
    M["lmP"] = pexact(om["mw_p"])
    M["lmD"] = f(om["cliffs_delta"], 3)
    M["lmCI"] = interval(*om["clustered_ci_median_diff_days"])
    M["lmMedZ"] = f(zm["median_ref_days"], 1)
    M["lmZP"] = pexact(zm["mw_p"])
    M["lmZD"] = f(zm["cliffs_delta"], 3)
    M["lmShareB"] = pct(lm30["original_B_share_multi_only_after_landmark"])
    M["lmShareA"] = pct(lm30["original_A_share_zero_at_landmark"])
    M["lmSignN"] = fint(st["n_projects_both_groups_at_landmark"])
    M["lmSignW"] = fint(st["projects_where_multi_longer"])
    M["lmSignP"] = pexact(st["binomial_p"])

    # --- label-validation audit (LLM-assisted, documented as such) ---
    M["vN"] = fint(va["n_sample"])
    M["vConf"] = fint(va["confirmed"]); M["vRej"] = fint(va["rejected"])
    M["vUnc"] = fint(va["uncertain"])
    M["vPrec"] = pct(va["precision_decided"])
    M["vCI"] = (rf"[{pct(va['wilson95_decided'][0])}, "
                rf"{pct(va['wilson95_decided'][1])}]")
    M["vPrecAll"] = pct(va["precision_incl_uncertain"])
    M["vCIAll"] = (rf"[{pct(va['wilson95_incl_uncertain'][0])}, "
                   rf"{pct(va['wilson95_incl_uncertain'][1])}]")

    tex = ["% GENERATED by src/build_report.py -- do not edit by hand."]
    for k, v in M.items():
        tex.append(rf"\newcommand{{\val{k}}}{{{v}}}")
    path = common.project_path("report_source", "numbers.tex")
    path.write_text("\n".join(tex) + "\n", encoding="utf-8")
    return path, M


ROB_NICE = {
    "baseline_strict_primary": "Baseline (strict label, matched links)",
    "R1_broad_bug_definition": "R1 broader bug definition",
    "R2_grouping_on_all_references": "R2 group on all link references",
    "R3_at_least_one_merged_PR": "R3 at least one merged linked PR",
    "R4_continuous_multiplicity": "R4 PR count continuous (Spearman)",
    "R5_projects_with_ge5_each_group": "R5 projects with $\\geq$5 per group",
    "R6_winsorized_at_180_days": "R6 winsorised at 180 days",
    "R6_winsorized_at_365_days": "R6 winsorised at 365 days",
    "R6b_drop_durations_under_1h": "R6b drop durations $<$1 hour",
    "R7_drop_top_project_per_group": "R7 drop largest project per group",
}
EXP_NICE = {
    "n_contrib": "Distinct linked-PR authors",
    "median_pr_time_days": "Median per-PR duration (days)",
    "n_commits": "Commits via linked PRs",
    "n_pr_comments": "Comments on linked PRs",
    "n_issue_comments": "Comments on the issue",
}


def book(rows, caption, label, colhead, notes=None, small=True, resize=None):
    size = r"\small " if small else ""
    if resize:
        head = [size + rf"\resizebox{{{resize}}}{{!}}{{%",
                r"\begin{tabular}{@{}" + colhead + r"@{}}"]
        tail = [r"\end{tabular}", r"}"]
    else:
        head = [size + r"\begin{tabular}{@{}" + colhead + r"@{}}"]
        tail = [r"\end{tabular}"]
    out = [r"\begin{table}[t]", r"\centering",
           rf"\caption{{{caption}\label{{tab:{label}}}}}"] + head + [r"\toprule"]
    out.extend(rows)
    out.append(r"\bottomrule")
    out.extend(tail)
    for nt in (notes or []):
        out.append(rf"\par\vspace{{2pt}}{{\footnotesize\textit{{Note: {nt}}}}}")
    out.append(r"\end{table}")
    return "\n".join(out) + "\n"


def tab_descriptives(prim):
    rows = [r"Group & $n$ & Mean & SD & Q1 & Median & Q3 & P90 & P95 & Max \\"
            r"\midrule"]
    for d in prim["descriptives"]:
        name = {"A_single": "A (1 PR)", "B_multi": "B ($\\geq$2 PRs)",
                "C_zero_excluded": "C (0 PRs, not analysed)"}[d["group"]]
        rows.append(" & ".join([
            name, fint(d["n"]), f(d["mean_days"], 1), f(d["sd_days"], 1),
            f(d["q1_days"], 2), rf"\textbf{{{f(d['median_days'], 1)}}}",
            f(d["q3_days"], 1), f(d["p90_days"], 0), f(d["p95_days"], 0),
            f(d["max_days"], 0)]) + r"\\")
    return book(rows, "Observed issue-to-closure duration (days) by group.",
                "descriptives", "llrrrrrrrrr")


def tab_stages(excl):
    rows = [r"Stage & Issues & Retained \\" + r"\midrule"]
    steps = [("S0", "All issues in issue.parquet", "S0_total_issues"),
             ("S1", "Carry a strict bug label", "S1_strict_bug"),
             ("S2", "State CLOSED with non-null closed\\_at",
              "S2_closed_nonnull_closed_at"),
             ("S3", "Timestamps parse, duration $\\geq$ 0",
              "S3_valid_nonneg_duration"),
             ("S4", "Analysis cohort", "S4_analysis_cohort")]
    prev = None
    for code, what, key in steps:
        v = excl[key]
        rows.append(rf"{code} & {fint(v)} & "
                    + (f"{f(100.0*v/excl['S0_total_issues'],1)}\\% of S0"
                       if prev is None or code != "S3" else "no change")
                    + r"\\")
        prev = v
    return book(rows, "Cohort construction and filtering stages for closed strict bug issues.",
                "stages", "lrl", small=True)


def tab_inventory(inv):
    er = inv["entity_rows"]
    items = [
        ("Projects (project.parquet)", fint(inv["projects_in_project_table"])),
        ("Issues", fint(er["issue"]) + " (unique composite keys)"),
        ("Issue comments", fint(er["issue_comment"])),
        ("Pull requests", fint(er["pr"])),
        ("PR comments", fint(er["pr_comment"])),
        ("issue\\_pr link rows", f"{fint(er['issue_pr'])} raw / "
         f"{fint(inv['issue_pr_unique_links'])} unique "
         f"({fint(inv['issue_pr_duplicate_rows'])} exact duplicates)"),
        ("pr\\_commit rows", fint(er["pr_commit"])),
        ("Commits in per-project detail files",
         f"{fint(inv['total_commits'])} across "
         f"{fint(inv['commit_detail_files'])} files"),
        ("Distinct issue labels", fint(inv["distinct_issue_labels"])),
        ("Issues with no labels", fint(inv["issues_with_no_labels"])),
        ("Issue creation range", "2000-12-12 to 2025-10-12"),
        ("PR merge range", "2010-10-16 to 2025-10-12"),
        ("Links not resolving to issue.parquet",
         fint(inv["links_unmatched_to_issue_table"])),
        ("Links not resolving to pr.parquet",
         fint(inv["links_unmatched_to_pr_table"])),
        ("Malformed timestamps (all 5 datetime columns)", "0"),
        ("PRs with closed\\_at $<$ created\\_at",
         fint(inv["pr_negative_closed_at_before_created"])),
        ("Comment tables expose created\\_at?", "no (see \\S\\ref{sec:threats})"),
    ]
    rows = ["Item & Value \\\\" + r"\midrule"]
    rows += [f"{a} & {b} \\\\" for a, b in items]
    return book(rows, "Selected inventory and data-quality findings for the OSSGameBench dataset (Version v2).",
                "inventory", "ll", small=True)


def tab_categories(cats):
    order = ["1", "2", "3", "4", "5+"]
    c = cats.set_index("kcat").loc[order].reset_index()
    rows = [r"Linked PRs & $n$ & Median duration (d) & P90 (d) \\" +
            r"\midrule"]
    for _, r_ in c.iterrows():
        rows.append(f"{r_['kcat']} & {fint(r_['n'])} & "
                    f"{f(r_['median_days'], 1)} & {f(r_['p90_days'], 0)} \\\\")
    return book(rows, "Median duration by number of matched linked PRs.",
                "categories", "lrrr",
                notes=[r"categories 4 and 5+ contain $n\leq 2$ issues"])


def tab_projects(proj, top=None, long=False):
    rows = []
    rows.append(r"Project & $n_A$ & $n_B$ & Med.\ A (d) & Med.\ B (d) & "
                r"Diff (d) \\")
    body = proj if top is None else proj.head(top)
    if long:
        env = [r"\begin{longtable}{@{}lrrrrr@{}}", r"\toprule",
               *rows, r"\midrule",
               *[f"{esc(r_['project'])} & {fint(r_['n_A'])} & "
                 f"{fint(r_['n_B'])} & {f(r_['median_A_days'], 1)} & "
                 f"{f(r_['median_B_days'], 1)} & {f(r_['diff_days'], 1)} \\\\"
                 for _, r_ in body.iterrows()],
               r"\bottomrule", r"\end{longtable}"]
        return ("\n".join(
            [r"{\small", *env, "}"]) + "\n")
    out = rows + [
        f"{esc(r_['project'])} & {fint(r_['n_A'])} & {fint(r_['n_B'])} & "
        f"{f(r_['median_A_days'], 1)} & {f(r_['median_B_days'], 1)} & "
        f"{f(r_['diff_days'], 1)} \\\\" for _, r_ in body.iterrows()]
    cap = ("Projects with the most Group-B issues "
           f"(top {len(body)} of {len(proj)}; the complete list of "
           f"{len(proj)} projects ships with the replication package "
           r"as \texttt{tables/tab\_projects\_all.tex}).")
    return book(out, cap, "projects-top", "lrrrrr", small=True)


def tab_projects_all(proj):
    header = [r"\begin{center}", r"{\footnotesize",
              r"\begin{longtable}{@{}lrrrrr@{}}",
              r"\caption{Complete list of the \valnProjBoth{} projects "
              r"contributing both groups, with within-project medians "
              r"(days).}\label{tab:projects-all}\\",
              r"\toprule",
              r"Project & $n_A$ & $n_B$ & Med.\ A & Med.\ B & Diff (d) \\",
              r"\midrule", r"\endfirsthead",
              r"\toprule",
              r"Project & $n_A$ & $n_B$ & Med.\ A & Med.\ B & Diff (d) \\",
              r"\midrule", r"\endhead"]
    body = [f"{esc(r_['project'])} & {fint(r_['n_A'])} & {fint(r_['n_B'])} & "
            f"{f(r_['median_A_days'], 1)} & {f(r_['median_B_days'], 1)} & "
            f"{f(r_['diff_days'], 1)} \\\\" for _, r_ in proj.iterrows()]
    return "\n".join(header + body +
                     [r"\bottomrule", r"\end{longtable}", "}",
                      r"\end{center}"]) + "\n"


def tab_robustness(rob):
    rows = [r"Check & $n_A$ & $n_B$ & Med.\ A & Med.\ B & Diff & $p$ & "
            r"Cliff's $\delta$ & Same dir.? \\" + r"\midrule"]
    for _, r_ in rob.iterrows():
        name = ROB_NICE.get(r_["check"], r_["check"])
        if pd.isna(r_["median_diff_days"]):
            rows.append(
                name + r" & \multicolumn{4}{l}{" + fint(r_["n"])
                + r" issues with $\geq 1$ link} & Spearman "
                + rf"$\rho$={f(r_['spearman_rho'], 3)}"
                + rf" & ${pexact(r_['p'])}$ & -- & "
                + ("yes" if r_["direction_preserved"] else "no") + r" \\")
        else:
            rows.append(
                rf"{name} & {fint(r_['n_A'])} & {fint(r_['n_B'])} & "
                rf"{f(r_['median_A_days'], 1)} & {f(r_['median_B_days'], 1)}"
                rf" & {f(r_['median_diff_days'], 1)} & ${pexact(r_['p'])}$ & "
                rf"{f(r_['cliffs_delta'], 3)} & "
                rf"{'yes' if r_['direction_preserved'] else 'no'} \\")
    return book(rows, "Robustness checks R1--R7 for the primary "
                     "comparison (all tests Mann--Whitney $U$ unless noted).",
                "robustness", "lrrrrrccc", resize=r"\textwidth",
                notes=[r"R2 is identical to the baseline by construction:",
                       r"the cohort contains zero unmatched link references."])


def tab_exploratory(prim):
    rows = [r"Measure (B vs.\ A) & Med.\ A & Med.\ B & Cliff's $\delta$ & "
            r"$p_{\text{raw}}$ & $p_{\text{Holm}}$ \\" + r"\midrule"]
    for t in prim["resolution_path_exploratory"]["tests"]:
        rows.append(rf"{EXP_NICE[t['measure']]} & "
                    rf"{f(t['median_A'], 2)} & {f(t['median_B'], 2)} & "
                    rf"{f(t['cliffs_delta'], 3)} & ${pexact(t['p_raw'])}$ & "
                    rf"${pexact(t['p_holm'])}$ \\")
    m = prim["resolution_path_exploratory"]["merge_rate"]
    rows.append(rf"Issue has $\geq$1 merged linked PR (\%) & "
                rf"{f(100*m['rate_A'], 1)} & {f(100*m['rate_B'], 1)} & -- & "
                rf"${pexact(m['p_raw'])}$ & ${pexact(m['p_holm'])}$ \\")
    return book(rows, "Exploratory resolution-path profile by group "
                     "(Holm-adjusted within this family). Comment and "
                     "commit counts are activity proxies, not effort "
                     "measures.",
                "exploratory", "llllll", small=True)


def tab_datadic():
    dd = pd.read_csv(Path(common.PROJECT_ROOT) / "outputs" / "tables" /
                     "data_dictionary.csv")
    rows = ["Field & Type & Definition \\" + r"\midrule"]
    rows += [rf"\texttt{{{esc(r_['field'])}}} & {esc(r_['type'])} & "
             rf"{esc(r_['definition'])} \\" for _, r_ in dd.iterrows()]
    return book(rows, "Data dictionary of the analysis dataset "
                     r"(outputs/tables/data\_dictionary.csv).",
                "datadic", "llp{0.46\\textwidth}", small=True)


def tab_landmark(lm):
    rows = [r"Landmark & Still open & Closed before & $n_0$ & $n_1$ & "
            r"$n_{\geq 2}$ & Med.\ 1-PR & Med.\ $\geq$2-PR & Diff. & "
            r"$\delta$ & $p$ & 95\% CI (days) \\" + r"\midrule"]
    for L in ["7", "30", "90"]:
        e = lm["by_landmark"][L]
        pv = {g["lg"]: g["count"] for g in e["group_sizes"]}
        om = e["primary_one_vs_multi"]
        ci = om["clustered_ci_median_diff_days"]
        rows.append(
            rf"{L} d & {fint(e['n_at_risk'])} & "
            rf"{fint(e['n_closed_before_landmark'])} & "
            rf"{pv['L_zero']} & {pv['L_one']} & {pv['L_multi']} & "
            rf"{f(om['median_ref_days'], 1)} & "
            rf"{f(om['median_target_days'], 1)} & "
            rf"{f(om['median_diff_days'], 1)} & "
            rf"{f(om['cliffs_delta'], 3)} & ${pexact(om['mw_p'])}$ & "
            rf"[{f(ci[0], 1)}, {f(ci[1], 1)}] \\")
    return book(rows,
                "Fixed-landmark analysis: among issues still open $L$ days "
                "after creation, additional days from the landmark to "
                "closure, by number of linked PRs created by day $L$ "
                "(Mann--Whitney $U$; project-clustered bootstrap CI).",
                "landmark", "lrrrrrrrrrrc", resize=r"\textwidth",
                notes=[r"$n_0/n_1/n_{\geq 2}$: issues with 0, 1, "
                       r"$\geq$2 linked PRs created by day $L$; the "
                       r"comparison column pair uses only the 1-PR and "
                       r"$\geq$2-PR groups."])


def tab_validation(va):
    rows = [r"Outcome & $n$ (of " + fint(va["n_sample"]) + r") \\"
            + r"\midrule",
            rf"Confirmed as a genuine bug report & {fint(va['confirmed'])} \\",
            rf"Rejected (not a bug report) & {fint(va['rejected'])} \\",
            rf"Uncertain & {fint(va['uncertain'])} \\",
            r"\midrule",
            rf"Precision, decided cases only & \textbf{{{pct(va['precision_decided'])}}} "
            rf"(95\% Wilson CI {pct(va['wilson95_decided'][0])}--"
            rf"{pct(va['wilson95_decided'][1])}) \\",
            rf"Precision, uncertain counted as errors & "
            rf"{pct(va['precision_incl_uncertain'])} "
            rf"(CI {pct(va['wilson95_incl_uncertain'][0])}--"
            rf"{pct(va['wilson95_incl_uncertain'][1])}) \\"]
    return book(rows,
                "Label-validation audit of the strict-bug classification "
                "(seeded random sample of " + fint(va["n_sample"]) +
                " cohort issues; LLM-assisted single-reviewer screening, "
                "not independent double-human validation).",
                "validation", "lp{0.52\\textwidth}", small=True)


def tab_main(prim):
    pr = prim["primary"]
    pl_ = prim["project_level"]
    rg = prim["regression"]
    rows = [r"Quantity & Value \\" + r"\midrule",
            rf"Samples (Group A / Group B) & {fint(pr['n_A'])} / "
            rf"{fint(pr['n_B'])} issues \\",
            rf"Median days to closure (A / B) & {f(pr['median_A_days'], 1)} "
            rf"/ {f(pr['median_B_days'], 1)} \\",
            rf"Median difference (B $-$ A) & \textbf{{{f(pr['median_diff_days'], 1)}}}~days \\",
            rf"Hodges--Lehmann shift & {f(pr['hodges_lehmann_shift_days'], 1)}~days \\",
            rf"Mann--Whitney $U$ (two-sided $p$) & "
            rf"{fint(pr['mannwhitney_U']).replace(',', r'\,')} "
            rf"($p={pexact(pr['p_two_sided'])}$) \\",
            rf"Cliff's $\delta$ (magnitude) & {f(pr['cliffs_delta'], 3)} "
            rf"({esc(pr['cliffs_magnitude'])}) \\",
            rf"95\% CI for median difference, resampling issues & "
            rf"{interval(*pr['median_diff_ci95_issue_level'])} \\",
            rf"95\% CI for median difference, resampling projects & "
            rf"{interval(*pr['median_diff_ci95_project_cluster'])} \\",
            rf"Within-project sign test & B slower in "
            rf"{pl_['n_projects_B_higher']}/{pl_['n_projects_both_groups']} "
            rf"projects ($p={pexact(pl_['sign_test_p'])}$) \\",
            rf"Regression $\log(1+\text{{days}})$, project $\times$ year "
            rf"fixed effects & $\hat\beta={f(rg['coef_b_multi_log1p'], 3)}$, "
            rf"CI [{f(rg['ci95'][0], 2)}, {f(rg['ci95'][1], 2)}], "
            rf"$p={pexact(rg['p'])}$ \\"]
    return book(rows, "Primary comparison: single-PR (A) vs.\\ multi-PR (B) "
                     "closed strict bugs. Both confidence intervals use "
                     "10{,}000 bootstrap draws (seed 20261120); the "
                     "project-level one is wider and is the honest reading.",
                "main", "ll", small=True)


def write_tables(prim, excl, rob, cats, proj, inv, lm, va):
    tdir = Path(common.PROJECT_ROOT) / "report_source" / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "tab_main.tex").write_text(tab_main(prim))
    (tdir / "tab_descriptives.tex").write_text(tab_descriptives(prim))
    (tdir / "tab_stages.tex").write_text(tab_stages(excl))
    (tdir / "tab_inventory.tex").write_text(tab_inventory(inv))
    (tdir / "tab_categories.tex").write_text(tab_categories(cats))
    (tdir / "tab_projects_top.tex").write_text(tab_projects(proj, top=15))
    (tdir / "tab_projects_all.tex").write_text(tab_projects_all(proj))
    (tdir / "tab_robustness.tex").write_text(tab_robustness(rob))
    (tdir / "tab_exploratory.tex").write_text(tab_exploratory(prim))
    (tdir / "tab_datadic.tex").write_text(tab_datadic())
    (tdir / "tab_landmark.tex").write_text(tab_landmark(lm))
    (tdir / "tab_validation.tex").write_text(tab_validation(va))
    return tdir


def compile_report():
    src = Path(common.PROJECT_ROOT) / "report_source"
    subprocess.run(["tectonic", "main.tex"], cwd=src, check=True)
    out = Path(common.PROJECT_ROOT) / "report.pdf"
    (src / "main.pdf").replace(out)
    return out


def main() -> None:
    log = common.get_logger("build_report")
    (cfg, prim, excl, snap, rob, cats, proj, audit, inv, lm, va,
     df) = load_all()
    path, M = write_numbers(prim, excl, snap, audit, proj, lm, va, df)
    log.info("wrote %s (%d macros)", path, len(M))
    tdir = write_tables(prim, excl, rob, cats, proj, inv, lm, va)
    log.info("wrote tables to %s", tdir)
    log.info("wrote %s", compile_report())


if __name__ == "__main__":
    main()

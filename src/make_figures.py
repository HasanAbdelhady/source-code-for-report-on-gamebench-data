"""Figures for the OSSGameBench v2 bug-resolution report.

Palette: validated 2-slot categorical (blue #2a78d6 = Group A / single PR,
orange #eb6834 = Group B / multi PR); sequential blue ordinal ramp for the
PR-count categories; chart chrome per the data-viz method (recessive grid
#e1e0d9, baseline #c3c2b7, ink #0b0b0b / secondary #52514e, muted axes
#898781). Light surface only (the report is printed on paper white).

Outputs (identical set in both locations, PNG for review, PDF for LaTeX):
  outputs/figures/fig*.png   and   report_source/figures/fig*.pdf

Run:  python -I src/make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

# ---- validated palette (six-checks validator, light mode: ALL PASS) ----
BLUE = "#2a78d6"      # series 1: Group A (single linked PR)
ORANGE = "#eb6834"    # series 2: Group B (multiple linked PRs)
INK = "#0b0b0b"
SEC = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
GRAY = "#c3c2b7"
# ordinal blue ramp (validated sequential steps, light->dark, >= step 250)
ORD = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelcolor": SEC,
    "axes.edgecolor": BASE,
    "axes.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})


def save(fig, stem):
    png = common.project_path("outputs", "figures", f"{stem}.png")
    pdf = common.project_path("report_source", "figures", f"{stem}.pdf")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return png


def load_all():
    cfg = common.load_config(require_dataset=False)
    df = pl.read_parquet(
        common.out_path(cfg, "intermediate", "analysis_dataset.parquet"))
    prim = common.load_json(
        common.out_path(cfg, "statistics", "primary_results.json"))
    excl = common.load_json(
        common.out_path(cfg, "statistics", "exclusion_counts.json"))
    rob = pd.read_csv(
        common.out_path(cfg, "statistics", "robustness_table.csv"))
    cats = pd.read_csv(
        common.out_path(cfg, "statistics", "pr_count_categories_table.csv"))
    proj = pd.read_csv(
        common.out_path(cfg, "statistics", "project_comparison_table.csv"))
    proj["project"] = (proj["project"].astype(str)
                       .str.replace(r"^\('", "", regex=True)
                       .str.replace(r"',?\)$", "", regex=True))
    return cfg, df, prim, excl, rob, cats, proj


def fig1_funnel(excl, df):
    """S0->S4 cohort funnel and the A/B/C split inside the cohort."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.6, 2.5),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    stages = ["S0 all issues", "S1 strict bug label",
              "S2 CLOSED + closed_at", "S3-S4 valid duration"]
    vals = [excl["S0_total_issues"], excl["S1_strict_bug"],
            excl["S2_closed_nonnull_closed_at"], excl["S4_analysis_cohort"]]
    ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab"]
    ypos = np.arange(len(stages))[::-1]
    axa.barh(ypos, vals, height=0.62, color=ramp, zorder=3)
    for y, v in zip(ypos, vals):
        axa.text(v * 1.06, y, f"{v:,}", va="center", fontsize=8, color=INK)
    axa.set_yticks(ypos, stages)
    axa.set_xscale("log")
    axa.set_xlim(4e4, 9e5)
    axa.set_xlabel("issues (log scale)")
    axa.set_title("(a) cohort funnel (exclusion stages)", loc="left")
    axa.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    axa.set_axisbelow(True)
    axa.spines[["top", "right", "left"]].set_visible(False)

    gc = df["group_matched"].value_counts()
    counts = dict(zip(gc["group_matched"], gc["count"]))
    names = ["A_single", "B_multi", "C_zero"]
    labels = ["A\nsingle PR", "B\nmulti PR", "C\nno linked PR\n(not analysed)"]
    colors = [BLUE, ORANGE, GRAY]
    vals2 = [int(counts[n]) for n in names]
    xp = np.arange(3)
    axb.bar(xp, vals2, width=0.62, color=colors, zorder=3)
    for x, v in zip(xp, vals2):
        axb.text(x, v * 1.06, f"{v:,}", ha="center", fontsize=8, color=INK)
    axb.set_xticks(xp, labels)
    axb.set_yscale("log")
    axb.set_ylim(3e2, 1.2e5)
    axb.set_ylabel("issues (log scale)")
    axb.set_title("(b) group split inside cohort", loc="left")
    axb.set_axisbelow(True)
    axb.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    return save(fig, "fig1_cohort")


def ecdf(x):
    x = np.sort(np.asarray(x, float))
    return x, (np.arange(1, x.size + 1)) / x.size


def fig2_ecdf(df):
    a = df.filter(pl.col("group_matched") == "A_single")["duration_days"].to_numpy()
    b = df.filter(pl.col("group_matched") == "B_multi")["duration_days"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    for x, col, name, med in ((a, BLUE, "A single PR", np.median(a)),
                              (b, ORANGE, "B multi PR", np.median(b))):
        xs, ys = ecdf(x)
        ax.plot(np.log2(xs), ys, lw=2, color=col,
                label=f"{name} (n={x.size:,})")
        ax.axvline(np.log2(med), color=col, lw=1, ls=(0, (4, 3)), alpha=0.7)
        ax.annotate(f"median {med:,.1f} d", xy=(np.log2(med), 0.955),
                    xytext=(6 if col == ORANGE else -6, 0),
                    textcoords="offset points", color=col, fontsize=8,
                    ha="left" if col == ORANGE else "right", va="top")
    ticks = [2.0 ** k for k in range(-4, 13, 2)]
    ax.set_xticks(np.log2(ticks),
                  [f"{t:g}" if t >= 1 else f"{t:.2f}" for t in ticks])
    ax.set_xlim(-5, 13)
    ax.set_ylim(0, 1)
    ax.set_xlabel("observed issue-to-closure duration (days, log$_2$ scale)")
    ax.set_ylabel("ECDF")
    ax.legend(loc="lower right")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return save(fig, "fig2_ecdf")


def fig3_categories(cats):
    """Median duration by linked-PR count category (ordinal blue ramp)."""
    order = ["1", "2", "3", "4", "5+"]
    cats = cats.set_index("kcat").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    xp = np.arange(len(cats))
    ax.bar(xp, cats["median_days"], width=0.62, color=ORD[: len(cats)],
           zorder=3)
    for x, m, n in zip(xp, cats["median_days"], cats["n"]):
        ax.text(x, m * 1.25, f"{m:,.1f} d", ha="center", fontsize=8,
                color=INK)
    ax.set_xticks(xp, [f"{k}\nn={n:,}" for k, n in
                       zip(cats["kcat"], cats["n"])])
    ax.set_yscale("log")
    ax.set_ylim(4, 6e3)
    ax.set_ylabel("median duration (days, log scale)")
    ax.set_xlabel("number of matched linked PRs")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    return save(fig, "fig3_kcategory")


def fig4_projects(proj):
    """Per-project medians: issues where both groups are present."""
    x = np.maximum(proj["median_A_days"], 1e-4)
    y = np.maximum(proj["median_B_days"], 1e-4)
    hi = proj["diff_days"] > 0
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.scatter(np.log10(x[~hi]), np.log10(y[~hi]), s=16, color=BLUE,
               edgecolor="white", linewidth=0.6, zorder=3,
               label=f"B lower ({int((~hi).sum())})")
    ax.scatter(np.log10(x[hi]), np.log10(y[hi]), s=16, color=ORANGE,
               edgecolor="white", linewidth=0.6, zorder=4,
               label=f"B higher ({int(hi.sum())})")
    lim = [-4, 4]
    ax.plot(lim, lim, lw=1, color=MUTED, ls=(0, (4, 3)))
    ax.text(-3.9, 3.55, "B = A\n(equal medians)", fontsize=7.5, color=MUTED)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xticks(range(-4, 5), [f"10$^{k}$" for k in range(-4, 5)])
    ax.set_yticks(range(-4, 5), [f"10$^{k}$" for k in range(-4, 5)])
    ax.set_xlabel("project median duration, Group A (days)")
    ax.set_ylabel("project median duration, Group B (days)")
    ax.legend(loc="lower right")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return save(fig, "fig4_projects")


def fig5_timing(df):
    """Where linked-PR creation sits inside the observed issue lifetime."""
    a = df.filter(pl.col("group_matched") == "A_single")
    b = df.filter(pl.col("group_matched") == "B_multi")
    fa = (a["first_pr_days_after_issue"] / a["duration_days"]).drop_nulls()
    fb = (b["last_pr_days_after_issue"] / b["duration_days"]).drop_nulls()
    fa = fa.to_numpy().astype(float)
    fb = fb.to_numpy().astype(float)
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    bins = np.linspace(0, 2, 41)
    ax.hist(fa, bins=bins, density=True, color=BLUE, alpha=0.65,
            label=f"A: single linked PR (n={fa.size:,})", zorder=3)
    ax.hist(fb, bins=bins, density=True, color=ORANGE, alpha=0.85,
            label=f"B: last of multiple linked PRs (n={fb.size:,})",
            zorder=4)
    for arr, col, med_y in ((fa, BLUE, 9.9), (fb, ORANGE, 10.8)):
        m = np.median(arr)
        ax.axvline(m, color=col, lw=1.4, ls=(0, (4, 3)))
        ax.text(m + 0.025, med_y, f"median {m:.2f}", color=col,
                fontsize=8, ha="left", va="bottom")
    ax.set_ylim(0, 11.6)
    ax.set_xlim(0, 2)
    ax.set_xticks([0, 0.5, 1, 1.5, 2], ["0", "0.5", "1.0\n(closure)", "1.5",
                                        "2.0"])
    off_a = (float((fa < 0).mean()) + float((fa > 2).mean())) * 100
    off_b = (float((fb < 0).mean()) + float((fb > 2).mean())) * 100
    ax.set_xlabel("linked-PR creation time as a fraction of observed issue "
                  "lifetime (0 = issue created, 1 = closed; off-scale: "
                  f"A {off_a:.1f}%, B {off_b:.1f}%)")
    ax.set_ylabel("density")
    ax.legend(loc="upper left")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return save(fig, "fig5_timing")


def fig6_robustness(rob):
    """Median difference B - A (days) across robustness scenarios."""
    r = rob.dropna(subset=["median_diff_days"]).copy()
    nice = {
        "baseline_strict_primary": "Baseline (strict, matched links)",
        "R1_broad_bug_definition": "R1 broad bug definition",
        "R2_grouping_on_all_references": "R2 all link references",
        "R3_at_least_one_merged_PR": "R3 at least one merged PR",
        "R5_projects_with_ge5_each_group": "R5 projects with >=5 per group",
        "R6_winsorized_at_180_days": "R6 winsorised at 180 d",
        "R6_winsorized_at_365_days": "R6 winsorised at 365 d",
        "R6b_drop_durations_under_1h": "R6b drop durations < 1 h",
        "R7_drop_top_project_per_group": "R7 drop largest project per group",
    }
    r["label"] = r["check"].map(nice)
    r = r.dropna(subset=["label"]).set_index("check").loc[list(nice)].reset_index()
    base = float(r.loc[r["check"] == "baseline_strict_primary",
                       "median_diff_days"].iloc[0])
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    yp = np.arange(len(r))[::-1]
    ax.barh(yp, r["median_diff_days"], height=0.62, color=BLUE, zorder=3)
    for y, d, na, nb in zip(yp, r["median_diff_days"], r["n_A"], r["n_B"]):
        ax.text(d + 0.5, y, f"{d:.1f}  (A {na:,.0f} / B {nb:,.0f})",
                va="center", fontsize=7.5, color=INK)
    ax.axvline(base, color=BASE, lw=1.2, ls=(0, (4, 3)))
    ax.text(base + 0.5, len(r) - 0.2, f"baseline {base:.1f} d", color=MUTED,
            fontsize=7.5, ha="left", va="bottom")
    ax.set_yticks(yp, r["label"])
    ax.set_xlim(0, 48)
    ax.set_xlabel("median difference B - A (days)")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    return save(fig, "fig6_robustness")


def main() -> None:
    cfg = common.load_config(require_dataset=False)
    log = common.get_logger("make_figures")
    _, df, prim, excl, rob, cats, proj = load_all()
    log.info("wrote %s", fig1_funnel(excl, df))
    log.info("wrote %s", fig2_ecdf(df))
    log.info("wrote %s", fig3_categories(cats))
    log.info("wrote %s", fig4_projects(proj))
    log.info("wrote %s", fig5_timing(df))
    log.info("wrote %s", fig6_robustness(rob))
    log.info("make_figures complete")


if __name__ == "__main__":
    main()

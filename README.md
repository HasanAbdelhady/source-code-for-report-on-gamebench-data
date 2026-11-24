# Source code for report on gamebench data

Source code, analysis pipeline and replication package for the report
*Bug Resolution Dynamics in Open-Source Games*, built on the
**OSSGameBench** dataset. The research question: among closed,
explicitly bug-labelled issues with at least one linked pull request,
do issues with **multiple** linked PRs (Group B) show longer observed
issue-to-closure duration than issues with **exactly one** (Group A) —
and how much of any difference survives a fixed-landmark design that
blocks the "open issues accumulate PR links" channel?

Every number, table and figure in [`report.pdf`](report.pdf) is
produced by the scripts in `src/`, in the order documented below, from
the public dataset. There is no manual step in between:
`build_report.py` injects every statistic into the LaTeX source, so
re-running the pipeline regenerates the report from scratch. The
pipeline is **read-only** with respect to the dataset, fully **seeded**
(`seed = 20261120`, 10,000 bootstrap draws), and contains no
machine-specific absolute paths.

## Requirements

- Python 3.12+ (results were produced on 3.15.2)
- Dependencies pinned in [`requirements.txt`](requirements.txt):
  `duckdb`, `polars`, `pyarrow`, `pandas`, `numpy`, `scipy`,
  `statsmodels`, `matplotlib`, `seaborn`, `PyYAML`, `pytz`
- Optional (only for `build_report.py`):
  [Tectonic](https://tectonic-typesetting.github.io/) to compile the
  LaTeX report

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Getting the dataset

Download `OSSGameBench.zip` (65.8 GB) from Zenodo record
[17553539](https://zenodo.org/records/17553539) (DOI
`10.5281/zenodo.17553539`) and verify the archive:

```bash
md5sum OSSGameBench.zip
# expected: 40fd064b56ec327b6284039b18b19ea1
```

Extract it so the directory contains the entity tables
(`project.parquet`, `issue.parquet`, `issue_comment.parquet`,
`pr.parquet`, `pr_comment.parquet`, `pr_commit.parquet`,
`issue_pr.parquet`, `commit_metadata.parquet`) plus the `commits/`
directory of per-project commit-detail parquet files.

*Dataset version note.* The analysis runs on OSSGameBench Version v2 (Zenodo record 17553539, DOI `10.5281/zenodo.17553539`), which provides the full 65.8 GB data archive analyzed in this study. Section 2 of the report documents the dataset provenance and checksum verification.

Then tell the pipeline where the data lives (pick one):

```bash
# Option A — environment variable (recommended)
export OSSGAMEBENCH_DATA_ROOT=/path/to/extracted/OSSGameBench

# Option B — edit config/config.yaml → dataset.root
# (relative paths are resolved against the repository root)
```

## Running the analysis

Always run from the repository root; scripts are designed for
`python -I` (isolated mode).

```bash
for s in inspect_schema inventory classify_bugs prepare_data analyze \
         robustness landmark validate_labels make_figures build_report; do
    echo "=== $s ==="
    .venv/bin/python -I src/$s.py || break
done
```

What each stage does and produces:

| # | Script | Purpose | Main outputs |
|---|--------|---------|--------------|
| 1 | `common.py` | Shared config/path/logging utilities (imported, not run). | — |
| 2 | `inspect_schema.py` | Schema probe: columns and row counts of every entity table, one sample commit-detail file; provenance check of the archive. | `outputs/inventory/schema.json`, `schema_report.txt` |
| 3 | `inventory.py` | Data-quality inventory: timestamp nulls/unparsables, duplicate keys, unmatched issue–PR links, label distribution, commit totals. | `outputs/inventory/dataset_inventory.{csv,json}`, `label_distribution.csv` |
| 4 | `classify_bugs.py` | Builds the strict/broad bug-label tiers by mapping every observed label string to `strict`/`broad`/`no` (rule-based; documented in the report). | `outputs/audit/bug_label_mapping.csv`, `bug_label_mapping_summary.json` |
| 5 | `prepare_data.py` | Cohort construction S0→S4 (strict label → closed → valid non-negative duration) and per-issue PR-link attributes. | `outputs/intermediate/analysis_dataset.parquet`, `excluded_issues.csv`, `outputs/statistics/exclusion_counts.json` |
| 6 | `analyze.py` | Primary comparison Group A (1 linked PR) vs Group B (≥2): Mann–Whitney U, Cliff's δ, Hodges–Lehmann shift, issue- and project-clustered bootstraps, project sign test, project×year fixed-effects regression; PR-timing and labelled exploratory analyses. | `outputs/statistics/primary_results.json`, descriptive/comparison tables |
| 7 | `robustness.py` | Seven pre-specified robustness checks (R1–R7: broad tier, all-links grouping, merged-only PRs, winsorising, per-project minimum sizes, dropping the largest project, …). | `outputs/statistics/robustness_results.json`, `robustness_table.csv` |
| 8 | `landmark.py` | Fixed-landmark analysis: group membership fixed at day *L* after issue creation (L = 30 primary; 7/90 sensitivity), outcome = days from landmark to closure among issues still open at *L*. | `outputs/statistics/landmark_results.json` |
| 9 | `validate_labels.py` | Seeded simple random sample (n = 100) of cohort issues; scores the strict-tier precision (Wilson CIs) from the recorded review decisions shipped in `outputs/audit/`. | `outputs/audit/validation_sample.csv`, `outputs/statistics/validation_results.json` |
| 10 | `make_figures.py` | Renders the six report figures (cohort flow, ECDF, PR-count categories, per-project medians, PR timing, robustness forest) as PNG + PDF. | `outputs/figures/*.png`, `report_source/figures/*.pdf` |
| 11 | `build_report.py` | Regenerates `report_source/numbers.tex` and every `report_source/tables/tab_*.tex` **from the JSON/CSV outputs only**, then compiles `report.pdf` with Tectonic. | `report.pdf` |

### Running without the dataset

The figure and report stages read only the small artefacts shipped
under `outputs/`, so every figure and `report.pdf` can be regenerated
**without** the 65.8 GB archive (leave `OSSGAMEBENCH_DATA_ROOT` unset):

```bash
.venv/bin/python -I src/make_figures.py
.venv/bin/python -I src/build_report.py    # needs Tectonic for the PDF
```

Stages 2–9 read the raw parquet tables as well, so a full end-to-end
reproduction needs the dataset.

## Determinism

- One seed (`20261120` in `config/config.yaml`) drives every bootstrap
  and the validation sample; 10,000 draws.
- All joins are keyed on `(project, number)` /
  `(project, issue_number, pr_number)` with `DISTINCT` guards, so row
  order does not affect results.
- Re-running stages 6–11 reproduces every number in `report.pdf`
  bit-for-bit.

## Repository layout

- `config/config.yaml` — paths, seed, bootstrap iterations, provenance
- `requirements.txt` — pinned dependency versions
- `src/` — the pipeline scripts in the table above
- `outputs/inventory/` — schema + data-quality inventory
- `outputs/audit/` — all 4,273 distinct labels with rule/decision/reason;
  label-validation sample, per-issue decisions, sample IDs
- `outputs/intermediate/` — analysis dataset + excluded-issue ledger
- `outputs/statistics/` — primary, robustness, landmark, validation results
- `outputs/figures/` (PNG) + `report_source/figures/` (PDF)
- `report_source/` — LaTeX source; `numbers.tex` and `tables/*.tex`
  are generated by `build_report.py`, so prose cannot drift from data
- `logs/` — portable run logs, including the MD5/CRC provenance chain
- `report.pdf` — the compiled report

## Provenance and honest limitations

- The outer archive MD5 matches the published Zenodo value
  (verified; `logs/provenance.log`); entity tables are CRC-32-verified
  byte-identical to that archive.
- Dataset snapshot lower bound: 2025-10-12 22:28:19+09:00.
- Bug labels are rule-derived from issue labels; a seeded 100-issue
  audit of that rule was performed by a single reviewer assisted by an
  LLM (precision 97.9% among decided cases). This is **not**
  independent double-human validation; see `outputs/audit/` and
  Section 4.8 of the report.

## Reading guide

Primary result: Section 4 of `report.pdf`; robustness and the
fixed-landmark analysis: Sections 4.6–4.7; practical recommendations:
Section 5; limitations: Section 6; reproduction details: Appendix B.

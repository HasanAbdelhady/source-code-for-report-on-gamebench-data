"""Label-validation audit for the strict-bug classification.

Two modes:

  python -I src/validate_labels.py            # draw + export the sample
  python -I src/validate_labels.py --score    # score recorded decisions

Sample procedure (mode 1)
-------------------------
Simple random sample WITHOUT replacement of n = 100 issues from the
63,688 closed strict-bug cohort (analysis_dataset.parquet), seeded with
cfg analysis.seed (20261120) via a numpy Generator permutation — fully
reproducible.  For each sampled issue we export project, issue_number,
strict labels, title and body (truncated to 1200 characters), sorted by
(project, issue_number) into outputs/audit/validation_sample.csv, and the
drawn ids (in draw order) to validation_sample_ids.csv.

Scoring (mode 2)
----------------
Reads outputs/audit/validation_decisions.csv — one decision per sampled
issue, recorded by the human author after reviewing each issue's title
and description against the bug definition written in the report
(confirmed / rejected / uncertain + short reason) — and computes:
  * counts of confirmed / rejected / uncertain,
  * precision = confirmed / (confirmed + rejected) with a 95% Wilson
    score interval, also computed including 'uncertain' in the
    denominator,
  * per-label breakdown of the decisions.
Writes outputs/statistics/validation_results.json.

IMPORTANT: the review is automated/LLM-assisted (documented in the
report); it is NOT presented as independent double-human validation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

N_SAMPLE = 100
BODY_CHARS = 1200


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def draw(cfg, log) -> None:
    seed = cfg["analysis"]["seed"]
    ad = pl.read_parquet(
        common.out_path(cfg, "intermediate", "analysis_dataset.parquet"))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ad))[:N_SAMPLE]
    sample = ad[idx.tolist()].select(["project", "issue_number"])

    # recover which strict labels triggered classification + title/body
    mapping = str(common.out_path(cfg, "audit", "bug_label_mapping.csv"))
    iss = pl.read_parquet(str(common.dataset_table(cfg, "issue")),
                          columns=["project", "number", "title", "body", "labels"])
    mp = pl.read_csv(mapping)
    lab = (sample.join(iss, left_on=["project", "issue_number"],
                       right_on=["project", "number"], how="left")
           .select("project", "issue_number", "labels")
           .explode("labels").drop_nulls("labels")
           .join(mp.select("label", "decision"), left_on="labels",
                 right_on="label", how="left")
           .filter(pl.col("decision") == "strict")
           .group_by(["project", "issue_number"])
           .agg(pl.col("labels").str.join("; ").alias("strict_labels")))
    out = (sample.join(iss, left_on=["project", "issue_number"],
                       right_on=["project", "number"], how="left")
           .join(lab, on=["project", "issue_number"], how="left")
           .select("project", "issue_number", "title",
                   pl.col("strict_labels").fill_null(""),
                   pl.col("body").str.slice(0, BODY_CHARS).fill_null(""))
           .sort(["project", "issue_number"]))
    p1 = common.out_path(cfg, "audit", "validation_sample.csv")
    out.write_csv(p1)
    pl.DataFrame({"draw_index": range(N_SAMPLE),
                  "project": sample["project"], "issue_number": sample["issue_number"]}
                 ).write_csv(common.out_path(cfg, "audit", "validation_sample_ids.csv"))
    log.info("wrote %s (%d issues, seed=%d, body truncated to %d chars)",
             p1, len(out), seed, BODY_CHARS)


def score(cfg, log) -> None:
    dec = pl.read_csv(common.out_path(cfg, "audit", "validation_decisions.csv"),
                      infer_schema=False)
    sample = pl.read_csv(common.out_path(cfg, "audit", "validation_sample.csv"),
                         infer_schema=False)
    assert len(sample) == N_SAMPLE, f"sample file has {len(sample)} rows"
    joined = sample.select(["project", "issue_number"]).join(
        dec.select(["project", "issue_number", "decision"]),
        on=["project", "issue_number"], how="left")
    missing = joined.filter(pl.col("decision").is_null())
    if len(missing):
        raise SystemExit(f"missing decisions for {len(missing)} sampled issues")
    counts = joined["decision"].value_counts().sort("decision").to_dicts()
    n_conf = sum(c["count"] for c in counts if c["decision"] == "confirmed")
    n_rej = sum(c["count"] for c in counts if c["decision"] == "rejected")
    n_unc = sum(c["count"] for c in counts if c["decision"] == "uncertain")
    n_dec = n_conf + n_rej
    prec_dec = n_conf / n_dec if n_dec else float("nan")
    prec_all = n_conf / N_SAMPLE
    res = {
        "procedure": {
            "sampling": f"simple random sample of {N_SAMPLE} from 63,688 "
                        "closed strict-bug cohort issues, numpy default_rng "
                            f"(seed {cfg['analysis']['seed']}), without replacement",
            "reviewer": "single reviewer (report author) assisted by an LLM, "
                        "reviewing title + first 1200 chars of the issue body "
                        "against a written bug definition; automated review, "
                        "NOT independent double-human validation",
            "decision_scale": ["confirmed", "rejected", "uncertain"],
        },
        "n_sample": N_SAMPLE,
        "confirmed": n_conf, "rejected": n_rej, "uncertain": n_unc,
        "precision_decided": round(prec_dec, 4),
        "wilson95_decided": [round(x, 4) for x in wilson(n_conf, n_dec)],
        "precision_incl_uncertain": round(prec_all, 4),
        "wilson95_incl_uncertain": [round(x, 4) for x in wilson(n_conf, N_SAMPLE)],
        "notes": "precision = share of sampled strict-labelled issues judged "
                 "genuine bug reports; the complementary error is recall, "
                 "which this procedure does NOT estimate",
    }
    # per-label breakdown
    rows = (sample.with_columns(pl.col("strict_labels").cast(str))
            .join(dec.select(["project", "issue_number", "decision", "reason"]),
                  on=["project", "issue_number"], how="left"))
    common.save_json(res, common.out_path(cfg, "statistics",
                                          "validation_results.json"))
    rows.write_csv(common.out_path(cfg, "audit",
                                   "validation_decisions_joined.csv"))
    log.info("decisions: %s | precision(decided)=%.3f | incl-uncertain=%.3f",
             counts, prec_dec, prec_all)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    cfg = common.load_config()
    log = common.get_logger("validate_labels")
    score(cfg, log) if args.score else draw(cfg, log)


if __name__ == "__main__":
    main()

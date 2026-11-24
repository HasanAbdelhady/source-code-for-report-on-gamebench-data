"""Bug-label identification and classification audit (OSSGameBench v2).

Operational bug definitions (label-based only; titles/bodies never used):

  STRICT -- the issue carries >=1 label whose canonical form is a bare,
            unambiguous bug/defect type assertion (whitelist STRICT_CANON).
  BROAD  -- STRICT plus labels that state the issue is a bug with
            qualifiers (severity/area/confirmation) or name a canonical
            defect-symptom category (crash / error report / regression),
            guarded by explicit negation and status-label rules.

The mapping is rule-derived and deterministic. No human verification of
individual issues was performed (documented in the README and paper).

Outputs:
  outputs/audit/bug_label_mapping.csv
  outputs/audit/bug_label_mapping_summary.json
  outputs/audit/bug_label_examples.csv

Run:  python -I src/classify_bugs.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

WORD_RE = re.compile(r"[a-z]+")

# Canonical (punctuation/emoji/CJK stripped, lower-cased, collapsed) forms
# that unambiguously assert "this issue IS a bug/defect".
STRICT_CANON = {
    "bug", "bug report", "reported bug", "type bug", "type of bug",
    "t bug", "kind bug", "category bug", "issue bug", "status bug",
    "defect", "type defect", "t defect",
}

# Canonical forms accepted only under the broader definition: standard
# defect-symptom categories and bare "is a bug" statements with extra words.
BROAD_SYMPTOM_CANON = {
    "crash", "crash or hang", "crashbug", "error report",
    "regression", "t regression", "type regression", "category regression",
}

# Words that, when present alongside "bug", negate the bug claim.
NEG_WORDS = {
    "not", "wont", "won", "invalid", "feature", "intended", "blizzlike",
    "design", "works", "duplicate", "fake", "no",
}

# Labels beginning with resolution-status words describe what happened to the
# issue, not whether it is a bug.
STATUS_PREFIX = (
    "fixed", "fix", "fixes", "fixing", "resolution", "outcome",
    "r fixed", "reso", "status wontfix", "status won t fix", "wontfix",
    "closed", "done", "completed", "merged", "released",
)


def canon(label: str) -> str:
    words = WORD_RE.findall(str(label).lower())
    return " ".join(words)


def classify(label: str):
    """Return (canonical_form, decision, reason, confidence)."""
    c = canon(label)
    words = set(c.split())
    if c in STRICT_CANON:
        return c, "strict", "canonical whitelist: bare bug/defect type assertion", "high"
    if c in BROAD_SYMPTOM_CANON:
        return c, "broad", "standard defect-symptom category (crash/error/regression)", "medium"
    if "bug" in words:
        if words & NEG_WORDS:
            return c, "excluded", "contains 'bug' but also a negation word", "high"
        if c.startswith(STATUS_PREFIX):
            return c, "excluded", "resolution-status label, not a bug-type claim", "high"
        return c, "broad", "contains standalone token 'bug' (qualified bug statement)", "medium"
    return c, "excluded", "no explicit bug/defect claim in label", "high"


def main() -> None:
    cfg = common.load_config()
    log = common.get_logger("classify_bugs")
    con = common.duckdb_conn(cfg)
    I = common.quote_glob(cfg, "issue")

    labels = con.execute(
        f"""SELECT lbl AS label, COUNT(*) AS occurrences,
                   COUNT(DISTINCT struct_pack(project, number)) AS issues
            FROM (SELECT unnest(labels) lbl, project, number
                  FROM read_parquet('{I}')
                  WHERE labels IS NOT NULL AND len(labels) > 0)
            GROUP BY 1 ORDER BY occurrences DESC"""
    ).fetchdf()

    out = labels.label.map(classify).tolist()
    labels["normalized"] = [o[0] for o in out]
    labels["decision"] = [o[1] for o in out]
    labels["reason"] = [o[2] for o in out]
    labels["confidence"] = [o[3] for o in out]
    labels.to_csv(
        common.out_path(cfg, "audit", "bug_label_mapping.csv"), index=False
    )

    n_strict = int((labels.decision == "strict").sum())
    n_broad = int((labels.decision == "broad").sum())
    summary = {
        "distinct_labels_total": int(len(labels)),
        "labels_strict": n_strict,
        "labels_broad_only": n_broad,
        "labels_excluded": int(len(labels)) - n_strict - n_broad,
        "strict_label_list": sorted(
            labels.loc[labels.decision == "strict", "label"].str.lower().str.strip().unique().tolist()
        )[:80],
    }

    # issue-level counts under both definitions
    m = labels[["label", "decision", "reason"]]
    q = f"""
      WITH lab AS (
        SELECT project, number, unnest(labels) lbl
        FROM read_parquet('{I}') WHERE labels IS NOT NULL AND len(labels) > 0),
      j AS (
        SELECT lab.project, lab.number, m.decision
        FROM lab JOIN m ON lab.lbl = m.label)
      SELECT COUNT(DISTINCT struct_pack(project,number)) FILTER (WHERE decision='strict') ,
             COUNT(DISTINCT struct_pack(project,number)) FILTER (WHERE decision IN ('strict','broad'))
      FROM j"""
    con.register("m", m)
    s, b = con.execute(q).fetchone()
    summary["issues_strict"] = int(s)
    summary["issues_broad"] = int(b)
    summary["issues_strict_closed"] = int(con.execute(
        f"""WITH lab AS (SELECT project, number, unnest(labels) lbl, state
                         FROM read_parquet('{I}')
                         WHERE labels IS NOT NULL AND len(labels) > 0),
                j AS (SELECT lab.project, lab.number, lab.state, m.decision
                      FROM lab JOIN m ON lab.lbl = m.label)
              SELECT COUNT(DISTINCT struct_pack(project,number)) FROM j
              WHERE decision='strict' AND state='CLOSED'"""
    ).fetchone()[0])
    common.save_json(summary, common.out_path(cfg, "audit", "bug_label_mapping_summary.json"))

    # representative examples per decision class (titles truncated)
    examples = con.execute(
        f"""WITH lab AS (SELECT project, number, title, state, unnest(labels) lbl
                         FROM read_parquet('{I}')
                         WHERE labels IS NOT NULL AND len(labels) > 0),
                j AS (SELECT lab.*, m.decision, m.reason
                      FROM lab JOIN m ON lab.lbl = m.label)
              SELECT decision, project, number, state,
                     left(title, 90) AS title, lbl AS matching_label, reason
              FROM (SELECT *, row_number() OVER (PARTITION BY decision ORDER BY hash(project, number)) rn
                    FROM j)
              WHERE rn <= 20 ORDER BY decision, rn"""
    ).fetchdf()
    examples.to_csv(
        common.out_path(cfg, "audit", "bug_label_examples.csv"), index=False
    )

    log.info("distinct labels: %d | strict: %d | broad-only: %d",
             len(labels), n_strict, n_broad)
    log.info("issues strict: %d | broad: %d | strict+closed: %d",
             s, b, summary["issues_strict_closed"])
    log.info("top strict labels: %s", summary["strict_label_list"][:15])


if __name__ == "__main__":
    main()

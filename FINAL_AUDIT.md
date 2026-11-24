# Final Methodological and Factual Audit — OSSGameBench Coursework

## 1. Summary of Changes Made

### A. Critical Methodological Correction: Fixed-Landmark Analysis
- **Corrected Interpretative Overstatements**:
  - Removed all claims that the fixed-landmark analysis "completely vanishes" the closure gap, "eliminates reverse causality", or "proves zero difference".
  - Reframed the day-30 comparison ($L=30$) accurately: the observed median difference of $-0.8$ days is **not statistically distinguishable from zero** ($p = 0.832$).
  - Emphasized **estimation imprecision**: because only 33 multi-PR issues survived past day 30, the test has limited statistical power, and the 95% project-clustered confidence interval ($[-22.5, +29.2]$ days) is wide enough to accommodate meaningful differences in either direction. The result is inconclusive rather than proof of an exact null.
  - Documented **selection and survivor limitations**: the analysis is restricted to issues that eventually closed and eventually had at least one linked PR; issues still open at the snapshot and issues that never received a PR are excluded. Landmarking mitigates time-dependent exposure bias within this closed subset, but does not eliminate all selection bias.
  - Inspected and integrated multi-horizon results:
    - **Day 7 ($L=7$)**: multi-PR issues ($n=47$) closed a median of $5.1$ days *faster* ($p = 0.0154$, 95% CI $[-6.8, -1.4]$ days), capturing early maintainer focus.
    - **Day 30 ($L=30$)**: difference not statistically distinguishable from zero ($-0.8$ days, $p=0.832$, CI $[-22.5, +29.2]$ days, $n=33$).
    - **Day 90 ($L=90$)**: multi-PR issues closed $21.2$ days later, but non-significant ($p = 0.641$, CI $[-41.3, +189.0]$ days, $n=22$).
  - Synchronized across Abstract, Methodology (§3.5), Results (§4.8), Discussion (§5.3 & Table 11), Threats to Validity (§6.2), and Conclusion (§7).

### B. Corrected OSSGameBench Author Metadata
- Corrected author names in `report_source/references.bib` from `Marsad, Abdallah and Weeraddana, Dulanja` to **`Marsad, Faiz and Weeraddana, Nimmi`** for both:
  - The dataset record: *OSSGameBench: A Large-Scale Dataset of Contributor Activities in Open-Source Video Games (Version v2)*, Zenodo (record 17553539, DOI `10.5281/zenodo.17553539`).
  - The MSR '26 conference paper: *OSSGameBench: A Large-Scale Dataset of Development Activities in Open-Source Video Games*, Proceedings of the 23rd International Conference on Mining Software Repositories (MSR '26, Data and Tool Showcase), DOI `10.1145/3793302.3793313`.

### C. Consistency, URLs, and Layout
- Updated the replication repository URL to:
  `https://github.com/HasanAbdelhady/source-code-for-report-on-gamebench-data`
- Preserved self-contained replication instructions without exposing local directory paths or passwords.
- Resolved Figure 5 label overlap by adjusting callout positions and adding distinct visual badges for the median markers.

---

## 2. Statistics Independently Verified

| Metric / Cohort | Verified Value | Source / Verification |
|---|---|---|
| Initial issues ($S_0$) | 326,182 | `outputs/statistics/exclusion_counts.json` |
| Strict bug issues ($S_1$) | 73,035 | `outputs/statistics/exclusion_counts.json` |
| Closed strict bugs ($S_2$--$S_4$) | 63,688 | `outputs/statistics/exclusion_counts.json` |
| Group A (single-PR) size | 14,280 | `outputs/statistics/primary_results.json` |
| Group B (multi-PR) size | 662 | `outputs/statistics/primary_results.json` |
| Group C (no-PR) size | 48,746 | `outputs/statistics/primary_results.json` |
| Group A median closure duration | 8.0 days (7.99 d) | `outputs/statistics/primary_results.json` |
| Group B median closure duration | 35.3 days (35.27 d) | `outputs/statistics/primary_results.json` |
| Primary median difference ($B - A$) | 27.3 days | `outputs/statistics/primary_results.json` |
| Primary Cliff's $\delta$ | 0.284 | `outputs/statistics/primary_results.json` |
| Primary Hodges--Lehmann shift | 9.8 days | `outputs/statistics/primary_results.json` |
| Project-clustered 95% bootstrap CI | [13.9, 51.4] days | `outputs/statistics/primary_results.json` |
| Two-sided Mann--Whitney $U$ | 6,069,998 ($p = 3.31 \times 10^{-35}$) | `outputs/statistics/primary_results.json` |
| Both-group projects & sign test | 117 projects (88 higher, 29 lower; $p = 6.2 \times 10^{-12}$) | `outputs/statistics/primary_results.json` |
| Fixed-effects regression coefficient | $\hat\beta = 0.584$ (95% CI [0.354, 0.814], $p = 7.1 \times 10^{-7}$) | `outputs/statistics/primary_results.json` |
| Group B last PR lifetime fraction | 95.3% (median 0.9528) | `outputs/statistics/primary_results.json` |
| Landmark $L=7$ diff & $p$-value | $-5.1$ days, $p = 0.0154$ (CI $[-6.8, -1.4]$ d, $n=47$) | `outputs/statistics/landmark_results.json` |
| Landmark $L=30$ diff & $p$-value | $-0.8$ days, $p = 0.832$ (CI $[-22.5, +29.2]$ d, $n=33$) | `outputs/statistics/landmark_results.json` |
| Landmark $L=90$ diff & $p$-value | $+21.2$ days, $p = 0.641$ (CI $[-41.3, +189.0]$ d, $n=22$) | `outputs/statistics/landmark_results.json` |
| Label audit precision (decided) | 97.9% (95% Wilson CI [92.5%, 99.4%]) | `outputs/statistics/validation_results.json` |

---

## 3. Remaining Methodological Limitations

1. **Imprecision at Later Landmarks**:
   - Because 90.4% of multi-PR issues acquire their second PR only after day 30, the surviving multi-PR cell at day 30 is small ($n=33$), and shrinks to $n=22$ at day 90. The wide confidence intervals mean the data cannot distinguish between a moderate positive effect, zero difference, or a moderate negative effect.
2. **Survivor / Selection Conditioning**:
   - The analysis conditions on issues that eventually closed and eventually received at least one linked PR. Issues still open at the collection snapshot cutoff date (2025-06-25 UTC) are excluded.
3. **Observational Trace Data**:
   - Duration measures elapsed wall-clock calendar time ($\texttt{closed\_at} - \texttt{created\_at}$), which conflates active debugging with developer queueing, review latency, and dormancy. Neither calendar duration nor PR counts measure cognitive debugging effort.


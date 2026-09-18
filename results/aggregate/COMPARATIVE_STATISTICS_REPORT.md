Same intended identities and pipeline-specific development thresholds. End-to-end detection retains failed extraction and failed enrolment in its denominator.

| Pipeline | Conditional TPIR@1 | End-to-end detection | Mated coverage | False reviews / 1,000 scored new probes | Common-success TPIR@1 |
| --- | --- | --- | --- | --- | --- |
| YuNet + SFace | 92.57% | 87.20% | 94.20% | 5.25 | 92.57% |
| SCRFD + SFace | 94.99% | 94.90% | 99.90% | 6.36 | 95.44% |
| YuNet + ArcFace | 97.24% | 91.60% | 94.20% | 1.75 | 97.24% |
| SCRFD + ArcFace | 96.80% | 96.70% | 99.90% | 1.34 | 97.24% |

Common-success subset: 3,789 of 4,000 intended probes. Conditional comparisons across different surviving subsets cannot isolate a detector effect.

Paired changes below are right minus left in percentage points. Every bootstrap replicate uses the same identity draws across all methods.

| Left → right | End-to-end change (95% CI) | FPIR change (95% CI) |
| --- | --- | --- |
| YuNet + SFace → SCRFD + SFace | +7.70 [+5.80, +9.90] | +0.11 [-0.19, +0.44] |
| YuNet + SFace → YuNet + ArcFace | +4.40 [+2.80, +6.20] | -0.35 [-0.70, -0.07] |
| YuNet + SFace → SCRFD + ArcFace | +9.50 [+7.30, +12.10] | -0.39 [-0.71, -0.11] |
| SCRFD + SFace → YuNet + ArcFace | -3.30 [-5.30, -1.50] | -0.46 [-0.86, -0.13] |
| SCRFD + SFace → SCRFD + ArcFace | +1.80 [+1.00, +2.80] | -0.50 [-0.84, -0.23] |
| YuNet + ArcFace → SCRFD + ArcFace | +5.10 [+3.60, +6.90] | -0.04 [-0.21, +0.13] |

Detector-by-embedder interaction in end-to-end detection: -2.60 [-3.90, -1.50] percentage points. Contrast: (SCRFD+ArcFace - YuNet+ArcFace) - (SCRFD+SFace - YuNet+SFace).

Exploratory unadjusted 95% intervals. Fixed galleries and frozen policies; uncertainty excludes model fitting, threshold estimation, gallery selection and population shift. Common-success analysis describes a selected subset only. Zero-event bootstrap intervals cannot bound population FPIR.

An interval containing zero does not establish equivalence or absence of a contribution. The effects describe the complete configured detector, alignment, recogniser and calibrated-policy combinations.

# COM7014 Advanced Computing Project — research report

Auto-generated from the published artefacts. Ordered to show what each layer was intended to improve, and where it did not.

**Research objective.** Assess how detector and recogniser choice, enrolment images and threshold calibration affect duplicate detection, false-review burden, subgroup performance and computational cost. Controlled comparisons separate these factors where the protocol permits. The face networks remain pretrained and frozen; only the review classifier is trained here.

## 1. LFW 1:1 verification

LFW mean fold accuracy 99.28%, pooled FMR 0.33%, FNMR 1.11%, EER 0.78%, extraction failure 10.02%. This is a 1:1 pair task and its FMR is not comparable with the 1:N FPIR figures below. The official ten-fold protocol fits a threshold using the other nine folds for each held-out fold; the separate development-frozen transfer threshold is not used here.

## 2. CPLFW cross-pose transfer

Conditional accuracy 90.24% over 3,515 scored pairs, with 41.42% of the protocol never reaching comparison. Cross-pose *detection*, not comparison, is the dominant finding.

## 3. BFW single-image open-set control

FPIR 8.95%, TPIR@1 93.46%, 89.5 false reviews per 1,000. Reusing a 1:1 threshold for 1:N search refers a large share of genuinely new identities.

## 4. BFW three-image template, same threshold

FPIR 15.22%, TPIR@1 96.71%. Averaging three images changes both genuine and impostor score distributions. These renormalised templates do not necessarily become closer to every face. The separately calibrated one-versus-three-image comparison appears in the supplementary diagnostics.

## 5. BFW gallery-specific calibration

FPIR 0.52%, TPIR@1 92.57%, 5.2 false reviews per 1,000. Comparing this row with the preceding three-image row isolates the threshold change on the same templates.

## 6. Logistic-regression review classifier

FPIR 0.70% against the threshold method's 0.52%; TPIR@1 94.27% against 92.57%; 7.00 false reviews per 1,000 against 5.25.

The primary hypothesis was that the classifier would reduce false review referrals while retaining detection. That criterion is **not achieved**. The classifier raises identification while referring more innocent registrations, which is a trade-off rather than an improvement.

## 7. Female subgroup analysis

Pooled over identity outcomes, not by averaging subgroup percentages.

| Pipeline | Identities | FPIR | TPIR@1 | TPIR@5 | Mated scored/intended | Mated failures | Mated coverage | Non-mated scored/intended | Non-mated failures | Non-mated coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| insightface-scrfd-arcface-buffalo_l | 200 | 0.27% [0.00%–0.74%] | 97.40% [95.60%–99.20%] | 97.40% [95.60%–99.20%] | 500/500 | 0 | 100.00% [100.00%–100.00%] | 1496/1500 | 4 | 99.73% [99.47%–99.93%] |
| opencv-sface-2021dec-yunet-2023mar | 200 | 0.90% [0.28%–1.66%] | 90.64% [86.64%–94.38%] | 90.64% [86.64%–94.38%] | 481/500 | 19 | 96.20% [94.40%–97.80%] | 1442/1500 | 58 | 96.13% [94.67%–97.47%] |

## 8. Male subgroup analysis

Pooled over identity outcomes, not by averaging subgroup percentages.

| Pipeline | Identities | FPIR | TPIR@1 | TPIR@5 | Mated scored/intended | Mated failures | Mated coverage | Non-mated scored/intended | Non-mated failures | Non-mated coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| insightface-scrfd-arcface-buffalo_l | 200 | 0.00% [0.00%–0.00%] | 96.19% [94.20%–97.99%] | 96.19% [94.20%–97.99%] | 499/500 | 1 | 99.80% [99.40%–100.00%] | 1491/1500 | 9 | 99.40% [99.00%–99.73%] |
| opencv-sface-2021dec-yunet-2023mar | 200 | 0.14% [0.00%–0.35%] | 94.58% [92.41%–96.72%] | 94.58% [92.41%–96.72%] | 461/500 | 39 | 92.20% [88.60%–95.20%] | 1417/1500 | 83 | 94.47% [93.00%–95.73%] |

## 9. Profile-photo identity consistency

Exploratory threshold reuse. The operating threshold was frozen for open-set duplicate-profile screening and is applied here unchanged; no threshold was calibrated for profile-photo consistency and none of these figures has been separately validated. This is not a validated identity-authentication system and must not be reported as one.

| Pipeline | Consistency (cond.) | Consistency (end-to-end) | Open-set control detection (cond.) | Wrong-template detection (cond.) | Wrong-template false-consistency (cond.) | Same-person coverage |
| --- | --- | --- | --- | --- | --- | --- |
| insightface-scrfd-arcface-buffalo_l | 96.80% | 96.70% | 99.87% | 100.00% | 0.00% | 99.90% |
| opencv-sface-2021dec-yunet-2023mar | 92.57% | 87.20% | 99.48% | 100.00% | 0.00% | 94.20% |

The outcomes are not equivalent. A consistent photograph opens no case. An inconsistent one — a *low* similarity to the profile's own template — opens a consistency review. An extraction failure resolves nothing and is an unresolved outcome rather than a decision. Duplicate screening runs in the opposite direction: there a *high* similarity to another enrolled identity opens the review.

The two controls are also different questions. The open-set control searches an absent person against the whole gallery, which is the stricter test. The wrong-template control is the direct one-photograph-to-one-profile comparison and is supplementary.

A non-match indicates that the photograph is inconsistent with the enrolled facial template under the evaluated model and threshold. It does not prove that the photograph belongs to another person or that fraud occurred. Pose, lighting, occlusion, image quality, age difference, face-detection failure and model error can all produce the same result. An inconsistent photograph opens a human-review case; a consistent one does not, and an extraction failure resolves nothing.

## 10. YuNet + SFace against SCRFD + ArcFace

| Pipeline | Threshold | FPIR | TPIR@1 | Reviews/1,000 | Coverage |
| --- | --- | --- | --- | --- | --- |
| insightface-scrfd-arcface-buffalo_l | 0.393958 | 0.13% | 96.80% | 1.34 | 100.00% |
| opencv-sface-2021dec-yunet-2023mar | 0.477118 | 0.52% | 92.57% | 5.25 | 99.00% |

| Pipeline | End-to-end (95% CI) | Zero-face | Multiple-face | Embed mean | Complete mean | Model size |
| --- | --- | --- | --- | --- | --- | --- |
| insightface-scrfd-arcface-buffalo_l | 96.70% [95.40%–97.90%] | 2 | 12 | 52.78 ms | 80.73 ms | 182.4 MB |
| opencv-sface-2021dec-yunet-2023mar | 87.20% [84.30%–89.90%] | 189 | 0 | 17.89 ms | 21.72 ms | 37.1 MB |

Each pipeline was calibrated on its own development scores; the SFace threshold is never applied to ArcFace. This is a complete-pipeline comparison — detection, alignment, preprocessing, embedding width and runtime all differ — so no difference is attributable to the embedding model alone.

## 10a. The same two pipelines on one-to-one verification

Section 10 compared the pipelines on gallery search alone, so the conclusion rested on a single task. Experiments 9 and 10 put the comparison pipeline through the same one-to-one protocols. LFW uses official ten-fold cross-validation, fitting each threshold on the other nine folds. CPLFW uses each pipeline's separate frozen LFW development threshold. No threshold is shared between pipelines.

| Metric | LFW YuNet+SFace | LFW SCRFD+ArcFace | CPLFW YuNet+SFace | CPLFW SCRFD+ArcFace |
| --- | --- | --- | --- | --- |
| Correct among scored pairs | 99.28% | 99.79% | 90.24% | 93.13% |
| Reached comparison | 89.98% | 70.10% | 58.58% | 82.25% |
| Zero-face failures | 61 | 0 | 2,321 | 57 |
| Multiple-face failures | 540 | 1,794 | 164 | 1,008 |

Conditional accuracy must be read alongside extraction coverage: the pipelines score different subsets of pairs. The exactly-one-face rule rejects both zero-face and multiple-face detections. The breakdown records these outcomes but cannot establish whether additional detections are bystanders or false positives without manual annotation. These results compare complete pipelines and do not isolate the recogniser.


## 10b. The review classifier on both pipelines

Section 6 fitted the classifier on the baseline pipeline and section 10 compared the pipelines without it, so the framework's most elaborate addition and its strongest components were never combined. Experiment 11 runs the identical method over the comparison pipeline, under the same seed and therefore the same identity groups.

| Review burden per 1,000 new profiles | Threshold alone | With the classifier |
| --- | --- | --- |
| YuNet + SFace | 5.2 | 7.0 |
| SCRFD + ArcFace | 1.3 | 0.0 |

The classifier moves the burden in **opposite directions** on the two pipelines. Its effect is therefore a property of the components it runs on rather than of the classifier alone. The negative result in section 6 stands for the baseline pipeline, but it cannot be stated as a general finding about the method.

The zero on the second row is an observation over 2,987 scored new profiles, not a demonstration that the population rate is zero; the empirical zero-event bootstrap interval cannot bound population FPIR. A supplementary identity-level upper bound is reported with its assumptions. Detection fell from 96.80% to 95.90% in exchange.


## 10c. Detector, recogniser and coverage contributions

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

## 11. Performance against cost

A stronger pipeline is not free. Where it improves extraction and identification it also costs disk and latency, and the trade-off is shown in `implementation_layers_performance_latency` rather than omitted.

## 11a. Additional controlled diagnostics

The separate COMPARISON_DIAGNOSTICS_REPORT.md reports calibrated one- versus three-image enrolment, gallery-size sensitivity, classifier feature ablations and error analysis. These extensions were designed after the benchmark test results were inspected and are exploratory, not independent confirmation.

## 12. Limitations and policy

This is a benchmark-validated, human-review-only academic face-comparison study. It evaluates duplicate-profile screening and profile-photo facial consistency using frozen pretrained face-recognition pipelines and an identity-disjoint logistic-regression review classifier.

The two tasks refer in opposite directions, and a single threshold statement would misdescribe one of them:

- **Duplicate-profile screening** — a *high* similarity to some other enrolled identity opens a duplicate-profile review.
- **Profile-photo consistency** — a *low* similarity to the profile's own enrolled template opens an inconsistency review.

Neither is proof of fraud, ownership or identity, and an extraction failure resolves nothing in either direction.

No face-detection or face-recognition network is trained or fine-tuned. Experiment 7 trains a small logistic-regression review classifier on identity-disjoint BFW development data and evaluates it on untouched held-out identities.

- This remains a proof of concept. No result here proves fraud, misuse or misrepresentation by any person.
- No automatic sanction is applied. Every outcome opens a case for human review and nothing else. In this gallery-screening experiment the referral is triggered by a high similarity to another enrolled identity; profile-photo consistency refers a low similarity to the profile's own template instead, and an extraction failure makes no decision at all.
- The BFW open-set evaluation uses a protocol defined by this project. BFW publishes verification and bias-analysis protocols, not an open-set identification protocol.
- Development and test identities are completely disjoint, and the operating threshold was frozen before the held-out test partition was scored.
- Extraction failures are counted as coverage failures, never as genuine no-match decisions.
- Confidence intervals describe sampling uncertainty over these benchmark identities only. They do not extend to any other population.
- Benchmark demographics do not represent any real deployed user population, so subgroup figures must not be read as deployment estimates.
- The review classifier was fitted separately on each pipeline and moved the review burden in opposite directions on the two. Its contribution therefore depends on the components underneath it, and neither result should be read as a general property of the method.
- Coverage differences between the two detectors are shaped by this protocol's requirement that exactly one face be found. A detector that recovers faint faces also recovers bystanders, so a coverage loss is not on its own evidence of weaker detection.

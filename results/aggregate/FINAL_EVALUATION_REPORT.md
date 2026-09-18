# Final evaluation report

Generated from saved evaluation results on 2026-09-17T14:04:34.760132+00:00. Every number below is read directly from the corresponding `results/aggregate/*.json` file, each of which embeds its own software, model and dataset provenance (`software_environment`, `model_sha256`, `protocol_sha256`, `evaluated_image_set_sha256`, `dataset_archive_md5`/`dataset_archive_sha256`).

## Experiments 1–2 — threshold calibration and selection

Experiment 1 (`pairsDevTrain.txt`) generated candidate thresholds and wrote them with status `"candidates"`; it never selects a winner. Experiment 2 (`pairsDevTest.txt`) evaluated every candidate and only then selected and froze **balanced_accuracy** at threshold **0.363012**, by the rule: "Maximum balanced accuracy on the development split (pairsDevTest.txt); ties broken by lower development-split false match rate, then by candidate name, for full determinism."

## Experiment 2 — LFW development validation (`pairsDevTest.txt`)

Scored 896 / 1000 pairs (failure rate 10.40%). Accuracy **99.22%**, F1 0.9920, ROC-AUC 0.9966, EER 1.36%.

## Experiment 3 — official LFW ten-fold evaluation (`pairs.txt`)

Scored 5399 / 6000 pairs (failure rate 10.02%). **Mean fold accuracy 99.28%** (standard error 0.14%); pooled out-of-fold accuracy 99.28%, precision 99.66%, recall 98.89%, false match rate 0.33%, false non-match rate 1.11%, ROC-AUC 0.9975, EER 0.78%. Confusion matrix: {'false_negative': 30, 'false_positive': 9, 'true_negative': 2695, 'true_positive': 2665}. Mean embedding time 21.72 ms (p95 22.24 ms) over 7162 unique images.

## Experiment 4 — CPLFW cross-pose generalisation (same frozen threshold, no recalibration)

Of the 6,000 raw CPLFW protocol pairs, 3,515 produced valid similarity scores and 2,485 failed during face extraction. The extraction-failure rate was therefore **41.42%** (2,485 ÷ 6,000). These failed pairs were retained in the protocol total and reported separately rather than being silently discarded.

The 2,485 failures comprised 974 zero-face detections on the left image, 1,347 zero-face detections on the right image, 115 multiple-face detections on the left image and 49 multiple-face detections on the right image. Each failed pair carries exactly one category: sides are attempted left first and the pair is abandoned at the first terminal failure, so a right-side category means the left image had already yielded one valid face.

Accuracy, precision, recall, F1-score, ROC-AUC and EER are conditional on the 3,515 pairs for which both images produced exactly one valid face: accuracy 90.24%, F1 0.8949, false match rate 1.95%, false non-match rate 17.46%, ROC-AUC 0.9521, EER 9.77%. An extraction failure is not a verification error: the pipeline never produced a similarity score for those pairs, so they can be neither correct nor incorrect.

## Experiment 5 — real 1:N duplicate-profile gallery (LFW, seed 20260727)

Gallery size 986; 1047 duplicate probes (105 extraction failures); 3080 unknown probes (167 extraction failures).

- Gallery enrolment: 986 enrolled of 1047 intended (failure rate 5.83%); 61 mated probes had no enrolled reference.
- Duplicate detection rate (conditional, scored mated probes only): **99.36%**
- Duplicate detection rate (end-to-end, all intended mated probes): **89.40%**
- Rank-1 identification rate: 97.98%
- Rank-5 identification rate (conditional): 98.83%
- True duplicate miss rate: 0.64%
- **False duplicate-review rate: 52.56%**

Duplicate-profile screening: a high similarity to another enrolled gallery identity opens a case for human review only. It is not evidence of scam activity and does not ban, reject or accuse any identity. This polarity applies to gallery screening; profile-photo consistency refers a *low* similarity instead, and an extraction failure makes no decision at all.

The gallery uses the development-frozen 1:1 threshold as a transfer control. Final LFW uses ten independently fitted fold thresholds; its FMR is not the FMR of this transferred policy. The observed gallery false-review rate directly measures transfer failure at this gallery size without assuming independent pairwise comparisons.

## Limitations

These figures describe LFW and CPLFW's own demographic composition, not any real user base; the gallery experiment is research-scale, not production-scale; and "duplicate profile" here means "same face detected in the gallery", not a legal or investigative finding.

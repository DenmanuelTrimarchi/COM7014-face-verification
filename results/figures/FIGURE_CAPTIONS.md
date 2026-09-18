# Figure captions

Generated from the published JSON and CSV artefacts by `ACP_arden.py`. No value is typed by hand. Every figure is PNG at 300 dpi plus SVG, with PNG text metadata stripped, and passes the project privacy scan.

**Denominators (BFW held-out test):** 198 of 200 gallery identities enrolled, 942 scored mated probes, 2859 scored non-mated probes.

**Metric definitions.** FPIR is the proportion of non-mated searches returning at least one candidate above threshold — a 1:N quantity that compounds with gallery size, and never interchangeable with the 1:1 false-match rate. TPIR@k is the proportion of mated searches placing the correct identity within rank k *and* above threshold; a referral to another identity is a referral, not an identification. End-to-end detection divides by every intended mated probe, so extraction failures reduce it; conditional rates divide by those actually scored.

**Confidence intervals.** All intervals are 95% percentile bounds from an identity-cluster bootstrap: identities are resampled with replacement, carrying their complete protocol outcomes, with subgroup stratification preserved. Images are never resampled independently, which would treat correlated probes of one person as independent observations and understate the intervals.

**Coefficients describe association inside the fitted classifier on these benchmark identities. They are not causal and do not transfer to another population.**

**Experiment 8 status:** `evaluated_non_commercial_academic_research`.

## Result order

Captions follow the order in which the study developed, so each layer's intent is visible before the pretrained comparison.

### 1. LFW 1:1 verification

Official ten-fold pairwise cross-validation with a training-only threshold per fold, reported as accuracy, FMR and FNMR over scored pairs. A 1:1 quantity that never appears on an FPIR axis: one comparison, no competing candidates, no ranking. Conditional on scored pairs.

### 2. CPLFW cross-pose evaluation

The same frozen threshold on raw cross-pose images. Conditional accuracy only, always quoted with its extraction-failure rate; cross-pose detection rather than comparison is the dominant effect.

## Implementation layers (results 3-6)

The five layers share the BFW open-set protocol and show the order the project developed them. Coverage and calibration can also differ; adjacent contrasts are descriptive. The separately calibrated enrolment comparison and the crossed detector/recogniser experiment address these confounders:

3. Single-image gallery, transferred 1:1 threshold
4. Three-image gallery, transferred 1:1 threshold — observed TPIR and FPIR both increase under this transferred policy; this does not establish an inherent disadvantage of averaging normalised embeddings
5. Three-image gallery, BFW development calibration — isolates the policy change while keeping the three-image representation fixed
6. Logistic-regression review classifier, frozen probability threshold
7. SCRFD + ArcFace, its own frozen BFW development calibration

LFW and CPLFW are 1:1 verification and are deliberately absent from this series: mixing an FMR into an FPIR axis would compare different quantities.

- **implementation_layers_fpir** — false review referrals per 1,000 non-mated searches over 2859 scored probes. Lower is better.
- **implementation_layers_duplicate_detection** — TPIR@1, TPIR@5 and end-to-end detection, kept as separate bars because they use different denominators. Higher is better.
- **implementation_layers_coverage** — gallery, mated and non-mated coverage. The remainder in each bar is extraction failure, which is shown rather than hidden.
- **implementation_layers_performance_latency** — end-to-end duplicate detection is plotted against mean complete-pipeline latency in milliseconds. Marker area represents false reviews per 1,000 non-mated searches, so a larger marker means more false reviews. Colour identifies the pipeline: layers 1-4 share YuNet + SFace, layer 5 is SCRFD + ArcFace. Each point is labelled with its layer number and a short method name. Better operating points lie towards the upper-left with smaller markers. The latency axis begins at zero so the cost difference reads as a ratio; the detection axis is padded around the observed values rather than spanning 0-100%, because every layer lies above 85% and a full range would hide the differences. Layers 2, 3 and 4 record an identical latency because they reuse the same extraction pipeline and differ only in threshold or decision rule, so their markers share one horizontal position. Latency excludes one-time model loading and is specific to the recorded local evaluation environment; it is not a portable performance claim. A layer without a measured complete-pipeline latency is omitted from this figure rather than given an invented value.

## Same-person and profile-photo figures (Figures E-F)

- **mated_non_mated_similarity_distributions** — one panel per evaluated pipeline, each marking its own frozen threshold. Mated scores are similarity to the probe's own enrolled template; non-mated scores are top-1 similarity against a gallery the person is not in. Aggregate histograms only: bin edges and counts, never an individual score, identifier or path.
- **profile_photo_consistency_outcomes** — every evaluated pipeline, over 1000 same-person photographs per pipeline. Outcomes are not equivalent: a **consistent** photograph opens no case; an **inconsistent** one — a *low* similarity to the profile's own enrolled template — opens a consistency review; an **extraction failure** resolves nothing and is a separate unresolved outcome rather than a decision. Two different controls appear separately. The **open-set control** searches a person absent from the gallery against every enrolled profile, and is the stricter test; the **wrong-template control** compares one photograph with exactly one deterministically assigned wrong profile, and is the direct one-to-one comparison. An inconsistent result is **not** proof of photo theft or fraud: pose, lighting, occlusion, image quality, age difference, detection failure and model error all produce it.

## 7-8. Female and male subgroup evaluation

Sex is an evaluation dimension only: never a classifier feature, threshold input, calibration variable, or reason to apply a different decision policy. The female panel covers asian, black, indian and white females; the male panel covers the same four categories. Both use identical metric order, units and interval format so they compare fairly.

FPIR is plotted on its own axis with a metric-specific upper bound, not on the 0-100% axis used for TPIR and coverage: these FPIR values are fractions of one per cent, and compressing them against a 0-100% scale would flatten every bar to the baseline and hide the difference the experiment is about. The female and male companion figures share identical FPIR axis limits, computed across both sexes and both pipelines before either figure is drawn.

- **female_subgroup_pipeline_comparison** / **male_subgroup_pipeline_comparison** — FPIR (lower better, own axis), TPIR@1 and TPIR@5 (higher better), mated coverage and non-mated coverage, each with 95% identity-cluster bounds.
- **female_male_aggregate_comparison** — pooled from underlying identity outcomes, not by averaging four subgroup percentages, which would weight a small subgroup as heavily as a large one. FPIR occupies a separate panel for the same reason.

> A zero-event percentile-bootstrap interval such as 0%–0% means that no false referral was observed among the resampled benchmark identities. It does not establish that the population error probability is exactly zero.

These are binary dataset categories. They do not represent the full range of gender identities, every identity, or any real deployed population.

## 9. Profile-photo consistency analysis

A same-identity probe stands for a photograph belonging to the enrolled person. Two different controls appear, and they are not interchangeable. The **open-set non-mated gallery control** searches a person absent from the gallery against every enrolled profile, testing whether they avoid matching any of them; it is the stricter test. The **wrong-profile-template control** compares one photograph with exactly one deterministically assigned wrong profile, and is the direct one-to-one comparison. Referral is the correct outcome for both.

> A non-match indicates that the photograph is inconsistent with the enrolled facial template under the evaluated model and threshold. It does not prove that the photograph belongs to another person or that fraud occurred.

## 10-11. Pipeline comparison and the latency trade-off

- **pipeline_coverage_and_latency** — both pipelines once Experiment 8 is evaluated, each with its own frozen development threshold; the SFace threshold is never applied to ArcFace. A complete-pipeline comparison: detection, alignment, preprocessing and embedding width all differ, so no difference is attributable to the embedding model alone.
- **implementation_layers_performance_latency** — end-to-end duplicate detection is plotted against mean complete-pipeline latency in milliseconds. Marker area represents false reviews per 1,000 non-mated searches, so a larger marker means more false reviews. Colour identifies the pipeline: layers 1-4 share YuNet + SFace, layer 5 is SCRFD + ArcFace. Each point is labelled with its layer number and a short method name. Better operating points lie towards the upper-left with smaller markers. The latency axis begins at zero so the cost difference reads as a ratio; the detection axis is padded around the observed values rather than spanning 0-100%, because every layer lies above 85% and a full range would hide the differences. Layers 2, 3 and 4 record an identical latency because they reuse the same extraction pipeline and differ only in threshold or decision rule, so their markers share one horizontal position. Latency excludes one-time model loading and is specific to the recorded local evaluation environment; it is not a portable performance claim. A layer without a measured complete-pipeline latency is omitted from this figure rather than given an invented value.

## The later comparisons (Experiments 9 to 12)

- **pipeline_across_datasets** — both pipelines on one-to-one verification, accuracy on the left and the share of photographs reaching comparison on the right, for LFW and CPLFW. The two panels move in opposite directions on LFW and must not be read as one quantity: the accuracy panel is conditional on scored pairs, while coverage counts every intended pair. The coverage difference on LFW includes rejections under the exactly-one-face rule; the failure categories do not establish which detections are true faces.
- **detector_embedder_crossed** — the four detector and embedder combinations on the same held-out identities, each at a threshold frozen on the development identities. Colour identifies the embedder, so a pair of bars of one colour shows the detector varied at a fixed embedder. Detection carries 95% identity-cluster bootstrap bounds. Referral points are descriptive; paired referral differences and their intervals appear in the paired report. The conditional detection axis starts at 85%, explicitly truncating the scale. Scored subsets differ, so read this alongside the next figure.
- **paired_pipeline_comparison** — all four pipelines on a common 0–100% scale: mated extraction coverage, detection across all intended mated probes, and TPIR@1 within the common-success subset. Bars show 95% identity-cluster intervals. The accompanying paired report estimates differences using the same identity draws for every pipeline; interval overlap alone is not a test.

## Open-set operating points and the review classifier

- **open_set_operating_curve** — TPIR@1 (per cent, higher is better) against FPIR (a proportion of non-mated searches, log scale, lower is better). Two series: development, on which the threshold was selected, and the held-out test, which never influenced selection. The log axis cannot show zero, so an observed FPIR of zero is drawn at 1e-4; such a point marks the absence of an observed false referral, not a measured rate of 1e-4.
- **duplicate_detection_by_method** — conditional TPIR@1, end-to-end duplicate detection and gallery enrolment coverage as separate bars on a 0-100% axis. They are kept apart because they use different denominators: conditional rates divide by what was scored, end-to-end rates by every intended probe. Higher is better in all three.
- **false_reviews_per_1000_by_method** — false human-review referrals per 1,000 non-mated searches, comparing the single-image control with the three-image proposed method. Lower is better. The count, not a percentage, is the operationally meaningful quantity for a review queue.
- **ml_review_classifier_coefficients** — standardised logistic-regression coefficients, one horizontal bar per feature. Blue is a positive coefficient, which raises the modelled referral probability; red is negative, which lowers it. Bar length is the magnitude of the standardised coefficient, so features are comparable with one another. Feature names are the model's own, as published in ml_review_model.json. These are associations within this benchmark and are not causal claims; no demographic attribute is a feature.
- **subgroup_fpir_tpir_with_confidence_intervals** — the review classifier's FPIR and TPIR@1 for each of the eight BFW subgroups, with 95% identity-cluster bootstrap bounds. Each panel is bounded by its own observed interval rather than a shared 0-100% axis, on which a sub-one-per-cent FPIR and a 95% TPIR would both be unreadable; the axis range therefore differs between the two panels and should be read from the tick labels. Subgroups appear in a fixed alphabetical order shared with the other subgroup figures.

## 12. Limitations

## Limitations common to every figure

- Each rate is conditional on the coverage reported beside it.
- Subgroup intervals are wide once the partition is divided eight ways; overlapping intervals are not evidence of equality.
- These are benchmark identities, not a user population.
- A referral opens human review only. Nothing here proves duplication, fraud, ownership or identity.

## Note on the classifier

The classifier referred 20 non-mated searches in error against the calibrated threshold's 15. Its primary hypothesis — fewer false referrals — is not achieved.

# Comparison diagnostics

Designed after inspection of the benchmark test results. Exploratory analyses without independent confirmatory data. Gallery repeats vary galleries, not the development/test identity split.

All variants use development FPIR target 0.30%; actual test FPIR can differ.

## YuNet + SFace

### Calibrated enrolment

| Images | Test FPIR | Conditional TPIR@1 | End-to-end detection | Mated coverage |
| --- | --- | --- | --- | --- |
| one image | 0.14% | 86.07% | 80.30% | 93.30% |
| three images | 0.52% | 92.57% | 87.20% | 94.20% |

Three minus one image, end-to-end difference: +6.90 percentage points (paired 95% CI +4.30 to +9.60).

### Gallery size

Fixed 25-identity mated probe core and all protocol non-mated probes within each repeat. Three seeds; ranges are repeat ranges, not confidence intervals.

| Intended identities | Mean test FPIR (range) | Mean end-to-end detection | Mean search ms |
| --- | --- | --- | --- |
| 25 | 0.28% (0.10%–0.38%) | 89.60% | 0.115 |
| 50 | 0.17% (0.07%–0.35%) | 89.07% | 0.229 |
| 100 | 0.23% (0.21%–0.24%) | 88.80% | 0.467 |
| 200 | 0.47% (0.45%–0.49%) | 88.80% | 0.921 |

### Classifier features

Same train/calibration identities, complete feature rows and fixed hyperparameters; outputs are referral scores, not calibrated real-world probabilities.

| Features | Test FPIR | Conditional TPIR@1 | End-to-end detection |
| --- | --- | --- | --- |
| similarity only | 1.78% | 94.69% | 89.20% |
| similarity and margin | 1.05% | 94.16% | 88.70% |
| all features | 0.70% | 94.27% | 88.80% |

### Error analysis

Development quartile bins; whole-image Laplacian variance at 128x128 is a blur proxy, not a causal or face-quality label. Missing detection metadata retains unresolved failures. Pose is assessed separately by CPLFW; lighting is not labelled.

| Feature | Development-defined range | Intended probes | Processing failures | Conditional TPIR@1 | End-to-end detection |
| --- | --- | --- | --- | --- | --- |
| image short side pixels | value < 77 | 967 | 129 | 88.94% | 76.76% |
| image short side pixels | 77 ≤ value < 113 | 950 | 38 | 91.40% | 86.70% |
| image short side pixels | 113 ≤ value < 172 | 1025 | 17 | 92.49% | 89.31% |
| image short side pixels | 172 ≤ value | 1058 | 15 | 96.54% | 95.08% |
| image short side pixels | Measurement unavailable | 0 | 0 | n/a | n/a |
| laplacian variance | value < 239 | 933 | 69 | 90.26% | 83.41% |
| laplacian variance | 239 ≤ value < 392 | 925 | 39 | 91.94% | 85.84% |
| laplacian variance | 392 ≤ value < 640 | 1008 | 49 | 92.16% | 85.45% |
| laplacian variance | 640 ≤ value | 1134 | 42 | 95.02% | 92.71% |
| laplacian variance | Measurement unavailable | 0 | 0 | n/a | n/a |
| probe detection confidence | value < 0.922 | 914 | 0 | 85.90% | 85.90% |
| probe detection confidence | 0.922 ≤ value < 0.931 | 930 | 0 | 90.05% | 90.05% |
| probe detection confidence | 0.931 ≤ value < 0.939 | 910 | 0 | 95.43% | 95.43% |
| probe detection confidence | 0.939 ≤ value | 1047 | 0 | 98.13% | 98.13% |
| probe detection confidence | Measurement unavailable | 199 | 199 | n/a | 0.00% |
| probe face area ratio | value < 0.619 | 1074 | 0 | 93.93% | 93.93% |
| probe face area ratio | 0.619 ≤ value < 0.686 | 890 | 0 | 93.36% | 93.36% |
| probe face area ratio | 0.686 ≤ value < 0.764 | 958 | 0 | 93.93% | 93.93% |
| probe face area ratio | 0.764 ≤ value | 879 | 0 | 88.74% | 88.74% |
| probe face area ratio | Measurement unavailable | 199 | 199 | n/a | 0.00% |

## SCRFD + ArcFace

### Calibrated enrolment

| Images | Test FPIR | Conditional TPIR@1 | End-to-end detection | Mated coverage |
| --- | --- | --- | --- | --- |
| one image | 0.33% | 95.90% | 95.80% | 99.90% |
| three images | 0.13% | 96.80% | 96.70% | 99.90% |

Three minus one image, end-to-end difference: +0.90 percentage points (paired 95% CI +0.00 to +2.20).

### Gallery size

Fixed 25-identity mated probe core and all protocol non-mated probes within each repeat. Three seeds; ranges are repeat ranges, not confidence intervals.

| Intended identities | Mean test FPIR (range) | Mean end-to-end detection | Mean search ms |
| --- | --- | --- | --- |
| 25 | 0.09% (0.00%–0.27%) | 95.73% | 0.117 |
| 50 | 0.10% (0.00%–0.30%) | 95.73% | 0.231 |
| 100 | 0.01% (0.00%–0.03%) | 95.73% | 0.466 |
| 200 | 0.01% (0.00%–0.03%) | 95.73% | 0.947 |

### Classifier features

Same train/calibration identities, complete feature rows and fixed hyperparameters; outputs are referral scores, not calibrated real-world probabilities.

| Features | Test FPIR | Conditional TPIR@1 | End-to-end detection |
| --- | --- | --- | --- |
| similarity only | 0.00% | 96.10% | 96.00% |
| similarity and margin | 0.54% | 97.30% | 97.20% |
| all features | 0.00% | 95.90% | 95.80% |

### Error analysis

Development quartile bins; whole-image Laplacian variance at 128x128 is a blur proxy, not a causal or face-quality label. Missing detection metadata retains unresolved failures. Pose is assessed separately by CPLFW; lighting is not labelled.

| Feature | Development-defined range | Intended probes | Processing failures | Conditional TPIR@1 | End-to-end detection |
| --- | --- | --- | --- | --- | --- |
| image short side pixels | value < 77 | 967 | 5 | 93.33% | 92.95% |
| image short side pixels | 77 ≤ value < 113 | 950 | 3 | 96.14% | 96.14% |
| image short side pixels | 113 ≤ value < 172 | 1025 | 4 | 98.09% | 98.09% |
| image short side pixels | 172 ≤ value | 1058 | 2 | 99.24% | 99.24% |
| image short side pixels | Measurement unavailable | 0 | 0 | n/a | n/a |
| laplacian variance | value < 239 | 933 | 3 | 96.19% | 95.73% |
| laplacian variance | 239 ≤ value < 392 | 925 | 4 | 96.46% | 96.46% |
| laplacian variance | 392 ≤ value < 640 | 1008 | 2 | 96.73% | 96.73% |
| laplacian variance | 640 ≤ value | 1134 | 5 | 97.57% | 97.57% |
| laplacian variance | Measurement unavailable | 0 | 0 | n/a | n/a |
| probe detection confidence | value < 0.822 | 1006 | 0 | 93.90% | 93.90% |
| probe detection confidence | 0.822 ≤ value < 0.848 | 992 | 0 | 97.20% | 97.20% |
| probe detection confidence | 0.848 ≤ value < 0.869 | 1034 | 0 | 98.80% | 98.80% |
| probe detection confidence | 0.869 ≤ value | 954 | 0 | 97.22% | 97.22% |
| probe detection confidence | Measurement unavailable | 14 | 14 | n/a | 0.00% |
| probe face area ratio | value < 0.631 | 1083 | 0 | 99.62% | 99.62% |
| probe face area ratio | 0.631 ≤ value < 0.703 | 1006 | 0 | 95.72% | 95.72% |
| probe face area ratio | 0.703 ≤ value < 0.791 | 902 | 0 | 97.18% | 97.18% |
| probe face area ratio | 0.791 ≤ value | 995 | 0 | 94.74% | 94.74% |
| probe face area ratio | Measurement unavailable | 14 | 14 | n/a | 0.00% |

A referral is a model-generated review signal. It is not proof that two profiles belong to the same person, that a photograph was stolen or that fraud occurred.

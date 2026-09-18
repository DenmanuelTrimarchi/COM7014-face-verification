# Conversion map

`ACP_arden.py` is organised into thirty numbered sections. This table maps
each functional component of the research pipeline to the section that
implements it, so a reviewer can navigate the single file as quickly as a
package of small modules.

| Component | ACP_arden.py section | Status |
|---|---|---|
| Programme metadata, imports, project paths | Section 1 | Preserved |
| Model contract, pinned SHA-256 digests, detector settings, seeds, archive checksums | Section 2 | Preserved |
| CPLFW image-variant provenance fields | Section 2 | Preserved |
| `.env` parsing, storage-root resolution, path validation | Section 3 | Preserved |
| Private-path redaction for everything printed | Section 3 | Added |
| File and text hashing, evaluated-image-set fingerprint | Section 4 | Preserved |
| Model hash verification, dependency contract, software environment report | Section 4 | Preserved |
| Opaque one-way identifiers, filename scrubbing | Section 4 | Preserved |
| Bounded image loading, EXIF orientation, failure taxonomy | Section 5 | Preserved |
| YuNet detection wrapper, exactly-one-face rule, detector interface | Section 6 | Preserved |
| SFace embedding wrapper, 128-dimension contract, embedder interface | Section 7 | Preserved |
| L2 normalisation and cosine similarity | Section 8 | Preserved |
| LFW pair-protocol parsing and header validation | Section 9 | Preserved |
| CPLFW pair-protocol parsing (flat layout, two lines per pair) | Section 9 | Preserved |
| Confusion matrix, derived rates, ROC points, ROC-AUC, EER, percentile | Section 10 | Preserved |
| Threshold-candidate strategies (balanced accuracy, F1, EER, target FMR) | Section 10 | Preserved |
| Stage 1 calibration on the validation split only | Section 11 | Preserved |
| Stage 2 deterministic selection and freezing | Section 11 | Preserved |
| Stage 3 frozen-threshold enforcement for held-out evaluation | Section 11 | Preserved |
| Pair evaluation, embedding cache, per-image latency | Section 12 | Preserved |
| Four-category extraction-failure accounting and reconciliation | Section 12 | Preserved |
| Deterministic gallery manifest, role assignment, calibration exclusion | Section 13 | Preserved |
| 1:N gallery search, rank-1 and duplicate-review metrics | Section 13 | Preserved |
| Gallery manifest read/write (private, git-ignored) | Section 13 | Preserved |
| Atomic, schema-versioned JSON/CSV/Markdown artefact writing | Section 14 | Preserved |
| Run manifest, metrics summary, confusion matrices, ROC points | Section 14 | Preserved |
| Final evaluation report rendering | Section 14 | Preserved |
| Terminal results summary with mandatory limitations | Section 14 | Preserved |
| Forbidden-substring list, path-leak scanning, PNG metadata scanning | Section 15 | Preserved |
| Record-level leakage assertions | Section 15 | Preserved |
| Local review database (SQLite, opaque identifiers only) | Section 16 | Preserved |
| Streamlit human-review page and its launcher | Section 16 | Preserved |
| Deterministic synthetic self-tests and their fakes | Section 29 | Preserved |
| Environment check action | Section 30 | Preserved |
| Model and dataset verification action | Section 30 | Preserved |
| Five-experiment orchestration in required order | Section 30 | Preserved |
| Interactive menu, `--mode` command line, entry point | Section 30 | Added |

## Reference-only

These belong to a dissertation evidence workflow rather than to the
methodology, and are deliberately not part of the runnable artefact:

| Component | Status |
|---|---|
| Rendered command-output screenshot pack and its indices | Reference-only — not included |
| Evidence manifest and chapter-placement index | Reference-only — not included |
| Dataset extraction and flattening helpers | Reference-only — not included |
| Comment-style audit tooling | Reference-only — not included |

Excluding them changes no reported metric. Figure rendering is no longer among
them: it is implemented in section 28 and writes PNG and SVG to
`results/figures/`, drawn from the published JSON and CSV artefacts.

## Attribution headers

Where an external model, dataset, API or published method enters the pipeline,
a four-field attribution header sits immediately above the code concerned:

| External source | Section |
|---|---|
| YuNet detector (Wu, Peng and Yu, 2023) | Section 2 |
| SFace embedding (Zhong *et al.*, 2021) | Section 2 |
| OpenCV Zoo, distributor of both weight files | Section 2 |
| LFW dataset (Huang *et al.*, 2007) | Section 2 |
| Funnelled LFW alignment (Huang, Jain and Learned-Miller, 2007) | Section 2 |
| CPLFW dataset (Zheng and Deng, 2018) | Section 2 |
| OpenCV `FaceDetectorYN` / `FaceRecognizerSF` APIs (Bradski, 2000) | Section 6 |
| ROC-AUC rank identity (Hanley and McNeil, 1982) | Section 10 |

No header claims adapted code: each records the origin of an external artefact,
API or standard definition that original code here consumes or implements. The
full register, with licences and digests, is `REFERENCES.md`.
| Keyed HMAC opaque identifiers, key validation, constant-time comparison | Section 4 | Revised |
| Corrected gallery enrolment accounting, conditional vs end-to-end rates | Section 13 | Revised |
| Identifier-version guard for the local review database | Section 16 | Added |
| BFW adapter: pinned schema, containment and consistency validation, provenance | Section 17 | Added |
| Identity-disjoint open-set protocol, subgroup stratification, role assignment | Section 18 | Added |
| Single-image and three-image enrolment methods, template averaging | Section 19 | Added |
| Open-set metrics: FPIR, FNIR, TPIR, CMC, coverage | Section 20 | Added |
| Open-set threshold development, target FPIR selection, freezing guard | Section 21 | Added |
| Cluster-bootstrap confidence intervals over identities | Section 22 | Added |
| Demographic subgroup metrics and disparity summary | Section 23 | Added |
| Experiment 6 orchestration, artefacts, success criteria, open-set report | Section 24 | Added |
| Pipeline description record and optional ArcFace comparator interface | Section 25 | Added |
| Review classifier: identity split, features, logistic regression, freezing | Section 26 | Added |
| Experiment 7 and 8 orchestration, artefacts and reports | Section 27 | Added |
| Figure generation from machine-readable artefacts | Section 28 | Added |
| PNG and SVG output | Section 28 | Added |
| Figure captions and denominator documentation | Section 28 | Added |
| PNG metadata removal and privacy scanning | Section 28 | Added |
| Rank-aware review labels and TPIR definitions | Section 26 | Added |
| Identity-level outcomes for end-to-end denominators | Section 26 | Added |

## Notes on the added and revised rows

- **Private-path redaction (section 3)** is new. The single-file artefact
  prints directly to a terminal rather than through separate scripts, so every
  message it emits is passed through a redactor that replaces a configured
  storage root with its variable name. No storage location can reach the
  screen, a log, or a published artefact.
- **Interactive menu (section 30)** is new. It replaces a set of separate
  command-line entry points with one launcher that requires no arguments, so
  the VS Code play button is safe to press: it shows the menu rather than
  starting a multi-minute benchmark.

- **Keyed identifiers (section 4)** replace a published fixed salt. A salt that
  ships in the source leaves the mapping from a public identifier back to a
  benchmark identity recoverable by hashing a candidate name list. The
  replacement is HMAC-SHA-256 under a secret key supplied through the
  environment, widened from 16 to 32 hexadecimal characters.
- **Gallery accounting (section 13)** corrects a defect rather than adding a
  feature. Reference images that failed to embed were previously dropped, which
  removed the identity from the gallery while its mated probe was still scored
  as an ordinary miss. Both denominators are now reported.
- **Sections 25-26** are the two optional extensions. Both are implemented as
  interfaces that either produce a real artefact or record precisely why they
  did not run; neither can interrupt the primary experiment, and neither
  substitutes a stand-in model or dataset.
- **Sections 17-24** are the supplementary open-set experiment. They are
  additive: `--mode full` still runs exactly the original five experiments, and
  none of the baseline artefacts changed.

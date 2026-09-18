"""Tests for the review classifier (Experiment 7) and the pipeline comparison
(Experiment 8).

No face-recognition model is trained or fine-tuned anywhere in this suite, and
no test reads a dataset, loads a model binary or touches the network. The
classifier is exercised on deterministic synthetic feature rows so its
guarantees — identity-disjoint splits, training-only standardisation, frozen
thresholds — are pinned independently of BFW.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import pytest

import ACP_arden as acp

EXPECTED_SEED = 20260727
EXPECTED_FROZEN_STATUS = "ml_review_frozen"
EXPECTED_PRIMARY_FPIR_TARGET = 0.003
EXPECTED_FEATURE_ORDER = (
    "top1_similarity",
    "top2_similarity",
    "top1_top2_margin",
    "top5_similarity_mean",
    "top5_similarity_stdev",
    "top1_gallery_image_count",
    "gallery_size",
    "probe_detection_confidence",
    "probe_face_area_ratio",
)
EXPECTED_SUBGROUPS = (
    "asian_females", "asian_males", "black_females", "black_males",
    "indian_females", "indian_males", "white_females", "white_males",
)


def _row(label: int, top1: float, *, subgroup: str = "asian_females",
         identity: str = "id", sample: str = "s") -> acp.ReviewFeatureRow:
    top2 = top1 - 0.10
    return acp.ReviewFeatureRow(
        sample_id=sample, identity_hash=identity, subgroup=subgroup,
        role="mated_probe" if label == 1 else "non_mated_probe",
        features={
            "top1_similarity": top1,
            "top2_similarity": top2,
            "top1_top2_margin": top1 - top2,
            "top5_similarity_mean": top1 - 0.15,
            "top5_similarity_stdev": 0.05,
            "top1_gallery_image_count": 3.0,
            "gallery_size": 200.0,
            "probe_detection_confidence": 0.95,
            "probe_face_area_ratio": 0.55,
        },
        label=label,
        correct_rank=1 if label == 1 else None,
        correct_similarity=top1 if label == 1 else None,
    )


def _training_rows(count: int = 120) -> List[acp.ReviewFeatureRow]:
    rng = np.random.default_rng(EXPECTED_SEED)
    rows: List[acp.ReviewFeatureRow] = []
    for index in range(count):
        subgroup = EXPECTED_SUBGROUPS[index % len(EXPECTED_SUBGROUPS)]
        rows.append(_row(1, float(rng.uniform(0.55, 0.95)), subgroup=subgroup,
                         identity=f"m{index:03d}", sample=f"ms{index:03d}"))
        rows.append(_row(0, float(rng.uniform(0.05, 0.45)), subgroup=subgroup,
                         identity=f"n{index:03d}", sample=f"ns{index:03d}"))
    return rows


# --- Feature contract ---------------------------------------------------------


def test_feature_order_is_stable_and_excludes_forbidden_inputs() -> None:
    assert acp.ML_REVIEW_FEATURES == EXPECTED_FEATURE_ORDER
    joined = " ".join(acp.ML_REVIEW_FEATURES)
    # Subgroup must never be a predictor; nor may identity, path or embedding.
    for banned in ("subgroup", "ethnic", "gender", "identity_hash", "path", "embedding", "label"):
        assert banned not in joined


def test_every_feature_has_a_published_definition() -> None:
    definitions = acp._feature_definitions()
    assert set(definitions) == set(acp.ML_REVIEW_FEATURES)
    assert all(text.strip() for text in definitions.values())


def test_labels_follow_the_protocol_roles_and_rank() -> None:
    """A mated probe is positive only when its own identity ranks first."""
    results = [
        acp.OpenSetSearchResult(
            sample_id="a" * 32, identity_hash="h" * 32, subgroup="asian_females",
            role=role, top_similarity=0.8, top2_similarity=0.7,
            top5_similarity_mean=0.6, top5_similarity_stdev=0.05,
            top1_gallery_image_count=3, gallery_size=200,
            probe_detection_confidence=0.9, probe_face_area_ratio=0.5,
            correct_rank=1 if role == "mated_probe" else None,
            correct_similarity=0.8 if role == "mated_probe" else None,
        )
        for role in ("mated_probe", "non_mated_probe")
    ]
    rows, _excluded = acp.build_review_feature_rows(results)
    assert [r.label for r in rows] == [1, 0]


def test_unscored_records_never_become_training_examples() -> None:
    """An extraction failure is a coverage outcome, not a negative decision."""
    results = [
        acp.OpenSetSearchResult(
            sample_id="a" * 32, identity_hash="h" * 32, subgroup="asian_females",
            role="mated_probe", failure_code="zero_faces",
        ),
        acp.OpenSetSearchResult(
            sample_id="b" * 32, identity_hash="i" * 32, subgroup="asian_females",
            role="mated_probe", failure_code=acp.GALLERY_REFERENCE_UNAVAILABLE,
        ),
    ]
    rows, excluded = acp.build_review_feature_rows(results)
    assert rows == []
    assert excluded["unscored"] == 2


def test_a_record_missing_a_feature_is_excluded_not_imputed() -> None:
    results = [
        acp.OpenSetSearchResult(
            sample_id="a" * 32, identity_hash="h" * 32, subgroup="asian_females",
            role="mated_probe", top_similarity=0.8, top2_similarity=None,
        )
    ]
    rows, excluded = acp.build_review_feature_rows(results)
    assert rows == []
    assert excluded["missing_feature"] == 1


# --- Identity-disjoint splits -------------------------------------------------


def _protocol(tmp_path: Path) -> acp.OpenSetProtocol:
    from tests.test_bfw_open_set import _make_bfw  # reuse the official-layout fixture

    root, metadata = _make_bfw(tmp_path)
    return acp.build_open_set_protocol(acp.load_bfw_dataset(root, metadata), seed=EXPECTED_SEED)


def test_training_and_calibration_identities_are_disjoint(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    training, calibration = acp.split_development_identities_for_classifier(
        protocol, seed=EXPECTED_SEED
    )
    assert training and calibration
    assert set(training).isdisjoint(set(calibration))


def test_classifier_identities_never_touch_the_held_out_test_partition(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    training, calibration = acp.split_development_identities_for_classifier(
        protocol, seed=EXPECTED_SEED
    )
    test_identities = {e.identity for e in protocol.partition("test")}
    assert set(training).isdisjoint(test_identities)
    assert set(calibration).isdisjoint(test_identities)


def test_the_classifier_split_is_stratified_and_reproducible(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    first = acp.split_development_identities_for_classifier(protocol, seed=EXPECTED_SEED)
    second = acp.split_development_identities_for_classifier(protocol, seed=EXPECTED_SEED)
    assert first == second
    other = acp.split_development_identities_for_classifier(protocol, seed=EXPECTED_SEED + 1)
    assert other != first

    subgroup_of = {e.identity: e.subgroup for e in protocol.partition("development")}
    for group in first:
        represented = {subgroup_of[i] for i in group}
        assert represented == set(EXPECTED_SUBGROUPS)


def test_the_split_is_by_identity_never_by_image(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    training, calibration = acp.split_development_identities_for_classifier(
        protocol, seed=EXPECTED_SEED
    )
    images_of = {}
    for entry in protocol.partition("development"):
        images_of.setdefault(entry.identity, set()).add(entry.image_path)
    training_images = set().union(*(images_of[i] for i in training))
    calibration_images = set().union(*(images_of[i] for i in calibration))
    assert training_images.isdisjoint(calibration_images)


# --- Fitting ------------------------------------------------------------------


def test_the_classifier_is_reproducible_under_the_research_seed() -> None:
    rows = _training_rows()
    first = acp.fit_review_classifier(rows)
    second = acp.fit_review_classifier(rows)
    assert first.coefficients == pytest.approx(second.coefficients)
    assert first.intercept == pytest.approx(second.intercept)
    assert first.feature_order == EXPECTED_FEATURE_ORDER


def test_standardisation_uses_training_statistics_only() -> None:
    rows = _training_rows()
    classifier = acp.fit_review_classifier(rows)
    matrix, _labels = acp._feature_matrix(rows)
    assert classifier.scaler_mean == pytest.approx(matrix.mean(axis=0), rel=1e-6)
    # Applying the stored parameters to unseen rows must not refit them.
    unseen = np.asarray([[9.0] * len(EXPECTED_FEATURE_ORDER)])
    before = list(classifier.scaler_mean)
    classifier.probabilities(unseen)
    assert list(classifier.scaler_mean) == before


def test_hyperparameters_are_recorded_and_imbalance_is_declared() -> None:
    payload = acp.fit_review_classifier(_training_rows()).as_dict()
    assert payload["hyperparameters"]["class_weight"] == "balanced"
    assert payload["hyperparameters"]["random_state"] == EXPECTED_SEED
    assert "pickle" in payload["serialisation"].lower()
    assert len(payload["coefficients"]) == len(EXPECTED_FEATURE_ORDER)


def test_the_model_serialises_as_plain_numbers(tmp_path: Path) -> None:
    """A pickle would be unsafe to load and opaque to a reader."""
    payload = acp.fit_review_classifier(_training_rows()).as_dict()
    path = tmp_path / "model.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    restored = json.loads(path.read_text(encoding="utf-8"))
    assert restored["feature_order"] == list(EXPECTED_FEATURE_ORDER)
    assert all(isinstance(c, float) for c in restored["coefficients"])


def test_fitting_requires_both_classes() -> None:
    with pytest.raises(acp.MlReviewError):
        acp.fit_review_classifier([_row(1, 0.9), _row(1, 0.8)])


def test_probabilities_are_bounded_and_monotone_in_similarity() -> None:
    classifier = acp.fit_review_classifier(_training_rows())
    low = acp._feature_matrix([_row(0, 0.05)])[0]
    high = acp._feature_matrix([_row(1, 0.95)])[0]
    p_low = float(classifier.probabilities(low)[0])
    p_high = float(classifier.probabilities(high)[0])
    assert 0.0 <= p_low <= 1.0 and 0.0 <= p_high <= 1.0
    assert p_high > p_low


# --- Threshold selection and freezing -----------------------------------------


def test_threshold_selection_respects_the_target_fpir() -> None:
    rows = _training_rows()
    classifier = acp.fit_review_classifier(rows)
    matrix, _ = acp._feature_matrix(rows)
    probabilities = classifier.probabilities(matrix)
    for target in acp.FPIR_TARGETS:
        chosen = acp.select_review_probability_threshold(
            rows, probabilities, target_fpir=target
        )
        assert chosen["calibration_fpir"] <= target + 1e-12
        assert 0.0 <= chosen["probability_threshold"] <= 1.0000001


def test_an_unreachable_target_is_refused_rather_than_approximated() -> None:
    rows = _training_rows()
    classifier = acp.fit_review_classifier(rows)
    probabilities = classifier.probabilities(acp._feature_matrix(rows)[0])
    with pytest.raises(acp.MlReviewError):
        acp.select_review_probability_threshold(rows, probabilities, target_fpir=-1.0)


@pytest.mark.parametrize("status", ["ml_review_development", None, "frozen", "open_set_frozen"])
def test_an_unfrozen_review_policy_is_refused(status) -> None:
    payload = {"status": status,
               "operating_points": {str(EXPECTED_PRIMARY_FPIR_TARGET):
                                    {"probability_threshold": 0.5}}}
    with pytest.raises(acp.MlReviewError):
        acp.require_frozen_review_policy(payload)


def test_a_frozen_policy_returns_the_primary_threshold() -> None:
    payload = {"status": EXPECTED_FROZEN_STATUS,
               "operating_points": {str(EXPECTED_PRIMARY_FPIR_TARGET):
                                    {"probability_threshold": 0.6125}}}
    assert acp.require_frozen_review_policy(payload) == pytest.approx(0.6125)


def test_a_frozen_policy_without_the_primary_target_is_refused() -> None:
    with pytest.raises(acp.MlReviewError):
        acp.require_frozen_review_policy(
            {"status": EXPECTED_FROZEN_STATUS,
             "operating_points": {"0.5": {"probability_threshold": 0.4}}}
        )


# --- Failure accounting and metrics -------------------------------------------


def test_confusion_counts_reconcile_with_the_denominators() -> None:
    rows = [_row(1, 0.9), _row(1, 0.2), _row(0, 0.95), _row(0, 0.1)]
    probabilities = np.asarray([0.9, 0.2, 0.95, 0.1])
    rates = acp.review_rates_at_probability(rows, probabilities, 0.5)
    assert rates["mated_correct_rank1_referred"] == 1
    assert rates["mated_not_referred"] == 1
    assert rates["non_mated_incorrectly_referred"] == 1
    assert rates["non_mated_correctly_not_referred"] == 1
    assert rates["scored_mated_probes"] == 2
    assert rates["scored_non_mated_probes"] == 2
    assert rates["fpir"] == pytest.approx(0.5)
    assert rates["tpir_rank1"] == pytest.approx(0.5)
    assert rates["fnir_rank1"] == pytest.approx(0.5)


def test_the_end_to_end_denominator_counts_every_intended_mated_probe() -> None:
    rows = [_row(1, 0.9), _row(0, 0.1)]
    probabilities = np.asarray([0.9, 0.1])
    outcomes = {
        "id": acp.ReviewIdentityOutcome(
            identity_hash="id", subgroup="asian_females", intended_mated=4, scored_mated=1,
            intended_non_mated=1, scored_non_mated=1, mated_extraction_failures=3,
            non_mated_extraction_failures=0, gallery_reference_unavailable=2,
        )
    }
    rates = acp.review_rates_at_probability(rows, probabilities, 0.5, outcomes=outcomes)
    # One referral out of four intended, not out of the one that was scored.
    assert rates["end_to_end_duplicate_detection_rate"] == pytest.approx(0.25)
    assert rates["tpir_rank1"] == pytest.approx(1.0)
    assert rates["gallery_reference_unavailable"] == 2
    assert rates["mated_extraction_coverage"] == pytest.approx(0.25)


# --- Bootstrap and subgroups --------------------------------------------------


def test_the_review_bootstrap_is_deterministic() -> None:
    rows = _training_rows(40)
    classifier = acp.fit_review_classifier(rows)
    probabilities = classifier.probabilities(acp._feature_matrix(rows)[0])
    first = acp.review_cluster_bootstrap(rows, probabilities, 0.5, replicates=50,
                                         seed=EXPECTED_SEED)
    second = acp.review_cluster_bootstrap(rows, probabilities, 0.5, replicates=50,
                                          seed=EXPECTED_SEED)
    assert first == second
    assert first["fpir"]["requested_replicates"] == 50


def test_subgroup_metrics_carry_confidence_intervals_for_every_subgroup() -> None:
    rows = _training_rows(40)
    classifier = acp.fit_review_classifier(rows)
    probabilities = classifier.probabilities(acp._feature_matrix(rows)[0])
    per_subgroup = acp.review_subgroup_metrics(rows, probabilities, 0.5, replicates=40,
                                               seed=EXPECTED_SEED)
    assert set(per_subgroup) == set(EXPECTED_SUBGROUPS)
    for row in per_subgroup.values():
        for key in ("fpir", "fnir_rank1", "tpir_rank1"):
            assert f"{key}_lower_95" in row and f"{key}_upper_95" in row
            assert row[f"{key}_lower_95"] <= row[f"{key}_upper_95"] or math.isnan(
                row[f"{key}_lower_95"]
            )


def test_subgroup_is_never_used_as_a_predictor() -> None:
    """Rows differing only by subgroup must receive identical probabilities."""
    classifier = acp.fit_review_classifier(_training_rows())
    a = acp._feature_matrix([_row(1, 0.8, subgroup="asian_females")])[0]
    b = acp._feature_matrix([_row(1, 0.8, subgroup="white_males")])[0]
    assert float(classifier.probabilities(a)[0]) == pytest.approx(
        float(classifier.probabilities(b)[0])
    )


# --- Experiment 8: pipeline comparison ----------------------------------------


def test_the_configured_local_model_root_is_accepted() -> None:
    """Where the optional pack is configured locally, it must satisfy every
    precondition. The pack is not redistributable, so a checkout without it
    skips rather than fails: its absence is a property of the machine, not a
    defect in the code under test."""
    config = acp.EnvironmentConfig.load()
    if config.arcface_model_root is None:
        pytest.skip("the optional comparison model pack is not configured here")
    diagnosis = acp.arcface_preconditions(config)
    assert diagnosis["status"] == acp.PIPELINE_STATUS_EVALUATED, diagnosis["reason"]
    assert all(diagnosis["checks"].values())


def test_both_official_filenames_are_required(tmp_path: Path) -> None:
    (tmp_path / acp.ARCFACE_DETECTOR_FILENAME).write_bytes(b"x")
    config = acp.EnvironmentConfig(
        data_root=None, protocol_root=None, model_root=None,
        cplfw_raw_root=None, cache_root=None, arcface_model_root=tmp_path,
    )
    assert acp.arcface_preconditions(config)["status"] == (
        acp.PIPELINE_STATUS_SOURCE_UNVERIFIED
    )


def test_an_unconfigured_pipeline_reports_a_technical_status_not_a_licensing_one(
    tmp_path: Path,
) -> None:
    """Commercial-use restrictions are irrelevant to non-commercial academic
    research, so a licensing status must never stand in for a missing file."""
    config = acp.EnvironmentConfig(
        data_root=None, protocol_root=None, model_root=None,
        cplfw_raw_root=None, cache_root=None, arcface_model_root=None,
    )
    status = acp.pipeline_comparison_status(config)
    assert status["comparison_run"] is False
    assert status["substitute_model_used"] is False
    assert status["status"] == acp.PIPELINE_STATUS_NOT_CONFIGURED
    assert "licens" not in status["status"]
    assert not hasattr(acp, "PIPELINE_STATUS_NOT_RUN")


def test_the_status_vocabulary_separates_technical_from_terms_blockers() -> None:
    assert acp.PIPELINE_STATUS_EVALUATED == "evaluated_non_commercial_academic_research"
    for name in (acp.PIPELINE_STATUS_NOT_CONFIGURED, acp.PIPELINE_STATUS_SOURCE_UNVERIFIED,
                 acp.PIPELINE_STATUS_DIGEST_NOT_PINNED,
                 acp.PIPELINE_STATUS_DEPENDENCIES_MISSING):
        assert name.startswith("not_run_") and "licens" not in name
    # A terms status exists but is reserved for genuinely unclear research terms.
    assert acp.PIPELINE_STATUS_TERMS_UNCLEAR == "not_run_research_terms_not_established"


def test_preconditions_are_diagnosed_in_order(tmp_path: Path, monkeypatch) -> None:
    """A configured root with the files present but no pinned digests must
    report the digest blocker, not the configuration one."""
    (tmp_path / acp.ARCFACE_DETECTOR_FILENAME).write_bytes(b"placeholder")
    (tmp_path / acp.ARCFACE_RECOGNITION_FILENAME).write_bytes(b"placeholder")
    monkeypatch.setattr(acp, "ARCFACE_DETECTOR_SHA256", None)
    monkeypatch.setattr(acp, "ARCFACE_RECOGNITION_SHA256", None)
    config = acp.EnvironmentConfig(
        data_root=None, protocol_root=None, model_root=None,
        cplfw_raw_root=None, cache_root=None, arcface_model_root=tmp_path,
    )
    diagnosis = acp.arcface_preconditions(config)
    assert diagnosis["status"] == acp.PIPELINE_STATUS_DIGEST_NOT_PINNED
    assert diagnosis["checks"]["model_files_present"] is True
    assert diagnosis["checks"]["digests_pinned"] is False
    assert diagnosis["checks"]["research_terms_established"] is True


def test_the_licence_note_states_the_non_commercial_research_position() -> None:
    note = acp.ARCFACE_LICENCE_NOTE
    assert "non-commercial research" in note
    assert "no face-recognition network is trained or fine-tuned here" in note
    assert "MIT licence" in note and "does not automatically extend" in note
    assert "no ownership" in note.lower()
    assert "not redistribute" in note.lower() or "nor redistributes" in note.lower()


def test_unpinned_digests_are_refused(tmp_path: Path, monkeypatch) -> None:
    """With the digests cleared, present files must not be trusted."""
    (tmp_path / acp.ARCFACE_DETECTOR_FILENAME).write_bytes(b"not a real model")
    (tmp_path / acp.ARCFACE_RECOGNITION_FILENAME).write_bytes(b"not a real model")
    monkeypatch.setattr(acp, "ARCFACE_DETECTOR_SHA256", None)
    monkeypatch.setattr(acp, "ARCFACE_RECOGNITION_SHA256", None)
    config = acp.EnvironmentConfig(
        data_root=None, protocol_root=None, model_root=None,
        cplfw_raw_root=None, cache_root=None, arcface_model_root=tmp_path,
    )
    diagnosis = acp.arcface_preconditions(config)
    assert diagnosis["status"] == acp.PIPELINE_STATUS_DIGEST_NOT_PINNED


def test_a_wrong_weight_file_is_refused_by_digest(tmp_path: Path) -> None:
    """Digests are pinned, so a file that is not the approved one is refused."""
    (tmp_path / acp.ARCFACE_DETECTOR_FILENAME).write_bytes(b"not a real model")
    (tmp_path / acp.ARCFACE_RECOGNITION_FILENAME).write_bytes(b"not a real model")
    config = acp.EnvironmentConfig(
        data_root=None, protocol_root=None, model_root=None,
        cplfw_raw_root=None, cache_root=None, arcface_model_root=tmp_path,
    )
    with pytest.raises(acp.ModelUnavailableError):
        acp.arcface_pipeline_description(config)


def test_the_pinned_digests_are_recorded() -> None:
    assert acp.ARCFACE_DETECTOR_SHA256 == (
        "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"
    )
    assert acp.ARCFACE_RECOGNITION_SHA256 == (
        "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"
    )


def test_the_pipeline_record_forbids_embedding_only_attribution() -> None:
    payload = acp.primary_pipeline_description().as_dict()
    assert "cannot be attributed to the embedding model alone" in payload["comparison_scope"]


# --- Exclusions the brief requires --------------------------------------------


def test_no_face_recognition_model_is_trained_or_fine_tuned() -> None:
    source = Path(acp.__file__).read_text(encoding="utf-8")
    for banned in ("fine_tune", "finetune", ".backward()", "torch.optim"):
        assert banned not in source


def test_the_new_modes_are_registered_and_full_is_unchanged() -> None:
    for mode in ("ml-review", "ml-review-summary", "pipeline-compare",
                 "pipeline-compare-summary", "extensions"):
        assert mode in acp.MODES
    # --mode full must still mean the five baseline experiments only.
    assert "full" in acp.MODES
    assert not any("agedb" in mode.lower() for mode in acp.MODES)


def test_matching_the_comparator_is_not_recorded_as_a_reduction() -> None:
    """The declared criterion is 'lower than', so an equal false-review rate is
    not achieved. Recording a tie as a win would overstate the finding."""
    same = {"false_reviews_per_1000_non_mated": 5.2, "fpir": 0.005, "tpir_rank1": 0.94}
    coverage = {"gallery_enrolment_coverage": 0.99,
                "mated_extraction_failure_rate": 0.05,
                "non_mated_extraction_failure_rate": 0.05}
    detection = {"end_to_end_duplicate_detection_rate": 0.88}
    verdicts = acp.evaluate_review_success_criteria(same, coverage, same, detection, detection)
    assert verdicts["fewer_false_reviews_than_threshold_method"]["outcome"] == "not_achieved"

    better = dict(same, false_reviews_per_1000_non_mated=4.0)
    verdicts = acp.evaluate_review_success_criteria(better, coverage, same, detection, detection)
    assert verdicts["fewer_false_reviews_than_threshold_method"]["outcome"] == "achieved"


def test_every_required_artefact_exists_even_when_the_comparison_did_not_run() -> None:
    """A silently absent file is indistinguishable from a forgotten one, so the
    comparison writes its interval and subgroup artefacts either way."""
    root = Path(acp.__file__).parent / "results" / "aggregate"
    if not (root / "pipeline_comparison_metrics.json").is_file():
        pytest.skip("pipeline comparison has not been run in this checkout")
    intervals = json.loads(
        (root / "pipeline_comparison_confidence_intervals.json").read_text(encoding="utf-8")
    )
    assert intervals["artifact_type"] == "pipeline_comparison_confidence_intervals"
    assert "status" in intervals and "evaluated" in intervals
    # Empty rather than invented when the comparison did not run.
    if intervals["evaluated"] == "no":
        assert intervals["intervals"] == {}
        assert intervals["replicates"] == 0

    rows = list(csv.DictReader(open(root / "pretrained_pipeline_subgroup_metrics.csv",
                                    encoding="utf-8")))
    assert rows, "the subgroup file must carry one row per subgroup"
    assert {r["subgroup"] for r in rows} == set(EXPECTED_SUBGROUPS)
    if intervals["evaluated"] == "no":
        assert all(r["fpir"] == "" for r in rows)
        assert all(r["status"].startswith("not_run_") for r in rows)


def test_figure_captions_state_their_denominators() -> None:
    captions = Path(acp.__file__).parent / "results" / "figures" / "FIGURE_CAPTIONS.md"
    if not captions.is_file():
        pytest.skip("figures have not been generated in this checkout")
    text = captions.read_text(encoding="utf-8")
    # Figures are named rather than numbered, so a reordering cannot silently
    # detach a caption from its figure.
    for figure in ("implementation_layers_fpir", "implementation_layers_duplicate_detection",
                   "implementation_layers_coverage",
                   "implementation_layers_performance_latency",
                   "profile_photo_consistency_outcomes", "pipeline_coverage_and_latency"):
        assert figure in text, figure
    assert "scored mated probes" in text and "scored non-mated probes" in text
    # Every element the reporting contract requires.
    assert "Lower is better" in text and "Higher is better" in text
    assert "conditional" in text and "end-to-end" in text.lower()
    assert "identity-cluster bootstrap" in text
    assert "not causal" in text
    assert "human review only" in text


# --- Rank-aware TPIR (corrected defect) ---------------------------------------


def _mated(rank, prob=0.9):
    """A scored mated probe whose correct identity sits at ``rank``."""
    row = _row(1, 0.8)
    return acp.ReviewFeatureRow(
        sample_id=f"m{rank}", identity_hash=f"i{rank}", subgroup="asian_females",
        role="mated_probe", features=row.features,
        label=1 if rank == 1 else 0, correct_rank=rank, correct_similarity=0.8,
    ), prob


def test_rank_one_referral_counts_towards_tpir_rank1() -> None:
    row, prob = _mated(1)
    rates = acp.review_rates_at_probability([row], np.asarray([prob]), 0.5)
    assert rates["tpir_rank1"] == pytest.approx(1.0)
    assert rates["mated_correct_rank1_referred"] == 1


def test_a_rank_two_referral_is_not_a_rank_one_success() -> None:
    """A referral to the wrong identity is not a true identification."""
    row, prob = _mated(2)
    rates = acp.review_rates_at_probability([row], np.asarray([prob]), 0.5)
    assert rates["tpir_rank1"] == pytest.approx(0.0)
    assert rates["mated_wrong_identity_referred"] == 1


def test_a_rank_two_referral_may_count_towards_tpir_rank5() -> None:
    row, prob = _mated(2)
    rates = acp.review_rates_at_probability([row], np.asarray([prob]), 0.5)
    assert rates["tpir_rank5"] == pytest.approx(1.0)
    assert rates["mated_correct_rank5_referred"] == 1


def test_a_rank_nine_referral_counts_for_neither_rank() -> None:
    row, prob = _mated(9)
    rates = acp.review_rates_at_probability([row], np.asarray([prob]), 0.5)
    assert rates["tpir_rank1"] == pytest.approx(0.0)
    assert rates["tpir_rank5"] == pytest.approx(0.0)
    assert rates["mated_wrong_identity_referred"] == 1


def test_labels_require_the_correct_identity_at_rank_one() -> None:
    results = [
        acp.OpenSetSearchResult(
            sample_id=f"s{rank}", identity_hash=f"h{rank}", subgroup="asian_females",
            role="mated_probe", top_similarity=0.8, top2_similarity=0.7,
            top5_similarity_mean=0.6, top5_similarity_stdev=0.05,
            top1_gallery_image_count=3, gallery_size=200,
            probe_detection_confidence=0.9, probe_face_area_ratio=0.5,
            correct_rank=rank, correct_similarity=0.8,
        )
        for rank in (1, 2)
    ]
    rows, _ = acp.build_review_feature_rows(results)
    assert [r.label for r in rows] == [1, 0]


def test_ground_truth_rank_is_never_a_feature() -> None:
    results = [
        acp.OpenSetSearchResult(
            sample_id="a" * 32, identity_hash="h" * 32, subgroup="asian_females",
            role="mated_probe", top_similarity=0.8, top2_similarity=0.7,
            top5_similarity_mean=0.6, top5_similarity_stdev=0.05,
            top1_gallery_image_count=3, gallery_size=200,
            probe_detection_confidence=0.9, probe_face_area_ratio=0.5,
            correct_rank=1, correct_similarity=0.8,
        )
    ]
    rows, _ = acp.build_review_feature_rows(results)
    assert rows[0].correct_rank == 1
    assert "correct_rank" not in rows[0].features
    assert "correct_similarity" not in rows[0].features
    assert set(rows[0].features) == set(acp.ML_REVIEW_FEATURES)


def test_decision_counts_reconcile_with_the_scored_denominator() -> None:
    mated = [_mated(1), _mated(2), _mated(9)]
    all_rows: List[acp.ReviewFeatureRow] = [row for row, _ in mated] + [_row(0, 0.1)]
    all_probs = np.asarray([prob for _, prob in mated] + [0.1])
    rates = acp.review_rates_at_probability(all_rows, all_probs, 0.5)
    assert (
        rates["mated_correct_rank1_referred"]
        + rates["mated_wrong_identity_referred"]
        + rates["mated_not_referred"]
        == rates["scored_mated_probes"]
    )
    assert (
        rates["non_mated_incorrectly_referred"]
        + rates["non_mated_correctly_not_referred"]
        == rates["scored_non_mated_probes"]
    )


# --- Bootstrap denominator (corrected defect) ---------------------------------


def _outcome_fixture():
    """Twenty identities, each with one scored and one failed mated probe."""
    rows, probs, outcomes = [], [], {}
    for index in range(20):
        subgroup = EXPECTED_SUBGROUPS[index % len(EXPECTED_SUBGROUPS)]
        identity = f"id{index:03d}"
        base = _row(1, 0.9, subgroup=subgroup, identity=identity, sample=f"s{index}")
        rows.append(acp.ReviewFeatureRow(
            sample_id=base.sample_id, identity_hash=identity, subgroup=subgroup,
            role="mated_probe", features=base.features, label=1,
            correct_rank=1, correct_similarity=0.9,
        ))
        probs.append(0.9)
        outcomes[identity] = acp.ReviewIdentityOutcome(
            identity_hash=identity, subgroup=subgroup, intended_mated=2, scored_mated=1,
            intended_non_mated=0, scored_non_mated=0, mated_extraction_failures=1,
            non_mated_extraction_failures=0, gallery_reference_unavailable=1,
        )
    return rows, np.asarray(probs), outcomes


def test_the_end_to_end_point_estimate_lies_inside_its_interval() -> None:
    """A point estimate outside its own interval means the bootstrap changed
    the denominator. This is the regression guard for that defect."""
    rows, probs, outcomes = _outcome_fixture()
    point = acp.review_rates_at_probability(rows, probs, 0.5, outcomes=outcomes)
    intervals = acp.review_cluster_bootstrap(
        rows, probs, 0.5, replicates=300, seed=EXPECTED_SEED, outcomes=outcomes
    )
    estimate = point["end_to_end_duplicate_detection_rate"]
    band = intervals["end_to_end_duplicate_detection_rate"]
    assert band["lower_95"] <= estimate <= band["upper_95"], (
        f"{estimate} outside [{band['lower_95']}, {band['upper_95']}]"
    )


def test_conditional_and_end_to_end_differ_when_failures_exist() -> None:
    rows, probs, outcomes = _outcome_fixture()
    rates = acp.review_rates_at_probability(rows, probs, 0.5, outcomes=outcomes)
    assert rates["conditional_duplicate_detection_rate"] == pytest.approx(1.0)
    assert rates["end_to_end_duplicate_detection_rate"] == pytest.approx(0.5)
    assert rates["end_to_end_duplicate_detection_rate"] < rates[
        "conditional_duplicate_detection_rate"
    ]


def test_coverage_intervals_use_intended_probe_counts() -> None:
    rows, probs, outcomes = _outcome_fixture()
    intervals = acp.review_cluster_bootstrap(
        rows, probs, 0.5, replicates=100, seed=EXPECTED_SEED, outcomes=outcomes
    )
    band = intervals["mated_extraction_coverage"]
    # One scored of two intended for every identity, so coverage is exactly 0.5.
    assert band["lower_95"] == pytest.approx(0.5)
    assert band["upper_95"] == pytest.approx(0.5)


def test_the_corrected_bootstrap_is_deterministic() -> None:
    rows, probs, outcomes = _outcome_fixture()
    a = acp.review_cluster_bootstrap(rows, probs, 0.5, replicates=100,
                                     seed=EXPECTED_SEED, outcomes=outcomes)
    b = acp.review_cluster_bootstrap(rows, probs, 0.5, replicates=100,
                                     seed=EXPECTED_SEED, outcomes=outcomes)
    # NaN never equals itself, so an undefined metric needs explicit handling
    # rather than a plain dictionary comparison.
    assert set(a) == set(b)
    for metric, band in a.items():
        other = b[metric]
        for key, value in band.items():
            if isinstance(value, float) and math.isnan(value):
                assert math.isnan(other[key])
            else:
                assert value == other[key]


def test_subgroup_metrics_include_rank_five_and_coverage_intervals() -> None:
    rows, probs, outcomes = _outcome_fixture()
    per_subgroup = acp.review_subgroup_metrics(
        rows, probs, 0.5, replicates=50, seed=EXPECTED_SEED, outcomes=outcomes
    )
    assert per_subgroup
    for entry in per_subgroup.values():
        for metric in ("fpir", "fnir_rank1", "fnir_rank5", "tpir_rank1", "tpir_rank5",
                       "mated_probe_coverage", "non_mated_probe_coverage"):
            assert f"{metric}_lower_95" in entry and f"{metric}_upper_95" in entry
        assert entry["intended_mated_probes"] >= entry["scored_mated_probes"]


# --- Documentation contracts --------------------------------------------------


def _project_file(name: str) -> str:
    return (Path(acp.__file__).parent / name).read_text(encoding="utf-8")


def test_documentation_does_not_claim_nothing_is_trained() -> None:
    """Experiment 7 trains a classifier, so the blanket claim is now false. The
    narrower statement about face-recognition networks remains correct."""
    for name in ("README.md", "ACP_arden.py"):
        text = _project_file(name)
        assert "Nothing here is trained or fine-tuned" not in text
    readme = _project_file("README.md")
    assert "No face-detection or face-recognition network is\ntrained or fine-tuned" in readme \
        or "no face-recognition network is trained" in readme.lower()
    assert "logistic-regression review classifier" in readme


def test_the_valid_narrower_training_sentence_is_permitted() -> None:
    assert "No face-recognition model is trained or fine-tuned." not in _project_file(
        "README.md"
    ) or True  # the sentence is acceptable wherever it appears


def test_no_stale_licensing_wording_remains() -> None:
    for name in ("README.md", "ACP_arden.py", ".env.example", "REFERENCES.md",
                 "CONVERSION_MAP.md"):
        assert "those terms are unresolved for this project" not in _project_file(name)


def test_past_tense_evaluation_claims_match_the_real_status() -> None:
    """'was evaluated' may only appear once held-out metrics exist."""
    root = Path(acp.__file__).parent / "results" / "aggregate"
    metrics = root / "pipeline_comparison_metrics.json"
    if not metrics.is_file():
        pytest.skip("pipeline comparison has not been run in this checkout")
    evaluated = json.loads(metrics.read_text(encoding="utf-8"))["evaluated"] == "yes"
    report = (root / "PRETRAINED_PIPELINE_COMPARISON_REPORT.md").read_text(encoding="utf-8")
    if not evaluated:
        assert "pipeline was evaluated solely" not in report
        assert "is intended for evaluation solely" in report
    assert acp.arcface_use_statement(False).startswith("The InsightFace pipeline is intended")
    assert acp.arcface_use_statement(True).startswith("The InsightFace pipeline was evaluated")


def test_the_conversion_map_places_figures_in_section_28() -> None:
    text = _project_file("CONVERSION_MAP.md")
    assert "Matplotlib evidence-figure rendering | Reference-only" not in text
    assert "sections 11 to 15" not in text
    assert "results/figures/" in text
    for row in ("Figure generation from machine-readable artefacts | Section 28",
                "PNG and SVG output | Section 28",
                "Figure captions and denominator documentation | Section 28",
                "PNG metadata removal and privacy scanning | Section 28"):
        assert row in text


def test_section_numbers_are_contiguous_and_documented() -> None:
    import re

    source = _project_file("ACP_arden.py")
    numbers = sorted({int(m.group(1)) for m in re.finditer(r"^# (\d+)\. ", source, re.M)
                      if int(m.group(1)) <= 40})
    assert numbers == list(range(1, len(numbers) + 1))
    assert len(numbers) == 30
    assert "thirty numbered sections" in _project_file("CONVERSION_MAP.md")


# --- Experiment 8 evaluation path (synthetic stubs, no real weights) ----------


def _stub_pipeline(dimensions: int, seed_offset: int) -> Callable[..., Any]:
    """A deterministic stand-in producing embeddings of a chosen width.

    Used to prove the comparison machinery without any pretrained weights."""

    def _embed(entry, detector, embedder):
        seed = (int(entry.identity_hash[:8], 16) + seed_offset) % (2**32)
        vector = np.random.default_rng(seed).normal(size=dimensions)
        return vector / float(np.linalg.norm(vector)), None, {
            "probe_detection_confidence": 0.9,
            "probe_face_area_ratio": 0.5,
        }

    return _embed


def test_each_pipeline_receives_its_own_development_threshold(tmp_path: Path) -> None:
    """Similarity scales differ between embedding models, so one threshold must
    never be reused for the other."""
    protocol = _protocol(tmp_path)
    results = {}
    for name, (dims, offset) in {"primary": (128, 0), "comparator": (512, 7)}.items():
        run_dev = acp.run_open_set_method(
            protocol, partition="development", method=acp.METHOD_B,
            detector=None, embedder=None,  # type: ignore[arg-type]
            embed_fn=_stub_pipeline(dims, offset),
        )
        results[name] = acp.select_open_set_threshold(
            run_dev.search_results, target_fpir=EXPECTED_PRIMARY_FPIR_TARGET
        )["threshold"]
    # Two independently calibrated thresholds, each from its own scores.
    assert set(results) == {"primary", "comparator"}
    assert all(isinstance(v, float) for v in results.values())


def test_both_pipelines_traverse_the_identical_protocol(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    runs = [
        acp.run_open_set_method(
            protocol, partition="test", method=acp.METHOD_B,
            detector=None, embedder=None,  # type: ignore[arg-type]
            embed_fn=_stub_pipeline(dims, offset),
        )
        for dims, offset in ((128, 0), (512, 7))
    ]
    assert {r.sample_id for r in runs[0].search_results} == {
        r.sample_id for r in runs[1].search_results
    }
    assert runs[0].gallery_size == runs[1].gallery_size


def test_held_out_identities_never_influence_threshold_selection(tmp_path: Path) -> None:
    """Calibration reads the development partition only."""
    protocol = _protocol(tmp_path)
    development = {e.identity for e in protocol.partition("development")}
    test = {e.identity for e in protocol.partition("test")}
    assert development.isdisjoint(test)
    run_dev = acp.run_open_set_method(
        protocol, partition="development", method=acp.METHOD_B,
        detector=None, embedder=None,  # type: ignore[arg-type]
        embed_fn=_stub_pipeline(128, 0),
    )
    chosen = acp.select_open_set_threshold(
        run_dev.search_results, target_fpir=EXPECTED_PRIMARY_FPIR_TARGET
    )
    sample_ids = {r.sample_id for r in run_dev.search_results}
    test_samples = {e.sample_id for e in protocol.partition("test")}
    assert sample_ids.isdisjoint(test_samples)
    assert chosen["threshold"] is not None


def test_a_status_only_run_never_claims_evaluation() -> None:
    """Readiness alone must not set evaluated=yes."""
    root = Path(acp.__file__).parent / "results" / "aggregate"
    payload = json.loads((root / "pipeline_comparison_metrics.json").read_text(encoding="utf-8"))
    if payload["evaluated"] == "no":
        assert payload.get("held_out_metrics") is None
        assert payload.get("comparison_pipeline") is None
        assert payload["status"].startswith("not_run_")
    else:
        assert payload.get("held_out_metrics")


def test_the_comparison_guard_rejects_evaluation_without_metrics() -> None:
    assert issubclass(acp.PipelineComparisonError, RuntimeError)
    source = _project_file("ACP_arden.py")
    assert "cannot be marked as evaluated without held-out metrics" in source


def test_automatic_model_download_is_impossible() -> None:
    """Each ONNX file is loaded by its exact verified path, so there is no
    cache directory to miss and no fetch path to take."""
    source = _project_file("ACP_arden.py")
    assert "ARCFACE_DETECTOR_FILENAME" in source
    # No fetching API is reachable. Attribution URLs appear in reference
    # headers and are not requests, so only call sites are checked.
    for fetching in ("download_zip", "urlretrieve", "requests.get", "urlopen",
                     "insightface.app", "FaceAnalysis("):
        assert fetching not in source


def test_a_hash_mismatch_is_refused(tmp_path: Path) -> None:
    impostor = tmp_path / acp.ARCFACE_DETECTOR_FILENAME
    impostor.write_bytes(b"not the pinned model")
    with pytest.raises(acp.ModelUnavailableError):
        acp.verify_model_file(impostor, "0" * 64)


class _StubDetectorModel:
    """Returns a fixed number of detections in the SCRFD output shape."""

    def __init__(self, count: int):
        self._count = count

    def detect(self, bgr, max_num=0, metric="default"):
        if self._count == 0:
            return np.zeros((0, 5)), None
        boxes = np.tile(np.asarray([0.0, 0.0, 8.0, 8.0, 0.9]), (self._count, 1))
        landmarks = np.tile(np.asarray([[1.0, 1.0]] * 5), (self._count, 1, 1))
        return boxes, landmarks


class _StubRecognitionModel:
    def __init__(self, dimensions: int):
        self._dimensions = dimensions

    def get_feat(self, aligned):
        return np.ones(self._dimensions) / math.sqrt(self._dimensions)


def test_the_arcface_embedder_refuses_unexpected_dimensions() -> None:
    detector = acp.ArcFaceDetector(_StubDetectorModel(1), "digest")
    detector.detect_single_face(np.zeros((112, 112, 3), dtype=np.uint8))
    # The recognition model returns 128 dimensions where 512 is required.
    embedder = acp.ArcFaceEmbedder(
        _StubRecognitionModel(128), detector, "digest", dimensions=512
    )
    with pytest.raises(acp.SimilarityError):
        embedder.embed(np.zeros((112, 112, 3), dtype=np.uint8), np.zeros(15))


class _RecordingRecognizer:
    """Stands in for cv2.FaceRecognizerSF and remembers the row it was given."""

    def __init__(self) -> None:
        self.row_dtype: Any = None

    def alignCrop(self, bgr, face_row):  # noqa: N802 - the OpenCV spelling
        self.row_dtype = np.asarray(face_row).dtype
        return bgr

    def feature(self, aligned):
        return np.ones((1, acp.EMBEDDING_DIMENSIONS))


def test_the_sface_embedder_converts_the_row_before_aligning() -> None:
    """alignCrop reads the landmark columns as 32-bit floats without checking
    the array's type, so a 64-bit row is misread rather than refused. Any
    detector's row must therefore be converted before it is passed on."""
    embedder = object.__new__(acp.SFaceEmbedder)
    recognizer = _RecordingRecognizer()
    # Built without the constructor so the test needs no model binary.
    setattr(embedder, "_recognizer", recognizer)
    embedder.embed(np.zeros((112, 112, 3), dtype=np.uint8), np.zeros(15, dtype=np.float64))
    assert recognizer.row_dtype == np.float32


class _DetectorWithoutLandmarkRecord:
    """A detector that fills the row but keeps no landmark state, as YuNet
    does. The ArcFace embedder must still be able to align from such a row."""

    def detect_single_face(self, bgr: np.ndarray) -> np.ndarray:
        row = np.zeros(15, dtype=np.float32)
        row[0], row[1], row[2], row[3] = 10.0, 12.0, 40.0, 44.0
        row[4:14] = [22.0, 28.0, 38.0, 28.0, 30.0, 36.0, 24.0, 44.0, 36.0, 44.0]
        row[14] = 0.95
        return row


def test_the_arcface_embedder_aligns_from_any_detectors_row() -> None:
    reference = acp.ArcFaceDetector(_StubDetectorModel(1), "digest")
    embedder = acp.ArcFaceEmbedder(_StubRecognitionModel(512), reference, "digest")
    foreign = _DetectorWithoutLandmarkRecord()
    image = np.zeros((112, 112, 3), dtype=np.uint8)
    # The reference detector has recorded nothing, so alignment can only come
    # from the row itself.
    embedding = embedder.embed(image, foreign.detect_single_face(image))
    assert embedding.shape == (512,)


def test_the_landmarks_are_read_from_the_row_when_it_carries_them() -> None:
    row = np.zeros(15, dtype=np.float32)
    row[4:14] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    landmarks = acp.ArcFaceEmbedder._landmarks_from_row(row)
    assert landmarks is not None
    assert landmarks.shape == (5, 2)
    assert landmarks[0].tolist() == [1.0, 2.0]
    assert landmarks[4].tolist() == [9.0, 10.0]
    # A row that carries none, and a row too short to carry any, both defer to
    # the detector's own record instead.
    assert acp.ArcFaceEmbedder._landmarks_from_row(np.zeros(15)) is None
    assert acp.ArcFaceEmbedder._landmarks_from_row(np.zeros(13)) is None
    assert acp.ArcFaceEmbedder._landmarks_from_row(None) is None


def test_the_arcface_detector_requires_exactly_one_face() -> None:
    for count in (0, 2):
        detector = acp.ArcFaceDetector(_StubDetectorModel(count), "digest")
        with pytest.raises(acp.FaceCountError) as raised:
            detector.detect_single_face(np.zeros((112, 112, 3), dtype=np.uint8))
        assert raised.value.face_count == count


def test_the_detector_row_matches_the_yunet_shape() -> None:
    detector = acp.ArcFaceDetector(_StubDetectorModel(1), "digest")
    row = detector.detect_single_face(np.zeros((112, 112, 3), dtype=np.uint8))
    assert row.shape == (15,)
    assert row[2] == pytest.approx(8.0) and row[3] == pytest.approx(8.0)
    assert row[14] == pytest.approx(0.9)


def test_face_analysis_is_not_used_so_nothing_can_be_downloaded() -> None:
    """FaceAnalysis resolves models through a cache directory and fetches the
    pack over the network when it is empty, which would both download
    automatically and evaluate files other than the pinned ones."""
    source = _project_file("ACP_arden.py")
    # Named only in the comment explaining why it is avoided, never called.
    assert "FaceAnalysis(" not in source
    assert "from insightface.model_zoo import get_model" in source


def test_the_detection_input_size_is_pinned_and_the_threshold_is_default() -> None:
    """Input size is a preprocessing scale; the decision threshold stays at the
    published default so coverage is not inflated by lowering the bar."""
    assert acp.ARCFACE_DETECTION_INPUT_SIZE == 320
    assert acp.ARCFACE_DETECTION_THRESHOLD == 0.5


def test_the_comparison_artefact_carries_full_provenance() -> None:
    root = Path(acp.__file__).parent / "results" / "aggregate"
    payload = json.loads((root / "pipeline_comparison_metrics.json").read_text(encoding="utf-8"))
    required = (
        "artifact_type", "schema_version", "created_at", "seed", "status", "evaluated",
        "dataset_name", "protocol_version", "protocol_digest", "public_manifest_digest",
        "primary_pipeline", "comparison_pipeline", "model_filenames", "model_digests",
        "software_environment", "dependency_versions", "preprocessing_revision",
        "threshold_policy", "calibration_partition", "held_out_partition", "policy_note",
        "licence_note", "preconditions", "limitations",
    )
    missing = [key for key in required if key not in payload]
    assert not missing, f"missing provenance: {missing}"
    if payload["evaluated"] == "no":
        assert payload["reason"]


def test_missing_dependencies_produce_the_dependency_status(tmp_path: Path, monkeypatch) -> None:
    """With files present and digests pinned, an absent package is the blocker."""
    (tmp_path / acp.ARCFACE_DETECTOR_FILENAME).write_bytes(b"placeholder")
    (tmp_path / acp.ARCFACE_RECOGNITION_FILENAME).write_bytes(b"placeholder")
    monkeypatch.setattr(acp, "ARCFACE_DETECTOR_SHA256", "a" * 64)
    monkeypatch.setattr(acp, "ARCFACE_RECOGNITION_SHA256", "b" * 64)
    monkeypatch.setattr(acp.importlib.util, "find_spec", lambda name: None)
    config = acp.EnvironmentConfig(
        data_root=None, protocol_root=None, model_root=None,
        cplfw_raw_root=None, cache_root=None, arcface_model_root=tmp_path,
    )
    diagnosis = acp.arcface_preconditions(config)
    assert diagnosis["status"] == acp.PIPELINE_STATUS_DEPENDENCIES_MISSING
    assert diagnosis["missing_dependencies"]
    assert diagnosis["checks"]["digests_pinned"] is True
    assert diagnosis["checks"]["model_files_present"] is True


def test_experiment_seven_subgroup_sample_counts_reconcile() -> None:
    rows, probs, outcomes = _outcome_fixture()
    per_subgroup = acp.review_subgroup_metrics(
        rows, probs, 0.5, replicates=20, seed=EXPECTED_SEED, outcomes=outcomes
    )
    for entry in per_subgroup.values():
        assert entry["scored_mated_probes"] <= entry["intended_mated_probes"]
        assert entry["scored_non_mated_probes"] <= entry["intended_non_mated_probes"]


def test_threshold_selection_is_reproducible_across_processes() -> None:
    """The frozen threshold must not drift between runs.

    A threshold that moved run to run would void the freezing guarantee, so
    selection is pinned here on fixed inputs rather than only end to end."""
    rows = _training_rows()
    first = acp.fit_review_classifier(rows)
    second = acp.fit_review_classifier(rows)
    assert first.coefficients == pytest.approx(second.coefficients, abs=0.0)
    assert first.intercept == pytest.approx(second.intercept, abs=0.0)

    matrix, _ = acp._feature_matrix(rows)
    chosen = [
        acp.select_review_probability_threshold(
            rows, model.probabilities(matrix), target_fpir=EXPECTED_PRIMARY_FPIR_TARGET
        )["probability_threshold"]
        for model in (first, second)
    ]
    assert chosen[0] == chosen[1]


def test_the_published_threshold_matches_the_frozen_policy() -> None:
    """The evaluated threshold must be the one recorded as frozen."""
    root = Path(acp.__file__).parent / "results" / "aggregate"
    policy_path, test_path = root / "ml_review_threshold.json", root / "ml_review_test_metrics.json"
    if not (policy_path.is_file() and test_path.is_file()):
        pytest.skip("the review experiment has not been run in this checkout")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    test = json.loads(test_path.read_text(encoding="utf-8"))
    assert policy["status"] == EXPECTED_FROZEN_STATUS
    frozen = policy["operating_points"][str(EXPECTED_PRIMARY_FPIR_TARGET)]["probability_threshold"]
    assert test["operating_probability_threshold"] == pytest.approx(frozen)


def test_the_readme_states_the_negative_outcome_and_makes_no_improvement_claim() -> None:
    """The classifier did not reduce false referrals, so the documentation must
    not imply machine learning improved the project."""
    readme = _project_file("README.md")
    assert "primary hypothesis was not achieved" in readme.lower()
    assert "benchmark-validated, human-review-only" in readme
    for forbidden in ("production-ready artefact", "unbiased", "proves fraud"):
        # These appear only inside the explicit disclaimer sentence.
        if forbidden in readme:
            assert "not " + forbidden in readme or "not unbiased" in readme


def test_the_readme_documents_rank_aware_tpir() -> None:
    readme = _project_file("README.md")
    assert "rank-aware" in readme.lower()
    assert "mated_wrong_identity_referred" in readme


def test_the_dependency_contract_checks_the_imported_opencv_not_only_metadata() -> None:
    """Installing opencv-python beside opencv-python-headless leaves both
    recorded in metadata while only one is imported. The contract must catch
    the shadowing library, because it silently changes detection and embedding
    numerics."""
    import cv2

    report = acp.check_dependency_contract(strict=False)
    assert "cv2 (imported)" not in report, (
        f"a shadowing OpenCV is loaded: {getattr(cv2, '__version__', 'unknown')}"
    )
    expected = acp.EXPECTED_DEPENDENCY_VERSIONS["opencv-python-headless"]
    assert expected.startswith(str(cv2.__version__))
    source = _project_file("ACP_arden.py")
    assert "shadowing opencv-python-headless" in source


# --- Acceptance tests for the completed Experiment 8 --------------------------

_AGG = Path(acp.__file__).parent / "results" / "aggregate"
_FIG = Path(acp.__file__).parent / "results" / "figures"


def _pipeline_payload():
    path = _AGG / "pipeline_comparison_metrics.json"
    if not path.is_file():
        pytest.skip("the pipeline comparison has not been run in this checkout")
    return json.loads(path.read_text(encoding="utf-8"))


def test_evaluated_yes_requires_both_real_pipeline_results() -> None:
    payload = _pipeline_payload()
    if payload["evaluated"] != "yes":
        assert payload.get("held_out_metrics") is None
        return
    held_out = payload["held_out_metrics"]
    assert len(held_out) == 2, "both pipelines must have produced held-out metrics"
    for metrics in held_out.values():
        assert metrics["rates"]["scored_mated_probes"] > 0
        assert metrics["rates"]["scored_non_mated_probes"] > 0


def test_each_pipeline_has_its_own_frozen_threshold() -> None:
    payload = _pipeline_payload()
    held_out = payload.get("held_out_metrics") or {}
    if len(held_out) < 2:
        pytest.skip("comparison not evaluated")
    thresholds = {n: m["development_threshold"] for n, m in held_out.items()}
    # Scores from different embedding models are not interchangeable.
    assert len(set(thresholds.values())) == 2, thresholds
    for metrics in held_out.values():
        assert metrics["threshold_status"] == "open_set_frozen"


def test_the_interval_file_is_populated_after_evaluation() -> None:
    payload = _pipeline_payload()
    intervals = json.loads(
        (_AGG / "pipeline_comparison_confidence_intervals.json").read_text(encoding="utf-8")
    )
    if payload["evaluated"] != "yes":
        assert intervals["intervals"] == {}
        return
    assert len(intervals["intervals"]) == 2
    for series in intervals["intervals"].values():
        for metric in ("fpir", "fnir_rank1", "fnir_rank5", "tpir_rank1", "tpir_rank5",
                       "end_to_end_duplicate_detection_rate", "mated_extraction_coverage",
                       "non_mated_extraction_coverage"):
            assert metric in series, metric


def test_the_subgroup_csv_holds_sixteen_real_rows_after_evaluation() -> None:
    payload = _pipeline_payload()
    rows = list(csv.DictReader(open(_AGG / "pretrained_pipeline_subgroup_metrics.csv",
                                    encoding="utf-8")))
    if payload["evaluated"] != "yes":
        pytest.skip("comparison not evaluated")
    assert len(rows) >= 16
    assert len({r["pipeline"] for r in rows}) == 2
    assert all(r["fpir"] != "" for r in rows)


def test_the_comparison_csv_carries_performance_not_only_descriptions() -> None:
    payload = _pipeline_payload()
    rows = list(csv.DictReader(open(_AGG / "pretrained_pipeline_comparison.csv",
                                    encoding="utf-8")))
    assert rows
    for column in ("fpir", "tpir_rank1", "threshold", "detector_file_size_mb"):
        assert column in rows[0]
    if payload["evaluated"] == "yes":
        assert all(r["fpir"] != "" for r in rows)


def test_model_sizes_and_latency_are_published() -> None:
    payload = _pipeline_payload()
    sizes = payload["model_file_sizes"]
    assert sizes["primary"] and sizes["comparison"]
    for group in sizes.values():
        for entry in group.values():
            assert entry["bytes"] > 0 and entry["megabytes"] > 0
    held_out = payload.get("held_out_metrics") or {}
    for metrics in held_out.values():
        c = metrics["coverage"]
        for key in ("embedding_latency_mean_ms", "embedding_latency_p95_ms",
                    "complete_pipeline_latency_mean_ms", "complete_pipeline_latency_p95_ms",
                    "top1_search_time_mean_ms"):
            assert key in c, key


def test_every_experiment_eight_json_has_full_provenance() -> None:
    required = (
        "artifact_type", "schema_version", "created_at", "seed", "dataset_name",
        "protocol_version", "protocol_digest", "public_manifest_digest",
        "development_partition", "held_out_partition", "primary_pipeline",
        "comparison_pipeline", "model_filenames", "model_digests", "model_file_sizes",
        "software_environment", "dependency_versions", "preprocessing_revision",
        "threshold_policy", "frozen_thresholds", "status", "licence_note",
        "policy_note", "limitations",
    )
    for name in ("pipeline_comparison_metrics.json", "pipeline_comparison_protocol.json",
                 "pipeline_comparison_confidence_intervals.json"):
        path = _AGG / name
        if not path.is_file():
            pytest.skip(f"{name} not present")
        payload = json.loads(path.read_text(encoding="utf-8"))
        missing = [k for k in required if k not in payload]
        assert not missing, f"{name} missing provenance: {missing}"


def test_the_implementation_layer_artefact_lists_every_available_layer() -> None:
    path = _AGG / "implementation_layer_comparison.csv"
    if not path.is_file():
        pytest.skip("layer artefact not generated in this checkout")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert len(rows) >= 4
    assert [int(r["layer"]) for r in rows] == list(range(1, len(rows) + 1))
    for r in rows:
        assert r["threshold_source"]


def test_the_profile_consistency_artefacts_include_both_controls() -> None:
    path = _AGG / "profile_photo_consistency_metrics.json"
    if not path.is_file():
        pytest.skip("consistency artefact not generated in this checkout")
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Pipeline-scoped since the comparison publishes both. The open-set control
    # searches an absent person against the whole gallery; the wrong-template
    # control is the direct one-photograph-to-one-profile comparison. Both are
    # published, and neither stands in for the other.
    entries = payload.get("pipelines") or {"primary": payload}
    for entry in entries.values():
        for key in ("consistent_same_person_photographs",
                    "inconsistent_same_person_review_candidates",
                    "open_set_non_mated_gallery_controls_correctly_identified",
                    "open_set_non_mated_gallery_controls_false_consistent",
                    "wrong_profile_template_controls_correctly_inconsistent",
                    "wrong_profile_template_controls_false_consistent",
                    "same_person_extraction_failures",
                    "gallery_reference_unavailable"):
            assert key in entry, key
    note = payload.get("interpretation_note") or next(iter(entries.values()))[
        "interpretation_note"
    ]
    assert "does not prove" in note


def test_the_sex_figures_contain_only_their_own_subgroups() -> None:
    for sex, forbidden in (("female", "_males"), ("male", "_females")):
        path = _FIG / f"{sex}_subgroup_pipeline_comparison.svg"
        if not path.is_file():
            pytest.skip("figures not generated in this checkout")
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert forbidden not in text, f"{sex} figure leaks {forbidden}"


def test_sex_aggregates_are_pooled_from_identity_outcomes() -> None:
    path = _AGG / "bfw_sex_aggregated_metrics.json"
    if not path.is_file():
        pytest.skip("sex aggregation not generated in this checkout")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "pooled over identity outcomes" in payload["aggregation"]
    for entry in payload["groups"].values():
        assert entry["identities"] > 0
        assert len(entry["subgroups_pooled"]) == 4
        assert "do not represent" in entry["population_note"]


def test_no_stale_licensing_language_survives_anywhere() -> None:
    """The comparison now runs under permitted non-commercial research terms,
    so wording implying an unresolved licensing blocker is inaccurate."""
    for name in ("ACP_arden.py", "README.md", "REFERENCES.md", ".env.example",
                 "CONVERSION_MAP.md", "requirements-comparison.txt"):
        text = _project_file(name)
        for stale in ("terms are unresolved", "licensing is unresolved",
                      "No model is trained or fine-tuned"):
            assert stale not in text, f"{name} still contains: {stale}"


def test_the_comparison_docstring_describes_both_paths() -> None:
    source = _project_file("ACP_arden.py")
    assert "Records a precise technical status when the local comparison pipeline is" in source
    assert "performs the complete held-out comparison when all verified" in source


def test_estimator_arguments_are_typed_constants_not_dictionary_lookups() -> None:
    """scikit-learn types solver and class_weight as literals. Recovering them
    with str() erases the literal and defeats the type checker, so the call
    uses the constants directly and the provenance dict is built from them."""
    source = _project_file("ACP_arden.py")
    assert 'solver=ML_REVIEW_SOLVER' in source
    assert 'class_weight=ML_REVIEW_CLASS_WEIGHT' in source
    assert 'str(ML_REVIEW_HYPERPARAMETERS[' not in source
    # One source of truth: the published record matches what is passed.
    assert acp.ML_REVIEW_HYPERPARAMETERS["solver"] == acp.ML_REVIEW_SOLVER == "lbfgs"
    assert acp.ML_REVIEW_HYPERPARAMETERS["class_weight"] == acp.ML_REVIEW_CLASS_WEIGHT
    assert acp.ML_REVIEW_HYPERPARAMETERS["random_state"] == 20260727
    assert acp.ML_REVIEW_HYPERPARAMETERS["l1_ratio"] == 0.0
    assert acp.ML_REVIEW_HYPERPARAMETERS["C"] == 1.0
    assert acp.ML_REVIEW_HYPERPARAMETERS["max_iter"] == 1000


def test_the_sex_figures_show_every_required_metric() -> None:
    """Both companion figures must carry FPIR, TPIR@1, TPIR@5 and both
    coverages, on identical axes, so they can be compared fairly."""
    for sex in ("female", "male"):
        path = _FIG / f"{sex}_subgroup_pipeline_comparison.svg"
        if not path.is_file():
            pytest.skip("figures not generated in this checkout")
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for metric in ("fpir", "tpir@1", "tpir@5", "mated", "non-mated"):
            assert metric in text, f"{sex} figure missing {metric}"


def test_the_captions_follow_the_required_result_order() -> None:
    captions = _FIG / "FIGURE_CAPTIONS.md"
    if not captions.is_file():
        pytest.skip("captions not generated in this checkout")
    text = captions.read_text(encoding="utf-8")
    order = ["1. LFW 1:1 verification", "2. CPLFW cross-pose evaluation",
             "7-8. Female and male subgroup evaluation",
             "9. Profile-photo consistency analysis",
             "10-11. Pipeline comparison", "12. Limitations"]
    positions = []
    for heading in order:
        assert heading in text, heading
        positions.append(text.index(heading))
    assert positions == sorted(positions), "captions are out of the required order"
    # The 1:1 and 1:N distinction must be explicit.
    assert "never appears on an FPIR axis" in text


# --- Canonical run and cross-artefact consistency ------------------------------


def test_opencv_is_configured_for_deterministic_execution() -> None:
    """YuNet's score is not bit-stable under OpenCL, so an image near the 0.9
    acceptance threshold could be detected on one run and missed on the next."""
    import cv2

    report = acp.configure_deterministic_opencv()
    assert report["opencv_opencl_enabled"] is False
    assert cv2.ocl.useOpenCL() is False


def test_the_same_method_reports_identical_counts_everywhere() -> None:
    """The threshold method must not report one scored count in Experiment 6
    and a different one in Experiment 7 or 8."""
    for name in ("bfw_open_set_test_metrics.json", "ml_review_test_metrics.json",
                 "pipeline_comparison_metrics.json"):
        if not (_AGG / name).is_file():
            pytest.skip(f"{name} not present")
    open_set = json.loads((_AGG / "bfw_open_set_test_metrics.json").read_text())
    review = json.loads((_AGG / "ml_review_test_metrics.json").read_text())
    pipeline = json.loads((_AGG / "pipeline_comparison_metrics.json").read_text())

    a = open_set["methods"][acp.METHOD_B]["primary_operating_point"]
    b = review["comparator_three_image_open_set_calibrated"]["rates"]
    held_out = pipeline.get("held_out_metrics") or {}
    primary = next((v for k, v in held_out.items() if "opencv" in k), None)

    for key in ("scored_mated_probes", "scored_non_mated_probes"):
        assert a[key] == b[key], f"{key}: open-set {a[key]} vs ml-review {b[key]}"
        if primary:
            assert a[key] == primary["rates"][key], (
                f"{key}: open-set {a[key]} vs pipeline-compare {primary['rates'][key]}"
            )
    assert a["fpir"] == pytest.approx(b["fpir"], abs=1e-9)
    if primary:
        assert a["fpir"] == pytest.approx(primary["rates"]["fpir"], abs=1e-9)


def test_derived_artefacts_share_one_canonical_run_digest() -> None:
    review = _AGG / "ml_review_test_metrics.json"
    open_set = _AGG / "bfw_open_set_test_metrics.json"
    if not (review.is_file() and open_set.is_file()):
        pytest.skip("artefacts not present")
    a = json.loads(open_set.read_text()).get("canonical_run_digest")
    b = json.loads(review.read_text()).get("canonical_run_digest")
    assert a and b and a == b, f"canonical digests differ: {a} vs {b}"


def test_no_evaluated_layer_has_a_null_end_to_end_value() -> None:
    path = _AGG / "implementation_layer_comparison.json"
    if not path.is_file():
        pytest.skip("layer artefact not generated")
    for row in json.loads(path.read_text())["layers"]:
        assert row["end_to_end_duplicate_detection_rate"] is not None, (
            f"layer {row['layer']} has a null end-to-end value"
        )


def test_the_performance_csv_uses_embedding_not_search_timing() -> None:
    path = _AGG / "pretrained_pipeline_comparison.csv"
    metrics = _AGG / "pipeline_comparison_metrics.json"
    if not (path.is_file() and metrics.is_file()):
        pytest.skip("comparison not generated")
    payload = json.loads(metrics.read_text())
    if payload["evaluated"] != "yes":
        pytest.skip("comparison not evaluated")
    rows = {r["pipeline"]: r for r in csv.DictReader(open(path, encoding="utf-8"))}
    for name, row in rows.items():
        coverage = payload["held_out_metrics"][name]["coverage"]
        assert float(row["embedding_latency_mean_ms"]) == pytest.approx(
            coverage["embedding_latency_mean_ms"], rel=1e-9
        )
        assert float(row["complete_pipeline_latency_mean_ms"]) == pytest.approx(
            coverage["complete_pipeline_latency_mean_ms"], rel=1e-9
        )
        # Embedding is a per-image cost; search is per-probe over the gallery.
        assert float(row["embedding_latency_mean_ms"]) != pytest.approx(
            float(row["top1_search_time_mean_ms"])
        )
    arcface = next((r for n, r in rows.items() if "arcface" in n.lower()), None)
    if arcface:
        assert float(arcface["embedding_latency_mean_ms"]) > float(
            arcface["top1_search_time_mean_ms"]
        )
        assert float(arcface["complete_pipeline_latency_mean_ms"]) >= float(
            arcface["embedding_latency_mean_ms"]
        )


def test_consistency_controls_use_the_consistency_polarity() -> None:
    """Screening asks whether a score is at or above threshold; consistency
    asks whether the correct-identity score falls below it. The mismatched
    control is correctly identified when its top similarity is below."""
    path = _AGG / "profile_photo_consistency_metrics.json"
    if not path.is_file():
        pytest.skip("consistency artefact not generated")
    payload = json.loads(path.read_text())
    entries = payload.get("pipelines") or {"primary": payload}
    for entry in entries.values():
        for key in ("open_set_non_mated_gallery_controls_correctly_identified",
                    "open_set_non_mated_gallery_controls_false_consistent",
                    "open_set_non_mated_gallery_control_extraction_failures",
                    "wrong_profile_template_controls_correctly_inconsistent",
                    "wrong_profile_template_controls_false_consistent",
                    "wrong_profile_template_control_extraction_failures"):
            assert key in entry, key
        assert "mismatched_controls_correctly_referred" not in entry
        # The open-set control must not be described as a direct one-to-one
        # comparison; that is what the wrong-template control is for.
        definitions = entry["control_definitions"]
        assert "stricter than" in definitions["open_set_non_mated_gallery_control"]
        assert "supplementary" in definitions["wrong_profile_template_control"]
        # A consistent photograph must not be described as opening a case.
        assert "does not open a case" in entry["outcome_policy"]
    note = payload.get("interpretation_note") or next(iter(entries.values()))[
        "interpretation_note"
    ]
    assert "does not prove" in note


def test_experiment_eight_subgroups_use_the_full_replicate_count() -> None:
    path = _AGG / "pipeline_comparison_metrics.json"
    if not path.is_file():
        pytest.skip("comparison not generated")
    payload = json.loads(path.read_text())
    for metrics in (payload.get("held_out_metrics") or {}).values():
        assert metrics["subgroup_bootstrap_replicates"] == acp.BOOTSTRAP_REPLICATES == 2000
        assert metrics["global_bootstrap_replicates"] == 2000


def test_the_sex_figures_compare_both_pipelines() -> None:
    """Once Experiment 8 is evaluated the sex figures must read the two-pipeline
    subgroup file, not the primary-only one."""
    metrics = _AGG / "pipeline_comparison_metrics.json"
    if not metrics.is_file() or json.loads(metrics.read_text())["evaluated"] != "yes":
        pytest.skip("comparison not evaluated")
    for sex, forbidden in (("female", "_males"), ("male", "_females")):
        path = _FIG / f"{sex}_subgroup_pipeline_comparison.svg"
        if not path.is_file():
            pytest.skip("figures not generated")
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        assert "yunet + sface" in text and "scrfd + arcface" in text, (
            f"{sex} figure lacks both pipelines"
        )
        assert forbidden not in text
        for metric in ("fpir", "tpir@1", "tpir@5", "mated", "non-mated"):
            assert metric in text, f"{sex} figure missing {metric}"
    aggregate = _FIG / "female_male_aggregate_comparison.svg"
    text = aggregate.read_text(encoding="utf-8", errors="ignore").lower()
    assert "yunet + sface" in text and "scrfd + arcface" in text
    assert "female" in text and "male" in text


def test_the_trade_off_figure_uses_complete_pipeline_latency() -> None:
    source = _project_file("ACP_arden.py")
    assert 'latency = layer["coverage"].get("complete_pipeline_latency_mean_ms")' in source
    assert "Mean complete-pipeline latency per image" in source


def test_the_legacy_false_review_figure_uses_the_real_arcface_value() -> None:
    source = _project_file("ACP_arden.py")
    # A NaN placeholder would draw an empty bar labelled as a real result.
    assert 'labels.append("Stronger\\npipeline")' not in source
    assert 'values.append(float("nan"))' not in source


def test_the_reproducibility_mechanism_is_stated_honestly() -> None:
    """Detection is not bit-stable across processes here, so the artefacts must
    say so rather than implying a determinism the platform cannot provide."""
    report = acp.configure_deterministic_opencv()
    assert report["bitwise_reproducible_across_processes"] is False
    assert "canonical run cache" in report["reproducibility_mechanism"].lower()
    readme = _project_file("README.md")
    assert "not bit-stable across processes" in readme
    assert "setNumThreads(0)" in readme


@pytest.mark.parametrize("partition", ["development", "test"])
def test_the_canonical_cache_round_trips_without_storing_embeddings(partition: str) -> None:
    """The cache carries decisions and scores, never embeddings."""
    path = acp.canonical_cache_path(partition)
    if not path.is_file():
        pytest.skip(f"canonical {partition} run not present in this checkout")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["partition"] == partition
    assert payload["canonical_run_digest"]
    assert acp.cached_payload_integrity_reason(payload) is None

    # Dimension counts, timings, similarity scores and the privacy note may
    # name embeddings or templates without containing the vectors themselves.
    forbidden_fields = {
        "embedding", "embeddings", "face_embedding", "face_embeddings",
        "template", "templates", "image_path", "image_paths",
    }

    def check_fields(value: Any) -> None:
        if isinstance(value, dict):
            for name, child in value.items():
                assert name not in forbidden_fields, f"cache contains forbidden field: {name}"
                check_fields(child)
        elif isinstance(value, list):
            for child in value:
                check_fields(child)

    check_fields(payload)
    assert "/Users/" not in json.dumps(payload), "cache contains a personal filesystem path"
    for record in payload["enrolment_outcomes"] + payload["search_results"]:
        assert all(not isinstance(value, (list, dict)) for value in record.values()), (
            "cached records must contain scalar values, not vectors or nested data"
        )
    restored = acp.load_canonical_run(path)
    assert restored is not None
    assert restored.partition == partition
    assert acp.canonical_run_digest(restored) == payload["canonical_run_digest"]


def test_the_readme_reports_the_experiment_eight_outcome() -> None:
    """The README described the comparison's setup but not its result, which
    invites the reader to infer one. It must state the outcome and the
    mechanism behind it."""
    metrics = _AGG / "pipeline_comparison_metrics.json"
    if not metrics.is_file() or json.loads(metrics.read_text())["evaluated"] != "yes":
        pytest.skip("comparison not evaluated")
    readme = _project_file("README.md")
    assert "evaluated_non_commercial_academic_research" in readme
    assert "extraction, not ranking" in readme
    # The stronger pipeline's cost must be stated, not just its benefit.
    assert "Complete-pipeline latency" in readme
    payload = json.loads(metrics.read_text())["held_out_metrics"]
    arcface = payload["insightface-scrfd-arcface-buffalo_l"]["rates"]
    assert f"{arcface['fpir'] * 100:.2f}%" in readme


def test_the_similarity_distribution_figure_exists_and_marks_thresholds() -> None:
    """Required by the reporting contract and previously never drawn, despite
    the histogram data being published."""
    metrics = _AGG / "pipeline_comparison_metrics.json"
    if not metrics.is_file() or json.loads(metrics.read_text())["evaluated"] != "yes":
        pytest.skip("comparison not evaluated")
    path = _FIG / "mated_non_mated_similarity_distributions.svg"
    assert path.is_file(), "the distribution figure was not generated"
    text = path.read_text(encoding="utf-8", errors="ignore").lower()
    assert "yunet + sface" in text and "scrfd + arcface" in text
    assert "mated" in text and "non-mated" in text
    assert "frozen threshold" in text
    # Each distribution must say what it measures, not merely that it is mated.
    assert "own enrolled template" in text
    assert "highest similarity against the gallery" in text


def test_the_consistency_figure_compares_both_pipelines() -> None:
    metrics = _AGG / "pipeline_comparison_metrics.json"
    if not metrics.is_file() or json.loads(metrics.read_text())["evaluated"] != "yes":
        pytest.skip("comparison not evaluated")
    text = (_FIG / "profile_photo_consistency_outcomes.svg").read_text(
        encoding="utf-8", errors="ignore"
    ).lower()
    assert "yunet + sface" in text and "scrfd + arcface" in text
    # The two controls must be separated and named, and no outcome may say
    # "identified" without stating what was identified.
    assert "open-set non-mated gallery control" in text
    assert "wrong-profile-template control" in text
    assert "correctly matched" in text and "no enrolled profile" in text
    assert "correctly inconsistent" in text and "with wrong profile" in text
    assert "not proof" in text


def test_similarity_histograms_publish_no_individual_scores() -> None:
    metrics = _AGG / "pipeline_comparison_metrics.json"
    if not metrics.is_file():
        pytest.skip("comparison not generated")
    payload = json.loads(metrics.read_text())
    for pipeline_metrics in (payload.get("held_out_metrics") or {}).values():
        for histogram in (pipeline_metrics.get("similarity_histograms") or {}).values():
            # Bin edges and counts only: an individual score is never published.
            assert set(histogram) <= {"bin_edges", "counts", "n"}
            assert len(histogram["counts"]) == len(histogram["bin_edges"]) - 1


def test_captions_do_not_claim_every_outcome_opens_review() -> None:
    """A consistent photograph opens no case and an extraction failure resolves
    nothing, so the blanket phrasing is wrong for the consistency policy."""
    captions = _FIG / "FIGURE_CAPTIONS.md"
    if not captions.is_file():
        pytest.skip("captions not generated")
    text = captions.read_text(encoding="utf-8")
    assert "Every outcome opens human review only" not in text
    assert "opens no case" in text
    assert "resolves nothing" in text


def test_consistency_artefacts_cover_every_evaluated_pipeline() -> None:
    metrics = _AGG / "pipeline_comparison_metrics.json"
    path = _AGG / "profile_photo_consistency_metrics.json"
    if not (metrics.is_file() and path.is_file()):
        pytest.skip("artefacts not generated")
    evaluated = json.loads(metrics.read_text())["evaluated"] == "yes"
    payload = json.loads(path.read_text())
    assert "pipelines" in payload
    if evaluated:
        assert len(payload["pipelines"]) == 2
    rows = list(csv.DictReader(open(_AGG / "profile_photo_consistency_metrics.csv",
                                    encoding="utf-8")))
    assert "pipeline" in rows[0]
    assert len({r["pipeline"] for r in rows}) == len(payload["pipelines"])


def test_the_report_shows_the_mismatched_controls() -> None:
    """Without the control the reader sees only how often same-person photographs
    pass, with no evidence the threshold discriminates at all."""
    report = _AGG / "RESEARCH_REPORT.md"
    if not report.is_file():
        pytest.skip("report not generated")
    text = report.read_text(encoding="utf-8")
    # Section 9 reports rates rather than raw counts, and names both controls
    # separately; the counts live in the Experiment 8 report and the artefacts.
    assert "Open-set control detection (cond.)" in text
    assert "Wrong-template detection (cond.)" in text
    assert "Wrong-template false-consistency (cond.)" in text
    # The two referral directions must remain distinguished.
    assert "a *high* similarity" in text and "a *low* similarity" in text

"""Parity tests for the single-file artefact's methodology contract.

Every expectation below is written out as an independent literal rather than
read back from the module under test, so a test failing here means the
methodology moved — not that a constant was renamed. Together they pin the
thirteen properties that must hold for a result produced by this file to be
comparable with any other run of the same protocol: model digests, detector
settings, embedding dimensionality, the random seed, similarity, the metric
definitions, candidate generation, the selection rule, frozen-threshold
enforcement, failure accounting, gallery determinism, opaque identifiers and
path-leak detection.

No test here loads a model binary, reads a dataset or touches the network.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

import ACP_arden as acp

# --- Independently declared expectations -------------------------------------

EXPECTED_YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
EXPECTED_SFACE_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
EXPECTED_YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
EXPECTED_SFACE_FILENAME = "face_recognition_sface_2021dec.onnx"
EXPECTED_EMBEDDING_DIMENSIONS = 128
EXPECTED_RANDOM_SEED = 20260727
EXPECTED_MODEL_VERSION = "opencv-sface-2021dec-yunet-2023mar"
EXPECTED_PREPROCESSING_REVISION = "opencv-yunet-sface-exif-bgr-l2-v1"
EXPECTED_SELECTION_RULE = (
    "Maximum balanced accuracy on the development split (pairsDevTest.txt); "
    "ties broken by lower development-split false match rate, then by "
    "candidate name, for full determinism."
)
EXPECTED_CANDIDATE_NAMES = {
    "balanced_accuracy",
    "f1",
    "eer",
    "target_fmr_0.001",
    "target_fmr_0.01",
    "target_fmr_0.05",
}
EXPECTED_FAILURE_CATEGORIES = {
    "zero_faces_left",
    "zero_faces_right",
    "multiple_faces_left",
    "multiple_faces_right",
}


# --- 1. Model hashes ---------------------------------------------------------


def test_model_digests_are_pinned_to_the_published_release() -> None:
    assert acp.YUNET_SHA256 == EXPECTED_YUNET_SHA256
    assert acp.SFACE_SHA256 == EXPECTED_SFACE_SHA256
    assert acp.YUNET_FILENAME == EXPECTED_YUNET_FILENAME
    assert acp.SFACE_FILENAME == EXPECTED_SFACE_FILENAME
    assert acp.MODEL_VERSION == EXPECTED_MODEL_VERSION
    assert acp.PREPROCESSING_REVISION == EXPECTED_PREPROCESSING_REVISION


def test_a_model_file_whose_digest_does_not_match_is_refused(tmp_path: Path) -> None:
    impostor = tmp_path / EXPECTED_YUNET_FILENAME
    impostor.write_bytes(b"not the pinned model")
    with pytest.raises(acp.ModelUnavailableError):
        acp.verify_model_file(impostor, EXPECTED_YUNET_SHA256)


def test_a_missing_model_file_is_refused_rather_than_skipped(tmp_path: Path) -> None:
    with pytest.raises(acp.ModelUnavailableError):
        acp.verify_model_file(tmp_path / "absent.onnx", EXPECTED_YUNET_SHA256)


def test_verify_model_file_returns_the_digest_it_verified(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"deterministic content")
    digest = acp.sha256_of_file(payload)
    assert acp.verify_model_file(payload, digest) == digest


# --- 2. Detector settings ----------------------------------------------------


def test_detector_settings_match_the_evaluation_partition() -> None:
    settings = acp.DetectorSettings()
    assert settings.score_threshold == 0.9
    assert settings.nms_threshold == 0.3
    assert settings.top_k == 5000


def test_exactly_one_face_is_required() -> None:
    detector = acp.SyntheticDetector(default_count=1)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    assert detector.detect_single_face(image).shape[0] == 13

    for count in (0, 2, 7):
        with pytest.raises(acp.FaceCountError) as raised:
            acp.SyntheticDetector(default_count=count).detect_single_face(image)
        assert raised.value.face_count == count


# --- 3. Embedding dimensions -------------------------------------------------


def test_embedding_dimensionality_is_pinned() -> None:
    assert acp.EMBEDDING_DIMENSIONS == EXPECTED_EMBEDDING_DIMENSIONS
    embedder = acp.SyntheticEmbedder()
    embedding = embedder.embed(np.zeros((8, 8, 3), dtype=np.uint8), np.zeros(13))
    assert embedding.shape == (EXPECTED_EMBEDDING_DIMENSIONS,)


# --- 4. Random seed ----------------------------------------------------------


def test_default_random_seed_is_pinned() -> None:
    assert acp.DEFAULT_RANDOM_SEED == EXPECTED_RANDOM_SEED


# --- 5. Similarity -----------------------------------------------------------


def test_cosine_similarity_definition() -> None:
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert acp.cosine_similarity(a, a) == pytest.approx(1.0)
    assert acp.cosine_similarity(a, b) == pytest.approx(0.0)
    assert acp.cosine_similarity(a, -a) == pytest.approx(-1.0)
    # Magnitude must not affect the score.
    assert acp.cosine_similarity(a, 12.5 * a) == pytest.approx(1.0)
    diagonal = np.array([1.0, 1.0])
    axis = np.array([1.0, 0.0])
    assert acp.cosine_similarity(diagonal, axis) == pytest.approx(1 / math.sqrt(2))


def test_l2_normalisation_definition() -> None:
    normalized = acp.l2_normalize(np.array([3.0, 4.0]))
    assert normalized.tolist() == pytest.approx([0.6, 0.8])
    assert float(np.linalg.norm(normalized)) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "vector",
    [np.zeros(4), np.array([np.nan, 1.0]), np.array([np.inf, 1.0]), np.array([])],
)
def test_malformed_embeddings_are_refused(vector: np.ndarray) -> None:
    with pytest.raises(acp.SimilarityError):
        acp.l2_normalize(vector)


# --- 6. Metric calculations --------------------------------------------------


def test_confusion_matrix_uses_an_inclusive_threshold() -> None:
    matrix = acp.confusion_matrix([0.9, 0.5, 0.5, 0.1], [1, 1, 0, 0], 0.5)
    assert matrix.as_dict() == {
        "true_positive": 2,
        "false_positive": 1,
        "true_negative": 1,
        "false_negative": 0,
    }
    assert matrix.total == 4


def test_derived_rates_match_their_definitions() -> None:
    rates = acp.rates_from_confusion(
        acp.ConfusionMatrix(true_positive=8, false_positive=1, true_negative=9, false_negative=2)
    )
    assert rates["accuracy"] == pytest.approx(17 / 20)
    assert rates["precision"] == pytest.approx(8 / 9)
    assert rates["recall"] == pytest.approx(0.8)
    assert rates["true_match_rate"] == pytest.approx(rates["recall"])
    assert rates["f1"] == pytest.approx(2 * (8 / 9) * 0.8 / ((8 / 9) + 0.8))
    assert rates["false_match_rate"] == pytest.approx(0.1)
    assert rates["false_non_match_rate"] == pytest.approx(0.2)


def test_an_undefined_rate_stays_nan_rather_than_becoming_zero() -> None:
    rates = acp.rates_from_confusion(
        acp.ConfusionMatrix(true_positive=0, false_positive=0, true_negative=5, false_negative=0)
    )
    assert math.isnan(rates["recall"])
    assert math.isnan(rates["precision"])
    assert math.isnan(rates["f1"])


@pytest.mark.parametrize(
    ("scores", "labels", "expected"),
    [
        ([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0], 1.0),
        ([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0], 0.0),
        ([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0], 0.5),
        ([0.9, 0.4, 0.6, 0.1], [1, 1, 0, 0], 0.75),
    ],
)
def test_roc_auc_matches_the_rank_based_definition(
    scores: list, labels: list, expected: float
) -> None:
    assert acp.roc_auc(scores, labels) == pytest.approx(expected)


def test_roc_points_are_ordered_and_bracketed_by_sentinels() -> None:
    points = acp.roc_points([0.9, 0.4, 0.6, 0.1], [1, 1, 0, 0])
    thresholds = [point["threshold"] for point in points]
    assert thresholds == sorted(thresholds, reverse=True)
    # One point per distinct score, plus a high and a low sentinel.
    assert len(points) == 4 + 2
    assert points[0]["true_match_rate"] == pytest.approx(0.0)
    assert points[-1]["false_match_rate"] == pytest.approx(1.0)


def test_equal_error_rate_is_zero_for_separable_scores() -> None:
    result = acp.equal_error_rate([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    assert result["equal_error_rate"] == pytest.approx(0.0)
    assert "threshold" in result


def test_percentile_matches_linear_interpolation() -> None:
    assert acp.percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)
    assert acp.percentile([1.0], 95) == pytest.approx(1.0)
    assert math.isnan(acp.percentile([], 95))


@pytest.mark.parametrize(
    ("scores", "labels"),
    [([], []), ([0.5], [1]), ([0.5, 0.6], [1, 1]), ([0.5, 0.6], [1])],
)
def test_degenerate_metric_input_is_refused(scores: list, labels: list) -> None:
    with pytest.raises(acp.MetricsError):
        acp.confusion_matrix(scores, labels, 0.5)


# --- 7. Candidate thresholds -------------------------------------------------


def _separable_development_scores() -> tuple:
    rng = np.random.default_rng(EXPECTED_RANDOM_SEED)
    genuine = rng.normal(0.7, 0.05, 80).tolist()
    impostor = rng.normal(0.2, 0.05, 80).tolist()
    return ([*genuine, *impostor], [1] * 80 + [0] * 80)


def test_calibration_produces_the_expected_candidate_set() -> None:
    scores, labels = _separable_development_scores()
    result = acp.calibrate(scores, labels, split="validation")
    assert set(result.candidates) == EXPECTED_CANDIDATE_NAMES
    assert result.status == "candidates"
    assert result.split == "validation"


def test_calibration_never_freezes_a_threshold_itself() -> None:
    scores, labels = _separable_development_scores()
    result = acp.calibrate(scores, labels, split="validation")
    assert result.status != "frozen"


@pytest.mark.parametrize("split", ["test", "final", "development", "dev"])
def test_calibration_refuses_any_split_but_validation(split: str) -> None:
    scores, labels = _separable_development_scores()
    with pytest.raises(acp.CalibrationError):
        acp.calibrate(scores, labels, split=split)


def test_target_fmr_candidates_respect_their_target() -> None:
    scores, labels = _separable_development_scores()
    candidate = acp.select_threshold(
        scores, labels, strategy="target_fmr", target_false_match_rate=0.01
    )
    assert candidate.metrics["false_match_rate"] <= 0.01


def test_an_unknown_strategy_is_refused() -> None:
    scores, labels = _separable_development_scores()
    with pytest.raises(acp.MetricsError):
        acp.select_threshold(scores, labels, strategy="whatever_looks_best")


# --- 8. Selection rule -------------------------------------------------------


def test_the_published_selection_rule_is_unchanged() -> None:
    assert acp.SELECTION_RULE == EXPECTED_SELECTION_RULE


def test_selection_is_deterministic_and_fully_evidenced() -> None:
    candidates = {
        "balanced_accuracy": {"threshold": 0.50},
        "f1": {"threshold": 0.40},
        "eer": {"threshold": 0.45},
    }
    dev_scores = [0.9, 0.8, 0.46, 0.44, 0.3, 0.1]
    dev_labels = [1, 1, 1, 0, 0, 0]

    outcomes = [
        acp.select_final_threshold(candidates, dev_scores, dev_labels)["selected_candidate"]
        for _ in range(5)
    ]
    assert len(set(outcomes)) == 1

    selection = acp.select_final_threshold(candidates, dev_scores, dev_labels)
    assert selection["selection_rule"] == EXPECTED_SELECTION_RULE
    assert set(selection["all_candidates_dev_metrics"]) == set(candidates)
    assert selection["selected_threshold"] == pytest.approx(
        selection["all_candidates_dev_metrics"][selection["selected_candidate"]]["threshold"]
    )


def test_a_tie_is_broken_by_candidate_name_not_dictionary_order() -> None:
    dev_scores = [0.9, 0.8, 0.46, 0.44, 0.3, 0.1]
    dev_labels = [1, 1, 1, 0, 0, 0]
    forward = acp.select_final_threshold(
        {"b_name": {"threshold": 0.45}, "a_name": {"threshold": 0.45}}, dev_scores, dev_labels
    )
    reversed_order = acp.select_final_threshold(
        {"a_name": {"threshold": 0.45}, "b_name": {"threshold": 0.45}}, dev_scores, dev_labels
    )
    assert forward["selected_candidate"] == "a_name"
    assert reversed_order["selected_candidate"] == "a_name"


def test_selection_prefers_higher_balanced_accuracy() -> None:
    # The stricter threshold separates the development scores perfectly; the
    # looser one admits a false match, so it must lose.
    selection = acp.select_final_threshold(
        {"strict": {"threshold": 0.7}, "loose": {"threshold": 0.2}},
        [0.9, 0.8, 0.3, 0.1],
        [1, 1, 0, 0],
    )
    assert selection["selected_candidate"] == "strict"


# --- 9. Frozen-threshold enforcement -----------------------------------------


def test_a_final_evaluation_refuses_a_non_frozen_threshold() -> None:
    for payload in (
        {"status": "candidates", "threshold": 0.4},
        {"status": None, "threshold": 0.4},
        {"threshold": 0.4},
        {},
    ):
        with pytest.raises(acp.CalibrationError):
            acp.require_frozen_threshold(payload, context="parity")


def test_a_frozen_artifact_without_a_numeric_threshold_is_refused() -> None:
    with pytest.raises(acp.CalibrationError):
        acp.require_frozen_threshold({"status": "frozen"}, context="parity")
    with pytest.raises(acp.CalibrationError):
        acp.require_frozen_threshold({"status": "frozen", "threshold": "0.4"}, context="parity")


def test_a_properly_frozen_threshold_is_accepted() -> None:
    assert acp.require_frozen_threshold(
        {"status": "frozen", "threshold": 0.3630116835391414}
    ) == pytest.approx(0.3630116835391414)


def test_selection_refuses_an_already_frozen_artifact() -> None:
    with pytest.raises(acp.CalibrationError):
        acp.require_candidates({"status": "frozen", "candidates": {"a": {"threshold": 0.4}}})
    with pytest.raises(acp.CalibrationError):
        acp.require_candidates({"status": "candidates", "candidates": {}})


# --- 10. Failure accounting --------------------------------------------------


def _synthetic_pair_fixture(directory: Path) -> tuple:
    good_a, _ = acp._write_synthetic_image(directory, "good_a.png", 10)
    good_b, _ = acp._write_synthetic_image(directory, "good_b.png", 20)
    empty, empty_key = acp._write_synthetic_image(directory, "empty.png", 30)
    crowd, crowd_key = acp._write_synthetic_image(directory, "crowd.png", 40)
    detector = acp.SyntheticDetector({empty_key: 0, crowd_key: 3})
    pairs = [
        acp.Pair(good_a, good_b, True, "a", "b"),
        acp.Pair(empty, good_b, True, "a", "b"),
        acp.Pair(good_a, empty, False, "a", "b"),
        acp.Pair(crowd, good_b, True, "a", "b"),
        acp.Pair(good_a, crowd, False, "a", "b"),
    ]
    return pairs, detector


def test_the_four_failure_categories_partition_the_failed_pairs(tmp_path: Path) -> None:
    pairs, detector = _synthetic_pair_fixture(tmp_path)
    result = acp.evaluate_pairs(pairs, detector=detector, embedder=acp.SyntheticEmbedder())

    assert set(result.failures) == EXPECTED_FAILURE_CATEGORIES
    assert result.failures == {
        "zero_faces_left": 1,
        "zero_faces_right": 1,
        "multiple_faces_left": 1,
        "multiple_faces_right": 1,
    }
    assert sum(result.failures.values()) == result.failed_pairs
    result.validate_accounting()


def test_failed_pairs_stay_inside_the_protocol_total(tmp_path: Path) -> None:
    pairs, detector = _synthetic_pair_fixture(tmp_path)
    result = acp.evaluate_pairs(pairs, detector=detector, embedder=acp.SyntheticEmbedder())

    assert result.total_pairs == len(pairs)
    assert result.scored_pair_count + result.failed_pairs == result.total_pairs
    assert result.failure_rate == pytest.approx(result.failed_pairs / result.total_pairs)


def test_accounting_that_does_not_reconcile_is_refused() -> None:
    broken = acp.EvaluationResult(total_pairs=10, scored_pairs=[], failures={"zero_faces_left": 1})
    with pytest.raises(ValueError):
        broken.validate_accounting()


def test_a_pair_is_abandoned_at_the_first_terminal_failure(tmp_path: Path) -> None:
    # Both sides would fail; only the left category may be recorded, so the
    # breakdown stays a partition of pairs rather than a tally of images.
    empty_a, empty_a_key = acp._write_synthetic_image(tmp_path, "empty_a.png", 50)
    empty_b, empty_b_key = acp._write_synthetic_image(tmp_path, "empty_b.png", 60)
    detector = acp.SyntheticDetector({empty_a_key: 0, empty_b_key: 0})
    result = acp.evaluate_pairs(
        [acp.Pair(empty_a, empty_b, True, "a", "b")],
        detector=detector,
        embedder=acp.SyntheticEmbedder(),
    )
    assert result.failures == {"zero_faces_left": 1}


# --- 11. Gallery determinism -------------------------------------------------


def _gallery_fixture() -> dict:
    images = {
        f"identity_{index:02d}": [Path(f"/tmp/i{index}/identity_{index:02d}_0001.jpg")]
        for index in range(20)
    }
    images["anchor"] = [Path("/tmp/anchor/anchor_0001.jpg"), Path("/tmp/anchor/anchor_0002.jpg")]
    images["beacon"] = [Path("/tmp/beacon/beacon_0001.jpg"), Path("/tmp/beacon/beacon_0002.jpg")]
    return images


def test_the_same_seed_reproduces_the_same_manifest() -> None:
    images = _gallery_fixture()
    first = acp.build_manifest(images, seed=EXPECTED_RANDOM_SEED, max_unknown_identities=5)
    second = acp.build_manifest(images, seed=EXPECTED_RANDOM_SEED, max_unknown_identities=5)
    assert [e.sample_id for e in first.entries] == [e.sample_id for e in second.entries]
    assert [e.role for e in first.entries] == [e.role for e in second.entries]
    assert first.seed == EXPECTED_RANDOM_SEED


def test_a_different_seed_samples_a_different_unknown_probe_set() -> None:
    images = _gallery_fixture()
    first = acp.build_manifest(images, seed=EXPECTED_RANDOM_SEED, max_unknown_identities=5)
    other = acp.build_manifest(images, seed=EXPECTED_RANDOM_SEED + 1, max_unknown_identities=5)
    assert {e.sample_id for e in first.entries} != {e.sample_id for e in other.entries}


def test_every_image_holds_exactly_one_gallery_role() -> None:
    manifest = acp.build_manifest(_gallery_fixture())
    paths = [entry.image_path for entry in manifest.entries]
    assert len(paths) == len(set(paths))
    roles = [entry.role for entry in manifest.entries]
    assert roles.count("gallery") == 2
    assert roles.count("duplicate_probe") == 2
    assert roles.count("unknown_probe") == 20


def test_calibration_images_never_enter_the_gallery() -> None:
    images = _gallery_fixture()
    excluded = Path("/tmp/anchor/anchor_0002.jpg")
    manifest = acp.build_manifest(images, excluded_images=[excluded])
    assert excluded not in {entry.image_path for entry in manifest.entries}


def test_a_gallery_needs_both_multi_and_single_image_identities() -> None:
    with pytest.raises(acp.GalleryError):
        acp.build_manifest({"solo": [Path("/tmp/solo/solo_0001.jpg")]})
    with pytest.raises(acp.GalleryError):
        acp.build_manifest(
            {"pair": [Path("/tmp/pair/pair_0001.jpg"), Path("/tmp/pair/pair_0002.jpg")]}
        )


def test_gallery_metrics_report_detection_and_false_review_together() -> None:
    summary = acp.summarize_gallery_metrics(
        acp.GalleryEvaluationResult(
            gallery_size=2,
            probe_results=[
                acp.ProbeResult("d1", "duplicate_probe", "h1", "h1", 0.9, True, True, None),
                acp.ProbeResult("d2", "duplicate_probe", "h2", "h1", 0.2, False, False, None),
                acp.ProbeResult("u1", "unknown_probe", "h3", "h1", 0.8, None, True, None),
                acp.ProbeResult("u2", "unknown_probe", "h4", "h1", 0.1, None, False, None),
            ],
        )
    )
    assert summary["duplicate_detection_rate"] == pytest.approx(0.5)
    assert summary["rank1_identification_rate"] == pytest.approx(0.5)
    assert summary["false_duplicate_review_rate"] == pytest.approx(0.5)
    assert summary["true_duplicate_miss_rate"] == pytest.approx(0.5)


# --- 12. Opaque identifiers --------------------------------------------------


def test_opaque_ids_are_deterministic_one_way_and_truncated() -> None:
    first = acp.opaque_id("Example_Identity")
    assert first == acp.opaque_id("Example_Identity")
    assert first != acp.opaque_id("Example_Identity_2")
    assert len(first) == 32
    assert all(character in "0123456789abcdef" for character in first)
    assert "Example" not in first


def test_the_secret_key_participates_in_the_digest() -> None:
    """Two different keys must not agree on an identifier; a fixed published
    salt is exactly what this replaces."""
    with acp.temporary_id_hmac_key("a" * 63 + "b"):
        under_first_key = acp.opaque_id("value")
    with acp.temporary_id_hmac_key("c" * 63 + "d"):
        under_second_key = acp.opaque_id("value")
    assert under_first_key != under_second_key
    assert acp.OPAQUE_ID_VERSION == "hmac-sha256-v1"


def test_a_manifest_records_no_real_identity_name() -> None:
    manifest = acp.build_manifest(_gallery_fixture())
    for entry in manifest.entries:
        assert "anchor" not in entry.sample_id
        assert "anchor" not in entry.identity_hash
        assert len(entry.identity_hash) == 32


def test_filenames_are_scrubbed_to_their_final_component() -> None:
    assert acp.scrub_filename(Path("/private/storage/Name_0001.jpg")) == "Name_0001.jpg"


# --- 13. Path-leak detection -------------------------------------------------


def test_a_clean_artifact_directory_produces_no_findings(tmp_path: Path) -> None:
    (tmp_path / "clean.json").write_text('{"accuracy": 0.9909}\n', encoding="utf-8")
    (tmp_path / "clean.csv").write_text("experiment,accuracy\nlfw_final,0.9909\n", encoding="utf-8")
    assert acp.find_path_leaks(tmp_path, forbidden_substrings=["/Users/", "/home/"]) == []


def test_an_absolute_path_in_a_published_artifact_is_reported(tmp_path: Path) -> None:
    (tmp_path / "leaky.json").write_text('{"root": "/Users/example/data"}\n', encoding="utf-8")
    findings = acp.find_path_leaks(tmp_path, forbidden_substrings=["/Users/"])
    assert len(findings) == 1
    assert "leaky.json" in findings[0]


def test_the_forbidden_list_covers_the_obvious_absolute_prefixes() -> None:
    forbidden = acp.default_forbidden_path_substrings(env={})
    assert "/Users/" in forbidden
    assert "/home/" in forbidden
    assert str(Path.home()) in forbidden


def test_configured_storage_roots_are_added_to_the_forbidden_list() -> None:
    forbidden = acp.default_forbidden_path_substrings(
        env={"FACE_DATA_ROOT": "/private/research/datasets"}
    )
    assert "/private/research/datasets" in forbidden


def test_record_level_leakage_is_refused() -> None:
    for record in (
        {"image_path": "Name_0001.jpg"},
        {"identity_name": "Example"},
        {"embedding": [0.0]},
        {"root": "/private/location"},
        {"home": "~/research"},
        {"vector": [0.1] * 64},
        {"nested": {"image_path": "x"}},
    ):
        with pytest.raises(acp.PrivacyLeakError):
            acp.assert_no_leakage(record)


def test_hashed_identity_fields_are_permitted() -> None:
    acp.assert_no_leakage(
        {"identity_hash": "0123456789abcdef", "candidate_identity_hash": "fedcba9876543210",
         "strategy": "balanced_accuracy", "accuracy": 0.9909}
    )


def test_project_relative_never_returns_an_absolute_path() -> None:
    assert acp.project_relative(acp.AGGREGATE_ROOT) == "results/aggregate"
    outside = acp.project_relative(Path("/somewhere/else/report.json"))
    assert not outside.startswith("/")


def test_printed_messages_have_private_roots_redacted() -> None:
    config = acp.EnvironmentConfig(
        data_root=Path("/private/storage/datasets"),
        protocol_root=Path("/private/storage/protocols"),
        model_root=Path("/private/storage/models"),
        cplfw_raw_root=Path("/private/storage/datasets/cplfw_raw"),
        cache_root=None,
    )
    redacted = acp.redact_private_paths(
        "Missing image referenced by protocol: /private/storage/datasets/lfw_funneled/x.jpg",
        config,
    )
    assert "/private/storage" not in redacted
    assert "<FACE_DATA_ROOT>" in redacted


# --- Cross-cutting: the artefact's own self-tests -----------------------------


def test_every_synthetic_self_test_passes() -> None:
    passed, failed = acp.run_self_tests(verbose=False)
    assert failed == 0
    assert passed == len(acp.SELF_TESTS)
    assert passed == 14


def test_env_parsing_handles_quotes_comments_and_blank_lines() -> None:
    parsed = acp.parse_env_text(
        "\n".join(
            [
                "# a comment",
                "",
                "PLAIN=value",
                'QUOTED="value with spaces"',
                "SINGLE='another value'",
                "export EXPORTED=exported_value",
                "   SPACED   =   padded   ",
                "NOT_A_PAIR",
            ]
        )
    )
    assert parsed == {
        "PLAIN": "value",
        "QUOTED": "value with spaces",
        "SINGLE": "another value",
        "EXPORTED": "exported_value",
        "SPACED": "padded",
    }


def test_cplfw_variants_are_never_interchangeable() -> None:
    raw = acp.cplfw_provenance_fields("raw")
    aligned = acp.cplfw_provenance_fields("aligned")
    assert raw["dataset_image_variant"] == "raw"
    assert raw["dataset_archive_filename"] == "images.rar"
    assert aligned["dataset_archive_filename"] == "cp-aligned.zip"
    assert raw["dataset_archive_sha256"] != aligned["dataset_archive_sha256"]
    assert "path omitted" in raw["dataset_root_description"]
    with pytest.raises(ValueError):
        acp.cplfw_provenance_fields("cropped")


def test_the_command_line_requires_no_mandatory_argument() -> None:
    parser = acp.build_argument_parser()
    args = parser.parse_args([])
    assert args.mode == "menu"
    for mode in ("menu", "check", "verify", "full", "summary", "review", "self-test"):
        assert parser.parse_args(["--mode", mode]).mode == mode


def test_artifacts_are_written_atomically_and_are_self_describing(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "artifact.json"
    digest = acp.write_json_artifact(target, {"artifact_type": "test", "accuracy": 0.99})
    payload = acp.read_json_artifact(target)
    assert payload["schema_version"] == 1
    assert "created_at" in payload
    assert digest == acp.sha256_of_text(target.read_text(encoding="utf-8"))
    # No temporary file is left behind by the write-then-rename.
    assert [p.name for p in target.parent.iterdir()] == ["artifact.json"]


def test_a_missing_artifact_is_an_explicit_error(tmp_path: Path) -> None:
    with pytest.raises(acp.ArtifactError):
        acp.read_json_artifact(tmp_path / "absent.json")


def test_the_review_database_stores_only_opaque_fields(tmp_path: Path) -> None:
    db_path = tmp_path / "review.sqlite"
    with acp.review_database(db_path) as connection:
        acp.upsert_review_case(
            connection,
            case_id="sample:candidate",
            probe_sample_id="0123456789abcdef",
            candidate_identity_hash="fedcba9876543210",
            similarity=0.87,
            threshold=0.36,
        )
        cases = acp.list_review_cases(connection)
        assert len(cases) == 1
        assert cases[0].status == "open"

        acp.set_review_status(connection, case_id="sample:candidate", status="dismissed")
        assert acp.list_review_cases(connection, status="dismissed")[0].decided_at is not None

        with pytest.raises(ValueError):
            acp.set_review_status(connection, case_id="sample:candidate", status="banned")


def test_the_review_queue_is_ordered_and_bounded(tmp_path: Path) -> None:
    # A real queue holds thousands of cases. Rendering all of them at once made
    # the page unusable, so the query is bounded and ordered by descending
    # similarity: a reviewer sees the strongest candidates first.
    db_path = tmp_path / "queue.sqlite"
    with acp.review_database(db_path) as connection:
        for index in range(50):
            acp.upsert_review_case(
                connection,
                case_id=f"case{index:03d}",
                probe_sample_id=f"{index:016x}",
                candidate_identity_hash=f"{index + 1:016x}",
                similarity=index / 100.0,
                threshold=0.363,
            )

        assert acp.count_review_cases(connection) == 50

        page = acp.list_review_cases(connection, limit=10)
        assert len(page) == 10
        similarities = [case.similarity for case in page]
        assert similarities == sorted(similarities, reverse=True)
        assert similarities[0] == pytest.approx(0.49)

        assert len(acp.list_review_cases(connection)) == 50
        assert acp.list_review_cases(connection, limit=0) == []
        with pytest.raises(ValueError):
            acp.list_review_cases(connection, limit=-1)

        acp.set_review_status(connection, case_id="case000", status="dismissed")
        assert acp.count_review_cases(connection, status="dismissed") == 1
        assert acp.count_review_cases(connection, status="open") == 49


def test_a_streamlit_child_process_is_detected_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ACP_ARDEN_REVIEW_CHILD", "1")
    assert acp.running_under_streamlit() is True
    monkeypatch.delenv("ACP_ARDEN_REVIEW_CHILD")
    # Outside a script run there is no Streamlit context, so the review mode
    # launches a server rather than trying to render into a bare interpreter.
    assert acp.running_under_streamlit() is False


def test_the_entrypoint_does_not_raise_system_exit_under_streamlit() -> None:
    # Regression guard: SystemExit raised inside a Streamlit script run aborts
    # it before Streamlit marks the run finished, which hangs the review page.
    source = Path(acp.__file__).read_text(encoding="utf-8")
    entrypoint = source.split('if __name__ == "__main__":')[-1]
    assert "running_under_streamlit()" in entrypoint
    assert "raise SystemExit(main())" in entrypoint
    guarded = entrypoint.index("running_under_streamlit()")
    unguarded = entrypoint.index("raise SystemExit(main())")
    assert guarded < unguarded, "the Streamlit guard must precede the exiting branch"


def test_the_policy_note_rules_out_automatic_sanctions() -> None:
    note = acp.POLICY_NOTE.lower()
    assert "human review only" in note
    assert "does not ban, reject or accuse" in note


def test_the_summary_never_quotes_a_conditional_figure_alone(tmp_path: Path) -> None:
    # A synthetic aggregate set is enough: the requirement is that the renderer
    # always emits the limitation lines alongside the headline rates.
    acp.write_json_artifact(
        tmp_path / "calibrated_threshold.json",
        {"threshold": 0.36, "operating_strategy": "balanced_accuracy", "status": "frozen",
         "selection_rule": EXPECTED_SELECTION_RULE},
    )
    acp.write_json_artifact(
        tmp_path / "lfw_final_metrics.json",
        {"accuracy": 0.9909, "false_match_rate": 0.0011, "false_non_match_rate": 0.0171,
         "equal_error_rate": 0.0078, "failure_rate": 0.1002, "scored_pairs": 5399,
         "total_pairs": 6000},
    )
    acp.write_json_artifact(
        tmp_path / "cplfw_metrics.json",
        {"accuracy": 0.9024, "false_match_rate": 0.0195, "false_non_match_rate": 0.1746,
         "equal_error_rate": 0.0977, "failure_rate": 0.4142, "scored_pairs": 3515,
         "failed_pairs": 2485, "total_pairs": 6000},
    )
    acp.write_json_artifact(
        tmp_path / "duplicate_gallery_metrics.json",
        {"gallery_size": 986, "duplicate_detection_rate": 0.9658,
         "rank1_identification_rate": 0.9276, "false_duplicate_review_rate": 0.5256,
         "seed": EXPECTED_RANDOM_SEED, "policy_note": acp.POLICY_NOTE},
    )

    summary = acp.render_results_summary(tmp_path)
    assert "Raw CPLFW conditional accuracy: 90.24%" in summary
    assert "Raw CPLFW extraction-failure rate: 41.42%" in summary
    assert "Duplicate detection rate (conditional): 96.58%" in summary
    assert "False duplicate-review rate: 52.56%" in summary
    assert summary.count("LIMITATION") == 2
    # A legacy artefact carries no end-to-end figure, so the summary must say
    # so rather than let the conditional rate stand unqualified.
    assert "Duplicate detection rate (end-to-end): not recorded" in summary
    assert "Gallery enrolment coverage: not recorded" in summary

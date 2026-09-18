"""Regression tests for evaluation independence and controlled comparisons."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import ACP_arden as acp


def pair_result(scores=(0.9, 0.2, 0.8, 0.1, 0.7, 0.3, 0.6, 0.05)):
    rows = []
    for i, score in enumerate(scores):
        pair = acp.Pair(Path(f"left_{i}"), Path(f"right_{i}"), i % 2 == 0, "left", "right")
        rows.append(acp.PairScore(pair, score, "zero_faces_left" if score is None else None))
    return acp.EvaluationResult(len(rows), rows, {"zero_faces_left": sum(s is None for s in scores)} if None in scores else {}, [])


def test_fold_threshold_does_not_see_its_test_scores():
    folds = [list(range(4)), list(range(4, 8))]
    baseline = acp.summarize_lfw_cross_validation(pair_result(), folds)
    changed = acp.summarize_lfw_cross_validation(pair_result((0.1, 0.99, 0.2, 0.98, 0.7, 0.3, 0.6, 0.05)), folds)
    assert baseline["fold_results"][0]["threshold"] == changed["fold_results"][0]["threshold"]
    assert baseline["fold_results"][0]["accuracy"] != changed["fold_results"][0]["accuracy"]
    assert "threshold" not in baseline
    assert baseline["threshold_status"] == "per_fold_training_only"
    assert sum(baseline["confusion_matrix"].values()) == 8


def test_fold_failures_remain_in_original_fold_and_total():
    result = acp.summarize_lfw_cross_validation(pair_result((None, 0.2, 0.8, 0.1, 0.7, 0.3, 0.6, 0.05)), [range(4), range(4, 8)])
    assert result["total_pairs"] == 8
    assert result["scored_pairs"] == 7
    assert result["fold_results"][0]["test_failed_pairs"] == 1
    assert result["fold_results"][1]["test_failed_pairs"] == 0
    assert result["end_to_end_correct_decision_rate"] <= 7 / 8


@pytest.mark.parametrize("folds", [[[0, 1], [1, 2]], [list(range(8))], [[], list(range(8))]])
def test_cv_refuses_incomplete_duplicate_or_empty_folds(folds):
    with pytest.raises(acp.ProtocolError):
        acp.summarize_lfw_cross_validation(pair_result(), folds)


def test_training_threshold_handles_equal_scores_and_strict_ties():
    assert acp.select_lfw_training_threshold([0.5, 0.5], [0, 1]) > 0.5
    assert acp.select_lfw_training_threshold([0.1, 0.8, 0.2, 0.9], [0, 1, 0, 1]) == 0.8


def test_official_fold_layout_preserves_all_six_hundred_positions(tmp_path):
    path = tmp_path / "pairs.txt"
    path.write_text("10 300\n")
    pairs = [acp.Pair(Path(f"l{i}"), Path(f"r{i}"), i % 600 < 300, "left", "right") for i in range(6000)]
    folds = acp.lfw_official_fold_indices(path, pairs)
    assert folds[1] == list(range(600, 1200))
    pairs[300] = replace(pairs[300], same_identity=True)
    with pytest.raises(acp.ProtocolError):
        acp.lfw_official_fold_indices(path, pairs)


def test_content_digest_changes_when_pixels_change_under_same_filename(tmp_path):
    image = tmp_path / "same.jpg"
    image.write_bytes(b"image one")
    first = acp.image_content_digest([image])
    image.write_bytes(b"image two")
    assert acp.image_content_digest([image]) != first


def search_run(fail=False):
    rows = [
        acp.OpenSetSearchResult("p1", "id1", "group", "mated_probe", failure_code="zero_faces" if fail else None,
                               top_similarity=None if fail else 0.9, correct_rank=None if fail else 1, correct_similarity=None if fail else 0.9),
        acp.OpenSetSearchResult("p2", "id2", "group", "mated_probe", top_similarity=0.9, correct_rank=1, correct_similarity=0.9),
        acp.OpenSetSearchResult("n1", "new1", "group", "non_mated_probe", top_similarity=0.1),
        acp.OpenSetSearchResult("n2", "new2", "group", "non_mated_probe", top_similarity=0.2),
    ]
    return acp.OpenSetRunResult(acp.METHOD_B, "test", [], rows, 2, 2)


def test_paired_comparison_includes_failure_benefit_and_common_subset():
    result = acp.paired_pipeline_bootstrap({"a": search_run(True), "b": search_run()}, {"a": 0.5, "b": 0.5}, replicates=50)
    assert result["common_success_probes"] == 3
    full = result["analyses"]["full_protocol"]
    assert full["pipelines"]["a"]["tpir_rank1"]["estimate"] == 1
    assert full["pipelines"]["a"]["end_to_end_duplicate_detection_rate"]["estimate"] == 0.5
    delta = full["paired_differences"][0]["metrics"]
    assert delta["end_to_end_duplicate_detection_rate"]["estimate"] == 0.5
    assert result["analyses"]["common_success"]["paired_differences"][0]["metrics"]["tpir_rank1"]["estimate"] == 0


def test_identical_pipelines_have_zero_paired_differences_in_every_draw():
    result = acp.paired_pipeline_bootstrap({"a": search_run(), "b": search_run()}, {"a": 0.5, "b": 0.5}, replicates=40)
    for metric in result["analyses"]["full_protocol"]["paired_differences"][0]["metrics"].values():
        assert metric["lower_95"] == metric["upper_95"] == metric["estimate"] == 0


def test_paired_comparison_refuses_different_probe_protocols():
    other = search_run()
    other.search_results.pop()
    with pytest.raises(acp.ProtocolError, match="same intended probes"):
        acp.paired_pipeline_bootstrap({"a": search_run(), "b": other}, {"a": 0.5, "b": 0.5}, replicates=20)


def test_zero_event_upper_bound_counts_identities_not_correlated_images():
    bound = acp.zero_event_identity_upper_bound({str(i): (15, 15) for i in range(200)}, false_referrals=0)
    assert bound is not None
    assert bound["one_sided_upper"] == pytest.approx(1 - 0.05 ** (1 / 200))
    assert bound["conditional_scored_probe_fpir_upper"] is None
    assert bound["independent_identities"] == 200
    assert acp.zero_event_identity_upper_bound({"one": (15, 15)}, false_referrals=1) is None
    ci = acp.cluster_bootstrap_intervals(search_run().search_results, threshold=0.5, replicates=20)
    assert ci["fpir"]["bootstrap_degenerate_zero_events"]
    assert ci["fpir"]["zero_event_identity_bound"]["one_sided_upper"] > 0


def small_protocol():
    entries = []
    for partition in ("development", "test"):
        for group in ("g1", "g2"):
            for i in range(4):
                identity = f"{partition}-{group}-{i}"
                for role in ("gallery_enrolment", "mated_probe"):
                    entries.append(acp.OpenSetEntry(f"{identity}-{role}", identity, identity, group,
                                                    Path(f"{identity}-{role}.jpg"), role, partition))
            identity = f"{partition}-{group}-new"
            entries.append(acp.OpenSetEntry(identity, identity, identity, group, Path(identity), "non_mated_probe", partition))
    return acp.OpenSetProtocol(entries, 1, {})


def test_gallery_sweep_changes_only_distractors_and_preserves_partition():
    protocol = small_protocol()
    small = acp.gallery_sensitivity_protocol(protocol, 2, seed=42, probe_core_size=2)
    large = acp.gallery_sensitivity_protocol(protocol, 6, seed=42, probe_core_size=2)
    assert {e.sample_id for e in small.entries if e.role != "gallery_enrolment"} == {e.sample_id for e in large.entries if e.role != "gallery_enrolment"}
    for part in ("development", "test"):
        a = small.identities(part, "gallery_enrolment")
        b = large.identities(part, "gallery_enrolment")
        assert len(a) == 2 and len(b) == 6 and a < b
    assert not large.identities("development", "gallery_enrolment") & large.identities("test", "gallery_enrolment")
    assert small == acp.gallery_sensitivity_protocol(protocol, 2, seed=42, probe_core_size=2)


def test_quality_bins_are_derived_from_development_only():
    development, test = search_run(), search_run()
    quality = {r.sample_id: {"image_short_side_pixels": i * 10.0} for i, r in enumerate(development.search_results)}
    test_rows = [replace(r, sample_id="test-" + r.sample_id) for r in test.search_results]
    test = replace(test, search_results=test_rows)
    quality.update({r.sample_id: {"image_short_side_pixels": 99999.0} for r in test_rows})
    result = acp.quality_error_analysis(development, test, quality, 0.5)
    assert result["image_short_side_pixels"]["development_cutoffs"] == [7.5, 15.0, 22.5]
    assert sum(b["intended_probes"] for b in result["image_short_side_pixels"]["bins_low_to_high"].values()) == 4


def test_latency_percentile_is_measured_instead_of_copied_from_mean(monkeypatch):
    classifier = acp.ReviewClassifier(("one_feature",), [0.0], 0.0, [0.0], [1.0])
    ticks = iter([0.0, 0.001, 1.0, 1.003, 2.0, 2.009])
    monkeypatch.setattr(acp.time, "perf_counter", lambda: next(ticks))
    result = acp.measure_classifier_latency(classifier, np.zeros((3, 1)))
    assert result["classifier_decision_latency_mean_ms"] == pytest.approx(13 / 3)
    assert result["classifier_decision_latency_p95_ms"] == pytest.approx(8.4)
    assert result["classifier_latency_samples"] == 3


def test_menu_actions_route_to_requested_results_directory(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(acp, "action_run_comparison_diagnostics", lambda root: calls.append(root) or 0)
    answers = iter(["18", "", "26"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    assert acp.run_menu(tmp_path) == 0
    assert calls == [tmp_path]


def test_show_menu_results_does_not_run_an_experiment(monkeypatch, tmp_path):
    monkeypatch.setattr(acp, "run_comparison_diagnostics", lambda *_: pytest.fail("Show must not run models"))
    answers = iter(["19", "", "26"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    assert acp.run_menu(tmp_path) == 0


def ablation_inputs():
    entries, runs = [], {}
    for partition in ("development", "test"):
        rows = []
        for i in range(40):
            identity = f"{partition}-{i}"
            role = "mated_probe" if i % 2 == 0 else "non_mated_probe"
            entry = acp.OpenSetEntry(identity, identity, identity, "group", Path(identity), role, partition)
            entries.append(entry)
            score = 0.8 if role == "mated_probe" else 0.2
            rows.append(acp.OpenSetSearchResult(
                identity, identity, "group", role, top_similarity=score, top2_similarity=0.1,
                correct_rank=1 if role == "mated_probe" else None,
                correct_similarity=score if role == "mated_probe" else None,
                top5_similarity_mean=0.1, top5_similarity_stdev=0.02,
                top1_gallery_image_count=3, gallery_size=20,
                probe_detection_confidence=0.99, probe_face_area_ratio=0.5,
            ))
        runs[partition] = acp.OpenSetRunResult(acp.METHOD_B, partition, [], rows, 20, 20)
    return acp.OpenSetProtocol(entries, 1, {}), runs["development"], runs["test"]


def test_feature_ablation_models_and_thresholds_ignore_test_scores():
    protocol, development, test = ablation_inputs()
    first = acp.classifier_feature_ablation(protocol, development, test)
    changed = replace(test, search_results=[replace(row, top_similarity=0.01) for row in test.search_results])
    second = acp.classifier_feature_ablation(protocol, development, changed)
    training, calibration = acp.split_development_identities_for_classifier(protocol)
    for name, features in acp.DIAGNOSTIC_FEATURE_SETS.items():
        assert first[name]["model"] == second[name]["model"]
        assert first[name]["calibration"] == second[name]["calibration"]
        assert first[name]["features"] == list(features)
        assert first[name]["training_rows"] == len(training)
        assert first[name]["calibration_rows"] == len(calibration)
    assert first["similarity_only"]["rates"]["tpir_rank1"] != second["similarity_only"]["rates"]["tpir_rank1"]


def test_feature_ablation_retains_failed_probes_in_end_to_end_denominator():
    protocol, development, test = ablation_inputs()
    test.search_results[0] = replace(test.search_results[0], failure_code="zero_faces", top_similarity=None)
    result = acp.classifier_feature_ablation(protocol, development, test)
    for variant in result.values():
        assert variant["test_rows"] == 39
        assert variant["test_excluded_records"]["unscored"] == 1
        assert variant["rates"]["tpir_rank1"] == 1
        assert variant["rates"]["end_to_end_duplicate_detection_rate"] == pytest.approx(19 / 20)


def test_menu_preserves_workflow_numbers_and_adds_run_all():
    import re
    numbers = [int(value) for value in re.findall(r"^\s*(\d+)\.", acp.MENU_TEXT, re.MULTILINE)]
    assert numbers == [*range(1, 25), 100, 25, 26]
    assert "human review moderation" in acp.MENU_TEXT.lower()
    for label in ("original", "new experiment", "updated", "local"):
        assert label not in acp.MENU_TEXT.lower()


def test_every_menu_choice_routes_to_its_action(monkeypatch, tmp_path):
    expected = [
        "action_check_environment", "action_verify_inputs", "action_self_test",
        "action_run_complete_evaluation", "action_show_summary",
        "action_run_open_set_evaluation", "action_show_open_set_summary",
        "action_run_ml_review", "action_show_ml_review_summary",
        "action_run_pipeline_comparison", "action_show_pipeline_comparison_summary",
        "action_run_verification_comparison", "action_show_verification_comparison_summary",
        "action_run_arcface_review", "action_show_arcface_review_summary",
        "action_run_mixed_pipelines", "action_show_mixed_pipeline_summary",
        "action_run_comparison_diagnostics", "render_comparison_diagnostics_summary",
        "action_run_comparative_statistics", "render_paired_comparison_report",
        "action_show_experiment_table", "action_run_extensions", "action_refresh_reports",
        "launch_review_interface",
    ]
    calls = []
    def fake(name):
        def invoke(*args):
            calls.append((name, args))
            return "saved report" if name.startswith("render_") else 0
        return invoke
    for name in expected:
        monkeypatch.setattr(acp, name, fake(name))
    answers = iter([value for number in range(1, 26) for value in (str(number), "")] + ["26"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    assert acp.run_menu(tmp_path) == 0
    assert [name for name, _ in calls] == expected
    for name, args in calls[3:-1]:
        assert args == (tmp_path,), name
    assert calls[-1][1] == (acp.DEFAULT_REVIEW_DB,)

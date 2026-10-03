"""Checks for the deployment-oriented evaluation summary.

Every figure in that summary is read back from saved artefacts, so these tests
write small synthetic artefacts whose numbers differ from the published ones.
A value can then only appear in the output if it was read from the file. No
scientific result is checked here.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict

import pytest

import ACP_arden as acp

AGG = Path(acp.__file__).parent / "results" / "aggregate"
METHOD = acp.METHOD_B

PARTS = (
    (acp.render_threshold_validation_part, "HOW EVERY THRESHOLD WAS CALIBRATED AND VALIDATED"),
    (acp.render_dataset_limits_part, "WHAT THESE DATASETS CAN AND CANNOT SHOW"),
    (acp.render_cost_and_workload_part, "WHAT EACH COMBINATION COSTS AND WHAT IT BUYS"),
    (acp.render_detector_settings_part, "HOW DETECTOR SETTINGS SHAPE THE FAILURES"),
    (acp.render_statistical_support_part, "IS THE BEST COMBINATION REALLY BETTER?"),
)


def _bfw_threshold(threshold: float, development_fpir: float) -> Dict[str, Any]:
    # A second operating point proves the primary target is looked up, not
    # taken from whichever point happens to come first.
    return {"primary_fpir_target": 0.003, "operating_points": {
        "0.001": {"target_fpir": 0.001, "threshold": 0.9, "development_fpir": 0.0},
        "0.003": {"target_fpir": 0.003, "threshold": threshold,
                  "development_fpir": development_fpir},
    }}


def _bfw_test(fpir: float, **coverage: Any) -> Dict[str, Any]:
    return {"methods": {METHOD: {"primary_operating_point": {"fpir": fpir},
                                 "coverage": coverage}}}


def _interval(estimate: float, lower: float, upper: float) -> Dict[str, float]:
    return {"estimate": estimate, "lower_95": lower, "upper_95": upper}


def _difference(left: str, right: str, detection, fpir) -> Dict[str, Any]:
    return {"direction": "right minus left", "left": left, "right": right,
            "metrics": {"end_to_end_duplicate_detection_rate": _interval(*detection),
                        "fpir": _interval(*fpir)}}


def _fixture() -> Dict[str, Any]:
    folds = lambda *values: [{"threshold": value} for value in values]
    return {
        "lfw_final_metrics.json": {
            "protocol_file": "folds.txt", "fold_count": 4,
            "fold_results": folds(0.31, 0.34, 0.32, 0.33),
            "failure_breakdown": {"multiple_faces_left": 7, "zero_faces_right": 2}},
        "verification_comparison/lfw_final_metrics.json": {
            "protocol_file": "folds.txt", "fold_count": 4,
            "fold_results": folds(0.27, 0.28, 0.275, 0.276),
            "failure_breakdown": {"multiple_faces_left": 50, "multiple_faces_right": 25}},
        "calibrated_threshold.json": {
            "threshold": 0.111111, "protocol_file": "devtrain.txt",
            "frozen_from_protocol": "devtest.txt", "operating_strategy": "balanced_accuracy"},
        "verification_comparison/calibrated_threshold.json": {
            "threshold": 0.222222, "protocol_file": "devtrain.txt",
            "frozen_from_protocol": "devtest.txt", "operating_strategy": "balanced_accuracy"},
        "cplfw_metrics.json": {
            "protocol_file": "cplfw_pairs.txt",
            "failure_breakdown": {"zero_faces_left": 30, "zero_faces_right": 12,
                                  "multiple_faces_left": 5}},
        "verification_comparison/cplfw_metrics.json": {
            "protocol_file": "cplfw_pairs.txt",
            "failure_breakdown": {"multiple_faces_left": 40, "zero_faces_left": 3}},
        "bfw_open_set_threshold.json": _bfw_threshold(0.401, 0.0011),
        "mixed_pipelines/scrfd-sface/bfw_open_set_threshold.json": _bfw_threshold(0.402, 0.0012),
        "mixed_pipelines/yunet-arcface/bfw_open_set_threshold.json": _bfw_threshold(0.403, 0.0013),
        "arcface_review/bfw_open_set_threshold.json": _bfw_threshold(0.404, 0.0014),
        "bfw_open_set_test_metrics.json": _bfw_test(
            0.0042, intended_gallery_identities=77,
            probe_failure_breakdown={"zero_faces": 9, "gallery_reference_unavailable": 4}),
        "mixed_pipelines/scrfd-sface/bfw_open_set_test_metrics.json": _bfw_test(0.0021),
        "mixed_pipelines/yunet-arcface/bfw_open_set_test_metrics.json": _bfw_test(0.0022),
        "ml_review_threshold.json": {"primary_fpir_target": 0.003, "operating_points": {
            "0.003": {"target_fpir": 0.003, "probability_threshold": 0.612345}}},
        "arcface_review/ml_review_threshold.json": {"primary_fpir_target": 0.003, "operating_points": {
            "0.003": {"target_fpir": 0.003, "probability_threshold": 0.712345}}},
        "pipeline_comparison_metrics.json": {
            "held_out_metrics": {
                "opencv-sface-yunet-test": {
                    "coverage": {"detection_latency_mean_ms": 2.0, "embedding_latency_mean_ms": 10.0,
                                 "complete_pipeline_latency_mean_ms": 13.0},
                    "rates": {"fpir": 0.0042}},
                "insightface-scrfd-arcface-test": {
                    "coverage": {"detection_latency_mean_ms": 20.0, "embedding_latency_mean_ms": 40.0,
                                 "complete_pipeline_latency_mean_ms": 61.0,
                                 "probe_failure_breakdown": {"multiple_faces": 6, "zero_faces": 1}},
                    "rates": {"fpir": 0.0023}},
            },
            "model_file_sizes": {
                "primary": {acp.YUNET_FILENAME: {"megabytes": 1.0},
                            acp.SFACE_FILENAME: {"megabytes": 10.0}},
                "comparison": {acp.ARCFACE_DETECTOR_FILENAME: {"megabytes": 5.0},
                               acp.ARCFACE_RECOGNITION_FILENAME: {"megabytes": 50.0}},
            },
        },
        "comparative_statistics.json": {
            "replicates": 500,
            "analyses": {"full_protocol": {
                "pipelines": {
                    "YuNet + SFace": {
                        "end_to_end_duplicate_detection_rate": _interval(0.80, 0.75, 0.85),
                        "fpir": _interval(0.006, 0.002, 0.010)},
                    "SCRFD + SFace": {
                        "end_to_end_duplicate_detection_rate": _interval(0.85, 0.80, 0.90),
                        "fpir": _interval(0.007, 0.003, 0.011)},
                    "YuNet + ArcFace": {
                        "end_to_end_duplicate_detection_rate": _interval(0.88, 0.84, 0.92),
                        "fpir": _interval(0.002, 0.0005, 0.004)},
                    "SCRFD + ArcFace": {
                        "end_to_end_duplicate_detection_rate": _interval(0.93, 0.90, 0.96),
                        "fpir": _interval(0.001, 0.0, 0.003)},
                },
                # Saved as right minus left. The second entry puts the best
                # combination on the left, so both orientations are exercised.
                "paired_differences": [
                    _difference("YuNet + SFace", "SCRFD + ArcFace",
                                (0.13, 0.09, 0.17), (-0.005, -0.009, -0.001)),
                    _difference("SCRFD + ArcFace", "SCRFD + SFace",
                                (-0.08, -0.12, -0.04), (0.006, 0.001, 0.011)),
                    _difference("YuNet + ArcFace", "SCRFD + ArcFace",
                                (0.05, -0.01, 0.11), (-0.001, -0.003, 0.002)),
                ],
            }},
        },
    }


def _write(root: Path, files: Dict[str, Any]) -> None:
    for name, payload in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _row(text: str, starts: str) -> str:
    # Table rows are indented; wrapped sentences are not, so a sentence that
    # happens to begin with a combination name is never mistaken for a row.
    rows = [" ".join(line.split()) for line in text.splitlines()
            if line.startswith("  ") and line.strip().startswith(starts)]
    assert len(rows) == 1, (starts, rows)
    return rows[0]


@pytest.fixture
def saved(tmp_path: Path) -> Path:
    _write(tmp_path, _fixture())
    return tmp_path


def test_every_part_reads_its_values_from_the_saved_files(saved: Path) -> None:
    before = {path: path.read_bytes() for path in saved.rglob("*") if path.is_file()}
    text = _flat(acp.render_deployment_evaluation_summary(saved))
    expected = (
        # Part A: thresholds, rules and the development against test line.
        "3 folds of LFW folds.txt", "0.310000 to 0.340000 over 4 folds",
        "0.270000 to 0.280000 over 4 folds", "LFW devtrain.txt",
        "highest balanced accuracy on devtest.txt", "CPLFW cplfw_pairs.txt",
        "0.111111", "0.222222", "0.401000", "0.402000", "0.403000", "0.404000",
        "highest detection with FPIR at most 0.30%",
        "0.612345 (probability)", "0.712345 (probability)",
        "YuNet + SFace 0.11% against 0.42%", "SCRFD + SFace 0.12% against 0.21%",
        "YuNet + ArcFace 0.13% against 0.22%", "SCRFD + ArcFace 0.14% against 0.23%",
        "above the 0.30% development target for YuNet + SFace.",
        # Part B: the gallery size.
        "holds only 77 identities",
        # Part C: intervals, cost, the two swaps and the workload at scale.
        "80.00% (75.00 to 85.00)", "93.00% (90.00 to 96.00)",
        "6.00 (2.00 to 10.00)", "1.00 (0.00 to 3.00)",
        "swapping YuNet for SCRFD adds 18.00 ms per image and 4.0 MB of storage, "
        "and raises end to end detection from 80.00% to 85.00%",
        "Swapping SFace for ArcFace adds 30.00 ms per image and 40.0 MB of storage, "
        "and cuts false reviews from 6.00 to 2.00 per 1,000 new profiles",
        "YuNet + SFace is the cheapest combination and SCRFD + ArcFace detects the most",
        "YuNet + SFace would need about 22 minutes of processing, and "
        "SCRFD + ArcFace about 1 hour 42 minutes",
        "roughly 500 fewer false reviews", "about 130 more",
        # Part D: settings from the code, counts from the breakdowns.
        f"YuNet {acp.DETECTOR_SCORE_THRESHOLD:g} each photograph's own size "
        "zero faces: CPLFW 42 pairs, BFW 9 photographs",
        f"SCRFD {acp.ARCFACE_DETECTION_THRESHOLD:g} "
        f"{acp.ARCFACE_DETECTION_INPUT_SIZE} by {acp.ARCFACE_DETECTION_INPUT_SIZE} pixels "
        "multiple faces: LFW 75 pairs, CPLFW 40 pairs, BFW 6 photographs",
        "YuNet lost 7 pairs to multiple faces and SCRFD lost 75",
        "YuNet lost 42 pairs to zero faces and SCRFD lost 3",
        # Part E: the best combination and the number of resamples.
        "SCRFD + ArcFace has the highest end to end detection, 93.00%",
        "resampling those identities 500 times",
    )
    for fragment in expected:
        assert fragment in text, fragment
    assert "not available" not in text
    after = {path: path.read_bytes() for path in saved.rglob("*") if path.is_file()}
    assert after == before, "rendering the summary must not write any file"


def test_each_part_reports_not_available_when_its_files_are_missing(tmp_path: Path) -> None:
    for render, heading in PARTS:
        text = render(tmp_path)
        assert text.startswith(heading)
        assert "not available" in text, heading
    assert "The gallery size is not available." in acp.render_dataset_limits_part(tmp_path)
    # The detector settings are code constants, so they survive missing files.
    settings = acp.render_detector_settings_part(tmp_path)
    assert f"{acp.ARCFACE_DETECTION_INPUT_SIZE} by {acp.ARCFACE_DETECTION_INPUT_SIZE} pixels" in settings
    summary = acp.render_deployment_evaluation_summary(tmp_path)
    assert all(heading in summary for _, heading in PARTS)


def test_one_missing_file_costs_only_its_own_cells(saved: Path) -> None:
    (saved / "mixed_pipelines/scrfd-sface/bfw_open_set_threshold.json").unlink()
    (saved / "comparative_statistics.json").unlink()
    thresholds = acp.render_threshold_validation_part(saved)
    row = _row(thresholds, "BFW gallery search, SCRFD + SFace")
    assert row.count("not available") == 4
    assert "0.401000" in _row(thresholds, "BFW gallery search, YuNet + SFace")
    assert "SCRFD + SFace not available against 0.21%" in _flat(thresholds)

    cost = acp.render_cost_and_workload_part(saved)
    row = _row(cost, "YuNet + SFace ")
    assert row.count("not available") == 2
    assert row.endswith("12.00 11.0")
    assert "is not available" in _flat(cost)


def test_crossed_combinations_are_timed_as_estimates(saved: Path) -> None:
    profile = acp.combination_cost_profile(saved)
    assert {name: entry["estimated"] for name, entry in profile.items()} == {
        "YuNet + SFace": False, "SCRFD + SFace": True,
        "YuNet + ArcFace": True, "SCRFD + ArcFace": False,
    }
    assert profile["SCRFD + SFace"]["time_ms"] == pytest.approx(30.0)
    assert profile["YuNet + ArcFace"]["time_ms"] == pytest.approx(42.0)
    assert profile["SCRFD + ArcFace"]["storage_mb"] == pytest.approx(55.0)

    text = acp.render_cost_and_workload_part(saved)
    assert "30.00 (estimated)" in _row(text, "SCRFD + SFace")
    assert "42.00 (estimated)" in _row(text, "YuNet + ArcFace")
    for measured in ("YuNet + SFace", "SCRFD + ArcFace"):
        assert "(estimated)" not in _row(text, measured), measured


@pytest.mark.parametrize("lower, upper, verdict", [
    (0.01, 0.05, "supported"),
    (-0.05, -0.01, "supported"),
    (-0.01, 0.02, "not distinguishable"),
    (0.0, 0.02, "not distinguishable"),
    (-0.02, 0.0, "not distinguishable"),
    (0.0, 0.0, "not distinguishable"),
    (None, 0.02, "not available"),
])
def test_the_verdict_follows_the_signs_of_the_interval(lower, upper, verdict) -> None:
    assert acp.interval_verdict(lower, upper) == verdict


def test_paired_differences_are_turned_to_read_best_minus_other(saved: Path) -> None:
    text = acp.render_statistical_support_part(saved)
    assert _row(text, "YuNet + SFace") == (
        "YuNet + SFace 13.00 (9.00 to 17.00) supported 0.50 (0.10 to 0.90) supported")
    # Saved with the best combination on the left, so both signs are flipped.
    assert _row(text, "SCRFD + SFace") == (
        "SCRFD + SFace 8.00 (4.00 to 12.00) supported 0.60 (0.10 to 1.10) supported")
    assert _row(text, "YuNet + ArcFace") == (
        "YuNet + ArcFace 5.00 (-1.00 to 11.00) not distinguishable "
        "0.10 (-0.20 to 0.30) not distinguishable")
    flat = _flat(text)
    assert ("The reduction in FPIR is supported against YuNet + SFace and SCRFD + SFace, "
            "and not distinguishable against YuNet + ArcFace.") in flat
    assert "not adjusted for making several comparisons" in flat
    assert "cannot bound the population FPIR" in flat


def test_the_best_combination_is_chosen_from_the_data(saved: Path) -> None:
    assert acp.best_combination_by_detection({
        "Alpha": {"end_to_end_duplicate_detection_rate": {"estimate": 0.5}},
        "Beta": {"end_to_end_duplicate_detection_rate": {"estimate": 0.7}},
        "Gamma": {"end_to_end_duplicate_detection_rate": {}},
    }) == "Beta"
    assert acp.best_combination_by_detection(None) is None

    files = _fixture()
    pipelines = files["comparative_statistics.json"]["analyses"]["full_protocol"]["pipelines"]
    pipelines["YuNet + SFace"]["end_to_end_duplicate_detection_rate"] = _interval(0.99, 0.98, 1.0)
    _write(saved, files)
    text = acp.render_statistical_support_part(saved)
    assert "YuNet + SFace has the highest end to end detection, 99.00%" in _flat(text)
    # The saved difference is SCRFD + ArcFace minus YuNet + SFace, so it now
    # reads the other way round and counts against the new best combination.
    assert _row(text, "SCRFD + ArcFace").startswith(
        "SCRFD + ArcFace -13.00 (-17.00 to -9.00) supported")
    assert "supported in the other direction against SCRFD + ArcFace" in _flat(text)


def test_no_new_sentence_uses_a_hyphen_or_a_dash(saved: Path, tmp_path_factory) -> None:
    roots = [saved, tmp_path_factory.mktemp("empty")]
    if (AGG / "comparative_statistics.json").is_file():
        roots.append(AGG)
    for root in roots:
        text = acp.render_deployment_evaluation_summary(root)
        assert not re.search(r"[A-Za-z]-[A-Za-z]", text), re.findall(r"\S*[A-Za-z]-[A-Za-z]\S*", text)
        assert "–" not in text and "—" not in text


def test_combination_names_are_never_split_across_lines(saved: Path) -> None:
    text = acp.render_deployment_evaluation_summary(saved)
    assert not re.search(r"\+\s*\n|\n\s*\+ ", text)


def test_the_summary_follows_the_processing_coverage_section(tmp_path: Path, capsys) -> None:
    assert acp.action_show_experiment_table(tmp_path) == 0
    out = capsys.readouterr().out
    coverage = out.index("WHY SOME PHOTOGRAPHS ARE NOT PROCESSED")
    summary = out.index("DEPLOYMENT ORIENTED EVALUATION")
    assert coverage < summary < out.index("THE MODELS COMPARED")


def test_the_published_results_fill_every_cell() -> None:
    if not (AGG / "comparative_statistics.json").is_file():
        pytest.skip("the comparison experiments have not been run in this checkout")
    assert "not available" not in acp.render_deployment_evaluation_summary(AGG)

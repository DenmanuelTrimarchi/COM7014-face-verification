"""Checks for the terminal explanation of saved processing coverage."""

import json
from pathlib import Path

import ACP_arden as acp


def test_processing_coverage_reads_saved_values_and_handles_missing_files(tmp_path: Path) -> None:
    empty = acp.render_processing_coverage_explanation(tmp_path)
    missing_rows = [line for line in empty.splitlines() if "(Exp " in line]
    assert len(missing_rows) == 9
    assert all(line.count("not available") == 5 for line in missing_rows)

    pair = {
        "total_pairs": 10, "scored_pairs": 8, "accuracy": 0.75,
        "confusion_matrix": {"true_positive": 4, "true_negative": 2},
        "failure_breakdown": {"multiple_faces_left": 1, "multiple_faces_right": 1},
    }
    gallery = {
        "intended_gallery_size": 5, "embedded_gallery_size": 4,
        "duplicate_probe_count": 4, "unknown_probe_count": 6,
        "duplicate_probe_failures": 1, "unknown_probe_failures": 1,
        "gallery_entry_failure_count": 1, "gallery_failure_breakdown": {"zero_faces_gallery": 1},
        "conditional_duplicate_detection_rate": 0.75,
        "false_duplicate_review_rate": 0.2, "end_to_end_duplicate_detection_rate": 0.5,
    }
    coverage = {
        "intended_gallery_identities": 5, "enrolled_gallery_identities": 4,
        "intended_mated_probes": 4, "intended_non_mated_probes": 6,
        "scored_mated_probes": 3, "scored_non_mated_probes": 5,
        "probe_failure_breakdown": {"zero_faces": 1, "gallery_reference_unavailable": 1},
    }
    rates = {"tpir_rank1": 0.75, "fpir": 0.2}
    method = {"coverage": coverage, "primary_operating_point": rates,
              "end_to_end_duplicate_detection_rate": 0.5}
    files = {
        "lfw_final_metrics.json": pair,
        "verification_comparison/lfw_final_metrics.json": pair,
        "cplfw_metrics.json": pair,
        "verification_comparison/cplfw_metrics.json": pair,
        "duplicate_gallery_metrics_v2.json": gallery,
        "bfw_open_set_test_metrics.json": {"methods": {"three_image_open_set_calibrated": method}},
        "pipeline_comparison_metrics.json": {"held_out_metrics": {
            "test-arcface-pipeline": {"coverage": coverage, "rates": rates,
                                      "end_to_end_duplicate_detection_rate": 0.5}}},
        "mixed_pipelines/scrfd-sface/bfw_open_set_test_metrics.json": {
            "methods": {"three_image_open_set_calibrated": method}},
        "mixed_pipelines/yunet-arcface/bfw_open_set_test_metrics.json": {
            "methods": {"three_image_open_set_calibrated": method}},
    }
    for name, payload in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    before = {name: (tmp_path / name).read_bytes() for name in files}
    text = acp.render_processing_coverage_explanation(tmp_path)
    rows = [" ".join(line.split()) for line in text.splitlines() if "(Exp " in line]
    assert len(rows) == 9
    for row in rows[:4]:
        assert "10 pairs 8 (80.00%) 2: multiple faces 2 75.00% correct 60.00% correct" in row
    assert "5 profiles; 10 search photos 4 profiles; 8 photos 1 profiles (zero faces 1); 2 search photos" in rows[4]
    assert "75.00% detected; 20.00% false reviews 50.00% detected" in rows[4]
    for row in rows[5:]:
        assert "5 profiles; 10 search photos 4 profiles; 8 photos 2 photos:" in row
        assert "profile missing 1" in row and "zero faces 1" in row
        assert "TPIR@1 75.00%; FPIR 20.00% 50.00% detected" in row
    flat = " ".join(text.split())
    assert "missed 1 faces" in flat and "pixels first and missed 1." in flat
    assert "BFW search photos are 4 known duplicates and 6 new profiles" in flat
    assert "most YuNet failures were zero-face detections" not in flat
    assert {name: (tmp_path / name).read_bytes() for name in files} == before

    pair["confusion_matrix"]["true_positive"] = 3
    pair["accuracy"] = 0.625
    (tmp_path / "lfw_final_metrics.json").write_text(json.dumps(pair), encoding="utf-8")
    (tmp_path / "pipeline_comparison_metrics.json").unlink()
    changed = acp.render_processing_coverage_explanation(tmp_path)
    assert "62.50% correct" in changed and "50.00% correct" in changed
    arcface_row = next(line for line in changed.splitlines() if "(Exp 8)" in line)
    assert arcface_row.count("not available") == 5
    assert "(Exp 12b)" in changed

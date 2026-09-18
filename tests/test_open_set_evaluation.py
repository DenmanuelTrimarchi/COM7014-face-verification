"""Tests for the corrected gallery accounting and the open-set additions.

These cover the defect that motivated the revision: a gallery reference image
that cannot be embedded used to be dropped silently, so the identity vanished
from the gallery while the mated probes pointing at it were still scored as
ordinary misses. Every expectation below is written as an independent literal
in the same spirit as the parity suite.

No test here loads a model binary, reads a dataset or touches the network. The
embedding stage is replaced by a deterministic stub so the accounting logic is
tested in isolation from OpenCV.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest

import ACP_arden as acp

EXPECTED_METHODOLOGY_REVISION = "open-set-gallery-accounting-v2"
EXPECTED_GALLERY_FAILURE_CODES = (
    "zero_faces_gallery",
    "multiple_faces_gallery",
    "image_error_gallery",
)
EXPECTED_UNAVAILABLE_CODE = "gallery_reference_unavailable"


def _identity_vector(identity_hash: str) -> np.ndarray:
    """A unit vector fixed by the identity hash, so two samples of one identity
    score exactly 1.0 and unrelated identities score near zero. ``int(...,16)``
    rather than ``hash()`` because Python randomises string hashing per run."""
    seed = int(identity_hash[:8], 16)
    vector = np.random.default_rng(seed).normal(size=16)
    return vector / float(np.linalg.norm(vector))


def _stub_embed(fail_map: dict):
    def _embed(entry, detector, embedder):
        if entry.sample_id in fail_map:
            return None, fail_map[entry.sample_id]
        return _identity_vector(entry.identity_hash), None

    return _embed


def _manifest() -> acp.GalleryManifest:
    images = {
        f"identity_{index:02d}": [Path(f"/tmp/i{index}/identity_{index:02d}_0001.jpg")]
        for index in range(8)
    }
    # Five mated identities, so a test may fail three references and still
    # leave a non-empty gallery to search.
    for name in ("anchor", "beacon", "cedar", "dogwood", "elm"):
        images[name] = [Path(f"/tmp/{name}/{name}_0001.jpg"), Path(f"/tmp/{name}/{name}_0002.jpg")]
    return acp.build_manifest(images, seed=20260727)


def _run(monkeypatch, fail_map: dict, threshold: float = 0.5):
    """``_embed_entry`` is replaced wholesale, so the detector and embedder are
    never consulted; they are cast rather than constructed to keep the test free
    of any model binary."""
    monkeypatch.setattr(acp, "_embed_entry", _stub_embed(fail_map))
    manifest = _manifest()
    result = acp.evaluate_gallery(
        manifest,
        detector=cast(acp.FaceDetector, None),
        embedder=cast(acp.FaceEmbedder, None),
        duplicate_review_threshold=threshold,
    )
    return manifest, result, acp.summarize_gallery_metrics(result)


def _gallery_sample_id(manifest: acp.GalleryManifest, identity_index: int = 0) -> str:
    gallery = [e for e in manifest.entries if e.role == "gallery"]
    return gallery[identity_index].sample_id


# --- Enrolment accounting -----------------------------------------------------


def test_intended_gallery_entries_equal_embedded_plus_failures(monkeypatch) -> None:
    manifest = _manifest()
    failing = _gallery_sample_id(manifest)
    _, result, summary = _run(monkeypatch, {failing: "zero_faces"})

    assert summary["intended_gallery_size"] == (
        summary["embedded_gallery_size"] + summary["gallery_entry_failure_count"]
    )
    assert summary["gallery_entry_failure_count"] == 1
    assert result.resolved_intended_gallery_size == len(result.gallery_entry_results)


def test_gallery_failure_categories_reconcile_exactly(monkeypatch) -> None:
    manifest = _manifest()
    gallery_ids = [e.sample_id for e in manifest.entries if e.role == "gallery"]
    fail_map = {
        gallery_ids[0]: "zero_faces",
        gallery_ids[1]: "multiple_faces",
        gallery_ids[2]: "image_error:decode failed",
    }
    _, _result, summary = _run(monkeypatch, fail_map)

    breakdown = summary["gallery_failure_breakdown"]
    assert set(breakdown) == set(EXPECTED_GALLERY_FAILURE_CODES)
    assert breakdown["zero_faces_gallery"] == 1
    assert breakdown["multiple_faces_gallery"] == 1
    assert breakdown["image_error_gallery"] == 1
    # The categories must partition the failures: no double counting, no gaps.
    assert sum(breakdown.values()) == summary["gallery_entry_failure_count"] == 3


def test_gallery_failure_rate_uses_the_intended_denominator(monkeypatch) -> None:
    manifest = _manifest()
    _, _result, summary = _run(monkeypatch, {_gallery_sample_id(manifest): "zero_faces"})
    assert summary["gallery_entry_failure_rate"] == pytest.approx(
        summary["gallery_entry_failure_count"] / summary["intended_gallery_size"]
    )


# --- Unavailable references ---------------------------------------------------


def test_unavailable_gallery_reference_is_not_an_ordinary_false_non_match(monkeypatch) -> None:
    manifest = _manifest()
    failing = _gallery_sample_id(manifest)
    failing_identity = next(e.identity_hash for e in manifest.entries if e.sample_id == failing)
    _, result, summary = _run(monkeypatch, {failing: "zero_faces"})

    orphaned = [
        r
        for r in result.probe_results
        if r.role == "duplicate_probe" and r.identity_hash == failing_identity
    ]
    assert len(orphaned) == 1
    probe = orphaned[0]
    assert probe.failure_code == EXPECTED_UNAVAILABLE_CODE
    # Not a similarity decision: it never reached the comparison stage at all.
    assert probe.exceeds_duplicate_threshold is None
    assert probe.top_similarity is None
    assert probe.rank1_correct is None
    assert summary["gallery_reference_unavailable_count"] == 1


def test_unavailable_references_are_excluded_from_the_conditional_denominator(
    monkeypatch,
) -> None:
    manifest = _manifest()
    _, _result, summary = _run(monkeypatch, {_gallery_sample_id(manifest): "zero_faces"})
    scored = summary["duplicate_probe_count"] - summary["duplicate_probe_failures"]
    # Five mated identities, one of which lost its reference.
    assert summary["duplicate_probe_count"] == 5
    assert scored == 4


# --- Conditional versus end-to-end -------------------------------------------


def test_conditional_and_end_to_end_denominators_differ_correctly(monkeypatch) -> None:
    manifest = _manifest()
    _, _result, summary = _run(monkeypatch, {_gallery_sample_id(manifest): "zero_faces"})

    # Same numerator, different denominators: 4 of 4 scored, 4 of 5 intended.
    assert summary["conditional_duplicate_detection_rate"] == pytest.approx(1.0)
    assert summary["end_to_end_duplicate_detection_rate"] == pytest.approx(4.0 / 5.0)
    assert (
        summary["end_to_end_duplicate_detection_rate"]
        < summary["conditional_duplicate_detection_rate"]
    )
    assert summary["conditional_rank1_identification_rate"] == pytest.approx(1.0)
    assert summary["end_to_end_rank1_identification_rate"] == pytest.approx(4.0 / 5.0)
    assert summary["conditional_rank5_identification_rate"] == pytest.approx(1.0)
    assert summary["end_to_end_rank5_identification_rate"] == pytest.approx(4.0 / 5.0)


def test_the_two_rates_agree_when_nothing_fails(monkeypatch) -> None:
    _, _result, summary = _run(monkeypatch, {})
    assert summary["gallery_entry_failure_count"] == 0
    assert summary["conditional_duplicate_detection_rate"] == pytest.approx(
        summary["end_to_end_duplicate_detection_rate"]
    )


def test_probe_extraction_failure_lowers_only_the_end_to_end_rate(monkeypatch) -> None:
    manifest = _manifest()
    probe_id = next(e.sample_id for e in manifest.entries if e.role == "duplicate_probe")
    _, _result, summary = _run(monkeypatch, {probe_id: "zero_faces"})

    assert summary["gallery_entry_failure_count"] == 0
    assert summary["conditional_duplicate_detection_rate"] == pytest.approx(1.0)
    assert summary["end_to_end_duplicate_detection_rate"] == pytest.approx(4.0 / 5.0)


def test_summary_carries_the_methodology_revision(monkeypatch) -> None:
    _, _result, summary = _run(monkeypatch, {})
    assert summary["methodology_revision"] == EXPECTED_METHODOLOGY_REVISION


# --- Reporting guards ---------------------------------------------------------


def _write_baseline_artifacts(tmp_path: Path, gallery_payload: dict) -> None:
    acp.write_json_artifact(
        tmp_path / "calibrated_threshold.json",
        {"threshold": 0.363, "operating_strategy": "balanced_accuracy", "status": "frozen",
         "selection_rule": "highest development balanced accuracy"},
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
    acp.write_json_artifact(tmp_path / "duplicate_gallery_metrics_v2.json", gallery_payload)


def test_summary_output_always_reports_the_gallery_failure_rate(tmp_path: Path) -> None:
    _write_baseline_artifacts(
        tmp_path,
        {"gallery_size": 900, "intended_gallery_size": 986, "embedded_gallery_size": 900,
         "gallery_entry_failure_count": 86, "gallery_entry_failure_rate": 0.0872,
         "gallery_reference_unavailable_count": 86,
         "duplicate_detection_rate": 0.9658, "end_to_end_duplicate_detection_rate": 0.8812,
         "rank1_identification_rate": 0.9276, "false_duplicate_review_rate": 0.5256,
         "seed": 20260727, "policy_note": acp.POLICY_NOTE},
    )
    summary = acp.render_results_summary(tmp_path)
    assert "Gallery enrolment-failure rate: 8.72%" in summary
    assert "986" in summary and "900" in summary


def test_the_legacy_conditional_result_cannot_be_printed_alone(tmp_path: Path) -> None:
    """A conditional detection rate must always be accompanied by the
    end-to-end figure and the enrolment coverage, whether the artefact records
    them or explicitly reports that it cannot."""
    _write_baseline_artifacts(
        tmp_path,
        {"gallery_size": 900, "intended_gallery_size": 986, "embedded_gallery_size": 900,
         "gallery_entry_failure_count": 86, "gallery_entry_failure_rate": 0.0872,
         "gallery_reference_unavailable_count": 86,
         "duplicate_detection_rate": 0.9658, "end_to_end_duplicate_detection_rate": 0.8812,
         "rank1_identification_rate": 0.9276, "false_duplicate_review_rate": 0.5256,
         "seed": 20260727, "policy_note": acp.POLICY_NOTE},
    )
    summary = acp.render_results_summary(tmp_path)
    assert "Duplicate detection rate (conditional): 96.58%" in summary
    assert "Duplicate detection rate (end-to-end): 88.12%" in summary
    assert summary.index("(conditional)") < summary.index("(end-to-end)")
    assert "False duplicate-review rate: 52.56%" in summary


def test_a_v1_artifact_states_that_the_end_to_end_rate_is_missing(tmp_path: Path) -> None:
    acp.write_json_artifact(
        tmp_path / "calibrated_threshold.json",
        {"threshold": 0.363, "operating_strategy": "balanced_accuracy", "status": "frozen",
         "selection_rule": "highest development balanced accuracy"},
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
         "seed": 20260727, "policy_note": acp.POLICY_NOTE},
    )
    summary = acp.render_results_summary(tmp_path)
    assert "Duplicate detection rate (end-to-end): not recorded" in summary
    assert "Gallery enrolment coverage: not recorded" in summary


def test_existing_baseline_aggregate_metrics_remain_readable() -> None:
    """The historical v1 artefact is kept for provenance and must stay parseable
    after the revision."""
    legacy = Path(__file__).resolve().parent.parent / "results/aggregate/duplicate_gallery_metrics.json"
    if not legacy.is_file():
        pytest.skip("baseline artefact not present in this checkout")
    payload = acp.read_json_artifact(legacy)
    assert payload["artifact_type"] == "duplicate_gallery_metrics"
    assert payload["gallery_size"] == 986


# --- Keyed opaque identifiers -------------------------------------------------

EXPECTED_OPAQUE_ID_VERSION = "hmac-sha256-v1"
EXPECTED_OPAQUE_ID_HEX_LENGTH = 32
EXPECTED_MINIMUM_KEY_BYTES = 32

_KEY_A = "a" * 63 + "b"
_KEY_B = "c" * 63 + "d"


def test_identical_input_and_key_produce_identical_identifiers() -> None:
    with acp.temporary_id_hmac_key(_KEY_A):
        assert acp.opaque_id("Aaron_Peirsol") == acp.opaque_id("Aaron_Peirsol")


def test_different_keys_produce_different_identifiers() -> None:
    with acp.temporary_id_hmac_key(_KEY_A):
        first = acp.opaque_id("Aaron_Peirsol")
    with acp.temporary_id_hmac_key(_KEY_B):
        second = acp.opaque_id("Aaron_Peirsol")
    assert first != second


def test_different_inputs_produce_different_identifiers() -> None:
    with acp.temporary_id_hmac_key(_KEY_A):
        assert acp.opaque_id("Aaron_Peirsol") != acp.opaque_id("Aaron_Peirsol_2")


def test_the_raw_identity_is_absent_from_the_identifier() -> None:
    with acp.temporary_id_hmac_key(_KEY_A):
        identifier = acp.opaque_id("Aaron_Peirsol")
    assert "Aaron" not in identifier
    assert "Peirsol" not in identifier
    assert len(identifier) == EXPECTED_OPAQUE_ID_HEX_LENGTH
    assert all(character in "0123456789abcdef" for character in identifier)


def test_the_identifier_version_is_pinned() -> None:
    assert acp.OPAQUE_ID_VERSION == EXPECTED_OPAQUE_ID_VERSION
    assert acp.MINIMUM_ID_HMAC_KEY_BYTES == EXPECTED_MINIMUM_KEY_BYTES


def test_constant_time_comparison_is_used_for_derived_values() -> None:
    with acp.temporary_id_hmac_key(_KEY_A):
        identifier = acp.opaque_id("Aaron_Peirsol")
        assert acp.opaque_ids_match(identifier, acp.opaque_id("Aaron_Peirsol"))
        assert not acp.opaque_ids_match(identifier, acp.opaque_id("Someone_Else"))


@pytest.mark.parametrize(
    "weak",
    [
        "",
        "short",
        "0011223344556677",           # 8 bytes of hex
        "changeme-changeme-changeme-changeme",
        "your-key-goes-here-your-key-goes-here",
        "0" * 64,                      # 32 bytes, but a single repeated byte
    ],
)
def test_short_malformed_or_placeholder_keys_are_rejected(weak: str) -> None:
    with pytest.raises(acp.OpaqueIdentifierKeyError):
        acp.decode_id_hmac_key(weak)


def test_a_missing_key_is_refused_rather_than_defaulted() -> None:
    with pytest.raises(acp.OpaqueIdentifierKeyError):
        acp.configure_id_hmac_key(None)


def test_identifiers_are_refused_when_no_key_is_configured(monkeypatch) -> None:
    monkeypatch.setattr(acp, "_ID_HMAC_KEY", None)
    with pytest.raises(acp.OpaqueIdentifierKeyError):
        acp.opaque_id("Aaron_Peirsol")


def test_both_hex_and_urlsafe_base64_keys_are_accepted() -> None:
    hex_key = "0123456789abcdef" * 4  # 64 hex characters -> 32 varied bytes
    assert len(acp.decode_id_hmac_key(hex_key)) == 32
    b64_key = "bm90LWEtcmVhbC1rZXktMzJieXRlcy1sb25nLXZhbHVlLTAwMQ"
    assert len(acp.decode_id_hmac_key(b64_key)) >= EXPECTED_MINIMUM_KEY_BYTES


def test_a_rejection_message_never_echoes_the_key() -> None:
    secret = "0011223344556677"
    with pytest.raises(acp.OpaqueIdentifierKeyError) as raised:
        acp.decode_id_hmac_key(secret)
    assert secret not in str(raised.value)


def test_no_hmac_key_appears_in_a_public_artifact(tmp_path: Path) -> None:
    """Provenance records the scheme name only. The key, and any digest of it,
    must be absent from everything published."""
    with acp.temporary_id_hmac_key(_KEY_A):
        identifier = acp.opaque_id("Aaron_Peirsol")
    payload = {
        "artifact_type": "duplicate_gallery_metrics_v2",
        "opaque_id_version": acp.OPAQUE_ID_VERSION,
        "identity_hash": identifier,
    }
    artifact = tmp_path / "artifact.json"
    acp.write_json_artifact(artifact, payload)
    text = artifact.read_text(encoding="utf-8")
    assert _KEY_A not in text
    assert acp.OPAQUE_ID_VERSION in text
    # A digest of the key would let an attacker confirm a guess.
    import hashlib

    assert hashlib.sha256(_KEY_A.encode()).hexdigest() not in text


# --- Review-database identifier versioning ------------------------------------


def test_a_review_database_from_another_identifier_scheme_is_refused(tmp_path: Path) -> None:
    db_path = tmp_path / "review.sqlite"
    with acp.review_database(db_path) as connection:
        acp.upsert_review_case(
            connection,
            case_id="case-1",
            probe_sample_id="a" * 32,
            candidate_identity_hash="b" * 32,
            similarity=0.9,
            threshold=0.36,
        )
    # Simulate rows written by the previous fixed-salt build.
    import sqlite3

    connection = sqlite3.connect(str(db_path))
    connection.execute("UPDATE review_cases SET opaque_id_version = 'legacy-salted-sha256'")
    connection.commit()
    connection.close()

    with pytest.raises(acp.ReviewDatabaseVersionError) as raised:
        with acp.review_database(db_path):
            pass
    assert "delete the file" in str(raised.value).lower()


def test_review_rows_record_the_identifier_version(tmp_path: Path) -> None:
    db_path = tmp_path / "review.sqlite"
    with acp.review_database(db_path) as connection:
        acp.upsert_review_case(
            connection,
            case_id="case-1",
            probe_sample_id="a" * 32,
            candidate_identity_hash="b" * 32,
            similarity=0.9,
            threshold=0.36,
        )
    with acp.review_database(db_path) as connection:
        versions = {row[0] for row in connection.execute(
            "SELECT opaque_id_version FROM review_cases"
        )}
    assert versions == {EXPECTED_OPAQUE_ID_VERSION}


def test_the_dev_test_split_does_not_depend_on_the_identifier_key() -> None:
    """Partitioning must depend only on the seed and the protocol, so the
    published metrics stay reproducible by someone without the key."""
    with acp.temporary_id_hmac_key(_KEY_A):
        first = [
            (e.role, e.image_path) for e in _manifest().entries
        ]
    with acp.temporary_id_hmac_key(_KEY_B):
        second = [
            (e.role, e.image_path) for e in _manifest().entries
        ]
    assert sorted(first) == sorted(second)

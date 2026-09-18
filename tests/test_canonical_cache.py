"""Acceptance tests for the canonical run cache, its context validation and
the outcome digest.

The cache is what guarantees Experiments 6, 7 and 8 report identical figures
for the same method, so it must be rejected whenever any input capable of
changing its result differs. Matching probe identifiers is not sufficient: the
same images scored with a different model, detector setting or OpenCV build
produce different outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

import ACP_arden as acp

EXPECTED_SEED = 20260727
EXPECTED_CACHE_SCHEMA_VERSION = 3


def _run(partition: str = "test") -> acp.OpenSetRunResult:
    """A small deterministic run standing in for a scored partition."""
    results: List[acp.OpenSetSearchResult] = []
    outcomes: List[acp.EnrolmentOutcome] = []
    for index in range(6):
        subgroup = acp.BFW_SUBGROUPS[index % len(acp.BFW_SUBGROUPS)]
        results.append(acp.OpenSetSearchResult(
            sample_id=f"m{index:031d}", identity_hash=f"{index:032d}", subgroup=subgroup,
            role="mated_probe", top_similarity=0.8, top2_similarity=0.7,
            correct_rank=1, correct_similarity=0.8, top5_similarity_mean=0.6,
            top5_similarity_stdev=0.05, top1_gallery_image_count=3, gallery_size=6,
            probe_detection_confidence=0.9, probe_face_area_ratio=0.5,
            top1_time_seconds=0.001, top5_time_seconds=0.002,
        ))
        results.append(acp.OpenSetSearchResult(
            sample_id=f"n{index:031d}", identity_hash=f"x{index:031d}", subgroup=subgroup,
            role="non_mated_probe", top_similarity=0.2, top1_time_seconds=0.001,
        ))
        outcomes.append(acp.EnrolmentOutcome(
            identity_hash=f"{index:032d}", subgroup=subgroup, enrolled=True,
            attempted_images=3, embedded_images=3, failure_code=None,
        ))
    return acp.OpenSetRunResult(
        method=acp.METHOD_B, partition=partition, enrolment_outcomes=outcomes,
        search_results=results, gallery_size=6, comparisons_per_probe=6,
        stage_times_seconds={"complete_pipeline_seconds": [0.01, 0.02]},
    )


def _context() -> Dict[str, Any]:
    return {
        "cache_schema_version": EXPECTED_CACHE_SCHEMA_VERSION,
        "partition": "test",
        "dataset_metadata_sha256": "a" * 64,
        "evaluated_image_set_sha256": "b" * 64,
        "protocol_version": acp.BFW_PROTOCOL_VERSION,
        "public_manifest_digest": "c" * 64,
        "private_cache_protocol_context_digest": "d" * 64,
        "pipeline_revision": acp.CANONICAL_PIPELINE_REVISION,
        "model_filenames": [acp.YUNET_FILENAME, acp.SFACE_FILENAME],
        "model_sha256": {"yunet": acp.YUNET_SHA256, "sface": acp.SFACE_SHA256},
        "preprocessing_revision": acp.PREPROCESSING_REVISION,
        "detector_initial_input_size": [320, 320],
        "detector_input_strategy": "native_image_dimensions_per_image",
        "detector_score_threshold": acp.DETECTOR_SCORE_THRESHOLD,
        "detector_nms_threshold": acp.DETECTOR_NMS_THRESHOLD,
        "detector_top_k": acp.DETECTOR_TOP_K,
        "embedding_dimensions": acp.EMBEDDING_DIMENSIONS,
        "gallery_images_per_identity": acp.MULTI_IMAGE_ENROLMENT,
        "seed": EXPECTED_SEED,
        "opencv_version": "4.13.0",
        "gallery_enrolment_samples": ["g1", "g2"],
        "mated_probe_samples": ["m1"],
        "non_mated_probe_samples": ["n1"],
    }


# --- Cache context validation -------------------------------------------------


def test_a_matching_context_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    acp.save_canonical_run(_run(), path, _context())
    assert acp.cache_invalidation_reason(path, _context()) is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_metadata_sha256", "f" * 64),
        ("private_cache_protocol_context_digest", "f" * 64),
        ("pipeline_revision", "opencv-yunet-sface-open-set-v999"),
        ("detector_nms_threshold", 0.9),
        ("detector_top_k", 10),
        ("detector_input_strategy", "fixed_320"),
        ("preprocessing_revision", "changed-revision"),
        ("detector_score_threshold", 0.5),
        ("detector_initial_input_size", [640, 640]),
        ("embedding_dimensions", 512),
        ("opencv_version", "5.0.0"),
        ("gallery_enrolment_samples", ["g1", "g3"]),
        ("mated_probe_samples", ["m2"]),
        ("non_mated_probe_samples", ["n2"]),
        ("seed", 1),
        ("gallery_images_per_identity", 1),
    ],
)
def test_any_changed_context_field_invalidates_the_cache(
    tmp_path: Path, field: str, value: Any
) -> None:
    path = tmp_path / "run.json"
    acp.save_canonical_run(_run(), path, _context())
    changed = {**_context(), field: value}
    reason = acp.cache_invalidation_reason(path, changed)
    assert reason is not None, f"changing {field} did not invalidate the cache"
    assert field in reason


def test_a_changed_model_digest_invalidates_the_cache(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    acp.save_canonical_run(_run(), path, _context())
    changed = {**_context(), "model_sha256": {"yunet": "0" * 64, "sface": acp.SFACE_SHA256}}
    reason = acp.cache_invalidation_reason(path, changed)
    assert reason is not None and "model_sha256" in reason


def test_a_changed_cache_schema_invalidates_the_cache(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    acp.save_canonical_run(_run(), path, _context())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cache_schema_version"] = EXPECTED_CACHE_SCHEMA_VERSION + 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    reason = acp.cache_invalidation_reason(path, _context())
    assert reason is not None and "schema" in reason


def test_a_missing_or_contextless_cache_is_rejected(tmp_path: Path) -> None:
    assert acp.cache_invalidation_reason(tmp_path / "absent.json", _context()) is not None
    path = tmp_path / "run.json"
    acp.save_canonical_run(_run(), path, None)
    assert acp.cache_invalidation_reason(path, _context()) is not None


def test_the_invalidation_reason_never_carries_context_values(tmp_path: Path) -> None:
    """Values can include private material, so only field names are reported."""
    path = tmp_path / "run.json"
    acp.save_canonical_run(_run(), path, _context())
    changed = {**_context(), "dataset_metadata_sha256": "f" * 64}
    reason = acp.cache_invalidation_reason(path, changed) or ""
    assert "f" * 64 not in reason and "a" * 64 not in reason


# --- Canonical outcome digest -------------------------------------------------


def test_timing_only_changes_do_not_alter_the_digest() -> None:
    run = _run()
    baseline = acp.canonical_run_digest(run)
    retimed = acp.OpenSetRunResult(
        method=run.method, partition=run.partition,
        enrolment_outcomes=run.enrolment_outcomes,
        search_results=[
            acp.OpenSetSearchResult(**{
                **{f.name: getattr(r, f.name) for f in __import__("dataclasses").fields(r)},
                "top1_time_seconds": 99.0, "top5_time_seconds": 99.0,
            })
            for r in run.search_results
        ],
        gallery_size=run.gallery_size, comparisons_per_probe=run.comparisons_per_probe,
        stage_times_seconds={"complete_pipeline_seconds": [9.9]},
    )
    assert acp.canonical_run_digest(retimed) == baseline


def test_reordering_equivalent_records_does_not_alter_the_digest() -> None:
    run = _run()
    reordered = acp.OpenSetRunResult(
        method=run.method, partition=run.partition,
        enrolment_outcomes=list(reversed(run.enrolment_outcomes)),
        search_results=list(reversed(run.search_results)),
        gallery_size=run.gallery_size, comparisons_per_probe=run.comparisons_per_probe,
        stage_times_seconds=run.stage_times_seconds,
    )
    assert acp.canonical_run_digest(reordered) == acp.canonical_run_digest(run)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("failure_code", "zero_faces"), ("top_similarity", 0.55),
        ("top2_similarity", 0.11), ("correct_rank", 4), ("correct_similarity", 0.33),
        ("top5_similarity_mean", 0.42), ("top5_similarity_stdev", 0.9),
        ("top1_gallery_image_count", 1), ("gallery_size", 99),
        ("probe_detection_confidence", 0.1), ("probe_face_area_ratio", 0.99),
        ("subgroup", "white_males"), ("role", "non_mated_probe"),
        ("top_identity_hash", "z" * 32),
    ],
)
def test_every_non_timing_search_field_affects_the_digest(field: str, value: Any) -> None:
    import dataclasses

    run = _run()
    baseline = acp.canonical_run_digest(run)
    first = run.search_results[0]
    mutated = acp.OpenSetSearchResult(**{
        **{f.name: getattr(first, f.name) for f in dataclasses.fields(first)}, field: value
    })
    changed = acp.OpenSetRunResult(
        method=run.method, partition=run.partition,
        enrolment_outcomes=run.enrolment_outcomes,
        search_results=[mutated] + list(run.search_results[1:]),
        gallery_size=run.gallery_size, comparisons_per_probe=run.comparisons_per_probe,
    )
    assert acp.canonical_run_digest(changed) != baseline, f"{field} did not affect the digest"


@pytest.mark.parametrize(
    ("field", "value"),
    [("enrolled", False), ("attempted_images", 1), ("embedded_images", 0),
     ("failure_code", "insufficient_gallery_images"), ("subgroup", "black_males")],
)
def test_every_enrolment_field_affects_the_digest(field: str, value: Any) -> None:
    import dataclasses

    run = _run()
    baseline = acp.canonical_run_digest(run)
    first = run.enrolment_outcomes[0]
    mutated = acp.EnrolmentOutcome(**{
        **{f.name: getattr(first, f.name) for f in dataclasses.fields(first)}, field: value
    })
    changed = acp.OpenSetRunResult(
        method=run.method, partition=run.partition,
        enrolment_outcomes=[mutated] + list(run.enrolment_outcomes[1:]),
        search_results=run.search_results,
        gallery_size=run.gallery_size, comparisons_per_probe=run.comparisons_per_probe,
    )
    assert acp.canonical_run_digest(changed) != baseline, f"{field} did not affect the digest"


def test_the_cache_records_its_privacy_position_accurately(tmp_path: Path) -> None:
    """It holds derived face-comparison scores, so describing it as carrying no
    biometric record at all would be wrong."""
    path = tmp_path / "run.json"
    acp.save_canonical_run(_run(), path, _context())
    note = json.loads(path.read_text(encoding="utf-8"))["privacy_note"]
    assert "privacy-sensitive derived face-comparison" in note
    assert "no raw photographs, face embeddings or enrolled templates" in note
    assert "excluded from Git" in note


def test_the_cache_location_is_git_ignored() -> None:
    ignore = (Path(acp.__file__).parent / ".gitignore").read_text(encoding="utf-8")
    assert "results/raw/" in ignore
    assert str(acp.CANONICAL_RUN_CACHE).startswith(str(acp.RAW_ROOT))


# --- Stored-payload integrity (section 1) -------------------------------------
#
# Validating the expected context against the stored context proves only that
# the cache was built for this configuration. It does not prove the contents
# are still the ones that were built. A cache edited afterwards would otherwise
# load silently and be republished under a freshly computed digest.


def _saved(tmp_path: Path) -> Path:
    path = tmp_path / "run.json"
    acp.save_canonical_run(_run(), path, _context())
    return path


def _rewrite(path: Path, mutate) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _first_row(payload: Dict[str, Any], role: str) -> Dict[str, Any]:
    return next(r for r in payload["search_results"] if r["role"] == role)


@pytest.mark.parametrize(
    ("label", "mutate"),
    [
        ("top similarity",
         lambda p: _first_row(p, "mated_probe").update(top_similarity=0.99)),
        ("correct rank",
         lambda p: _first_row(p, "mated_probe").update(correct_rank=7)),
        ("enrolment count",
         lambda p: p["enrolment_outcomes"][0].update(embedded_images=1)),
        ("stored outcome digest",
         lambda p: p.update(canonical_run_digest="0" * 64)),
        ("stored context digest",
         lambda p: p.update(canonical_context_sha256="0" * 64)),
        ("failure code",
         lambda p: _first_row(p, "non_mated_probe").update(failure_code="zero_faces")),
        ("gallery size", lambda p: p.update(gallery_size=999)),
        ("method", lambda p: p.update(method="method_a")),
    ],
)
def test_a_modified_cache_is_rejected(tmp_path: Path, label: str, mutate) -> None:
    path = _saved(tmp_path)
    assert acp.cache_invalidation_reason(path, _context()) is None
    _rewrite(path, mutate)
    reason = acp.cache_invalidation_reason(path, _context())
    assert reason is not None, f"modifying the {label} did not invalidate the cache"


@pytest.mark.parametrize("field", sorted(acp._CACHE_REQUIRED_FIELDS))
def test_a_removed_required_field_is_rejected(tmp_path: Path, field: str) -> None:
    path = _saved(tmp_path)
    _rewrite(path, lambda p: p.pop(field, None))
    reason = acp.cache_invalidation_reason(path, _context()) or ""
    assert reason, f"removing {field} did not invalidate the cache"
    assert field in reason or "schema" in reason


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [("gallery_size", "six"), ("search_results", {}), ("enrolment_outcomes", 3),
     ("canonical_run_digest", 17), ("canonical_context", "context"), ("method", 1)],
)
def test_a_retyped_required_field_is_rejected(
    tmp_path: Path, field: str, wrong_value: Any
) -> None:
    path = _saved(tmp_path)
    _rewrite(path, lambda p: p.update({field: wrong_value}))
    reason = acp.cache_invalidation_reason(path, _context()) or ""
    assert field in reason or "reconstructed" in reason


def test_an_unknown_schema_is_rejected(tmp_path: Path) -> None:
    path = _saved(tmp_path)
    _rewrite(path, lambda p: p.update(cache_schema_version=999))
    assert "schema" in (acp.cache_invalidation_reason(path, _context()) or "")


def test_the_integrity_reason_never_carries_a_stored_value(tmp_path: Path) -> None:
    path = _saved(tmp_path)
    _rewrite(path, lambda p: _first_row(p, "mated_probe").update(top_similarity=0.123456789))
    reason = acp.cache_invalidation_reason(path, _context()) or ""
    assert "0.123456789" not in reason
    assert not any(part in reason for part in ("m000", "n000", "0.8"))


def test_a_rebuilt_cache_replaces_a_corrupted_one(tmp_path: Path) -> None:
    """Rejection must lead to regeneration, not to a silent republication."""
    path = _saved(tmp_path)
    _rewrite(path, lambda p: p.update(canonical_run_digest="0" * 64))
    assert acp.cache_invalidation_reason(path, _context()) is not None
    digest = acp.save_canonical_run(_run(), path, _context())
    assert acp.cache_invalidation_reason(path, _context()) is None
    assert digest == acp.canonical_run_digest(_run())


# --- Private protocol-context digest (section 2) ------------------------------


def _protocol_entry(**overrides: Any) -> acp.OpenSetEntry:
    base: Dict[str, Any] = dict(
        sample_id="s" * 32, identity_hash="i" * 32, identity="private-name",
        subgroup="asian_females", image_path=Path("/private/store/x.jpg"),
        role="gallery_enrolment", partition="test",
    )
    base.update(overrides)
    return acp.OpenSetEntry(**base)


def _protocol(**overrides: Any) -> acp.OpenSetProtocol:
    entries = [
        _protocol_entry(sample_id="a" * 32, identity_hash="1" * 32),
        _protocol_entry(sample_id="b" * 32, identity_hash="1" * 32),
        _protocol_entry(sample_id="c" * 32, identity_hash="2" * 32,
                        subgroup="black_males"),
        _protocol_entry(sample_id="d" * 32, identity_hash="1" * 32,
                        role="mated_probe"),
        _protocol_entry(sample_id="e" * 32, identity_hash="3" * 32,
                        role="non_mated_probe", subgroup="white_males"),
    ]
    if overrides:
        index = overrides.pop("index", 0)
        entries[index] = _protocol_entry(**{
            **{f: getattr(entries[index], f)
               for f in ("sample_id", "identity_hash", "identity", "subgroup",
                         "image_path", "role", "partition")},
            **overrides,
        })
    return acp.OpenSetProtocol(entries=entries, seed=EXPECTED_SEED, provenance={})


def test_reassigning_a_sample_to_another_identity_changes_the_private_digest() -> None:
    """The published manifest cannot see this: sample, role and partition are
    unchanged, but the score now means something different."""
    baseline = _protocol()
    moved = _protocol(index=0, identity_hash="9" * 32)
    assert acp.public_manifest_digest(baseline) == acp.public_manifest_digest(moved)
    assert (acp.private_protocol_context_digest(baseline)
            != acp.private_protocol_context_digest(moved))


def test_reassigning_a_sample_to_another_subgroup_changes_the_private_digest() -> None:
    baseline = _protocol()
    moved = _protocol(index=0, subgroup="indian_males")
    assert acp.public_manifest_digest(baseline) == acp.public_manifest_digest(moved)
    assert (acp.private_protocol_context_digest(baseline)
            != acp.private_protocol_context_digest(moved))


def test_regrouping_enrolment_images_changes_the_private_digest() -> None:
    """The same images distributed differently across identities build
    different templates, so the grouping is part of the context."""
    baseline = _protocol()
    regrouped = _protocol(index=1, identity_hash="2" * 32)
    assert (acp.private_protocol_context_digest(baseline)
            != acp.private_protocol_context_digest(regrouped))


def test_an_identical_protocol_produces_an_identical_private_digest() -> None:
    assert (acp.private_protocol_context_digest(_protocol())
            == acp.private_protocol_context_digest(_protocol()))


def test_the_private_digest_hashes_no_private_material() -> None:
    """Only opaque identifiers may enter it. Changing the private identity name
    or the absolute image path must leave the digest alone, which is what
    proves neither was hashed in."""
    baseline = acp.private_protocol_context_digest(_protocol())
    renamed = acp.private_protocol_context_digest(
        _protocol(index=0, identity="a-completely-different-private-name")
    )
    moved = acp.private_protocol_context_digest(
        _protocol(index=0, image_path=Path("/somewhere/else/entirely.jpg"))
    )
    assert baseline == renamed == moved
    assert len(baseline) == 64 and all(c in "0123456789abcdef" for c in baseline)


# --- Exact float preservation (section 5) -------------------------------------


def test_floats_differing_beyond_twelve_significant_digits_differ_in_the_digest() -> None:
    """Decimal truncation to twelve significant digits would collide these."""
    import math

    near = 0.1234567890123456
    nearer = math.nextafter(near, 1.0)
    assert near != nearer, "the two doubles must genuinely differ"
    assert format(near, ".12g") == format(nearer, ".12g")
    assert acp._stable_float(near) != acp._stable_float(nearer)

    def digest_with(value: float) -> str:
        import dataclasses

        run = _run()
        first = run.search_results[0]
        mutated = acp.OpenSetSearchResult(**{
            **{f.name: getattr(first, f.name) for f in dataclasses.fields(first)},
            "top_similarity": value,
        })
        return acp.canonical_run_digest(acp.OpenSetRunResult(
            method=run.method, partition=run.partition,
            enrolment_outcomes=run.enrolment_outcomes,
            search_results=[mutated] + list(run.search_results[1:]),
            gallery_size=run.gallery_size,
            comparisons_per_probe=run.comparisons_per_probe,
        ))

    assert digest_with(near) != digest_with(nearer)


@pytest.mark.parametrize(
    ("left", "right"),
    [(float("nan"), 0.0), (float("inf"), float("-inf")), (0.0, -0.0),
     (float("inf"), 1e308), (float("nan"), float("inf"))],
)
def test_special_floats_are_distinguished(left: float, right: float) -> None:
    assert acp._stable_float(left) != acp._stable_float(right)


def test_the_same_special_float_is_represented_consistently() -> None:
    assert acp._stable_float(float("nan")) == acp._stable_float(float("nan"))
    assert acp._stable_float(float("inf")) == acp._stable_float(float("inf"))
    assert acp._stable_float(-0.0) == acp._stable_float(-0.0)


def test_exact_floats_survive_a_reordering() -> None:
    """Exactness must not come at the cost of order independence."""
    run = _run()
    reordered = acp.OpenSetRunResult(
        method=run.method, partition=run.partition,
        enrolment_outcomes=list(reversed(run.enrolment_outcomes)),
        search_results=list(reversed(run.search_results)),
        gallery_size=run.gallery_size, comparisons_per_probe=run.comparisons_per_probe,
    )
    assert acp.canonical_run_digest(reordered) == acp.canonical_run_digest(run)


# --- Atomic private write (section 6) -----------------------------------------


def test_the_cache_is_written_atomically(tmp_path: Path, monkeypatch) -> None:
    """A failure part-way through must leave the previous cache intact and no
    temporary file behind."""
    path = tmp_path / "run.json"
    acp.save_canonical_run(_run(), path, _context())
    original = path.read_text(encoding="utf-8")

    def explode(*args: Any, **kwargs: Any) -> None:
        raise OSError("interrupted")

    monkeypatch.setattr(acp.os, "replace", explode)
    with pytest.raises(OSError):
        acp.save_canonical_run(_run("development"), path, _context())
    assert path.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob(".*tmp")), "a temporary cache file was left behind"


def test_no_partially_written_cache_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    acp.save_canonical_run(_run(), path, _context())
    truncated = path.read_text(encoding="utf-8")[: len(path.read_text()) // 2]
    path.write_text(truncated, encoding="utf-8")
    assert acp.cache_invalidation_reason(path, _context()) is not None


def test_the_cache_is_owner_only_on_posix(tmp_path: Path) -> None:
    import os
    import stat

    if os.name != "posix":
        pytest.skip("POSIX mode bits are not meaningful on this platform")
    path = tmp_path / "nested" / "run.json"
    path.parent.mkdir()
    acp.save_canonical_run(_run(), path, _context())
    assert stat.S_IMODE(path.stat().st_mode) == acp.CANONICAL_CACHE_FILE_MODE
    assert stat.S_IMODE(path.parent.stat().st_mode) == acp.CANONICAL_CACHE_DIR_MODE


def test_the_cache_records_its_permission_position(tmp_path: Path) -> None:
    import os

    path = tmp_path / "run.json"
    acp.save_canonical_run(_run(), path, _context())
    note = json.loads(path.read_text(encoding="utf-8"))["permission_note"]
    assert "os.replace()" in note
    if os.name == "posix":
        assert "0700" in note and "0600" in note
    else:
        assert "not applied on this platform" in note


# --- Cache reuse is unconditional across experiments --------------------------


def test_no_experiment_forces_a_canonical_rebuild() -> None:
    """Experiment 6 produces the canonical runs, but must not force a refresh.

    Forcing one re-rolled the primary pipeline on every invocation. OpenCV
    detection is not bit-reproducible across processes, so a probe could change
    side between two runs of identical code, moving the fitted classifier and
    the calibration operating points. Rebuilds are driven by the context digest
    alone, which covers everything capable of changing the result."""
    source = Path(acp.__file__).read_text(encoding="utf-8")
    assert "refresh=True" not in source, (
        "an experiment forces a canonical rebuild instead of trusting the context digest"
    )


def test_a_second_call_reuses_the_cache_rather_than_recomputing(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    first = acp.save_canonical_run(_run(), path, _context())
    assert acp.cache_invalidation_reason(path, _context()) is None
    reloaded = acp.load_canonical_run(path)
    assert reloaded is not None
    assert acp.canonical_run_digest(reloaded) == first


def test_the_pipeline_revision_is_what_retires_a_stale_cache() -> None:
    """The escape hatch for a logic change that no model digest would catch."""
    assert acp.CANONICAL_PIPELINE_REVISION
    context = _context()
    assert context["pipeline_revision"] == acp.CANONICAL_PIPELINE_REVISION

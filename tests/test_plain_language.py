"""Acceptance tests for the plain-language terminal layer.

The programme is read by people who do not already know what FPIR, TPIR,
enrolment or a cluster bootstrap are. These tests check what the terminal
actually says: that every headline figure carries its denominator, that no
wording claims a fact the artefact does not establish, and that a missing
optional experiment produces an instruction rather than a crash.

They deliberately do not check any scientific value, which the existing
methodology tests already cover.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

import pytest

import ACP_arden as acp

ROOT = Path(acp.__file__).parent
AGG = ROOT / "results" / "aggregate"


def _summary(renderer, *args: Any) -> str:
    return renderer(*args) if args else renderer()


# --- Wording that must never appear -------------------------------------------

# The programme produces review signals. It establishes none of these facts,
# so the vocabulary is banned outright rather than discouraged.
FORBIDDEN_CLAIMS = (
    "duplicate images found",
    "fake profiles detected",
    "stolen photographs identified",
    "fraudulent users found",
    "matching profiles proved",
    "fraud detected",
    "confirmed fraud",
)

PLAIN_RENDERERS = (
    "render_baseline_plain_summary",
    "render_open_set_plain_summary",
    "render_ml_review_plain_summary",
    "render_pipeline_plain_summary",
    "render_overall_conclusion",
)


def _flat(text: str) -> str:
    """Collapse wrapping, so a phrase split across two terminal lines still
    matches. Presentation wraps text; the wording is what is under test."""
    return re.sub(r"\s+", " ", text)


def _all_plain_text() -> str:
    parts = [acp.MENU_TEXT, acp.PROGRAMME_INTRODUCTION, acp.render_glossary(),
             acp.render_model_overview(), acp.render_dataset_overview()]
    for name in PLAIN_RENDERERS:
        parts.append(getattr(acp, name)(AGG))
    for preview in acp.EXPERIMENT_PREVIEWS.values():
        parts.append(preview)
    return "\n".join(parts).lower()


@pytest.mark.parametrize("claim", FORBIDDEN_CLAIMS)
def test_no_output_claims_a_confirmed_finding(claim: str) -> None:
    assert claim not in _all_plain_text(), f"the terminal output claims: {claim}"


def test_the_referral_disclaimer_accompanies_every_result_summary() -> None:
    """A count of referrals without this sentence invites the wrong reading."""
    for name in ("render_baseline_plain_summary", "render_open_set_plain_summary",
                 "render_ml_review_plain_summary", "render_pipeline_plain_summary"):
        text = getattr(acp, name)(AGG)
        assert "model-generated review signal" in text, name
        assert "not proof" in text, name


def test_the_disclaimer_names_what_it_does_not_prove() -> None:
    disclaimer = acp.REFERRAL_DISCLAIMER.lower()
    for claim in ("belong to the same person", "stolen", "fraud occurred"):
        assert claim in disclaimer, claim


# --- Counts, denominators and percentages -------------------------------------


def test_format_count_and_percentage_always_shows_the_denominator() -> None:
    text = acp.format_count_and_percentage(942, 1000, noun="intended photographs")
    assert "942" in text and "1,000" in text and "94.20%" in text


def test_format_count_and_percentage_handles_absent_values() -> None:
    assert acp.format_count_and_percentage(None, 100) == "not available"
    assert acp.format_count_and_percentage(float("nan"), 100) == "not available"
    # A count with no denominator is shown as a count, never as a bare share.
    assert "%" not in acp.format_count_and_percentage(5, 0)
    assert "%" not in acp.format_count_and_percentage(5, None)


def test_format_count_and_percentage_can_carry_the_technical_name() -> None:
    text = acp.format_count_and_percentage(15, 2859, technical="FPIR")
    assert text.endswith("[FPIR]")
    assert "15 of 2,859" in text


@pytest.mark.parametrize(
    ("renderer", "artefact"),
    [("render_baseline_plain_summary", "lfw_final_metrics.json"),
     ("render_open_set_plain_summary", "bfw_open_set_test_metrics.json")],
)
def test_every_percentage_is_accompanied_by_a_count(renderer: str, artefact: str) -> None:
    """A percentage on its own cannot be checked by a reader.

    Percentages inside a comparison table are exempt: the table's own row
    labels and the counts printed beneath it supply the denominators."""
    if not (AGG / artefact).is_file():
        pytest.skip(f"{artefact} not present in this checkout")
    text = getattr(acp, renderer)(AGG)
    body, _, _table = text.partition("  ---")
    for line in body.splitlines():
        if "%" not in line or line.strip().startswith("|"):
            continue
        if "approximately" in line or "at least" in line.lower():
            continue
        assert re.search(r"\d[\d,]* of \d[\d,]*", line), (
            f"{renderer} shows a percentage with no denominator: {line!r}"
        )


def test_the_denominator_distinction_is_explained_where_both_appear() -> None:
    for name in ("render_baseline_plain_summary", "render_open_set_plain_summary",
                 "render_ml_review_plain_summary", "render_pipeline_plain_summary"):
        text = getattr(acp, name)(AGG)
        assert "Conditional results use only" in text, name
        assert "End-to-end results use every" in text, name


# --- Menu -----------------------------------------------------------------------


def test_the_menu_is_grouped_by_purpose() -> None:
    for heading in ("SETUP AND VALIDATION", "EXPERIMENTS AND RESULTS",
                    "CONTROLLED COMPARISONS"):
        assert heading in acp.MENU_TEXT, heading


def test_combined_run_is_described_as_experiments_seven_and_eight() -> None:
    """It runs Experiments 7 and 8, not every extension experiment."""
    line = next(l for l in acp.MENU_TEXT.splitlines() if l.strip().startswith("23."))
    assert "Experiments 7 and 8" in line
    assert "both extension experiments" not in acp.MENU_TEXT.lower()


def test_every_menu_option_remains_reachable() -> None:
    """Regrouping must not drop an option."""
    for option in ("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.",
                   "10.", "11.", "12.", "13."):
        assert option in acp.MENU_TEXT, option


def test_the_menu_names_the_pipelines_readably() -> None:
    assert "YuNet + SFace" in acp.MENU_TEXT
    assert "SCRFD + ArcFace" in acp.MENU_TEXT


# --- Introduction, previews and overviews ---------------------------------------


def test_the_introduction_states_the_limits_of_the_programme() -> None:
    text = _flat(acp.PROGRAMME_INTRODUCTION)
    assert "does not automatically identify fraud" in text
    assert "No face-recognition model is trained or fine-tuned" in text
    assert "public academic benchmark datasets" in text


@pytest.mark.parametrize(
    "key", ["full", "open-set", "ml-review", "pipeline-compare", "extensions", "review"]
)
def test_every_long_running_option_has_a_preview(key: str) -> None:
    preview = acp.render_experiment_preview(key)
    assert preview, key
    assert "Purpose:" in preview or "demonstration" in preview.lower()


def test_the_previews_are_wired_to_the_menu_options() -> None:
    for choice, key in acp.MENU_PREVIEW_KEYS.items():
        assert key in acp.EXPERIMENT_PREVIEWS, f"option {choice} has no preview"


def test_the_experiment_seven_preview_excludes_demographic_predictors() -> None:
    preview = acp.render_experiment_preview("ml-review")
    assert "does not use" in preview
    assert "sex, ethnicity" in preview


def test_the_experiment_eight_preview_states_the_comparison_is_whole_pipeline() -> None:
    preview = acp.render_experiment_preview("pipeline-compare")
    assert "cannot be attributed only" in preview
    assert "detection, alignment and preprocessing also" in preview


def test_the_model_overview_states_who_trained_each_model() -> None:
    text = acp.render_model_overview()
    for model in ("YuNet", "SFace", "SCRFD", "ArcFace", "Logistic regression"):
        assert model in text, model
    assert text.count("not trained by this project") == 4
    assert "Fitted by this project using BFW development identities only" in text


def test_the_dataset_overview_does_not_call_benchmark_people_users() -> None:
    text = acp.render_dataset_overview()
    for dataset in ("LFW", "CPLFW", "BFW"):
        assert dataset in text, dataset
    assert "not users of any deployed system" in text


# --- Glossary ------------------------------------------------------------------


@pytest.mark.parametrize(
    "term",
    ["Enrolment", "Gallery", "Mated probe", "Non-mated probe", "False review",
     "TPIR@1", "FPIR", "Conditional rate", "End-to-end rate", "Confidence interval"],
)
def test_the_glossary_defines_each_technical_term(term: str) -> None:
    glossary = acp.render_glossary()
    assert f"{term}:" in glossary, term
    definition = glossary.split(f"{term}:", 1)[1].split("\n\n", 1)[0].strip()
    assert len(definition) > 20, f"{term} has no usable definition"


# --- Experiment-specific content -------------------------------------------------


def test_the_experiment_seven_summary_states_the_negative_finding() -> None:
    text = _flat(acp.render_ml_review_plain_summary(AGG))
    assert "main hypothesis was not achieved" in text
    assert "valid negative research finding" in text
    assert "does not indicate that the programme failed to run" in text


def test_the_experiment_seven_criteria_are_readable() -> None:
    text = acp.render_ml_review_plain_summary(AGG)
    assert "Fewer unnecessary reviews than the similarity method" in text
    assert "At least 90% gallery enrolment coverage" in text


def test_the_experiment_eight_summary_carries_real_comparison_values() -> None:
    payload = json.loads((AGG / "pipeline_comparison_metrics.json").read_text())
    if payload.get("evaluated") != "yes":
        pytest.skip("comparison not evaluated in this checkout")
    text = acp.render_pipeline_plain_summary(AGG)
    for row in ("Known duplicate-profile test cases detected", "End-to-end detection",
                "New profiles incorrectly referred for review",
                "False reviews per 1,000", "Mean complete processing time",
                "Model storage", "embedding dimensions"):
        assert row in text, row
    # Section 1: the plain wording leads, the technical name follows it.
    for plain, technical in (
        ("Known duplicate-profile test cases detected", "TPIR@1"),
        ("New profiles incorrectly referred for review", "FPIR"),
    ):
        line = next(l for l in text.splitlines() if plain in l)
        assert line.index(plain) < line.index(technical), plain
    # Real values, not placeholders.
    assert "[value]" not in text and "not available" not in text
    assert "%" in text and "ms" in text and "MB" in text


def test_the_experiment_eight_summary_leads_with_readable_pipeline_names() -> None:
    payload = json.loads((AGG / "pipeline_comparison_metrics.json").read_text())
    if payload.get("evaluated") != "yes":
        pytest.skip("comparison not evaluated")
    text = acp.render_pipeline_plain_summary(AGG)
    heading = text.split("Technical pipeline identifiers:")[0]
    assert "YuNet + SFace" in heading and "SCRFD + ArcFace" in heading
    # The internal identifier belongs under the technical heading only.
    assert "opencv-sface-2021dec-yunet-2023mar" not in heading
    assert "opencv-sface-2021dec-yunet-2023mar" in text


def test_the_review_interface_warns_that_it_uses_the_baseline_method() -> None:
    preview = acp.render_experiment_preview("review")
    assert "not confirmed duplicate profiles" in preview
    assert "high false-review rate" in preview
    assert "not a production moderation decision" in preview


def test_the_overall_conclusion_covers_all_five_findings() -> None:
    text = _flat(acp.render_overall_conclusion(AGG))
    for number in ("1.", "2.", "3.", "4.", "5."):
        assert number in text, number
    assert "does not prove identity, photograph ownership, fraud or profile" in text


# --- Status wording --------------------------------------------------------------


def test_internal_status_values_are_translated_for_display_only() -> None:
    assert acp.plain_status("evaluated_non_commercial_academic_research") == (
        "Evaluation completed for non-commercial academic research."
    )
    assert acp.plain_status("ml_review_tested") == "Classifier evaluation completed."
    assert acp.plain_status("open_set_tested") == (
        "Held-out open-set evaluation completed."
    )
    # An unknown status is shown rather than hidden behind friendly wording.
    assert acp.plain_status("some_new_state") == "some_new_state"
    assert acp.plain_status(None) == "Status not recorded."


def test_the_stored_status_values_are_unchanged() -> None:
    """Display wording must never be written back into an artefact."""
    for name, expected in (
        ("pipeline_comparison_metrics.json", {
            "evaluated_non_commercial_academic_research",
            "not_run_licensing_unresolved", "not_run_models_unavailable"}),
        ("ml_review_test_metrics.json", {"ml_review_tested"}),
        ("bfw_open_set_test_metrics.json", {"open_set_tested"}),
    ):
        path = AGG / name
        if not path.is_file():
            continue
        status = json.loads(path.read_text()).get("status")
        assert status in expected, f"{name} stores an unexpected status: {status}"


# --- Missing artefacts ------------------------------------------------------------


@pytest.mark.parametrize(
    ("renderer", "expected"),
    [("render_baseline_plain_summary", "option 4"),
     ("render_open_set_plain_summary", "option 6"),
     ("render_ml_review_plain_summary", "option 8"),
     ("render_pipeline_plain_summary", "option 10")],
)
def test_a_missing_artefact_gives_an_instruction_not_a_crash(
    tmp_path: Path, renderer: str, expected: str
) -> None:
    text = getattr(acp, renderer)(tmp_path)
    assert "not available yet" in text
    assert expected in text


def test_the_overall_conclusion_survives_a_missing_comparison(tmp_path: Path) -> None:
    text = acp.render_overall_conclusion(tmp_path)
    assert "has not been run" in text
    assert "5." in text


# --- Privacy ----------------------------------------------------------------------


def test_no_plain_summary_prints_a_path_identity_or_record_level_score() -> None:
    text = _all_plain_text()
    for forbidden in ("/users/", "/home/", "\\users\\", ".jpg", ".png",
                      "identity_hash", "sample_id", "image_path"):
        assert forbidden not in text, f"the terminal output contains {forbidden}"
    # "embedding" may be named, but only to say it is not used as a predictor
    # or not published; never as a printed value.
    for sentence in re.split(r"(?<=[.;])\s", text):
        if "embedding" in sentence:
            assert any(
                phrase in sentence
                for phrase in ("does not use", "not published", "never", "no ",
                               "converts a detected face", "representation")
            ), f"embeddings mentioned without an exclusion: {sentence[:120]!r}"
    # Opaque identifiers are 32 or 64 hexadecimal characters; neither belongs
    # in a plain-language summary.
    assert not re.search(r"\b[0-9a-f]{32,}\b", text)


def test_the_review_status_wording_covers_every_stored_status() -> None:
    for status in acp.REVIEW_STATUSES:
        assert status in acp.REVIEW_STATUS_WORDING, status
    # The stored vocabulary itself is unchanged.
    assert acp.REVIEW_STATUSES == [
        "open", "confirmed_duplicate", "false_match", "dismissed"
    ]


# --- SCRFD warning handling --------------------------------------------------------


def test_the_scrfd_explanation_sets_expectations_without_hiding_failure() -> None:
    text = acp.SCRFD_WARNING_EXPLANATION
    assert "dynamic output dimensions" in text
    assert "do not indicate that the evaluation has failed" in text
    assert "stop if the returned detector outputs are invalid" in text


def test_scrfd_validation_rejects_malformed_detector_output() -> None:
    """The warning is only tolerable because the outputs are actually checked."""
    import numpy as np

    class _Detector:
        def __init__(self, outputs: Any) -> None:
            self._outputs = outputs

        def detect(self, *_args: Any, **_kwargs: Any) -> Any:
            if isinstance(self._outputs, Exception):
                raise self._outputs
            return self._outputs

    good = (np.array([[10.0, 10.0, 40.0, 40.0, 0.9]]),
            np.zeros((1, 5, 2), dtype=np.float64))
    acp.validate_scrfd_outputs(_Detector(good))          # must not raise
    acp.validate_scrfd_outputs(_Detector((None, None)))  # no detection is fine

    bad_cases = [
        ("wrong output count", (np.array([]),)),
        ("negative box", (np.array([[40.0, 40.0, 10.0, 10.0, 0.9]]), None)),
        ("non-finite box", (np.array([[np.nan, 0.0, 4.0, 4.0, 0.9]]), None)),
        ("narrow box", (np.array([[1.0, 2.0]]), None)),
        ("wrong landmark shape",
         (good[0], np.zeros((1, 3, 2), dtype=np.float64))),
        ("non-finite landmark",
         (good[0], np.full((1, 5, 2), np.inf, dtype=np.float64))),
        ("detector raises", RuntimeError("session failed")),
    ]
    for label, outputs in bad_cases:
        with pytest.raises(acp.PipelineUnavailableError):
            acp.validate_scrfd_outputs(_Detector(outputs))


# --- The formal reports are untouched -----------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["FINAL_EVALUATION_REPORT.md", "OPEN_SET_EVALUATION_REPORT.md",
     "ML_REVIEW_EVALUATION_REPORT.md", "PRETRAINED_PIPELINE_COMPARISON_REPORT.md",
     "RESEARCH_REPORT.md"],
)
def test_the_academic_reports_keep_their_technical_wording(name: str) -> None:
    """The plain layer is additional. The formal reports keep formal metrics."""
    path = AGG / name
    if not path.is_file():
        pytest.skip(f"{name} not generated in this checkout")
    text = path.read_text(encoding="utf-8")
    assert any(term in text for term in ("FPIR", "TPIR", "threshold", "coverage")), name
    # Each report keeps its methodological caveat, though not all use the same
    # heading: the comparison report states it as a scope limitation instead.
    assert any(
        phrase in text.lower()
        for phrase in ("limitation", "cannot be attributed", "not proof",
                       "does not prove")
    ), name


# --- Section 17: the two headings ------------------------------------------------


def test_both_section_headings_are_produced() -> None:
    """Summary and technical details use the same section separators."""
    plain = acp.render_plain_section("body")
    technical = acp.render_technical_section("body")
    assert "# Summary\n" in plain
    assert "TECHNICAL DETAILS" in technical
    assert plain.startswith("# " + "=" * 77)
    assert technical.startswith("# " + "=" * 77)


def test_the_reference_section_carries_the_overviews_and_glossary() -> None:
    text = acp.render_reference_section()
    assert "REFERENCE INFORMATION" in text
    assert "MODELS USED" in text
    assert "DATASETS USED" in text
    assert "TERMS USED" in text


@pytest.mark.parametrize(
    "action",
    ["action_show_summary", "action_show_open_set_summary",
     "action_show_ml_review_summary", "action_show_pipeline_comparison_summary"],
)
def test_every_summary_option_prints_all_three_sections(
    action: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Summary, then technical detail, then reference material - in that
    order, for every option that displays a saved result."""
    getattr(acp, action)(AGG)
    printed = capsys.readouterr().out
    for heading in ("# Summary\n", "TECHNICAL DETAILS",
                    "REFERENCE INFORMATION"):
        assert heading in printed, f"{action} omitted {heading}"
    assert (printed.index("# Summary\n")
            < printed.index("TECHNICAL DETAILS")
            < printed.index("REFERENCE INFORMATION")), action
    # The glossary must reach every summary, not only the first one.
    assert "TPIR@1:" in printed and "Conditional rate:" in printed


@pytest.mark.parametrize(
    "action",
    ["action_show_summary", "action_show_open_set_summary",
     "action_show_ml_review_summary", "action_show_pipeline_comparison_summary"],
)
def test_the_technical_section_retains_the_academic_figures(
    action: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Moving the technical block below the plain text must not remove it."""
    getattr(acp, action)(AGG)
    technical = capsys.readouterr().out.split("TECHNICAL DETAILS", 1)[1]
    assert any(term in technical for term in ("threshold", "Threshold", "FPIR", "TPIR"))


# --- Section 20: the specified helper names --------------------------------------


@pytest.mark.parametrize(
    "helper",
    ["format_count_and_percentage", "plain_metric_description",
     "render_plain_pipeline_table", "render_dataset_overview",
     "render_model_overview", "render_glossary", "render_overall_conclusion"],
)
def test_the_presentation_helpers_exist(helper: str) -> None:
    assert callable(getattr(acp, helper)), helper


def test_the_table_helper_aligns_its_columns() -> None:
    table = acp.render_plain_pipeline_table(
        ["Metric", "A", "B"], [["a very long row label", "1", "2"], ["short", "3", "4"]]
    )
    lines = table.splitlines()
    # The header and the rule beneath it define the column layout.
    assert len(lines[0]) == len(lines[1])
    assert set(lines[1].strip()) == {"-", " "}


def test_plain_metric_description_covers_the_reported_metrics() -> None:
    for key in ("fpir", "tpir_rank1", "end_to_end", "false_reviews_per_1000",
                "mated_coverage", "non_mated_coverage"):
        description = acp.plain_metric_description(key)
        assert description != key, key
        assert description[0].isupper(), key


# --- Section 2: the prescribed vocabulary -----------------------------------------


def test_the_prescribed_referral_vocabulary_is_used() -> None:
    text = _all_plain_text()
    for phrase in ("known duplicate-profile test cases correctly detected",
                   "new profiles incorrectly referred for review"):
        assert phrase in text, phrase


# --- Section 21: no scientific value is produced by the presentation layer ---------


def test_the_presentation_layer_reads_values_and_never_computes_them() -> None:
    """Every displayed figure must come from an artefact.

    Rendering twice from the same files must give byte-identical output, and
    rendering must not write anything back."""
    import hashlib

    def digest_tree() -> str:
        sha = hashlib.sha256()
        for path in sorted(AGG.rglob("*")):
            if path.is_file():
                sha.update(path.name.encode())
                sha.update(path.read_bytes())
        return sha.hexdigest()

    before = digest_tree()
    first = "\n".join(getattr(acp, name)(AGG) for name in PLAIN_RENDERERS)
    second = "\n".join(getattr(acp, name)(AGG) for name in PLAIN_RENDERERS)
    assert first == second, "the plain summaries are not deterministic"
    assert digest_tree() == before, "rendering a summary modified an artefact"


def test_no_result_number_is_hard_coded_in_the_presentation_layer() -> None:
    """The published figures must not be duplicated as literals in source."""
    source = Path(acp.__file__).read_text(encoding="utf-8")
    payload = json.loads((AGG / "bfw_open_set_test_metrics.json").read_text())
    primary = payload["methods"][acp.METHOD_B]["primary_operating_point"]
    for value in (primary["fpir"], primary["tpir_rank1"],
                  payload["operating_threshold"]):
        assert repr(float(value)) not in source, (
            f"{value} is hard-coded in source instead of read from the artefact"
        )
        assert f"{float(value) * 100:.2f}%" not in source, value


def test_the_plain_summaries_match_the_stored_values() -> None:
    """A spot check that the displayed figures are the stored ones."""
    payload = json.loads((AGG / "bfw_open_set_test_metrics.json").read_text())
    primary = payload["methods"][acp.METHOD_B]["primary_operating_point"]
    text = acp.render_open_set_plain_summary(AGG)
    assert f"{primary['fpir'] * 100:.2f}%" in text
    assert f"{primary['tpir_rank1'] * 100:.2f}%" in text
    coverage = payload["methods"][acp.METHOD_B]["coverage"]
    assert f"{coverage['scored_mated_probes']:,}" in text
    assert f"{coverage['intended_mated_probes']:,}" in text


# --- Experiment 9 and the overview table ------------------------------------------

VERIF = AGG / "verification_comparison"


def test_the_menu_names_the_models_for_each_experiment_group() -> None:
    """A reader choosing an option should know what will run before it starts."""
    text = acp.MENU_TEXT
    assert "models: YuNet + SFace" in text
    for phrase in ("YuNet + SFace on BFW",
                   "YuNet + SFace + logistic regression on BFW",
                   "Both pipelines on the same BFW identities",
                   "SCRFD + ArcFace through the same one-to-one chain"):
        assert phrase in text, phrase


def test_the_menu_offers_experiment_nine_and_the_overview() -> None:
    for option in ("14.", "15.", "16."):
        assert option in acp.MENU_TEXT, option
    assert "OVERVIEW" in acp.MENU_TEXT


def test_experiment_nine_has_a_preview_wired_to_its_option() -> None:
    assert acp.MENU_PREVIEW_KEYS.get("12") == "verification-compare"
    preview = acp.render_experiment_preview("verification-compare")
    assert "Purpose:" in preview
    # The whole point is that a threshold is never shared between pipelines.
    assert "baseline threshold is never reused" in preview
    assert "No model will be trained or fine-tuned." in preview


def test_the_new_modes_are_registered() -> None:
    for mode in ("verification-compare", "verification-compare-summary",
                 "experiment-table"):
        assert mode in acp.MODES, mode


def test_experiment_nine_writes_to_its_own_directory() -> None:
    """Experiments 1-5 must not be overwritten by the comparison chain: both
    write a file called calibrated_threshold.json."""
    assert acp.VERIFICATION_COMPARISON_DIRNAME == "verification_comparison"
    if not VERIF.is_dir():
        pytest.skip("Experiment 9 has not been run in this checkout")
    baseline = json.loads((AGG / "calibrated_threshold.json").read_text())
    compare = json.loads((VERIF / "calibrated_threshold.json").read_text())
    assert baseline["threshold"] != compare["threshold"], (
        "each pipeline must calibrate its own threshold"
    )
    assert baseline["status"] == compare["status"] == "frozen"


def test_experiment_nine_summary_reports_both_pipelines() -> None:
    if not VERIF.is_dir():
        pytest.skip("Experiment 9 has not been run")
    text = acp.render_verification_comparison_summary(AGG)
    for column in ("LFW: YuNet+SFace", "LFW: SCRFD+ArcFace",
                   "CPLFW: YuNet+SFace", "CPLFW: SCRFD+ArcFace"):
        assert column in text, column
    assert "Frozen threshold" in text
    assert "not available" not in text


def test_a_missing_experiment_nine_gives_an_instruction(tmp_path: Path) -> None:
    text = acp.render_verification_comparison_summary(tmp_path)
    assert "not available yet" in text
    assert "option 12" in text


@pytest.mark.parametrize("exp", ["1-2", "3", "4", "5", "6", "7", "8", "9"])
def test_the_overview_table_covers_every_experiment(exp: str) -> None:
    text = acp.render_experiment_comparison_table(AGG)
    rows = [l for l in text.splitlines() if l.strip().startswith(exp + " ")]
    assert rows, f"experiment {exp} missing from the overview table"


def test_the_overview_table_names_models_and_task_for_each_row() -> None:
    text = acp.render_experiment_comparison_table(AGG)
    assert "YuNet + SFace" in text and "SCRFD + ArcFace" in text
    assert "logistic regression" in text
    # Both task types must be distinguished, never pooled.
    assert "1:1" in text and "1:N" in text
    assert "never pooled" in text
    assert "Only the logistic regression in Experiment 7 is trained" in text


def test_the_overview_table_marks_unrun_experiments_rather_than_hiding_them(
    tmp_path: Path,
) -> None:
    """A missing row would read as a gap in the method, not in what was run."""
    text = acp.render_experiment_comparison_table(tmp_path)
    assert text.count("not run yet") >= 7
    for exp in ("1-2", "3", "9"):
        assert any(l.strip().startswith(exp + " ") for l in text.splitlines()), exp


# --- Experiment 11: the classifier on the comparison pipeline ---------------------

ARC_REVIEW = AGG / "arcface_review"


def _skip_without_experiment_eleven() -> None:
    """The directory is created when the run starts, so presence of the
    directory is not evidence the run finished."""
    if not (ARC_REVIEW / "ml_review_test_metrics.json").is_file():
        pytest.skip("Experiment 11 has not completed in this checkout")


def test_experiment_eleven_uses_a_separate_run_cache() -> None:
    """The canonical cache is keyed by partition alone, so a second pipeline
    sharing the default path would overwrite the baseline runs that
    Experiments 6 to 10 depend on."""
    assert acp.ARCFACE_RUN_CACHE != acp.CANONICAL_RUN_CACHE
    assert acp.ARCFACE_RUN_CACHE.name.startswith("canonical_arcface")
    baseline = acp.canonical_cache_path("development", acp.CANONICAL_RUN_CACHE)
    compare = acp.canonical_cache_path("development", acp.ARCFACE_RUN_CACHE)
    assert baseline != compare


def test_experiment_eleven_writes_to_its_own_directory() -> None:
    assert acp.ARCFACE_REVIEW_DIRNAME == "arcface_review"
    _skip_without_experiment_eleven()
    # Both chains write ml_review_test_metrics.json; they must not collide.
    assert (AGG / "ml_review_test_metrics.json").is_file()
    assert (ARC_REVIEW / "ml_review_test_metrics.json").is_file()


def test_experiment_eleven_artefacts_name_the_right_pipeline() -> None:
    """primary_pipeline_description reports the OpenCV names unconditionally,
    so an unguarded run would label these as YuNet + SFace."""
    _skip_without_experiment_eleven()
    for name in ("ml_review_test_metrics.json", "ml_review_model.json",
                 "bfw_open_set_threshold.json"):
        path = ARC_REVIEW / name
        if not path.is_file():
            continue
        pipeline = json.loads(path.read_text())["pipeline"]
        assert "arcface" in pipeline["pipeline_name"].lower(), name
        assert pipeline["embedding_dimensions"] == 512, name


def test_experiment_eleven_uses_its_own_comparator_threshold() -> None:
    """A cutoff calibrated for SFace means nothing in ArcFace's space."""
    _skip_without_experiment_eleven()
    baseline = json.loads((AGG / "bfw_open_set_threshold.json").read_text())
    compare = json.loads((ARC_REVIEW / "bfw_open_set_threshold.json").read_text())
    target = str(acp.PRIMARY_FPIR_TARGET)
    assert (baseline["operating_points"][target]["threshold"]
            != compare["operating_points"][target]["threshold"])
    assert compare["status"] == acp.OPEN_SET_STATUS_FROZEN


def test_experiment_eleven_summary_reports_both_pipelines() -> None:
    _skip_without_experiment_eleven()
    text = acp.render_arcface_review_summary(AGG)
    for column in ("SFace threshold", "SFace + classifier",
                   "ArcFace threshold", "ArcFace + classifier"):
        assert column in text, column
    # The finding is whether the classifier moves the burden the same way.
    assert "review burden" in text
    assert "not available" not in text


def test_a_missing_experiment_eleven_gives_an_instruction(tmp_path: Path) -> None:
    text = acp.render_arcface_review_summary(tmp_path)
    assert "not available yet" in text
    assert "option 14" in text


def test_experiment_eleven_is_wired_to_the_menu() -> None:
    assert acp.MENU_PREVIEW_KEYS.get("14") == "arcface-review"
    for mode in ("arcface-review", "arcface-review-summary"):
        assert mode in acp.MODES, mode
    preview = acp.render_experiment_preview("arcface-review")
    assert "separate directory" in preview and "separate run cache" in preview


# --- The 1:1 rows are separated by dataset ----------------------------------------


def test_the_overview_separates_lfw_from_cplfw() -> None:
    """Merging them would hide that the coverage difference runs in opposite
    directions on the two datasets."""
    text = acp.render_experiment_comparison_table(AGG)
    nine = next(l for l in text.splitlines() if l.strip().startswith("9 "))
    ten = next(l for l in text.splitlines() if l.strip().startswith("10 "))
    assert "LFW" in nine and "CPLFW" not in nine
    assert "CPLFW" in ten
    assert "SCRFD + ArcFace" in nine and "SCRFD + ArcFace" in ten


def test_the_overview_includes_experiment_eleven() -> None:
    text = acp.render_experiment_comparison_table(AGG)
    row = next(l for l in text.splitlines() if l.strip().startswith("11 "))
    assert "classifier" in row


# --- The model comparison table ---------------------------------------------------


@pytest.mark.parametrize(
    "model", ["YuNet", "SFace", "SCRFD", "ArcFace", "Logistic regression"])
def test_the_model_table_covers_every_model(model: str) -> None:
    assert model in acp.render_model_comparison_table(AGG), model


def test_the_model_table_marks_only_the_classifier_as_trained_here() -> None:
    text = acp.render_model_comparison_table(AGG)
    rows = [l for l in text.splitlines()
            if any(m in l for m in ("YuNet", "SFace", "SCRFD", "ArcFace", "Logistic"))]
    trained = [l for l in rows if l.rstrip().split()[-2:][0] == "yes"
               or " yes " in l]
    assert len(trained) == 1 and "Logistic" in trained[0]
    assert "Only the logistic regression is fitted by this project" in text


def test_the_model_table_reports_measured_cost() -> None:
    text = acp.render_model_comparison_table(AGG)
    for label in ("Detection, mean per image", "Embedding, mean per image",
                  "Complete pipeline, mean per image", "Weight files on disk"):
        assert label in text, label
    assert " ms" in text and " MB" in text
    # A threshold belongs to the model that produced it.
    assert "never applied to the other" in text


# --- Experiment 12: the crossed pipelines ------------------------------------

MIXED = AGG / "mixed_pipelines"


def _skip_without_experiment_twelve(crossing: str = "scrfd-sface") -> None:
    """The directory appears when the run starts, so its presence is not
    evidence that the crossing finished."""
    if not (MIXED / crossing / "bfw_open_set_test_metrics.json").is_file():
        pytest.skip(f"Experiment 12 ({crossing}) has not completed in this checkout")


def test_each_crossing_has_its_own_run_cache() -> None:
    """Sharing a cache base would let one crossing overwrite the scored run of
    another, or of the baseline the whole study depends on."""
    paths = {acp.mixed_run_cache(c) for c in acp.MIXED_CROSSINGS}
    assert len(paths) == len(acp.MIXED_CROSSINGS)
    assert acp.CANONICAL_RUN_CACHE not in paths
    assert acp.ARCFACE_RUN_CACHE not in paths


def _skip_without_the_comparison_models() -> None:
    """Describing a crossing re-verifies both model files, so the optional
    comparison pack must be present for these checks to mean anything."""
    config = acp.EnvironmentConfig.load()
    if not acp.arcface_preconditions(config)["ready"]:
        pytest.skip("The optional comparison models are not installed")


def test_the_crossings_are_described_with_the_models_they_actually_use() -> None:
    """A crossing labelled with the wrong pipeline name would misattribute its
    result to a pipeline that never produced it."""
    _skip_without_the_comparison_models()
    config = acp.EnvironmentConfig.load()
    scrfd_sface = acp.mixed_pipeline_description("scrfd-sface", config)
    assert scrfd_sface.embedding_dimensions == acp.EMBEDDING_DIMENSIONS
    assert "SCRFD" in scrfd_sface.detector_name
    assert "SFace" in scrfd_sface.embedding_model_name

    yunet_arcface = acp.mixed_pipeline_description("yunet-arcface", config)
    assert yunet_arcface.embedding_dimensions == 512
    assert "YuNet" in yunet_arcface.detector_name
    assert "ArcFace" in yunet_arcface.embedding_model_name

    # The two halves must come from different pipelines, or it is not a crossing.
    assert scrfd_sface.model_sha256["detector"] != yunet_arcface.model_sha256["detector"]
    assert scrfd_sface.model_sha256["recognition"] != yunet_arcface.model_sha256["recognition"]


def test_an_unknown_crossing_is_refused() -> None:
    with pytest.raises(ValueError):
        acp.mixed_pipeline_description("yunet-sface")


def test_the_crossed_summary_reports_its_absence_gracefully() -> None:
    """Every summary must say what to run rather than fail, because the
    crossings are optional and need the comparison models."""
    text = acp.render_mixed_pipeline_summary(Path("/nonexistent-results-root"))
    assert "not available yet" in text
    assert "mixed-pipelines" in text


def test_experiment_twelve_is_wired_into_the_menu() -> None:
    assert "mixed-pipelines" in acp.MODES
    assert "mixed-pipelines-summary" in acp.MODES
    assert acp.MENU_PREVIEW_KEYS["16"] == "mixed-pipelines"
    assert "mixed-pipelines" in acp.EXPERIMENT_PREVIEWS
    assert "16. Run Experiment 12" in acp.MENU_TEXT
    assert "17. Show the saved Experiment 12" in acp.MENU_TEXT


def test_experiment_twelve_artefacts_name_the_crossing_that_produced_them() -> None:
    _skip_without_experiment_twelve()
    expected = {"scrfd-sface": 128, "yunet-arcface": 512}
    for crossing, dimensions in expected.items():
        path = MIXED / crossing / "bfw_open_set_test_metrics.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        assert payload["pipeline"]["pipeline_name"] == f"mixed-{crossing}"
        assert payload["pipeline"]["embedding_dimensions"] == dimensions
        assert payload["crossing"] == crossing


def test_each_crossing_freezes_a_threshold_of_its_own() -> None:
    """A threshold belongs to the embedding space that produced it, so no two
    pipelines may report the same frozen operating point by inheritance."""
    _skip_without_experiment_twelve()
    target = str(acp.PRIMARY_FPIR_TARGET)
    baseline = json.loads((AGG / "bfw_open_set_threshold.json").read_text())
    baseline_threshold = baseline["operating_points"][target]["threshold"]
    for crossing in acp.MIXED_CROSSINGS:
        path = MIXED / crossing / "bfw_open_set_threshold.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        assert payload["status"] == acp.OPEN_SET_STATUS_FROZEN
        assert payload["operating_points"][target]["threshold"] != baseline_threshold


def test_the_crossings_do_not_overwrite_the_baseline_artefacts() -> None:
    _skip_without_experiment_twelve()
    baseline = json.loads((AGG / "bfw_open_set_test_metrics.json").read_text())
    # Experiment 6's own artefact still names the baseline pair, with the
    # digest keys that pipeline uses.
    assert baseline["pipeline_name"] == acp.MODEL_VERSION
    assert set(baseline["model_sha256"]) == {"yunet", "sface"}

"""Run-all orchestration tests; no models, datasets, or reports are opened."""

import pytest

import ACP_arden as acp


# Keep this expectation independent of the implementation's stage list: moving
# a producer after its consumers must fail the orchestration regression test.
EXPECTED_STAGES = (
    "action_check_environment",
    "action_verify_inputs",
    "action_self_test",
    "action_run_complete_evaluation",
    "action_run_open_set_evaluation",
    "action_run_ml_review",
    "action_run_pipeline_comparison",
    "action_run_verification_comparison",
    "action_run_arcface_review",
    "action_run_mixed_pipelines",
    "action_run_comparison_diagnostics",
    "action_run_comparative_statistics",
    "action_refresh_reports",
    "action_show_experiment_table",
)


@pytest.fixture
def recorded_stages(monkeypatch):
    calls = []

    def stub(name, uses_output_root):
        if uses_output_root:
            def action_with_root(output_root):
                calls.append((name, output_root))
                return 0
            return action_with_root

        def action():
            calls.append((name, None))
            return 0
        return action

    for index, name in enumerate(EXPECTED_STAGES):
        monkeypatch.setattr(acp, name, stub(name, index >= 3))

    def unexpected_action(*args, **kwargs):
        pytest.fail("Run all invoked a duplicate summary, extension, or interactive UI")

    for name in (
        "action_run_extensions",
        "launch_review_interface",
        "action_show_summary",
        "action_show_open_set_summary",
        "action_show_ml_review_summary",
        "action_show_pipeline_comparison_summary",
        "action_show_verification_comparison_summary",
        "action_show_arcface_review_summary",
        "action_show_mixed_pipeline_summary",
        "_print_saved_summary",
    ):
        monkeypatch.setattr(acp, name, unexpected_action)
    return calls


@pytest.mark.parametrize("use_default_root", [False, True])
def test_run_all_runs_each_stage_in_dependency_order(recorded_stages, tmp_path, capsys, use_default_root):
    output_root = acp.AGGREGATE_ROOT if use_default_root else tmp_path / "chosen-results"
    status = acp.action_run_all() if use_default_root else acp.action_run_all(output_root)

    assert status == 0
    assert recorded_stages == [
        (name, output_root if index >= 3 else None)
        for index, name in enumerate(EXPECTED_STAGES)
    ]
    assert "Run all complete" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [(7, 7), (acp.ConfigurationError("missing test prerequisite"), 1), (KeyboardInterrupt(), 130)],
    ids=["nonzero-status", "expected-error", "interrupted"],
)
def test_run_all_stops_before_dependent_stages(
    monkeypatch, recorded_stages, tmp_path, capsys, failure, expected_status
):
    failed_stage = "action_run_open_set_evaluation"

    def fail(output_root):
        recorded_stages.append((failed_stage, output_root))
        if isinstance(failure, BaseException):
            raise failure
        return failure

    monkeypatch.setattr(acp, failed_stage, fail)

    assert acp.action_run_all(tmp_path) == expected_status
    assert [name for name, _root in recorded_stages] == list(EXPECTED_STAGES[:5])
    captured = capsys.readouterr()
    output = (captured.out + captured.err).lower()
    assert "run all stopped" in output
    assert "experiment 6" in output
    assert "run all complete" not in output


def test_menu_option_100_runs_all_with_preview_and_preserves_status(monkeypatch, tmp_path):
    calls = []
    previews = []
    responses = iter(["100", "", "26"])

    def run_all(output_root):
        calls.append(output_root)
        return 7

    monkeypatch.setattr(acp, "action_run_all", run_all)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(responses))
    monkeypatch.setattr(acp, "render_experiment_preview", lambda key: previews.append(key) or "")

    assert acp.run_menu(tmp_path) == 7
    assert calls == [tmp_path]
    assert previews == ["all"]


def test_cli_all_honors_results_root_and_returns_action_status(monkeypatch, tmp_path):
    calls = []

    def run_all(output_root):
        calls.append(output_root)
        return 7

    monkeypatch.setattr(acp, "action_run_all", run_all)

    assert acp.main(["--mode", "all", "--results-root", str(tmp_path)]) == 7
    assert calls == [tmp_path]


@pytest.mark.parametrize(("evaluated", "expected_status"), [("yes", 0), ("no", 1)])
def test_pipeline_comparison_reports_status_without_hiding_summary(
    monkeypatch, tmp_path, capsys, evaluated, expected_status
):
    calls = []

    def run_comparison(output_root):
        calls.append(output_root)
        return {"evaluated": evaluated}

    monkeypatch.setattr(acp, "run_pipeline_comparison", run_comparison)
    monkeypatch.setattr(acp, "render_pipeline_plain_summary", lambda _root: "Plain summary")
    monkeypatch.setattr(acp, "render_pipeline_comparison_summary", lambda _root: "Technical summary")
    monkeypatch.setattr(acp, "render_plain_section", lambda text: text)
    monkeypatch.setattr(acp, "render_technical_section", lambda text: text)
    monkeypatch.setattr(acp, "render_reference_section", lambda: "References")

    assert acp.action_run_pipeline_comparison(tmp_path) == expected_status
    assert calls == [tmp_path]
    output = capsys.readouterr().out
    assert all(text in output for text in ("Plain summary", "Technical summary", "References"))

"""Regression for #136322: the configured temp directory can be inside the project."""

import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from agent.verification_evidence import (
    mark_workspace_edited,
    record_terminal_result,
    verification_status,
)
from agent.verification_stop import build_verify_on_stop_nudge


@pytest.mark.parametrize("nested_temp", [False, True])
@pytest.mark.parametrize("prefix", ["hermes-verify-", "hermes-ad-hoc-"])
@pytest.mark.parametrize("success", [False, True])
def test_real_temp_check_records_result_and_controls_stop_nudge(
    tmp_path, monkeypatch, nested_temp, prefix, success
):
    project = tmp_path / "project"
    project.mkdir()
    (project / "package.json").write_text("{}", encoding="utf-8")
    scratch = (project if nested_temp else tmp_path) / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "ledger"))
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "1")
    monkeypatch.setenv("TMPDIR", str(scratch))
    monkeypatch.setattr(tempfile, "tempdir", None)
    changed = str(project / "app.py")
    mark_workspace_edited(session_id="real-check", cwd=project, paths=[changed])
    assert build_verify_on_stop_nudge(session_id="real-check", changed_paths=[changed]) is not None

    with tempfile.NamedTemporaryFile(mode="w", prefix=prefix, suffix=".py", delete=False) as handle:
        script = Path(handle.name)
        handle.write(
            "import json\nfrom pathlib import Path\n"
            "assert json.loads(Path('package.json').read_text()) == {}\n"
            f"assert {success!r}, 'intentional failing control'\n"
            "print('AD_HOC_OK')\n"
        )
    try:
        args = [sys.executable, str(script)]
        result = subprocess.run(args, cwd=project, capture_output=True, text=True, timeout=20)
        event = record_terminal_result(
            command=shlex.join(args), cwd=project, session_id="real-check",
            exit_code=result.returncode, output=result.stdout + result.stderr,
        )
    finally:
        script.unlink()

    assert result.returncode == (0 if success else 1)
    assert event is not None
    assert event["kind"] == "ad_hoc" and event["scope"] == "targeted"
    assert verification_status(session_id="real-check", cwd=project)["status"] == (
        "passed" if success else "failed"
    )
    assert (build_verify_on_stop_nudge(session_id="real-check", changed_paths=[changed]) is None) == success


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("case", [
    "unprefixed", "outside_temp", "repo_under_shared_temp", "cat", "masked_failure", "unattributed_pipeline", "canonical_suite",
])
def test_non_verifying_commands_do_not_clear_nested_temp_workspace(tmp_path, monkeypatch, case):
    project = tmp_path / "project"
    project.mkdir()
    manifest = {"scripts": {"test": "echo canonical suite"}} if case == "canonical_suite" else {}
    (project / "package.json").write_text(json.dumps(manifest), encoding="utf-8")
    scratch = tmp_path if case == "repo_under_shared_temp" else project / "scratch"
    scratch.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "ledger"))
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "1")
    monkeypatch.setenv("TMPDIR", str(scratch))
    monkeypatch.setattr(tempfile, "tempdir", None)
    changed = str(project / "app.py")
    mark_workspace_edited(session_id="non-verifier", cwd=project, paths=[changed])
    script = (project if case in {"outside_temp", "repo_under_shared_temp"} else scratch) / (
        "ordinary.py" if case == "unprefixed" else "hermes-verify-control.py"
    )
    script.write_text("raise SystemExit(1)\n" if case == "masked_failure" else "assert True\n", encoding="utf-8")
    command = shlex.join([sys.executable, str(script)])
    if case == "cat":
        command = f"cat {shlex.quote(str(script))}"
    elif case == "masked_failure":
        command += " || true"
    elif case == "unattributed_pipeline":
        command += " | cat"
    try:
        result = subprocess.run(command, shell=True, cwd=project, capture_output=True, text=True, timeout=20)
        event = record_terminal_result(
            command=command, cwd=project, session_id="non-verifier",
            exit_code=result.returncode, output=result.stdout + result.stderr,
        )
    finally:
        script.unlink()

    assert result.returncode == 0
    assert event is None
    assert verification_status(session_id="non-verifier", cwd=project)["status"] == "unverified"
    assert build_verify_on_stop_nudge(session_id="non-verifier", changed_paths=[changed]) is not None

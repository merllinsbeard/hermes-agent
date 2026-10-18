from pathlib import Path
import json
import os
import subprocess
import sys

from scripts import run_tests_parallel as runner


def test_private_scratch_reaches_worker_and_cleanup(tmp_path):
    root = tmp_path / "owned"
    root.mkdir()
    report = tmp_path / "worker.json"
    probe = tmp_path / "test_probe.py"
    probe.write_text(
        "import json, os, tempfile\nfrom pathlib import Path\n"
        "def test_worker():\n"
        "    root = Path(os.environ['TMPDIR'])\n"
        "    actual = Path(tempfile.mkdtemp())\n"
        f"    Path({str(report)!r}).write_text(json.dumps([str(root), str(actual)]))\n"
        "    assert actual.parent == root\n", encoding="utf-8")
    command = [sys.executable, str(Path(runner.__file__)), "-j", "1", "--file-retries", "0",
               "--scratch-root", str(root), str(probe), "--", f"--confcutdir={tmp_path}", "-q"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    worker, actual = map(Path, json.loads(report.read_text(encoding="utf-8")))
    assert worker.parent == root
    assert actual.parent == worker
    assert not worker.exists()
    assert "1 tests passed" in result.stdout


def test_default_scratch_selection_is_preserved(tmp_path, monkeypatch):
    created = []
    monkeypatch.setattr(runner.os, "makedirs", lambda path, **kw: created.append(path))
    expected = (f"/var/tmp/hermes-pytest-{os.getuid()}" if os.name != "nt" and os.path.isdir("/var/tmp")
                else str(Path(runner.tempfile.gettempdir()) / "hermes-pytest"))
    assert runner._runner_scratch_root() == expected
    assert created == [expected]
    assert runner._runner_scratch_root(str(tmp_path / "private")) == str(tmp_path / "private")

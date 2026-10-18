from types import SimpleNamespace

import pytest

from tests.cron.test_job_inference_policy import authored, make_agent, sandbox


def test_native_title_paths_with_positive_interactive_control(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    from agent.job_inference_policy import read_admitted_inference
    from agent.turn_context import _maybe_title_session_at_turn_start
    from agent import title_generator
    from tests.waiting import wait_until

    calls = []
    def controlled_title_service(**kwargs):
        calls.append(kwargs["task"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content='{"title":"Synthetic interactive title"}'))])

    monkeypatch.setattr(title_generator, "call_llm", controlled_title_service)
    db = SessionDB(tmp_path / "synthetic-title-state.db")
    pin = read_admitted_inference(authored())
    assert pin is not None
    cron = make_agent(job_inference=pin, session_db=db)
    child = make_agent(job_inference=pin.descendant(), platform="subagent", session_db=db)
    interactive = make_agent(platform="cli", session_db=db)
    try:
        for agent in (cron, child, interactive):
            agent._ensure_db_session()
        _maybe_title_session_at_turn_start(cron, [{"role": "user", "content": "Synthetic cron task"}])
        _maybe_title_session_at_turn_start(child, [{"role": "user", "content": "Synthetic child task"}])
        child_row = db.get_session(child.session_id)
        assert child_row is not None and child_row["title"] == "Subagent: Synthetic child task"
        assert calls == []
        _maybe_title_session_at_turn_start(interactive, [{"role": "user", "content": "Synthetic interactive request"}])
        assert wait_until(lambda: (row := db.get_session(interactive.session_id)) is not None
                          and row["title"] == "Synthetic interactive title", timeout=10)
        assert calls == ["title_generation"]
        assert cron.job_inference == pin and child.job_inference == pin.descendant()
    finally:
        cron.close()
        child.close()
        interactive.close()
        db.close()


def test_cron_completion_titles_are_local_across_collision_and_lineage(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    from cron.scheduler import _set_cron_session_title
    from agent import title_generator

    monkeypatch.setattr(title_generator, "call_llm", lambda **kw: pytest.fail("cron title called a model"))
    db = SessionDB(tmp_path / "synthetic-cron-titles.db")
    try:
        for session_id in ("synthetic-first", "synthetic-second"):
            db.create_session(session_id, source="cron")
        assert _set_cron_session_title(db, "synthetic-first", "Cron: Synthetic check") == "Cron: Synthetic check"
        assert _set_cron_session_title(db, "synthetic-second", "Cron: Synthetic check") == "Cron: Synthetic check #2"
        db.end_session("synthetic-first", "compression")
        db.create_session("synthetic-tip", source="cron", parent_session_id="synthetic-first")
        assert _set_cron_session_title(db, "synthetic-tip", "Cron: Synthetic check") == "Cron: Synthetic check"
        tip = db.get_session("synthetic-tip")
        assert tip is not None and tip["title"] == "Cron: Synthetic check"
    finally:
        db.close()

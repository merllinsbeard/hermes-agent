import asyncio
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

from tests.cron.test_job_inference_policy import POLICY, authored
from tests.cron.test_job_inference_policy import sandbox


def test_exact_retry_cold_restore_and_compression_keep_admitted_selection(tmp_path, monkeypatch, request):
    scope = request.getfixturevalue("sandbox")
    from cron.jobs import create_job, update_job
    from gateway import run, session_cron
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from gateway.session_contract import Principal
    from gateway.session_local_recovery import restore_local_session
    from gateway.session_policy import restore_policy
    from hermes_state import SessionDB
    from hermes_state_local import local_receipt
    from hermes_state_runtime import begin_runtime_epoch, list_session_admissions, RuntimeStoreError
    from agent.job_inference_policy import read_admitted_inference

    job = create_job("Synthetic admission", "every 1h", **authored())
    config = {"model": {"default": "profile-original"}, "platform_toolsets": {"cli": []}}
    monkeypatch.setattr(run, "_load_gateway_config", lambda: config)
    db = SessionDB(tmp_path / "private-state.db")
    store = SessionStore(tmp_path / "private-sessions", GatewayConfig())
    runner = SimpleNamespace(_draining=False, session_store=store, adapters={})
    authority = SessionAuthority(runner, profile_id=str(scope.home), instance_id="synthetic-owner", db=db,
                                 epoch=begin_runtime_epoch(db, instance_id="synthetic-owner"))
    runner.session_authority = authority
    monkeypatch.setattr(authority, "_schedule", lambda ref: None)
    actor = Principal("synthetic-human", str(scope.home), frozenset({"session:create", "session:submit", "session:control"}), "test")
    params = {"job_id": job["id"], "request_id": "synthetic-fire", "extra_prompt": "synthetic"}

    async def admit_and_retry():
        receipt = await session_cron.operation(authority, "submit", params, actor)
        update_job(job["id"], {"model": "later-model", "inference_policy": ""})
        config["model"]["default"] = "profile-later"
        assert await session_cron.operation(authority, "submit", params, actor) == receipt
        with pytest.raises(RuntimeStoreError):
            await session_cron.operation(authority, "submit", {**params, "inference_policy": POLICY}, actor)
        return receipt

    try:
        receipt = asyncio.run(admit_and_retry())
        sid = receipt["session_id"]
        saved = local_receipt(db, sid)
        frozen = restore_policy(saved["policy"])
        pin = read_admitted_inference(json.loads(frozen.request_json)["cron_job"])
        assert (pin.route.model, pin.main_effort, pin.delegation_effort) == ("gpt-6.1-sol", "xhigh", "high")
        assert len(list_session_admissions(db, session_id=sid, pending_only=False)) == 1
        authority.sessions.clear()
        runner.adapters.clear()
        restored = restore_local_session(authority, sid)
        assert restored.session_id == sid
        assert runner.adapters[Platform.LOCAL].policies[sid].request_json == frozen.request_json

        db.end_session(sid, "compression")
        db.create_session("synthetic-tip", source="cron", parent_session_id=sid)
        authority.sessions.clear()
        restore_local_session(authority, sid)
        after = local_receipt(db, sid)
        assert after["entry"]["session_id"] == "synthetic-tip"
        restored_policy = runner.adapters[Platform.LOCAL].policies[sid]
        assert read_admitted_inference(json.loads(restored_policy.request_json)["cron_job"]) == pin
        assert len(list_session_admissions(db, session_id=sid, pending_only=False)) == 1
        from tests.cron.test_job_inference_policy import make_agent
        from tests.agent.test_job_inference_wire import built_request, controlled_client
        from agent.codex_runtime import run_codex_stream
        agent = make_agent(job_inference=read_admitted_inference(json.loads(restored_policy.request_json)["cron_job"]))
        sent = []
        try:
            with controlled_client(sent) as client:
                run_codex_stream(agent, built_request(agent), client=client)
            assert (sent[0]["body"]["model"], sent[0]["body"]["reasoning"]["effort"],
                    sent[0]["body"].get("service_tier")) == ("gpt-6.1-sol", "xhigh", None)
        finally:
            agent.close()

        corrupt = asdict(restored_policy)
        data = json.loads(corrupt["request_json"])
        data["cron_job"]["inference_policy"] = None
        corrupt["request_json"] = json.dumps(data)
        with pytest.raises(RuntimeStoreError, match="storage_unavailable"):
            restore_policy(corrupt)
        data["cron_job"].pop("inference_policy")
        corrupt["request_json"] = json.dumps(data)
        legacy = restore_policy(corrupt)
        assert read_admitted_inference(json.loads(legacy.request_json)["cron_job"]) is None
    finally:
        db.close()
        if store._db is not None and store._db is not db:
            store._db.close()

import asyncio
from dataclasses import asdict
import json
from types import SimpleNamespace
from typing import Any

import pytest

from tests.cron.test_job_inference_policy import POLICY, authored
from tests.cron.test_job_inference_policy import sandbox


def test_native_admission_cron_consumer_uses_profile_proxy(tmp_path, monkeypatch, request):
    import httpx
    from agent import credential_pool
    from agent.codex_runtime import run_codex_stream
    from cron import scheduler
    from cron.jobs import create_job
    from gateway import run, session_cron
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from gateway.session_contract import Principal
    from gateway.session_policy import restore_policy
    from hermes_cli import runtime_provider
    from hermes_state import SessionDB
    from hermes_state_local import local_receipt
    from hermes_state_runtime import begin_runtime_epoch
    from run_agent import AIAgent
    from tests.agent.test_job_inference_wire import built_request, response_stream

    scope = request.getfixturevalue("sandbox")
    config = {"model": {"provider": "openai-codex", "default": "gpt-6.1-sol",
                        "context_length": 128000,
                        "base_url": "https://proxy.example.invalid/v1"},
              "cron": {"preflight": False}, "agent": {"environment_probe": False},
              "platform_toolsets": {"cron": []}}
    (scope.home / "config.yaml").write_text(json.dumps(config))
    monkeypatch.setattr(run, "_load_gateway_config", lambda: config)
    monkeypatch.setattr(scheduler, "_init_cron_mcp_tools", lambda *a: None)
    entry = credential_pool.PooledCredential.from_dict("openai-codex", {
        "id": "synthetic-native", "label": "synthetic-native", "source": "manual", "priority": 0,
        "auth_type": "api_key", "access_token": "synthetic-native-key",
        "base_url": "https://chatgpt.com/backend-api/codex"})
    pool = credential_pool.CredentialPool("openai-codex", [entry])
    monkeypatch.setattr(runtime_provider, "load_pool", lambda provider: pool)
    monkeypatch.setattr(credential_pool, "load_pool", lambda provider: pool)
    monkeypatch.setattr(runtime_provider, "resolve_codex_runtime_credentials", lambda: {
        "api_key": "synthetic-native-key", "base_url": "https://chatgpt.com/backend-api/codex"})
    sent = []

    def outgoing(http_request):
        sent.append((str(http_request.url), http_request.headers["authorization"], json.loads(http_request.content)))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=response_stream())

    monkeypatch.setattr(AIAgent, "_build_keepalive_http_client", lambda *a, **k:
                        httpx.Client(transport=httpx.MockTransport(outgoing)))
    job = create_job("Synthetic native consumer", "every 1h", **authored(base_url=None), enabled_toolsets=[])
    db = SessionDB(tmp_path / "consumer.db")
    store = SessionStore(tmp_path / "consumer-sessions", GatewayConfig())
    runner = SimpleNamespace(_draining=False, session_store=store, adapters={})
    authority = SessionAuthority(runner, profile_id=str(scope.home), instance_id="synthetic-consumer", db=db,
        epoch=begin_runtime_epoch(db, instance_id="synthetic-consumer"))
    runner.session_authority = authority
    monkeypatch.setattr(authority, "_schedule", lambda ref: None)
    actor = Principal("synthetic-human", str(scope.home),
                      frozenset({"session:create", "session:submit", "session:control"}), "test")
    agent = None
    try:
        receipt = asyncio.run(session_cron.operation(authority, "submit", {
            "job_id": job["id"], "request_id": "native-consumer", "extra_prompt": None}, actor))
        policy = restore_policy(local_receipt(db, receipt["session_id"])["policy"])
        admitted = json.loads(policy.request_json)["cron_job"]
        jc = scheduler._load_cron_job_config(admitted, job["id"], job["name"])
        setup = scheduler._resolve_cron_agent_setup(admitted, job["id"], job["name"], jc)
        agent = scheduler._construct_cron_agent(AIAgent, admitted, jc.cfg, setup,
            workdir=None, session_id=None, session_db=None)
        result = run_codex_stream(agent, built_request(agent), client=agent.client)
        assert result.output[0].content[0].text == "synthetic-ok"
        assert [(url, key) for url, key, body in sent] == [
            ("https://proxy.example.invalid/v1/responses", "Bearer synthetic-native-key")]
        assert (sent[0][2]["model"], sent[0][2]["reasoning"]["effort"], sent[0][2].get("service_tier")) == (
            "gpt-6.1-sol", "xhigh", None)
        assert not admitted.get("base_url")
    finally:
        if agent is not None:
            agent.close()
        db.close()
        if store._db is not None and store._db is not db:
            store._db.close()


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
    config: dict[str, Any] = {"model": {"default": "profile-original"}, "platform_toolsets": {"cli": []}}
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
        assert pin is not None
        assert (pin.route.model, pin.main_effort, pin.delegation_effort) == ("gpt-6.1-sol", "xhigh", "high")
        assert len(list_session_admissions(db, session_id=sid, pending_only=False)) == 1
        authority.sessions.clear()
        runner.adapters.clear()
        restored = restore_local_session(authority, sid)
        assert restored.session_id == sid
        assert runner.adapters[Platform.LOCAL].policies[sid].request_json == frozen.request_json

        db.publish_compression_child(parent_session_id=sid, child_session_id="synthetic-tip", source="cron",
            messages=[{"role": "user", "content": "Synthetic compressed context"}], require_compression_lease=False)
        authority.sessions.clear()
        if store._db is not None and store._db is not db:
            store._db.close()
        store = SessionStore(tmp_path / "private-sessions", GatewayConfig())
        runner.session_store = store
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

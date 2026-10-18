import asyncio
from dataclasses import asdict
import json
from types import SimpleNamespace
from typing import Any

import pytest

from tests.cron.test_job_inference_policy import POLICY, authored
from tests.cron.test_job_inference_policy import sandbox
from tests.cron.test_job_inference_policy import native_binding


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


@pytest.mark.parametrize("explicit", [False, True])
def test_exact_retry_cold_restore_and_compression_keep_admitted_selection(tmp_path, monkeypatch, request, native_binding, explicit):
    scope = request.getfixturevalue("sandbox")
    from cron.jobs import create_job, update_job
    from gateway import run, session_cron
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from gateway.session_contract import Principal, SessionRef
    from gateway.session_local_recovery import restore_local_session
    from gateway.session_policy import restore_policy
    from hermes_state import SessionDB
    from hermes_state_local import local_receipt
    from hermes_state_runtime import begin_runtime_epoch, list_session_admissions, RuntimeStoreError
    from agent.job_inference_policy import read_admitted_inference

    from tests.cron.test_job_inference_policy import ENDPOINT
    env = native_binding
    env.install([("original", "synthetic-original", ENDPOINT, 0)])
    job = create_job("Synthetic admission", "every 1h", **authored(base_url=ENDPOINT if explicit else None), enabled_toolsets=[])
    config: dict[str, Any] = env.config
    config["model"]["extra_headers"] = {"X-Synthetic-Context": "synthetic-saved-context"}
    monkeypatch.setattr(run, "_load_gateway_config", lambda *a: config)
    db = SessionDB(scope.home / "private-state.db")
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
        config["model"]["base_url"] = "https://later-profile.example.invalid/v1"
        with monkeypatch.context() as retry:
            retry.setattr("cron.jobs.get_job", lambda *a: pytest.fail("exact retry read latest authored job"))
            retry.setattr(run, "_load_gateway_config", lambda *a: pytest.fail("exact retry read current config"))
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
        assert pin.route.endpoint == (ENDPOINT if explicit else None)
        assert frozen.config_secret_ref is not None
        assert json.loads(frozen.config_json)["model"]["extra_headers"] is None
        assert env.loads == []
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
        runner = SimpleNamespace(_draining=False, session_store=store, adapters={})
        authority = SessionAuthority(runner, profile_id=str(scope.home), instance_id="synthetic-restored", db=db,
            epoch=begin_runtime_epoch(db, instance_id="synthetic-restored"))
        runner.session_authority = authority
        monkeypatch.setattr(authority, "_schedule", lambda ref: None)
        restore_local_session(authority, sid)
        after = local_receipt(db, sid)
        assert after["entry"]["session_id"] == "synthetic-tip"
        restored_policy = runner.adapters[Platform.LOCAL].policies[sid]
        assert read_admitted_inference(json.loads(restored_policy.request_json)["cron_job"]) == pin
        assert len(list_session_admissions(db, session_id=sid, pending_only=False)) == 1
        from cron import scheduler
        from run_agent import AIAgent
        from hermes_cli import runtime_provider
        from tests.agent.test_job_inference_wire import built_request
        from agent.codex_runtime import run_codex_stream
        env.install([("current", "synthetic-current", ENDPOINT, 0)])
        monkeypatch.setattr(runtime_provider, "resolve_codex_runtime_credentials", lambda: {
            "api_key": "synthetic-current", "base_url": ENDPOINT})
        executions = []

        def consume(frozen_job, **kw):
            executions.append(frozen_job)
            jc = scheduler._load_cron_job_config(frozen_job, job["id"], job["name"])
            setup = scheduler._resolve_cron_agent_setup(frozen_job, job["id"], job["name"], jc)
            agent = scheduler._construct_cron_agent(AIAgent, frozen_job, jc.cfg, setup,
                workdir=None, session_id=None, session_db=None)
            try:
                assert jc.cfg["model"]["base_url"] == "https://proxy.example.invalid/v1"
                assert jc.cfg["model"]["extra_headers"] == {"X-Synthetic-Context": "synthetic-saved-context"}
                result = run_codex_stream(agent, built_request(agent), client=agent.client)
                return True, "synthetic-document", result.output[0].content[0].text, None
            finally:
                agent.close()

        monkeypatch.setattr(scheduler, "run_job", consume)
        from hermes_state_runtime import claim_session_input, settle_session_input
        row = claim_session_input(db, epoch=authority.epoch, session_id=sid)
        assert isinstance(row, dict) and row["status"] == "started"
        result = asyncio.run(session_cron.execute(authority, SessionRef(str(scope.home), sid), row, restored_policy))
        assert result == "synthetic-ok"
        settle_session_input(db, epoch=authority.epoch, admission_id=row["admission_id"], generation=row["generation"],
            outcome="completed", result={"final_response": result})
        assert asyncio.run(session_cron.operation(authority, "submit", params, actor)) == receipt
        assert claim_session_input(db, epoch=authority.epoch, session_id=sid) is None
        assert len(executions) == 1 and local_receipt(db, sid)["policy"]["request_json"] == restored_policy.request_json
        assert [(r["url"], r["authorization"], r["body"]["model"], r["body"]["reasoning"]["effort"],
                 r["body"].get("service_tier"), r["headers"].get("x-synthetic-context")) for r in env.sent] == [
            ((ENDPOINT if explicit else "https://proxy.example.invalid/v1") + "/responses", "Bearer synthetic-current",
             "gpt-6.1-sol", "xhigh", None, "synthetic-saved-context")]

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

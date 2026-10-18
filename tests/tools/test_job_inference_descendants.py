import json
import threading

import httpx
from openai import OpenAI
import pytest

from tests.cron.test_job_inference_policy import ENDPOINT, admitted, authored, construct_native, make_agent, sandbox
from tests.cron.test_job_inference_policy import native_binding
from tests.agent.test_job_inference_wire import response_event, response_stream

__all__ = ["native_binding", "sandbox"]


def install_transport(monkeypatch, observed, *, gate=None):
    from run_agent import AIAgent

    def client_for(agent, **kwargs):
        def send(request):
            if gate is not None:
                gate[0].set()
                assert gate[1].wait(20)
            body = json.loads(request.content)
            depth = getattr(agent, "_delegate_depth", 0)
            observed.append((depth, body["model"], agent.provider, body["reasoning"]["effort"], body.get("service_tier")))
            event = response_event()
            encoded = json.dumps(body.get("input", []))
            if depth == 1 and "nested synthetic" in encoded and "function_call_output" not in encoded:
                event["response"]["output"] = [{"type": "function_call", "id": "synthetic-function",
                    "call_id": "synthetic-call", "name": "delegate_task",
                    "arguments": json.dumps({"goal": "synthetic leaf"})}]
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  content=response_stream(event))
        return OpenAI(api_key="synthetic-not-a-credential", base_url=agent.base_url, max_retries=0,
                      http_client=httpx.Client(transport=httpx.MockTransport(send)))

    monkeypatch.setattr(AIAgent, "_create_request_openai_client", client_for)


def test_registry_single_batch_and_nested_inherit_whole_pin(monkeypatch):
    from agent.job_inference_policy import read_admitted_inference
    import tools.delegate_tool
    from tools.registry import registry
    from gateway.session_context import set_session_vars, clear_session_vars

    observed = []
    install_transport(monkeypatch, observed)
    tokens = set_session_vars(async_delivery=False)
    parent = make_agent(job_inference=read_admitted_inference(admitted()), enabled_toolsets=["delegation"])
    try:
        raw_first = registry.dispatch("delegate_task", {"goal": "synthetic single", "model": "ignored-model"}, parent_agent=parent)
        assert isinstance(raw_first, str)
        first = json.loads(raw_first)
        assert first["results"][0]["summary"] == "synthetic-ok"
        raw_batch = registry.dispatch("delegate_task", {"tasks": [
            {"goal": "synthetic batch one", "model": "ignored-task-model"}, {"goal": "synthetic batch two"}]}, parent_agent=parent)
        assert isinstance(raw_batch, str)
        batch = json.loads(raw_batch)
        assert [entry["summary"] for entry in batch["results"]] == ["synthetic-ok", "synthetic-ok"]
        raw_nested = registry.dispatch("delegate_task", {"goal": "nested synthetic"}, parent_agent=parent)
        assert isinstance(raw_nested, str)
        nested = json.loads(raw_nested)
        assert nested["results"][0]["summary"] == "synthetic-ok"
        assert sorted(row[0] for row in observed) == [1, 1, 1, 1, 1, 2]
        assert [row[1:] for row in observed] == [("gpt-6.1-sol", "openai-codex", "high", None)] * 6
    finally:
        parent.close()
        clear_session_vars(tokens)


def test_background_pin_survives_scope_beside_legacy_agent(monkeypatch):
    from agent.job_inference_policy import read_admitted_inference
    from tools.registry import registry
    import tools.delegate_tool
    from gateway.session_context import set_session_vars, clear_session_vars
    from tests.agent.test_job_inference_wire import controlled_client, built_request
    from agent.codex_runtime import run_codex_stream
    from tests.waiting import wait_until
    from tools.async_delegation import list_async_delegations

    started, release = threading.Event(), threading.Event()
    observed = []
    install_transport(monkeypatch, observed, gate=(started, release))
    parent = make_agent(job_inference=read_admitted_inference(admitted()), platform="cli")
    tokens = set_session_vars(async_delivery=True)
    try:
        raw_reply = registry.dispatch("delegate_task", {"goal": "synthetic detached"}, parent_agent=parent)
        assert isinstance(raw_reply, str)
        reply = json.loads(raw_reply)
        assert reply["mode"] == "background", reply
        assert started.wait(20)
        clear_session_vars(tokens)
        tokens = []
        parent.job_inference = None
        legacy = make_agent(reasoning_config={"enabled": True, "effort": "low"}, service_tier="priority",
                            request_overrides={"service_tier": "priority"})
        sent = []
        with controlled_client(sent) as client:
            result = run_codex_stream(legacy, built_request(legacy), client=client)
        assert result.output[0].content[0].text == "synthetic-ok"
        assert (sent[0]["body"]["reasoning"]["effort"], sent[0]["body"]["service_tier"]) == ("low", "priority")
        legacy.close()
        release.set()
        assert wait_until(lambda: any(e["delegation_id"] == reply["delegation_id"] and e["status"] == "completed"
                                      for e in list_async_delegations()), timeout=20)
        assert observed == [(1, "gpt-6.1-sol", "openai-codex", "high", None)]
    finally:
        release.set()
        clear_session_vars(tokens)
        parent.close()


def test_conflicting_internal_override_refused_before_credentials():
    from agent.job_inference_policy import read_admitted_inference, JobInferenceError
    from tools.delegate_tool import delegate_task
    parent = make_agent(job_inference=read_admitted_inference(admitted()))
    try:
        with pytest.raises(JobInferenceError):
            delegate_task(goal="synthetic", parent_agent=parent,
                          credentials_cfg={"provider": "custom", "base_url": parent.base_url, "model": parent.model})
        assert parent.reasoning_config == {"enabled": True, "effort": "xhigh"}
    finally:
        parent.close()


def test_public_lifecycle_uses_the_common_child_pin(monkeypatch):
    from agent.job_inference_policy import read_admitted_inference, JobInferenceError
    from agent.subagent_lifecycle import SubagentLifecycleService, SubagentState
    observed = []
    install_transport(monkeypatch, observed)
    parent = make_agent(job_inference=read_admitted_inference(admitted()))
    service = SubagentLifecycleService(lambda: parent)
    try:
        with pytest.raises(JobInferenceError):
            service.launch({"goal": "Synthetic conflict", "provider": "custom", "model": parent.model})
        handle = service.launch({"goal": "Synthetic public lifecycle"})
        assert service.wait(handle, timeout_seconds=20).completed
        result = service.result(handle)
        assert result.terminal_state == SubagentState.SUCCEEDED and result.summary == "synthetic-ok"
        assert observed == [(1, "gpt-6.1-sol", "openai-codex", "high", None)]
    finally:
        parent.close()


@pytest.mark.parametrize("kind", ["single", "batch", "nested", "background"])
def test_registry_children_use_native_shared_pool_material(native_binding, kind):
    from hermes_cli.runtime_provider import frozen_runtime_config
    from tools.registry import registry
    import tools.delegate_tool
    from gateway.session_context import set_session_vars, clear_session_vars
    from tests.waiting import wait_until
    from tools.async_delegation import list_async_delegations
    env = native_binding
    env.config["delegation"] = {"model": "conflicting-default", "provider": "openrouter", "reasoning_effort": "low",
        "orchestrator_enabled": True, "max_spawn_depth": 3}
    pool = env.install([("A", "synthetic-A", ENDPOINT, 0),
                        ("B", "synthetic-B", "https://second.example.invalid/v1", 10)])

    def respond(body):
        event = response_event()
        encoded = json.dumps(body.get("input", []))
        names = {tool.get("name") for tool in body.get("tools", [])}
        if "delegate_task" in names and "native nested" in encoded and "function_call_output" not in encoded:
            event["response"]["output"] = [{"type": "function_call", "id": "native-call", "call_id": "native-nested",
                "name": "delegate_task", "arguments": json.dumps({"goal": "native leaf"})}]
        return event

    env.state.response_factory = respond
    tokens = set_session_vars(async_delivery=kind == "background")
    with frozen_runtime_config(env.config):
        parent, _ = construct_native(authored(base_url=None, enabled_toolsets=["delegation"]), env.config)
        parent.platform = "cli" if kind == "background" else "cron"
        parent_lease = pool.acquire_lease("A")
        try:
            args = {"tasks": [{"goal": "native batch one"}, {"goal": "native batch two"}]} if kind == "batch" else {
                "goal": "native nested" if kind == "nested" else "native single"}
            raw = registry.dispatch("delegate_task", args, parent_agent=parent)
            assert isinstance(raw, str)
            result = json.loads(raw)
            if kind == "background":
                assert result["mode"] == "background"
                clear_session_vars(tokens)
                tokens = []
                assert wait_until(lambda: any(r["delegation_id"] == result["delegation_id"] and r["status"] == "completed"
                    for r in list_async_delegations()), timeout=20)
            else:
                assert [r["summary"] for r in result["results"]] == ["synthetic-ok"] * (2 if kind == "batch" else 1)
            assert len(env.sent) == {"single": 1, "batch": 2, "nested": 3, "background": 1}[kind]
            assert set((r["body"]["model"], r["body"]["reasoning"]["effort"], r["body"].get("service_tier"))
                       for r in env.sent) == {("gpt-6.1-sol", "high", None)}
            pairs = {(r["url"], r["authorization"]) for r in env.sent}
            assert ("https://second.example.invalid/v1/responses", "Bearer synthetic-B") in pairs
            assert pairs <= {("https://second.example.invalid/v1/responses", "Bearer synthetic-B"),
                             ("https://proxy.example.invalid/v1/responses", "Bearer synthetic-A")}
            assert env.loads == ["openai-codex"] and parent._credential_pool is pool
            assert (parent.base_url, parent.api_key) == ("https://proxy.example.invalid/v1", "synthetic-A")
        finally:
            pool.release_lease(parent_lease)
            parent.close()
            clear_session_vars(tokens)


def test_child_lease_uses_exact_id_after_sibling_moves_current(native_binding, monkeypatch):
    from hermes_cli.runtime_provider import frozen_runtime_config
    from tools.delegate_tool_child_run import _lease_child_credential
    from agent.codex_runtime import run_codex_stream
    from tests.agent.test_job_inference_wire import built_request
    env = native_binding
    pool = env.install([("A", "synthetic-A", ENDPOINT, 0),
                        ("B", "synthetic-B", "https://second.example.invalid/v1", 10)])
    with frozen_runtime_config(env.config):
        parent, _ = construct_native(authored(base_url=None), env.config)
        held = pool.acquire_lease("A")
        child = make_agent(job_inference=parent.job_inference.descendant(), native_runtime=parent._job_runtime)
        raw_acquire = pool.acquire_lease

        def sibling_moves_current(credential_id=None):
            chosen = raw_acquire(credential_id)
            sibling = raw_acquire("A")
            pool.release_lease(sibling)
            return chosen

        monkeypatch.setattr(pool, "acquire_lease", sibling_moves_current)
        leased = None
        try:
            returned_pool, leased = _lease_child_credential(child)
            assert returned_pool is pool and leased == "B" and pool.current().id == "A"
            run_codex_stream(child, built_request(child), client=child.client)
            assert [(r["url"], r["authorization"], r["body"]["reasoning"]["effort"]) for r in env.sent] == [
                ("https://second.example.invalid/v1/responses", "Bearer synthetic-B", "high")]
            assert child._credential_pool is pool and child._credential_pool_entry_id == "B"
        finally:
            if leased:
                pool.release_lease(leased)
            pool.release_lease(held)
            child.close()
            parent.close()


def test_named_pool_child_adopts_qualified_native_material(native_binding):
    from hermes_cli.runtime_provider import frozen_runtime_config
    import tools.delegate_tool
    from tools.registry import registry
    from gateway.session_context import set_session_vars, clear_session_vars
    env = native_binding
    endpoint = "https://named.example.invalid/v1"
    env.config["model"].update(provider="openrouter", api_mode="chat_completions")
    env.config["providers"] = {"synthetic": {"base_url": endpoint, "api_mode": "codex_responses"}}
    pool = env.install([("A", "synthetic-A", endpoint, 0),
                        ("B", "synthetic-B", "https://named-second.example.invalid/v1", 10)], "synthetic")
    tokens = set_session_vars(async_delivery=False)
    with frozen_runtime_config(env.config):
        parent, _ = construct_native(authored(provider="synthetic", base_url=None,
            enabled_toolsets=["delegation"]), env.config)
        held = pool.acquire_lease("A")
        try:
            raw = registry.dispatch("delegate_task", {"goal": "native named child"}, parent_agent=parent)
            assert isinstance(raw, str)
            result = json.loads(raw)
            assert result["results"][0]["summary"] == "synthetic-ok"
            assert [(r["url"], r["authorization"], r["body"]["model"], r["body"]["reasoning"]["effort"],
                     r["body"].get("service_tier")) for r in env.sent] == [(
                "https://named-second.example.invalid/v1/responses", "Bearer synthetic-B", "gpt-6.1-sol", "high", None)]
            assert parent._credential_pool is pool and parent._credential_pool_entry_id == "A"
            assert env.loads == ["synthetic"]
        finally:
            pool.release_lease(held)
            parent.close()
            clear_session_vars(tokens)

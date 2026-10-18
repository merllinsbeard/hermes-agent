import json
import threading

import httpx
from openai import OpenAI
import pytest

from tests.cron.test_job_inference_policy import ENDPOINT, authored, make_agent, sandbox
from tests.agent.test_job_inference_wire import response_event, response_stream


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
    parent = make_agent(job_inference=read_admitted_inference(authored()), enabled_toolsets=["delegation"])
    try:
        first = json.loads(registry.dispatch("delegate_task", {"goal": "synthetic single", "model": "ignored-model"}, parent_agent=parent))
        assert first["results"][0]["summary"] == "synthetic-ok"
        batch = json.loads(registry.dispatch("delegate_task", {"tasks": [
            {"goal": "synthetic batch one", "model": "ignored-task-model"}, {"goal": "synthetic batch two"}]}, parent_agent=parent))
        assert [entry["summary"] for entry in batch["results"]] == ["synthetic-ok", "synthetic-ok"]
        nested = json.loads(registry.dispatch("delegate_task", {"goal": "nested synthetic"}, parent_agent=parent))
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
    parent = make_agent(job_inference=read_admitted_inference(authored()), platform="cli")
    tokens = set_session_vars(async_delivery=True)
    try:
        reply = json.loads(registry.dispatch("delegate_task", {"goal": "synthetic detached"}, parent_agent=parent))
        assert reply["mode"] == "background", reply
        assert started.wait(20)
        clear_session_vars(tokens)
        tokens = []
        parent.job_inference = None
        legacy = make_agent(reasoning_config={"enabled": True, "effort": "low"}, service_tier="priority")
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
    parent = make_agent(job_inference=read_admitted_inference(authored()))
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
    parent = make_agent(job_inference=read_admitted_inference(authored()))
    service = SubagentLifecycleService(lambda: parent)
    try:
        with pytest.raises(JobInferenceError):
            service.launch({"goal": "Synthetic conflict", "provider": "custom", "model": parent.model})
        handle = service.launch({"goal": "Synthetic public lifecycle"})
        assert service.wait(handle, timeout_seconds=20).completed
        result = service.collect(handle)
        assert result.terminal_state == SubagentState.SUCCEEDED and result.summary == "synthetic-ok"
        assert observed == [(1, "gpt-6.1-sol", "openai-codex", "high", None)]
    finally:
        parent.close()

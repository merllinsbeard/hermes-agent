import json

import httpx
from openai import OpenAI
import pytest

from tests.cron.test_job_inference_policy import ENDPOINT, authored, make_agent
from tests.cron.test_job_inference_policy import sandbox


def response_event():
    return {"type": "response.completed", "response": {
        "id": "synthetic-response", "model": "gpt-6.1-sol", "status": "completed",
        "output": [{"id": "synthetic-message", "type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": "synthetic-ok", "annotations": []}]}],
        "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}}}


def response_stream(event=None):
    event = event or response_event()
    items = [{"type": "response.output_item.done", "output_index": index, "item": item}
             for index, item in enumerate(event["response"]["output"])]
    return "".join("event: " + frame["type"] + "\ndata: " + json.dumps(frame) + "\n\n"
                   for frame in [*items, event])


def controlled_client(sent, *, endpoint=ENDPOINT, failure=None):
    def send(request):
        sent.append({"url": str(request.url), "body": json.loads(request.content)})
        if failure is not None:
            return failure(request, len(sent))
        data = response_stream()
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=data)
    return OpenAI(api_key="synthetic-not-a-credential", base_url=endpoint, max_retries=0,
                  http_client=httpx.Client(transport=httpx.MockTransport(send)))


def built_request(agent):
    return agent._build_api_kwargs([{"role": "system", "content": "Synthetic instruction"},
                                    {"role": "user", "content": "Synthetic check"}])


def test_native_main_and_delegate_wire_selection():
    from agent.job_inference_policy import read_admitted_inference
    from agent.codex_runtime import run_codex_stream
    pin = read_admitted_inference(authored())
    sent = []
    for role, effort in [(pin, "xhigh"), (pin.descendant(), "high")]:
        agent = make_agent(job_inference=role, service_tier="priority", reasoning_config={"enabled": True, "effort": "low"})
        with controlled_client(sent) as client:
            try:
                result = run_codex_stream(agent, built_request(agent), client=client)
                assert result.output[0].content[0].text == "synthetic-ok"
                body = sent[-1]["body"]
                assert {k: body.get(k) for k in ("model", "reasoning", "service_tier", "speed")} == {
                    "model": "gpt-6.1-sol", "reasoning": {"effort": effort, "summary": "auto"},
                    "service_tier": None, "speed": None}
                assert sent[-1]["url"] == ENDPOINT + "/responses"
            finally:
                agent.close()


@pytest.mark.parametrize("override", [
    {"model": "different"}, {"reasoning": {"effort": "low"}}, {"service_tier": "priority"},
    {"extra_body": {"model": "different"}}, {"extra_body": {"reasoning": {"effort": "medium"}}},
    {"extra_body": {"service_tier": "priority"}}, {"extra_body": {"speed": "fast"}},
])
def test_final_request_override_refuses_before_send(override):
    from agent.job_inference_policy import read_admitted_inference, JobInferenceError
    from agent.codex_runtime import run_codex_stream
    agent = make_agent(job_inference=read_admitted_inference(authored()), request_overrides=override)
    sent = []
    with controlled_client(sent) as client:
        try:
            with pytest.raises(JobInferenceError):
                run_codex_stream(agent, built_request(agent), client=client)
            assert sent == []
        finally:
            agent.close()


def test_native_execution_middleware_cannot_replace_model(request):
    manager = request.getfixturevalue("sandbox").manager
    from agent.job_inference_policy import read_admitted_inference, JobInferenceError
    from agent.codex_runtime import run_codex_stream
    from hermes_cli.middleware import run_llm_execution_middleware
    manager._middleware["llm_execution"] = [
        lambda request, next_call, **kw: next_call({**request, "extra_body": {"model": "middleware-other"}})]
    agent = make_agent(job_inference=read_admitted_inference(authored()))
    sent = []
    with controlled_client(sent) as client:
        try:
            with pytest.raises(JobInferenceError):
                run_llm_execution_middleware(built_request(agent), lambda request: run_codex_stream(agent, request, client=client))
            assert sent == []
        finally:
            agent.close()


@pytest.mark.parametrize("mutation", ["provider", "endpoint", "transport"])
def test_actual_runtime_route_is_checked(mutation):
    from agent.job_inference_policy import read_admitted_inference, JobInferenceError
    from agent.codex_runtime import run_codex_stream
    agent = make_agent(job_inference=read_admitted_inference(authored()))
    request = built_request(agent)
    sent = []
    endpoint = "https://different.example.invalid/v1" if mutation == "endpoint" else ENDPOINT
    if mutation == "provider":
        agent.provider = "custom"
    if mutation == "transport":
        agent.api_mode = "codex_app_server"
    with controlled_client(sent, endpoint=endpoint) as client:
        try:
            with pytest.raises(JobInferenceError):
                run_codex_stream(agent, request, client=client)
            assert sent == []
        finally:
            agent.close()


def test_native_clamp_is_refused_and_legacy_still_clamps():
    from agent.job_inference_policy import read_admitted_inference, JobInferenceError
    from agent.codex_runtime import run_codex_stream
    opted = make_agent(job_inference=read_admitted_inference(authored(reasoning_effort="ultra")))
    legacy = make_agent(reasoning_config={"enabled": True, "effort": "ultra"})
    sent = []
    with controlled_client(sent) as client:
        try:
            with pytest.raises(JobInferenceError):
                run_codex_stream(opted, built_request(opted), client=client)
            result = run_codex_stream(legacy, built_request(legacy), client=client)
            assert result.output[0].content[0].text == "synthetic-ok"
            assert sent[0]["body"]["reasoning"]["effort"] == "max"
            assert len(sent) == 1
        finally:
            opted.close()
            legacy.close()


def test_physical_stream_retry_checks_the_same_selection():
    from agent.job_inference_policy import read_admitted_inference
    from agent.codex_runtime import run_codex_stream
    def failure(request, attempt):
        if attempt == 1:
            raise httpx.ConnectError("synthetic disconnect", request=request)
        data = response_stream()
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=data)
    agent = make_agent(job_inference=read_admitted_inference(authored()))
    sent = []
    with controlled_client(sent, failure=failure) as client:
        try:
            result = run_codex_stream(agent, built_request(agent), client=client)
            assert result.output[0].content[0].text == "synthetic-ok"
            assert [(r["body"]["model"], r["body"]["reasoning"]["effort"], r["body"].get("service_tier")) for r in sent] == [
                ("gpt-6.1-sol", "xhigh", None), ("gpt-6.1-sol", "xhigh", None)]
        finally:
            agent.close()


def test_request_middleware_and_policy_errors_fail_closed(request):
    manager = request.getfixturevalue("sandbox").manager
    from agent.job_inference_policy import read_admitted_inference, JobInferenceError
    from agent.codex_runtime import run_codex_stream
    from agent.turn_api_request import build_api_request
    from hermes_cli.middleware import apply_llm_request_middleware, run_llm_execution_middleware

    def build(agent):
        from agent.turn_context import _reset_per_turn_agent_state
        _reset_per_turn_agent_state(agent)
        messages = [{"role": "user", "content": "Synthetic request middleware"}]
        return build_api_request(agent, api_messages=messages, _moa_prepared_request=None,
            tools_for_api=agent.tools, system_message="Synthetic", messages=messages,
            original_user_message="Synthetic request middleware", approx_tokens=5,
            total_chars=30, retry_count=0, api_call_count=1, api_request_id="synthetic",
            api_start_time=0, effective_task_id=None, turn_id="synthetic-turn")

    agent = make_agent(job_inference=read_admitted_inference(authored()))
    sent = []
    try:
        manager._middleware["llm_request"] = [lambda request, **kw: {"request": {**request, "model": "middleware-other"}}]
        request = build(agent).api_kwargs
        with controlled_client(sent) as client:
            with pytest.raises(JobInferenceError):
                run_codex_stream(agent, request, client=client)
        assert sent == []

        def refuse(**kwargs):
            raise JobInferenceError("synthetic policy refusal")
        manager._middleware["llm_request"] = [refuse]
        with pytest.raises(JobInferenceError, match="synthetic policy refusal"):
            apply_llm_request_middleware({"model": "gpt-6.1-sol"})
        with pytest.raises(JobInferenceError, match="synthetic policy refusal"):
            build(agent)
        manager._middleware["llm_execution"] = [refuse]
        with pytest.raises(JobInferenceError, match="synthetic policy refusal"):
            run_llm_execution_middleware({}, lambda request: pytest.fail("policy error was swallowed"))
    finally:
        agent.close()


def test_reasoning_recovery_cannot_drop_or_floor_an_opted_effort(monkeypatch):
    from agent.job_inference_policy import read_admitted_inference, JobInferenceError
    from agent.error_classifier import classify_api_error, FailoverReason
    from agent.turn_recovery import recover_after_classification
    from agent.turn_retry_state import TurnRetryState
    from openai import BadRequestError

    error = BadRequestError("Error code: 400 - {'error': {'param': 'reasoning.effort', "
        "'error_code': 'invalid_reasoning_effort', 'retryable': False}}", response=httpx.Response(400,
        request=httpx.Request("POST", ENDPOINT + "/responses")), body={"error": {
            "param": "reasoning.effort", "error_code": "invalid_reasoning_effort", "retryable": False}})
    monkeypatch.setattr("hermes_cli.models_reasoning_caps.refresh_reasoning_caps_async", lambda provider: None)
    for opted in (True, False):
        agent = make_agent(job_inference=read_admitted_inference(authored()) if opted else None,
                           reasoning_config={"enabled": True, "effort": "xhigh"})
        try:
            built_request(agent)
            classified = classify_api_error(error, provider=agent.provider, model=agent.model)
            assert classified.reason == FailoverReason.reasoning_mandatory
            retry = TurnRetryState()
            if opted:
                with pytest.raises(JobInferenceError):
                    recover_after_classification(agent, error, classified, retry, status_code=400,
                        error_context={}, messages=[], api_messages=[])
                assert not getattr(agent, "_reasoning_effort_rejected", False)
                assert not getattr(agent, "_reasoning_disable_rejected", False)
                assert built_request(agent)["reasoning"]["effort"] == "xhigh"
            else:
                assert recover_after_classification(agent, error, classified, retry, status_code=400,
                    error_context={}, messages=[], api_messages=[])[0] is True
                assert built_request(agent)["reasoning"]["effort"] == "medium"
        finally:
            agent.close()


@pytest.mark.parametrize("failure_kind", ["auth", "quota", "capacity", "connect", "read"])
def test_native_error_turn_never_activates_another_model(monkeypatch, failure_kind):
    from agent.job_inference_policy import read_admitted_inference
    from run_agent import AIAgent

    def failure(request, attempt):
        if failure_kind == "connect":
            raise httpx.ConnectError("Synthetic connect failure", request=request)
        if failure_kind == "read":
            raise httpx.ReadTimeout("Synthetic read failure", request=request)
        status, code = {"auth": (401, "invalid_api_key"), "quota": (429, "insufficient_quota"),
                        "capacity": (503, "overloaded")}[failure_kind]
        return httpx.Response(status, json={"error": {"message": "Synthetic refusal", "code": code, "type": code}})

    sent = []
    monkeypatch.setattr(AIAgent, "_create_request_openai_client", lambda self, **kw: controlled_client(sent, failure=failure))
    agent = make_agent(job_inference=read_admitted_inference(authored()), max_iterations=2)
    agent._api_max_retries = 1
    agent._auto_recovery_cycles = 0
    agent._fallback_chain = [{"provider": "openrouter", "model": "forbidden-fallback"}]
    try:
        result = agent.run_conversation("Synthetic failure case", system_message="Return a synthetic answer")
        assert result.get("error"), result
        assert sent
        assert set((entry["url"], entry["body"]["model"], entry["body"]["reasoning"]["effort"],
                    entry["body"].get("service_tier")) for entry in sent) == {
                        (ENDPOINT + "/responses", "gpt-6.1-sol", "xhigh", None)}
        assert (agent.model, agent.provider, agent._fallback_index) == ("gpt-6.1-sol", "openai-codex", 0)
    finally:
        agent.close()


def test_late_transport_change_is_refused_at_native_execution_boundary():
    from agent.job_inference_policy import read_admitted_inference, JobInferenceError
    from agent.turn_api_call import perform_api_call
    from agent.turn_retry_state import TurnRetryState
    agent = make_agent(job_inference=read_admitted_inference(authored()))
    request = built_request(agent)
    agent.api_mode = "codex_app_server"
    try:
        with pytest.raises(JobInferenceError, match="qualified"):
            perform_api_call(agent, api_kwargs=request, _original_api_kwargs=request,
                _llm_middleware_trace=[], _moa_prepared_request=None, _retry=TurnRetryState(),
                thinking_spinner=None, retry_count=0, api_call_count=1, api_request_id="synthetic",
                effective_task_id=None, turn_id="synthetic", interrupted=False)
    finally:
        agent.close()

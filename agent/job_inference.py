from __future__ import annotations

from typing import Any

from agent.job_inference_policy import JobInferenceError


def require_job_agent(agent: Any, client: Any = None) -> None:
    pin = getattr(agent, "job_inference", None)
    if pin is None:
        return
    pin.require_transport(agent.api_mode)
    pin.require_route(agent.provider, agent.model, agent.base_url)
    if getattr(agent, "acp_command", None):
        raise JobInferenceError("inference_policy does not support an external process transport")
    if client is not None:
        pin.require_route(agent.provider, agent.model, str(client.base_url))


def refuse_effort_recovery(agent: Any, error: Exception) -> None:
    if getattr(agent, "job_inference", None) is not None:
        raise JobInferenceError("inference_policy refuses reasoning recovery that changes effort") from error


def guard_codex_request(agent: Any, client: Any, request: dict) -> None:
    pin = getattr(agent, "job_inference", None)
    if pin is None:
        return
    require_job_agent(agent, client)
    extra = request.get("extra_body")
    if extra is not None and not isinstance(extra, dict):
        raise JobInferenceError("inference_policy cannot validate a non-object extra_body")
    # This is the SDK's post-transform merge, including its Omit semantics.
    from openai._base_client import _merge_mappings
    body = _merge_mappings(request, extra or {})
    pin.require_route(agent.provider, body.get("model"), str(client.base_url))
    reasoning = body.get("reasoning")
    if not isinstance(reasoning, dict) or reasoning.get("effort") != pin.effort:
        raise JobInferenceError("inference_policy reasoning effort changed; refusing inference")
    if (body.get("service_tier") not in (None, "default") or body.get("speed") is not None
            or body.get("serviceTier") not in (None, "default")):
        raise JobInferenceError("inference_policy requires the default service tier")


def delegation_routing(parent: Any, supplied: dict | None) -> dict:
    pin = parent.job_inference
    child = pin.descendant()
    if supplied is not None:
        child.require_route(supplied.get("provider", child.route.provider),
                            supplied.get("model", child.route.model), supplied.get("base_url", child.route.endpoint))
        if (supplied.get("reasoning_effort", child.effort) != child.effort
                or supplied.get("fallback_providers") or supplied.get("command")):
            raise JobInferenceError("inference_policy conflicts with delegation override")
        child.require_transport(supplied.get("api_mode", "codex_responses"))
    return {"provider": child.route.provider, "model": child.route.model, "base_url": child.route.endpoint,
            "api_mode": parent.api_mode, "fallback_providers": [],
            "request_overrides": dict(getattr(parent, "request_overrides", {}) or {})}

from __future__ import annotations

from typing import Any
import json

from agent.job_inference_policy import AgentRolePin, JobInferenceError


def resolve_job_runtime(pin: AgentRolePin, native_runtime: dict[str, Any] | None, *,
                        provider: Any, model: Any, base_url: Any, api_mode: Any, command: Any) -> dict[str, Any]:
    from hermes_cli.runtime_provider import resolve_runtime_provider
    route = pin.route
    pin.require_route(provider or route.provider, model or route.model, base_url or route.endpoint)
    if api_mode is not None:
        pin.require_transport(api_mode)
    if command:
        raise JobInferenceError("inference_policy does not support an external process transport")
    runtime = (dict(native_runtime) if native_runtime is not None else resolve_runtime_provider(
        requested=route.provider, target_model=route.model, explicit_base_url=route.endpoint))
    require_runtime(pin, runtime)
    return runtime


def assemble_job_tools(agent: Any, tool_defs: list[dict]) -> list[dict]:
    from agent.agent_init import config_context_length_for_runtime
    from agent.model_metadata import get_model_context_length
    from hermes_cli.runtime_provider import load_config
    from tools.tool_search import ToolSearchConfig, assemble_tool_defs
    config = load_config()
    context_length = get_model_context_length(agent.model, provider=agent.provider,
        base_url=agent.base_url, api_key=agent.api_key,
        config_context_length=config_context_length_for_runtime(agent, config))
    tools_config = config.get("tools")
    search_config = ToolSearchConfig.from_raw(tools_config.get("tool_search") if isinstance(tools_config, dict) else None)
    return assemble_tool_defs(tool_defs, context_length=context_length, config=search_config).tool_defs


def require_runtime(pin: Any, runtime: dict) -> None:
    from hermes_cli.models import normalize_provider
    identity = runtime.get("requested_provider") or runtime.get("provider")
    pin.require_route(identity, pin.route.model, runtime.get("base_url"))
    provider = normalize_provider(runtime.get("provider"))
    expected = "custom" if pin.route.provider.startswith("custom:") else pin.route.provider
    if (provider != expected or not runtime.get("base_url") or not runtime.get("api_key")
            or not isinstance(runtime.get("api_key"), str)):
        raise JobInferenceError("inference_policy requires coherent native HTTP credentials")
    pin.require_transport(runtime.get("api_mode"))
    if runtime.get("command"):
        raise JobInferenceError("inference_policy does not support an external process transport")


def require_job_agent(agent: Any, client: Any = None) -> None:
    pin = getattr(agent, "job_inference", None)
    if pin is None:
        return
    pin.require_transport(agent.api_mode)
    runtime = agent._job_runtime
    require_runtime(pin, runtime)
    pin.require_route(agent.requested_provider, agent.model, agent.base_url)
    from hermes_cli.route_identity import normalize_route_base_url
    if (agent.provider != runtime["provider"] or agent.api_key != runtime["api_key"]
            or normalize_route_base_url(agent.base_url) != normalize_route_base_url(runtime["base_url"])
            or agent._credential_pool is not runtime.get("credential_pool")
            or (runtime.get("credential_pool_entry_id") is not None
                and agent._credential_pool_entry_id != runtime["credential_pool_entry_id"])):
        raise JobInferenceError("inference_policy native binding changed; refusing inference")
    if getattr(agent, "acp_command", None):
        raise JobInferenceError("inference_policy does not support an external process transport")
    if client is not None:
        if (normalize_route_base_url(str(client.base_url)) != normalize_route_base_url(runtime["base_url"])
                or client.api_key != runtime["api_key"]):
            raise JobInferenceError("inference_policy client binding changed; refusing inference")


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
    _require_body(pin, body)
    import httpx
    headers = httpx.Headers(_merge_mappings(client.default_headers, request.get("extra_headers") or {}))
    if headers.get("authorization") != "Bearer " + agent._job_runtime["api_key"]:
        raise JobInferenceError("inference_policy authorization changed; refusing inference")


def _require_body(pin: Any, body: dict) -> None:
    if body.get("model") != pin.route.model:
        raise JobInferenceError("inference_policy model changed; refusing inference")
    reasoning = body.get("reasoning")
    if not isinstance(reasoning, dict) or reasoning.get("effort") != pin.effort:
        raise JobInferenceError("inference_policy reasoning effort changed; refusing inference")
    if (body.get("service_tier") not in (None, "default") or body.get("speed") is not None
            or body.get("serviceTier") not in (None, "default")):
        raise JobInferenceError("inference_policy requires the default service tier")


def install_job_request_guard(agent: Any, client: Any) -> None:
    if getattr(agent, "job_inference", None) is None:
        return
    hooks = client._client.event_hooks

    def check(request):
        require_job_agent(agent, client)
        from urllib.parse import urlsplit
        expected = urlsplit(str(client.base_url))
        actual = urlsplit(str(request.url))
        if (actual.scheme, actual.netloc, actual.path) != (
                expected.scheme, expected.netloc, expected.path.rstrip("/") + "/responses"):
            raise JobInferenceError("inference_policy physical destination changed; refusing inference")
        if request.headers.get("authorization") != "Bearer " + agent._job_runtime["api_key"]:
            raise JobInferenceError("inference_policy physical authorization changed; refusing inference")
        content = request.content
        if request.headers.get("content-encoding") == "zstd":
            from compression import zstd
            content = zstd.decompress(content)
        _require_body(agent.job_inference, json.loads(content))

    client._client.event_hooks = {**hooks, "request": [*hooks.get("request", ()), check]}


def delegation_routing(parent: Any, supplied: dict | None) -> dict:
    pin = parent.job_inference
    child = pin.descendant()
    if supplied is not None:
        child.require_route(supplied.get("provider", child.route.provider),
                            supplied.get("model", child.route.model), supplied.get("base_url", child.route.endpoint))
        if (supplied.get("reasoning_effort", child.effort) != child.effort
                or supplied.get("fallback_providers") or supplied.get("command") or supplied.get("api_key")):
            raise JobInferenceError("inference_policy conflicts with delegation override")
        child.require_transport(supplied.get("api_mode", "codex_responses"))
    return {"provider": child.route.provider, "model": child.route.model, "base_url": child.route.endpoint,
            "api_mode": parent.api_mode, "fallback_providers": [],
            "request_overrides": dict(getattr(parent, "request_overrides", {}) or {})}

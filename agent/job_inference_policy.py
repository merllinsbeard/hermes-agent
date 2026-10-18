from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping
from urllib.parse import urlsplit


class JobInferenceError(ValueError):
    """The requested job selection cannot be preserved. No inference is allowed."""


UNSET_POLICY = object()


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise JobInferenceError(f"inference_policy requires explicit {field}")
    return value.strip()


def _effort(value: Any, field: str) -> str:
    from hermes_constants import parse_reasoning_effort
    word = _text(value, field).lower()
    parsed = parse_reasoning_effort(word)
    if parsed is None:
        raise JobInferenceError(f"invalid inference_policy {field}")
    return "none" if parsed.get("enabled") is False else parsed["effort"]


def validate_job_inference(job: Mapping[str, Any], *, allow_clear: bool = False) -> dict:
    """Validate the whole authored definition; absence alone selects legacy behavior."""
    result = dict(job)
    if "inference_policy" not in result:
        return result
    raw = result["inference_policy"]
    if allow_clear and raw == "":
        result.pop("inference_policy")
        return result
    if (not isinstance(raw, dict)
            or set(raw) != {"delegation_reasoning_effort", "service_tier"}
            or raw["service_tier"] != "default"):
        raise JobInferenceError(
            "inference_policy requires delegation_reasoning_effort and service_tier='default'")
    provider = _text(result.get("provider"), "provider")
    if provider.lower() == "auto":
        raise JobInferenceError("inference_policy does not allow provider auto")
    _text(result.get("model"), "model")
    result["reasoning_effort"] = _effort(result.get("reasoning_effort"), "reasoning_effort")
    result["inference_policy"] = {
        "delegation_reasoning_effort": _effort(raw["delegation_reasoning_effort"], "delegation_reasoning_effort"),
        "service_tier": "default",
    }
    return result


@dataclass(frozen=True, slots=True)
class PinnedRoute:
    provider: str
    model: str
    endpoint: str


def _route(provider: Any, model: Any, endpoint: Any) -> PinnedRoute:
    from hermes_cli.model_normalize import normalize_model_for_provider
    from hermes_cli.models import normalize_provider
    from hermes_cli.route_identity import normalize_route_base_url
    from agent.model_metadata import strip_codex_context_variant_suffix

    identity = normalize_provider(_text(provider, "provider"))
    if identity == "auto":
        raise JobInferenceError("inference_policy does not allow provider auto")
    url = normalize_route_base_url(endpoint)
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        parsed.port
    except ValueError as exc:
        raise JobInferenceError("invalid inference_policy endpoint") from exc
    if (parsed.scheme not in {"http", "https"} or not hostname or parsed.username
            or parsed.password or any(ord(char) <= 0x20 for char in url)):
        raise JobInferenceError("inference_policy requires a resolved HTTP endpoint without credentials")
    return PinnedRoute(identity, strip_codex_context_variant_suffix(
        normalize_model_for_provider(_text(model, "model"), identity)), url)


def normalize_admitted_job(job: Mapping[str, Any], *, provider_defaults: dict) -> dict:
    """Freeze the effective endpoint in the existing job snapshot, without credentials."""
    result = validate_job_inference(job)
    if "inference_policy" not in result:
        return result
    from gateway.session_route_identity import resolved_route
    from hermes_cli.providers import get_provider
    identity, endpoint = resolved_route(result, provider_defaults)
    if not endpoint:
        definition = get_provider(identity, allow_network=False)
        endpoint = definition.base_url if definition is not None else ""
    route = _route(identity, result["model"], endpoint)
    result.update(provider=route.provider, model=route.model, base_url=route.endpoint)
    return result


@dataclass(frozen=True, slots=True)
class AgentRolePin:
    route: PinnedRoute
    main_effort: str
    delegation_effort: str
    role: Literal["main", "delegate"] = "main"

    @property
    def effort(self) -> str:
        return self.main_effort if self.role == "main" else self.delegation_effort

    def descendant(self) -> AgentRolePin:
        return replace(self, role="delegate")

    def reasoning_config(self) -> dict | None:
        from hermes_constants import parse_reasoning_effort
        return parse_reasoning_effort(self.effort)

    def require_route(self, provider: Any, model: Any, endpoint: Any) -> None:
        if _route(provider, model, endpoint) != self.route:
            raise JobInferenceError("inference_policy route changed; refusing inference")

    def require_transport(self, api_mode: str) -> None:
        if api_mode != "codex_responses":
            raise JobInferenceError("inference_policy requires qualified codex_responses transport")


def read_admitted_inference(job: Mapping[str, Any]) -> AgentRolePin | None:
    """Decode both fresh and restored receipts without consulting current profile defaults."""
    validated = validate_job_inference(job)
    if "inference_policy" not in validated:
        return None
    return AgentRolePin(
        _route(validated["provider"], validated["model"], validated.get("base_url")),
        validated["reasoning_effort"], validated["inference_policy"]["delegation_reasoning_effort"],
    )

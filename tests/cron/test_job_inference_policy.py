from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import traceback
from types import SimpleNamespace
from typing import Any

import pytest

POLICY = {"delegation_reasoning_effort": "high", "service_tier": "default"}
ENDPOINT = "https://chatgpt.com/backend-api/codex"


def pytest_load_initial_conftests(early_config):
    if not early_config.known_args_namespace.noconftest:
        return
    if not early_config.known_args_namespace.disable_plugin_autoload:
        raise pytest.UsageError("isolated policy tests require --disable-plugin-autoload")
    if any(name.startswith("HERMES_SESSION_") for name in os.environ):
        raise pytest.UsageError("isolated policy tests cannot inherit HERMES_SESSION_* names")
    home = tempfile.mkdtemp(prefix="synthetic-hermes-", dir=os.environ["TMPDIR"])
    os.environ["HERMES_HOME"] = os.environ["HERMES_TEST_ISOLATION"] = home
    os.environ["HERMES_DISABLE_LAZY_INSTALLS"] = "1"
    os.environ["HERMES_RUNTIME_DIR"] = str(Path(home) / "no-pm-store")


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    home = tmp_path / "synthetic-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    monkeypatch.setenv("HERMES_TEST_ISOLATION", str(home))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setattr(Path, "home", lambda: home)
    unhandled = []

    def refuse(*args, **kwargs):
        unhandled.append(" > ".join(f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}"
                                    for frame in traceback.extract_stack()[-15:-1]))
        raise AssertionError("uncontrolled network or subprocess in synthetic job test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket.socket, "sendto", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr("openai._base_client.platform_headers", lambda *a, **k: {"X-Stainless-Lang": "python"})
    import hermes_cli.venv_sync as sync
    import hermes_cli._early_recovery as recovery
    from pm import environments
    monkeypatch.setattr(sync, "prepare_launch", lambda *a, **k: None)
    monkeypatch.setattr(recovery, "restore_interrupted_pull", lambda *a: False)
    monkeypatch.setattr(recovery, "recover_if_needed", lambda *a: None)
    monkeypatch.setattr(environments, "activate_dependencies", lambda *a: None)
    from hermes_cli import version_info
    monkeypatch.setattr(version_info, "_cached_version_info", version_info.VersionInfo(
        base_version="0.0.0", derived_version="synthetic", distance=0,
        commit=None, branch=None, source="unknown"))
    import providers
    from providers.base import ProviderProfile
    monkeypatch.setattr(providers, "_discovered", True)
    for name in ("_REGISTRY", "_ALIASES", "_SOURCES", "_HOME_LAYERS"):
        monkeypatch.setattr(providers, name, {})
    monkeypatch.setattr(providers, "_refresh_home_layer", lambda *a, **k: False)
    providers.register_provider(ProviderProfile(name="openai-codex", api_mode="codex_responses", base_url=ENDPOINT))
    import tools.registry as tool_registry
    monkeypatch.setattr(tool_registry, "discover_builtin_tools", lambda *a: [])
    from hermes_cli import plugins
    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    monkeypatch.setattr(plugins, "_plugin_managers_by_home", {})
    monkeypatch.setattr(plugins, "discover_plugins", lambda *a, **k: None)
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: manager)
    monkeypatch.setattr(plugins, "_ensure_plugins_discovered", lambda **k: manager)
    import hermes_cli.left_core_migration as left_core
    monkeypatch.setattr(left_core, "recover_at_startup", lambda **k: None)
    (home / "config.yaml").write_text(
        "model:\n  default: gpt-6.1-sol\n  provider: openai-codex\n  context_length: 128000\n"
        "agent:\n  environment_probe: false\n  service_tier: priority\n  auto_recovery_cycles: 0\n"
        "delegation:\n  model: profile-other\n  provider: openrouter\n  reasoning_effort: low\n"
        "  orchestrator_enabled: true\n  max_spawn_depth: 3\n"
        "  fallback_providers:\n    - provider: openrouter\n      model: fallback-other\n",
        encoding="utf-8")
    from cron import jobs
    assert Path(jobs.__file__).resolve().parents[1] == Path(__file__).resolve().parents[2]
    yield SimpleNamespace(home=home, manager=manager)
    assert unhandled == [], "\n".join(unhandled)


def authored(**changes: Any) -> dict[str, Any]:
    return {"model": "gpt-6.1-sol", "provider": "openai-codex", "reasoning_effort": "xhigh",
            "base_url": ENDPOINT, "inference_policy": dict(POLICY), **changes}


def admitted(**changes: Any) -> dict[str, Any]:
    from agent.job_inference_policy import normalize_admitted_job
    return normalize_admitted_job(authored(**changes), provider_defaults={})


def test_owner_writer_import_and_explicit_clear(sandbox):
    from cron.jobs import create_job, get_job, update_job
    from cron.job_definition import import_job_definitions
    job = create_job("Synthetic check", "every 1h", **authored())
    saved = get_job(job["id"])
    assert saved is not None and saved["inference_policy"] == POLICY
    edited = update_job(job["id"], {"prompt": "Updated synthetic check"})
    assert edited is not None
    assert edited["reasoning_effort"] == "xhigh"
    assert edited["inference_policy"] == POLICY
    replacement = {**job, "inference_policy": {**POLICY, "delegation_reasoning_effort": "medium"}}
    import_job_definitions({job["id"]: replacement}, paused_reason="synthetic import")
    imported = get_job(job["id"])
    assert imported is not None and imported["inference_policy"]["delegation_reasoning_effort"] == "medium"
    cleared = update_job(job["id"], {"inference_policy": ""})
    assert cleared is not None
    assert "inference_policy" not in cleared
    legacy = create_job("Legacy check", "every 1h")
    assert legacy["provider"] is None
    assert "inference_policy" not in legacy


@pytest.mark.parametrize("change", [
    {"inference_policy": None}, {"inference_policy": {}},
    {"inference_policy": {**POLICY, "model": "forbidden"}},
    {"inference_policy": {**POLICY, "service_tier": "priority"}},
    {"provider": "auto"}, {"reasoning_effort": None}, {"model": ""},
])
def test_invalid_owner_definition_cannot_be_saved(change):
    from cron.jobs import create_job, list_jobs
    with pytest.raises(ValueError):
        create_job("Invalid synthetic check", "every 1h", **authored(**change))
    assert list_jobs() == []


def test_invalid_edit_is_atomic():
    from cron.jobs import create_job, get_job, update_job
    job = create_job("Atomic check", "every 1h", **authored())
    with pytest.raises(ValueError):
        update_job(job["id"], {"provider": "", "prompt": "must not be saved"})
    saved = get_job(job["id"])
    assert saved is not None and saved["prompt"] == "Atomic check"


def test_cli_parser_roundtrip_and_null_refusal():
    from hermes_cli.subcommands.cron import build_cron_parser
    from hermes_cli.cron import _job_api_kwargs
    parser = argparse.ArgumentParser()
    build_cron_parser(parser.add_subparsers(), cmd_cron=lambda a: None)
    args = parser.parse_args(["cron", "create", "every 1h", "Synthetic", "--model", "gpt-6.1-sol",
                              "--provider", "openai-codex", "--reasoning-effort", "xhigh",
                              "--inference-policy", json.dumps(POLICY)])
    assert _job_api_kwargs(args)["inference_policy"] == POLICY
    edit = parser.parse_args(["cron", "edit", "synthetic", "--inference-policy", ""])
    assert _job_api_kwargs(edit)["inference_policy"] == ""
    edit.inference_policy = "null"
    with pytest.raises(ValueError, match="null is invalid"):
        _job_api_kwargs(edit)


def test_receipt_parser_is_strict_and_detached():
    from agent.job_inference_policy import normalize_admitted_job, read_admitted_inference, JobInferenceError
    job = authored(provider="OPENAI-CODEX", model="openai/gpt-6.1-sol", base_url=ENDPOINT.upper() + "/")
    frozen = normalize_admitted_job(job, provider_defaults={})
    pin = read_admitted_inference(frozen)
    assert pin is not None
    assert (pin.route.provider, pin.route.model, pin.route.endpoint, pin.effort, pin.descendant().effort) == (
        "openai-codex", "gpt-6.1-sol", "https://chatgpt.com/BACKEND-API/CODEX", "xhigh", "high")
    job["inference_policy"]["delegation_reasoning_effort"] = "low"
    assert pin.descendant().descendant().effort == "high"
    with pytest.raises(JobInferenceError):
        read_admitted_inference(authored(base_url=None))
    with pytest.raises(JobInferenceError):
        read_admitted_inference(authored(inference_policy=None))
    assert read_admitted_inference({"model": "legacy"}) is None


def test_receipt_intent_version_and_pure_decoder(monkeypatch):
    from agent.job_inference_policy import normalize_admitted_job, read_admitted_inference, JobInferenceError
    explicit = admitted()
    omitted = admitted(base_url=None)
    assert "base_url" not in omitted and explicit["base_url"] == ENDPOINT
    with pytest.raises(JobInferenceError, match="cannot be authored"):
        normalize_admitted_job(authored(_inference_intent_version=1), provider_defaults={})
    for version in (None, True, 0, 2, "1"):
        with pytest.raises(JobInferenceError, match="ambiguous endpoint intent"):
            read_admitted_inference({**explicit, "_inference_intent_version": version})
    with pytest.raises(JobInferenceError, match="ambiguous endpoint intent"):
        read_admitted_inference(authored())
    with pytest.raises(JobInferenceError, match="must omit"):
        read_admitted_inference({**omitted, "base_url": None})

    def forbidden(*a, **kw):
        pytest.fail("pure decoder consulted runtime/config/auth")

    monkeypatch.setattr("hermes_cli.config.load_config", forbidden)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", forbidden)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", forbidden)
    monkeypatch.setattr("hermes_cli.auth._load_auth_store", forbidden)
    monkeypatch.setattr("hermes_cli.auth._load_global_auth_store", forbidden)
    pin = read_admitted_inference(omitted)
    assert pin is not None
    assert (pin.route.provider, pin.route.model, pin.route.endpoint, pin.main_effort,
            pin.descendant().effort) == ("openai-codex", "gpt-6.1-sol", None, "xhigh", "high")
    assert read_admitted_inference({"model": "legacy"}) is None


def test_frozen_named_route_does_not_consult_later_profile(monkeypatch):
    from agent.job_inference_policy import normalize_admitted_job, read_admitted_inference, JobInferenceError
    endpoint = "https://synthetic.example.invalid/v1"
    frozen = normalize_admitted_job(authored(provider="custom:synthetic", base_url=endpoint), provider_defaults={
        "providers": {"synthetic": {"base_url": endpoint, "models": ["gpt-6.1-sol"]}}})
    monkeypatch.setattr("hermes_cli.config.load_config", lambda **kw: pytest.fail("frozen route read current profile"))
    pin = read_admitted_inference(frozen)
    assert pin is not None
    assert (pin.route.provider, pin.route.model, pin.route.endpoint) == ("custom:synthetic", "gpt-6.1-sol", endpoint)
    pin.require_route("custom:synthetic", "gpt-6.1-sol", endpoint + "/")
    for provider in ("auto", "custom", "openai-codex"):
        with pytest.raises(JobInferenceError):
            pin.require_route(provider, "gpt-6.1-sol", endpoint)


def test_inference_admission_inherits_native_profile_endpoint(monkeypatch):
    from agent.job_inference_policy import normalize_admitted_job
    from agent.secret_scope import set_secret_scope, reset_secret_scope
    from hermes_cli import runtime_provider

    config = {"model": {"provider": "openai-codex", "base_url": "https://proxy.example.invalid/v1"}}
    entry = SimpleNamespace(runtime_base_url=ENDPOINT, runtime_api_key="synthetic-not-a-credential", source="synthetic")
    pool = SimpleNamespace(provider="openai-codex", has_credentials=lambda: True, select=lambda **kw: entry)
    monkeypatch.setattr(runtime_provider, "load_pool", lambda provider: pool)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda **kw: config)
    token = set_secret_scope({})
    try:
        native = runtime_provider.resolve_runtime_provider(requested="openai-codex", target_model="gpt-6.1-sol")
        assert (native["provider"], native["base_url"]) == ("openai-codex", "https://proxy.example.invalid/v1")
        frozen = normalize_admitted_job(authored(base_url=None), provider_defaults=config)
        assert (frozen["provider"], frozen["model"], frozen.get("base_url")) == (
            "openai-codex", "gpt-6.1-sol", None)
        assert "base_url" not in frozen and frozen["_inference_intent_version"] == 1
    finally:
        reset_secret_scope(token)


def test_inference_native_status_is_not_runtime_selection(monkeypatch):
    from agent import credential_pool
    from agent.secret_scope import set_secret_scope, reset_secret_scope
    from hermes_cli import auth, runtime_provider

    config = {"model": {"provider": "openai-codex", "base_url": "https://profile.example.invalid/v1"},
              "credential_pool_strategies": {"openai-codex": "least_used"}}
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda **kw: config)
    monkeypatch.setattr(auth, "_load_auth_store", lambda *a, **kw: pytest.fail("unexpected auth-store read"))
    monkeypatch.setattr(auth, "_load_global_auth_store", lambda *a, **kw: pytest.fail("unexpected root auth-store read"))
    entries = [credential_pool.PooledCredential.from_dict("openai-codex", {
        "id": "synthetic-first", "label": "synthetic-first", "priority": 0, "source": "manual",
        "access_token": "synthetic-first-key", "auth_type": "api_key", "request_count": 10,
        "base_url": "https://first-pool.example.invalid/v1"}),
        credential_pool.PooledCredential.from_dict("openai-codex", {
            "id": "synthetic-second", "label": "synthetic-second", "priority": 1, "source": "manual",
            "access_token": "synthetic-second-key", "auth_type": "api_key", "request_count": 0,
            "base_url": "https://second-pool.example.invalid/v1"})]
    token = set_secret_scope({})
    try:
        with runtime_provider.frozen_runtime_config(config):
            pool = credential_pool.CredentialPool("openai-codex", entries)
            monkeypatch.setattr(credential_pool, "load_pool", lambda provider: pool)
            monkeypatch.setattr(runtime_provider, "load_pool", lambda provider: pool)
            status = auth.get_codex_auth_status()
            assert status["base_url"] == "https://first-pool.example.invalid/v1"
            native = runtime_provider.resolve_runtime_provider(requested="openai-codex", target_model="gpt-6.1-sol")
            assert native["base_url"] == "https://second-pool.example.invalid/v1"
    finally:
        reset_secret_scope(token)


def make_agent(**changes: Any):
    from run_agent import AIAgent
    options: dict[str, Any] = dict(api_key="synthetic-not-a-credential", base_url=ENDPOINT, provider="openai-codex",
                   model="gpt-6.1-sol", api_mode="codex_responses", platform="cron", enabled_toolsets=[],
                   skip_context_files=True, skip_memory=True, skip_background_review=True, quiet_mode=True)
    options.update(changes)
    if options.get("job_inference") is not None and "native_runtime" not in options:
        options["native_runtime"] = {key: options[key] for key in ("provider", "base_url", "api_mode", "api_key")}
    return AIAgent(**options)


@pytest.fixture
def native_binding(sandbox, monkeypatch):
    import httpx
    from agent import credential_pool
    from hermes_cli import auth, runtime_provider
    from agent import process_bootstrap
    from tests.agent.test_job_inference_wire import response_stream

    config = {"model": {"provider": "openai-codex", "default": "gpt-6.1-sol", "context_length": 128000,
                        "base_url": "https://proxy.example.invalid/v1"},
              "agent": {"environment_probe": False, "service_tier": "priority"},
              "cron": {"preflight": False}, "platform_toolsets": {"cron": []},
              "credential_pool_strategies": {"openai-codex": "least_used"}}
    sent, loads, persisted = [], [], []
    state = SimpleNamespace(response_factory=None)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda **kw: config)
    monkeypatch.setattr(auth, "_load_auth_store", lambda *a, **kw: {})
    monkeypatch.setattr(auth, "_load_global_auth_store", lambda *a, **kw: {})
    monkeypatch.setattr(credential_pool, "_load_auth_store", lambda *a, **kw: {})
    monkeypatch.setattr(credential_pool, "read_credential_pool", lambda *a: list(persisted))

    def persist(provider, rows, **kw):
        persisted[:] = rows
        return rows

    monkeypatch.setattr(credential_pool, "persist_pool_entries", persist)

    @contextmanager
    def metadata(url, **kw):
        yield httpx.Response(200, request=httpx.Request("GET", url), json={"data": [
            {"id": "gpt-6.1-sol", "context_length": 128000}]})

    monkeypatch.setattr("agent.model_metadata_http.stream", metadata)
    monkeypatch.setattr("agent.models_dev._fetch_models_dev_from_network", lambda **kw: ({"synthetic": {"models": {}}}, ""))

    def outgoing(http_request):
        content = http_request.content
        if http_request.headers.get("content-encoding") == "zstd":
            from compression import zstd
            content = zstd.decompress(content)
        sent.append({"url": str(http_request.url), "authorization": http_request.headers.get("authorization"),
                     "body": json.loads(content), "headers": dict(http_request.headers)})
        event = state.response_factory(sent[-1]["body"]) if state.response_factory is not None else None
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=response_stream(event))

    monkeypatch.setattr(process_bootstrap, "_get_shared_transport", lambda *a: httpx.MockTransport(outgoing))
    monkeypatch.setattr("cron.scheduler._init_cron_mcp_tools", lambda *a: None)
    monkeypatch.setattr(credential_pool, "load_pool", lambda *a: pytest.fail("independent pool reload"))

    def install(rows, provider="openai-codex"):
        pool = credential_pool.CredentialPool(provider, [
            credential_pool.PooledCredential.from_dict(provider, {
                "id": label, "label": label, "priority": index, "source": "manual", "auth_type": "api_key",
                "access_token": key, "base_url": url, "request_count": count})
            for index, (label, key, url, count) in enumerate(rows)])

        def load(provider):
            loads.append(provider)
            return pool

        monkeypatch.setattr(runtime_provider, "load_pool", load)
        return pool

    return SimpleNamespace(config=config, sent=sent, install=install, loads=loads, persisted=persisted, state=state)


def construct_native(job, config):
    from agent.job_inference_policy import normalize_admitted_job
    from cron import scheduler
    from run_agent import AIAgent
    admitted_job = normalize_admitted_job({"enabled_toolsets": [], **job}, provider_defaults=config)
    jc = scheduler._load_cron_job_config(admitted_job, "synthetic", "Synthetic native binding")
    setup = scheduler._resolve_cron_agent_setup(admitted_job, "synthetic", "Synthetic native binding", jc)
    return scheduler._construct_cron_agent(AIAgent, admitted_job, jc.cfg, setup,
        workdir=None, session_id=None, session_db=None), admitted_job


def test_native_tool_search_assembly_and_dispatch_before_first_send(native_binding, sandbox):
    import model_tools
    import tools.memory_tool
    from tools.registry import registry
    from tools.tool_search import BRIDGE_TOOL_NAMES
    from hermes_cli.runtime_provider import frozen_runtime_config
    from agent.codex_runtime import run_codex_stream
    from tests.agent.test_job_inference_wire import built_request
    env = native_binding
    env.install([("A", "synthetic-A", ENDPOINT, 0)])
    env.config["tools"] = {"tool_search": {"enabled": "on"}}
    (sandbox.home / "config.yaml").write_text(json.dumps({**env.config, "tools": {"tool_search": {"enabled": "off"}}}))
    name, toolset = "mcp_policy_lookup", "mcp-policy-proof"
    calls = []

    def lookup(args, **kw):
        calls.append(args)
        return json.dumps({"found": args["subject"]})

    registry.register(name=name, toolset=toolset, handler=lookup, schema={
        "name": name, "description": "Look up a synthetic policy subject", "parameters": {
            "type": "object", "properties": {"subject": {"type": "string"}}, "required": ["subject"]}})
    enabled = ["memory", toolset]
    try:
        with frozen_runtime_config(env.config):
            agent, _ = construct_native(authored(base_url=None, enabled_toolsets=enabled), env.config)
            try:
                names = {td["function"]["name"] for td in agent.tools}
                assert env.sent == []
                assert BRIDGE_TOOL_NAMES | {"memory"} <= names
                assert name not in names
                found = json.loads(model_tools.handle_function_call("tool_search",
                    {"queries": ["policy subject"]}, enabled_toolsets=enabled))
                assert name in found["tools"]
                result = json.loads(model_tools.handle_function_call("tool_call",
                    {"name": name, "arguments": {"subject": "synthetic"}}, enabled_toolsets=enabled))
                assert result == {"found": "synthetic"} and calls == [{"subject": "synthetic"}]
                assert env.sent == [] and env.loads == ["openai-codex"]
                run_codex_stream(agent, built_request(agent), client=agent.client)
                wire_names = {td["name"] for td in env.sent[0]["body"]["tools"]}
                assert {"hermes_tool_search", "tool_describe", "tool_call", "memory"} <= wire_names
                assert "tool_search" not in wire_names
                assert name not in wire_names
                assert (env.sent[0]["url"], env.sent[0]["authorization"]) == (
                    "https://proxy.example.invalid/v1/responses", "Bearer synthetic-A")
            finally:
                agent.close()
    finally:
        registry.deregister(name)


@pytest.mark.parametrize("case,expected_url,expected_key", [
    ("profile", "https://proxy.example.invalid/v1", "synthetic-A"),
    ("mismatch", ENDPOINT, "synthetic-A"),
    ("least_used", "https://second.example.invalid/v1", "synthetic-B"),
    ("scoped_env", "https://env.example.invalid/v1", "synthetic-B"),
    ("explicit", ENDPOINT + "/authored", "synthetic-explicit"),
    ("alias", "https://proxy.example.invalid/v1", "synthetic-A"),
])
def test_native_binding_precedence_on_physical_request(native_binding, monkeypatch, case, expected_url, expected_key):
    from hermes_cli import runtime_provider
    from agent.secret_scope import set_secret_scope, reset_secret_scope
    from agent.codex_runtime import run_codex_stream
    from tests.agent.test_job_inference_wire import built_request
    env = native_binding
    rows = [("A", "synthetic-A", ENDPOINT, 0)]
    if case in {"least_used", "scoped_env"}:
        rows = [("A", "synthetic-A", ENDPOINT, 10),
                ("B", "synthetic-B", "https://second.example.invalid/v1", 0)]
    pool = env.install(rows)
    if case == "mismatch":
        env.config["model"]["provider"] = "openrouter"
    if case == "explicit":
        monkeypatch.setattr(runtime_provider, "resolve_codex_runtime_credentials", lambda: {
            "api_key": "synthetic-explicit", "base_url": "https://credential.example.invalid/v1"})
    token = set_secret_scope({"HERMES_CODEX_BASE_URL": "https://env.example.invalid/v1"} if case == "scoped_env" else {})
    try:
        with runtime_provider.frozen_runtime_config(env.config):
            agent, job = construct_native(authored(base_url=expected_url if case == "explicit" else None,
                provider="chatgpt-codex" if case == "alias" else "openai-codex"), env.config)
            try:
                client = agent._create_request_openai_client(reason="synthetic-rebuild")
                run_codex_stream(agent, built_request(agent), client=client)
                assert [(r["url"], r["authorization"]) for r in env.sent] == [
                    (expected_url + "/responses", "Bearer " + expected_key)]
                assert (env.sent[0]["body"]["model"], env.sent[0]["body"]["reasoning"]["effort"],
                        env.sent[0]["body"].get("service_tier")) == ("gpt-6.1-sol", "xhigh", None)
                assert job.get("base_url") == (expected_url if case == "explicit" else None)
                assert env.loads == ([] if case == "explicit" else ["openai-codex"])
                assert agent._credential_pool is (None if case == "explicit" else pool)
                if case in {"least_used", "scoped_env"}:
                    assert agent._credential_pool_entry_id == "B"
                    assert {e.id: e.request_count for e in pool.entries()} == {"A": 10, "B": 1}
            finally:
                agent.close()
    finally:
        reset_secret_scope(token)


@pytest.mark.parametrize("mode", ["codex_responses", "chat_completions", "codex_app_server"])
def test_native_named_provider_qualified_transport(native_binding, mode):
    from hermes_cli.runtime_provider import frozen_runtime_config
    from agent.job_inference_policy import JobInferenceError
    from agent.codex_runtime import run_codex_stream
    from tests.agent.test_job_inference_wire import built_request
    env = native_binding
    env.install([])
    endpoint = "https://named.example.invalid/v1"
    env.config["model"].update(provider="custom:synthetic", base_url=endpoint, api_mode=mode)
    env.config["providers"] = {"synthetic": {"base_url": endpoint, "api_key": "synthetic-named-key", "api_mode": mode}}
    with frozen_runtime_config(env.config):
        if mode != "codex_responses":
            with pytest.raises(JobInferenceError, match="qualified"):
                construct_native(authored(provider="synthetic", base_url=None), env.config)
            assert env.sent == []
            return
        agent, job = construct_native(authored(provider="synthetic", base_url=None), env.config)
        try:
            run_codex_stream(agent, built_request(agent), client=agent.client)
            assert (job["provider"], job.get("base_url")) == ("custom:synthetic", None)
            assert (agent.provider, agent.requested_provider) == ("custom", "custom:synthetic")
            assert [(r["url"], r["authorization"], r["body"]["model"], r["body"]["reasoning"]["effort"])
                    for r in env.sent] == [(endpoint + "/responses", "Bearer synthetic-named-key", "gpt-6.1-sol", "xhigh")]
        finally:
            agent.close()


def test_native_scoped_env_is_construction_time_authority(native_binding, monkeypatch):
    from agent.job_inference_policy import read_admitted_inference
    from agent.secret_scope import set_secret_scope, reset_secret_scope, set_multiplex_context, reset_multiplex_context
    from hermes_cli.runtime_provider import frozen_runtime_config, resolve_runtime_provider
    from agent.codex_runtime import run_codex_stream
    from tests.agent.test_job_inference_wire import built_request
    env = native_binding
    env.install([("A", "synthetic-A", ENDPOINT, 0)])
    monkeypatch.setenv("HERMES_CODEX_BASE_URL", "https://process.example.invalid/v1")
    job = admitted(base_url=None)
    pin = read_admitted_inference(job)
    assert pin is not None and pin.route.endpoint is None
    with frozen_runtime_config(env.config):
        token = set_secret_scope({})
        try:
            single = resolve_runtime_provider(requested="openai-codex", target_model="gpt-6.1-sol")
            assert single["base_url"] == "https://process.example.invalid/v1"
            multiplex = set_multiplex_context(True)
            try:
                isolated = resolve_runtime_provider(requested="openai-codex", target_model="gpt-6.1-sol")
                assert isolated["base_url"] == "https://proxy.example.invalid/v1"
            finally:
                reset_multiplex_context(multiplex)
        finally:
            reset_secret_scope(token)
        for url in ("https://first-env.example.invalid/v1", "https://later-env.example.invalid/v1"):
            token = set_secret_scope({"HERMES_CODEX_BASE_URL": url})
            try:
                agent = make_agent(job_inference=pin, native_runtime=resolve_runtime_provider(
                    requested=pin.route.provider, target_model=pin.route.model))
                try:
                    run_codex_stream(agent, built_request(agent), client=agent.client)
                finally:
                    agent.close()
            finally:
                reset_secret_scope(token)
    assert [(r["url"], r["authorization"]) for r in env.sent] == [
        ("https://first-env.example.invalid/v1/responses", "Bearer synthetic-A"),
        ("https://later-env.example.invalid/v1/responses", "Bearer synthetic-A")]
    assert read_admitted_inference(job) == pin and "base_url" not in job


def test_native_pool_refresh_rotation_and_rebuild_keep_coherent_material(native_binding, monkeypatch):
    import base64
    import time
    from dataclasses import replace
    from hermes_cli import auth
    from hermes_cli.runtime_provider import frozen_runtime_config
    from agent.codex_runtime import run_codex_stream
    from tests.agent.test_job_inference_wire import built_request
    env = native_binding
    pool = env.install([("A", "synthetic-A", ENDPOINT, 0),
                        ("B", "synthetic-B", "https://second.example.invalid/v1", 10)])
    claims = base64.urlsafe_b64encode(json.dumps({"exp": time.time() + 3600}).encode()).decode().rstrip("=")
    token = "synthetic." + claims + ".signature"
    first = pool.entries()[0]
    pool._replace_entry(first, replace(first, auth_type="oauth", access_token=token, refresh_token="synthetic-refresh"))
    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", lambda access, refresh: {
        "access_token": "synthetic-refreshed", "refresh_token": "synthetic-rotated-refresh"})
    with frozen_runtime_config(env.config):
        agent, job = construct_native(authored(base_url=None), env.config)
        try:
            refreshed = pool.try_refresh_matching(credential_id="A", api_key_hint=token)
            assert refreshed is not None and refreshed.access_token == "synthetic-refreshed"
            assert agent._swap_credential(refreshed)
            run_codex_stream(agent, built_request(agent), client=agent.client)
            assert agent._swap_credential(pool.entries()[1])
            client = agent._create_request_openai_client(reason="synthetic-post-rotation")
            run_codex_stream(agent, built_request(agent), client=client)
            assert [(r["url"], r["authorization"]) for r in env.sent] == [
                ("https://proxy.example.invalid/v1/responses", "Bearer synthetic-refreshed"),
                ("https://second.example.invalid/v1/responses", "Bearer synthetic-B")]
            assert agent._credential_pool is pool and agent._credential_pool_entry_id == "B"
            assert {r["id"] for r in env.persisted} == {"A", "B"}
            assert next(r for r in env.persisted if r["id"] == "A")["refresh_token"] == "synthetic-rotated-refresh"
            assert job.get("base_url") is None
        finally:
            agent.close()


@pytest.mark.parametrize("case", ["implicit", "explicit_same", "explicit_refuse"])
def test_native_singleton_refresh_preserves_ownership_and_explicit_url(native_binding, monkeypatch, case):
    from hermes_cli import auth, runtime_provider
    from agent.codex_runtime import run_codex_stream
    from tests.agent.test_job_inference_wire import built_request
    env = native_binding
    env.install([("A", "synthetic-A", ENDPOINT, 0)])
    explicit = ENDPOINT + "/authored" if case.startswith("explicit") else None
    monkeypatch.setattr(runtime_provider, "resolve_codex_runtime_credentials", lambda: {
        "api_key": "synthetic-A", "base_url": ENDPOINT})
    with runtime_provider.frozen_runtime_config(env.config):
        agent, job = construct_native(authored(base_url=explicit), env.config)
        try:
            old_url = agent.base_url
            reads = []

            def foreign(**kw):
                reads.append(kw)
                return {"api_key": "synthetic-foreign", "base_url": old_url}

            monkeypatch.setattr(auth, "resolve_codex_runtime_credentials", foreign)
            assert agent._try_refresh_codex_client_credentials() is False
            assert reads == [{"refresh_if_expiring": False}]
            next_url = explicit if case == "explicit_same" else "https://refreshed.example.invalid/v1"

            def refresh(**kw):
                return {"api_key": "synthetic-A", "base_url": old_url} if "refresh_if_expiring" in kw else {
                    "api_key": "synthetic-refreshed", "base_url": next_url}

            monkeypatch.setattr(auth, "resolve_codex_runtime_credentials", refresh)
            assert agent._try_refresh_codex_client_credentials() is (case != "explicit_refuse")
            run_codex_stream(agent, built_request(agent), client=agent.client)
            assert [(r["url"], r["authorization"], r["body"]["reasoning"]["effort"]) for r in env.sent] == [
                ((old_url if case == "explicit_refuse" else next_url) + "/responses",
                 "Bearer synthetic-A" if case == "explicit_refuse" else "Bearer synthetic-refreshed", "xhigh")]
            assert job.get("base_url") == explicit
        finally:
            agent.close()


@pytest.mark.parametrize("field", ["api_key", "base_url", "headers", "physical_headers", "physical_body"])
def test_native_client_overrides_cannot_break_binding(native_binding, field):
    from hermes_cli.runtime_provider import frozen_runtime_config
    from agent.codex_runtime import run_codex_stream
    from tests.agent.test_job_inference_wire import built_request
    env = native_binding
    env.install([("A", "synthetic-A", ENDPOINT, 0)])
    with frozen_runtime_config(env.config):
        agent, _ = construct_native(authored(base_url=None), env.config)
        try:
            run_codex_stream(agent, built_request(agent), client=agent.client)
            if field in {"api_key", "base_url"}:
                agent._client_kwargs[field] = "synthetic-other" if field == "api_key" else "https://other.example.invalid/v1"
            if field == "headers":
                agent._client_kwargs["default_headers"] = {"authorization": "Bearer synthetic-other"}
            client = agent._create_request_openai_client(reason="synthetic-header-rebuild")
            if field.startswith("physical_"):
                def mutate(request):
                    if field == "physical_headers":
                        request.headers["authorization"] = "Bearer synthetic-other"
                    else:
                        body = json.loads(request.content)
                        body["model"] = "synthetic-other"
                        request._content = json.dumps(body).encode()

                hooks = client._client.event_hooks
                client._client.event_hooks = {**hooks, "request": [mutate, *hooks["request"]]}
            with pytest.raises(Exception):
                run_codex_stream(agent, built_request(agent), client=client)
            assert [(r["url"], r["authorization"], r["body"]["model"]) for r in env.sent] == [
                ("https://proxy.example.invalid/v1/responses", "Bearer synthetic-A", "gpt-6.1-sol")]
        finally:
            agent.close()


def test_native_constructor_applies_whole_pin_and_refuses_other_transport():
    from agent.job_inference_policy import read_admitted_inference, JobInferenceError
    pin = read_admitted_inference(admitted())
    agent = make_agent(job_inference=pin)
    try:
        assert (agent.model, agent.provider, agent.reasoning_config, agent.service_tier, agent._fallback_chain) == (
            "gpt-6.1-sol", "openai-codex", {"enabled": True, "effort": "xhigh"}, "", [])
    finally:
        agent.close()
    with pytest.raises(JobInferenceError, match="qualified"):
        make_agent(job_inference=pin, api_mode="codex_app_server")


def test_owner_cli_create_edit_clear_and_model_tool_shape(monkeypatch):
    from cron.jobs import list_jobs, get_job
    from hermes_cli import cron as cli
    from hermes_cli.subcommands.cron import build_cron_parser
    from tools import cronjob_tools
    monkeypatch.setattr(cli, "_warn_if_gateway_not_running", lambda: None)
    monkeypatch.setattr(cronjob_tools, "_gateway_liveness_notice", lambda **kw: {})
    monkeypatch.setattr(cronjob_tools, "_notify_provider_jobs_changed_safe", lambda: None)
    parser = argparse.ArgumentParser()
    build_cron_parser(parser.add_subparsers(), cmd_cron=lambda a: None)
    create = parser.parse_args(["cron", "create", "every 1h", "Synthetic owner CLI", "--paused",
        "--model", "gpt-6.1-sol", "--provider", "openai-codex", "--reasoning-effort", "xhigh",
        "--inference-policy", json.dumps(POLICY)])
    assert cli.cron_create(create) == 0
    job = list_jobs(include_disabled=True)[0]
    assert job["inference_policy"] == POLICY
    edit = parser.parse_args(["cron", "edit", job["id"], "--prompt", "Edited synthetic owner CLI"])
    assert cli.cron_edit(edit) == 0
    edited = get_job(job["id"])
    assert edited is not None and edited["inference_policy"] == POLICY
    clear = parser.parse_args(["cron", "edit", job["id"], "--inference-policy", ""])
    assert cli.cron_edit(clear) == 0
    cleared = get_job(job["id"])
    assert cleared is not None and "inference_policy" not in cleared
    parameters = cronjob_tools.CRONJOB_SCHEMA["parameters"]
    assert isinstance(parameters, dict)
    properties = parameters["properties"]
    assert "inference_policy" not in properties and "model" not in properties and "provider" not in properties
    assert "inference_policy" not in cronjob_tools._HANDLER_FORWARDED_ARGS
    rejected = json.loads(cronjob_tools.cronjob(action="create", schedule="every 1h",
        prompt="Invalid synthetic owner API", inference_policy=None))
    assert rejected["success"] is False
    assert len(list_jobs(include_disabled=True)) == 1


def test_cron_constructor_and_native_prefill_stay_profile_scoped(sandbox):
    from run_agent import AIAgent
    from cron.scheduler import _CronAgentSetup, _construct_cron_agent
    from cron.agent_config import _load_prefill_messages
    from agent.job_inference_policy import read_admitted_inference
    from agent.codex_runtime import run_codex_stream
    from tests.agent.test_job_inference_wire import built_request, controlled_client
    job = admitted(enabled_toolsets=[])
    pin = read_admitted_inference(job)
    (sandbox.home / "synthetic-prefill.json").write_text('[{"role":"user","content":"Synthetic prefill"}]')
    assert _load_prefill_messages({"prefill_messages_file": "synthetic-prefill.json"}, "synthetic", sandbox.home) == [
        {"role": "user", "content": "Synthetic prefill"}]
    setup = _CronAgentSetup(model="gpt-6.1-sol", job_inference=pin, runtime={
        "provider": "openai-codex", "base_url": ENDPOINT, "api_mode": "codex_responses",
        "api_key": "synthetic-not-a-credential"}, reasoning_config={"enabled": True, "effort": "low"})
    agent = _construct_cron_agent(AIAgent, job, {}, setup, workdir=None, session_id=None, session_db=None)
    sent = []
    try:
        with controlled_client(sent) as client:
            run_codex_stream(agent, built_request(agent), client=client)
        assert (sent[0]["body"]["model"], sent[0]["body"]["reasoning"]["effort"],
                sent[0]["body"].get("service_tier")) == ("gpt-6.1-sol", "xhigh", None)
        assert agent.job_inference == pin
    finally:
        agent.close()

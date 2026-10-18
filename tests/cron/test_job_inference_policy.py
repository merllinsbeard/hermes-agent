from __future__ import annotations

import argparse
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
        assert (frozen["provider"], frozen["model"], frozen["base_url"]) == (
            "openai-codex", "gpt-6.1-sol", "https://proxy.example.invalid/v1")
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
    return AIAgent(**(options | changes))


def test_native_constructor_applies_whole_pin_and_refuses_other_transport():
    from agent.job_inference_policy import read_admitted_inference, JobInferenceError
    pin = read_admitted_inference(authored())
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
    job = authored(enabled_toolsets=[])
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

import json

import pytest

from hermes_cli import plugin_catalog
from hermes_cli.plugins_cmd_catalog import read_catalog_sidecar


def test_unknown_install_does_not_fetch_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugins = tmp_path / "plugins"
    plugin = plugins / "unlisted"
    plugin.mkdir(parents=True)
    (plugins / ".install-metadata.json").write_text(json.dumps({
        "unlisted": {"source": "https://github.com/example/unlisted.git", "revision": "a" * 40},
    }))
    fetches = []
    monkeypatch.setattr(plugin_catalog, "fetch_live_catalog", lambda **_: fetches.append(True))

    assert read_catalog_sidecar(plugin) is None
    assert fetches == []


@pytest.mark.parametrize("source", [
    "https://github.com/NousResearch/hermes-plugin-holographic.git#main",
    "git@github.com:NousResearch/hermes-plugin-holographic.git",
])
def test_handoff_install_is_adopted_from_catalog(tmp_path, monkeypatch, source):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugins = tmp_path / "plugins"
    plugin = plugins / "holographic"
    plugin.mkdir(parents=True)
    metadata = plugins / ".install-metadata.json"
    metadata.write_text(json.dumps({"holographic": {"source": source, "revision": "A" * 40}}))
    entry = plugin_catalog.PluginCatalogEntry(
        name="holographic", repo="https://github.com/example/maintained", sha="b" * 40,
        description="", maintainer="example", tier="community",
    )
    monkeypatch.setattr(plugin_catalog, "fetch_live_catalog", lambda **_: None)
    monkeypatch.setattr(plugin_catalog, "load_catalog", lambda: [entry])

    record = read_catalog_sidecar(plugin)
    assert record == {"catalog_name": "holographic", "repo": source.split("#", 1)[0],
                      "tier": "community", "pin": "a" * 40, "sha": "a" * 40}
    assert json.loads(metadata.read_text())["holographic"]["catalog"]["name"] == "holographic"

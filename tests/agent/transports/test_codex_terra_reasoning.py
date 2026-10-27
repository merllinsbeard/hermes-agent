import pytest

from agent.transports import get_transport


@pytest.fixture
def transport():
    import agent.transports.codex
    return get_transport("codex_responses")


class TestTerraReasoning:
    def test_terra_6_1_max_effort_reaches_the_wire(self, transport):
        kw = transport.build_kwargs(
            model="gpt-6.1-terra",
            messages=[{"role": "user", "content": "Route check"}],
            tools=[],
            base_url="https://chatgpt.com/backend-api/codex",
            is_codex_backend=True,
            reasoning_config={"enabled": True, "effort": "max"},
        )

        assert kw["reasoning"]["effort"] == "max"

    def test_terra_6_1_explicit_none_is_sent_verbatim(self, transport):
        kw = transport.build_kwargs(
            model="gpt-6.1-terra",
            messages=[{"role": "user", "content": "Route check"}],
            tools=[],
            base_url="https://chatgpt.com/backend-api/codex",
            is_codex_backend=True,
            reasoning_config={"enabled": False, "effort": "none"},
        )

        assert kw["reasoning"]["effort"] == "none"

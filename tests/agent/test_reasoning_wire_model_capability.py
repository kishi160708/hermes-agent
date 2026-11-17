"""A route marked supports_reasoning: false must never receive a reasoning field.

NVIDIA minimax-m3.1-flash rejects every reasoning shape with HTTP 422
("input rejected by content safety screening"), including a disable
(reasoning_effort=none). The custom provider still emitted a field for a
disable, so Hermes fell back to another model. Model capabilities are the
authoritative signal: omit the fields entirely.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.fixture()
def stub_agent():
    def _make(provider="custom", model="nvidia/minimaxai/minimax-m3.1-flash", cfg=None):
        return SimpleNamespace(
            provider=provider,
            model=model,
            reasoning_config=cfg if cfg is not None else {"enabled": True, "effort": "ultra"},
        )

    return _make


def _patch_caps(monkeypatch, caps):
    import agent.models_dev as models_dev

    monkeypatch.setattr(models_dev, "get_model_capabilities", lambda *a, **k: caps)


def test_non_reasoning_model_omits_reasoning_fields(monkeypatch, stub_agent):
    from agent.chat_completion_helpers import _reasoning_config_for_wire

    _patch_caps(monkeypatch, SimpleNamespace(supports_reasoning=False))
    agent = stub_agent(cfg={"enabled": True, "effort": "ultra"})

    assert _reasoning_config_for_wire(agent) is None
    assert agent._wire_reasoning_config is None


def test_non_reasoning_model_omits_reasoning_even_when_disabled(monkeypatch, stub_agent):
    """A disable is still a field upstream; it must be omitted too."""
    from agent.chat_completion_helpers import _reasoning_config_for_wire

    _patch_caps(monkeypatch, SimpleNamespace(supports_reasoning=False))
    agent = stub_agent(cfg={"enabled": False, "effort": "none"})

    assert _reasoning_config_for_wire(agent) is None


def test_reasoning_model_keeps_config(monkeypatch, stub_agent):
    from agent.chat_completion_helpers import _reasoning_config_for_wire

    _patch_caps(monkeypatch, SimpleNamespace(supports_reasoning=True))
    agent = stub_agent(cfg={"enabled": True, "effort": "high"})

    assert _reasoning_config_for_wire(agent) == {"enabled": True, "effort": "high"}
    assert agent._wire_reasoning_config == {"enabled": True, "effort": "high"}


def test_unknown_capabilities_keep_config(monkeypatch, stub_agent):
    """Unknown model (no catalog entry, no override) must keep the old behavior."""
    from agent.chat_completion_helpers import _reasoning_config_for_wire

    _patch_caps(monkeypatch, None)
    agent = stub_agent(cfg={"enabled": True, "effort": "high"})

    assert _reasoning_config_for_wire(agent) == {"enabled": True, "effort": "high"}


def test_aux_call_omits_reasoning_for_non_reasoning_model(monkeypatch):
    """Aux calls (title generation, summaries) must not ship a reasoning field either."""
    import agent.auxiliary_client as aux

    _patch_caps(monkeypatch, SimpleNamespace(supports_reasoning=False))
    kwargs = aux._build_call_kwargs(
        provider="custom",
        model="nvidia/minimaxai/minimax-m3.1-flash",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_config={"enabled": True, "effort": "high"},
        base_url="http://127.0.0.1:20129/v1",
        task="title_generation",
    )

    assert "reasoning_effort" not in kwargs
    assert kwargs.get("_reasoning_config") is None
    assert "reasoning" not in (kwargs.get("extra_body") or {})


def test_aux_call_drops_task_reasoning_for_non_reasoning_model(monkeypatch):
    """A task-level reasoning block is dropped too, not just the config argument."""
    import agent.auxiliary_client as aux

    _patch_caps(monkeypatch, SimpleNamespace(supports_reasoning=False))
    kwargs = aux._build_call_kwargs(
        provider="custom",
        model="nvidia/minimaxai/minimax-m3.1-flash",
        messages=[{"role": "user", "content": "hi"}],
        extra_body={"reasoning": {"effort": "high"}},
        base_url="http://127.0.0.1:20129/v1",
        task="title_generation",
    )

    assert "reasoning_effort" not in kwargs
    assert "reasoning" not in (kwargs.get("extra_body") or {})

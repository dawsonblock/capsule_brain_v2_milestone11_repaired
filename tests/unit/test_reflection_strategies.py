import pytest

from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.llm.gateway import LLMGateway
from capsule_brain.llm.models import LLMResult
from capsule_brain.llm.providers.base import LLMProvider
from capsule_brain.memory.service import MemoryService
from capsule_brain.memory.sqlite_repository import SQLiteMemoryRepository
from capsule_brain.reflection.repository import ReflectionRepository
from capsule_brain.reflection.service import ReflectionService
from capsule_brain.reflection.strategies import (
    CodeSyntaxStrategy,
    DefaultReflectionStrategy,
    OperatorFeedbackStrategy,
    PytestFailureStrategy,
    StrategyContext,
    select_strategy,
)


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------

def test_select_strategy_syntax_error():
    strategy = select_strategy(
        "verification",
        {"verifier": "python_syntax", "summary": "SyntaxError: invalid syntax"},
    )
    assert isinstance(strategy, CodeSyntaxStrategy)


def test_select_strategy_pytest_failure():
    strategy = select_strategy(
        "verification",
        {"verifier": "pytest", "summary": "assertion failed"},
    )
    assert isinstance(strategy, PytestFailureStrategy)


def test_select_strategy_operator_feedback():
    strategy = select_strategy("feedback", {})
    assert isinstance(strategy, OperatorFeedbackStrategy)


def test_select_strategy_default_for_unknown_source():
    strategy = select_strategy("goal", {})
    assert isinstance(strategy, DefaultReflectionStrategy)


def test_select_strategy_explicit_override():
    strategy = select_strategy(
        "verification",
        {"strategy": "operator_feedback"},
    )
    assert isinstance(strategy, OperatorFeedbackStrategy)


def test_select_strategy_unknown_explicit_falls_back():
    strategy = select_strategy("verification", {"strategy": "nonexistent"})
    assert isinstance(strategy, DefaultReflectionStrategy)


# ---------------------------------------------------------------------------
# CodeSyntaxStrategy fast-path
# ---------------------------------------------------------------------------

class SyntaxRepairProvider(LLMProvider):
    """Returns valid Python on the second call so we exercise the loop."""
    name = "fake"

    def __init__(self):
        self.calls = 0

    async def generate(self, request, model_cfg):
        self.calls += 1
        if self.calls == 1:
            # First attempt: still broken.
            return LLMResult(
                text="def f(:\n    pass",
                model="fake",
                provider="fake",
                latency_ms=1,
            )
        # Second attempt: valid Python.
        return LLMResult(
            text="def f():\n    pass",
            model="fake",
            provider="fake",
            latency_ms=1,
        )


@pytest.mark.asyncio
async def test_code_syntax_strategy_fast_path_resolves_on_compile(tmp_path):
    bus = LocalEventBus()
    await bus.start()
    memory = MemoryService(
        bus,
        repository=SQLiteMemoryRepository(str(tmp_path / "memory.sqlite")),
        cfg={"auto_index_embeddings": False},
    )
    await memory.start()

    gateway = LLMGateway(
        {
            "default_model": "fake",
            "models": {
                "fake": {
                    "provider": "fake",
                    "model_name": "fake",
                    "capabilities": ["text"],
                }
            },
            "routing": {"routes": {"reflection": {"models": ["fake"]}}},
        },
        providers={"fake": SyntaxRepairProvider()},
    )
    await gateway.start()

    service = ReflectionService(
        bus,
        gateway,
        memory,
        cfg={"max_iterations": 4},
        repository=ReflectionRepository(str(tmp_path / "reflections.sqlite")),
    )
    await service.start()

    session = await service.reflect(
        seed="def f(:\n    pass",
        source="verification",
        metadata={"verifier": "python_syntax", "summary": "SyntaxError"},
    )

    assert session.stop_reason == "resolved"
    assert len(session.iterations) == 2  # broken -> fixed
    # The fast-path strategy uses 1 LLM call per iteration (no critique/evaluate).
    assert session.iterations[0].critique == "syntax_error_fast_path"

    await service.stop()
    await gateway.stop()
    await memory.stop()
    await bus.stop()


# ---------------------------------------------------------------------------
# PytestFailureStrategy
# ---------------------------------------------------------------------------

class PytestRepairProvider(LLMProvider):
    """Returns a revision that the self-check marks RESOLVED."""
    name = "fake"

    def __init__(self):
        self.calls = 0

    async def generate(self, request, model_cfg):
        self.calls += 1
        # Odd calls = repair, even calls = self-check.
        if self.calls % 2 == 1:
            return LLMResult(
                text="def add(a, b):\n    return a + b",
                model="fake",
                provider="fake",
                latency_ms=1,
            )
        return LLMResult(
            text="RESOLVED: the revision makes the assertion pass.",
            model="fake",
            provider="fake",
            latency_ms=1,
        )


@pytest.mark.asyncio
async def test_pytest_strategy_resolves(tmp_path):
    bus = LocalEventBus()
    await bus.start()
    memory = MemoryService(
        bus,
        repository=SQLiteMemoryRepository(str(tmp_path / "memory.sqlite")),
        cfg={"auto_index_embeddings": False},
    )
    await memory.start()

    gateway = LLMGateway(
        {
            "default_model": "fake",
            "models": {
                "fake": {
                    "provider": "fake",
                    "model_name": "fake",
                    "capabilities": ["text"],
                }
            },
            "routing": {"routes": {"reflection": {"models": ["fake"]}}},
        },
        providers={"fake": PytestRepairProvider()},
    )
    await gateway.start()

    service = ReflectionService(
        bus,
        gateway,
        memory,
        cfg={"max_iterations": 4},
        repository=ReflectionRepository(str(tmp_path / "reflections.sqlite")),
    )
    await service.start()

    session = await service.reflect(
        seed="def add(a, b):\n    return a - b\n\nassert add(1, 2) == 3",
        source="verification",
        metadata={
            "verifier": "pytest",
            "summary": "assertion failed: add(1, 2) == 3",
            "failure_details": {"assertion_diff": "assert 3 == 3"},
        },
    )

    assert session.stop_reason == "resolved"
    assert len(session.iterations) == 1
    # 2 LLM calls per iteration (repair + self-check), not 3.
    assert session.iterations[0].critique == "pytest_assertion_repair"

    await service.stop()
    await gateway.stop()
    await memory.stop()
    await bus.stop()


# ---------------------------------------------------------------------------
# Default strategy still works (regression)
# ---------------------------------------------------------------------------

class ScriptedProvider(LLMProvider):
    name = "fake"

    def __init__(self):
        self.calls = 0

    async def generate(self, request, model_cfg):
        self.calls += 1
        responses = [
            "The thought lacks a measurable next step.",
            "Add a measurable next step and verify the outcome.",
            "RESOLVED: the revision addresses the missing next step.",
        ]
        return LLMResult(
            text=responses[self.calls - 1],
            model="fake",
            provider="fake",
            latency_ms=1,
        )


@pytest.mark.asyncio
async def test_default_strategy_still_used_for_operator_feedback(tmp_path):
    bus = LocalEventBus()
    await bus.start()
    memory = MemoryService(
        bus,
        repository=SQLiteMemoryRepository(str(tmp_path / "memory.sqlite")),
        cfg={"auto_index_embeddings": False},
    )
    await memory.start()

    gateway = LLMGateway(
        {
            "default_model": "fake",
            "models": {
                "fake": {
                    "provider": "fake",
                    "model_name": "fake",
                    "capabilities": ["text"],
                }
            },
            "routing": {"routes": {"reflection": {"models": ["fake"]}}},
        },
        providers={"fake": ScriptedProvider()},
    )
    await gateway.start()

    service = ReflectionService(
        bus,
        gateway,
        memory,
        cfg={"max_iterations": 4},
        repository=ReflectionRepository(str(tmp_path / "reflections.sqlite")),
    )
    await service.start()

    session = await service.reflect(
        seed="Improve this plan.",
        source="operator",  # not feedback/verification -> default strategy
    )

    assert session.stop_reason == "resolved"
    assert len(session.iterations) == 1

    health = await service.health()
    assert health.details["strategy_runs"].get("default") == 1

    await service.stop()
    await gateway.stop()
    await memory.stop()
    await bus.stop()

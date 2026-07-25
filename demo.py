"""Live demo of Capsule Brain v2.

Runs the full pipeline against a fake LLM provider (no API key required):
  1. Start the application via build_application()
  2. Operator sends a chat message -> assistant response with provenance
  3. Operator gives negative feedback -> reflection triggered
  4. Verification failure -> reflection triggered
  5. Goal request -> decomposed into tasks
  6. Print health + persisted state from every SQLite store

Run:  python demo.py
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from capsule_brain.events.models import EventEnvelope
from capsule_brain.llm.models import LLMRequest, LLMResult
from capsule_brain.llm.providers.base import LLMProvider
from capsule_brain.runtime.bootstrap import build_application


class FakeProvider(LLMProvider):
    """Deterministic provider so the demo needs no network/API key."""

    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, request: LLMRequest, model_cfg: dict) -> LLMResult:
        self.calls += 1
        # Structured (JSON) requests get a valid goal-decomposition object.
        if request.response_format == "json":
            text = json.dumps(
                {"tasks": ["Define success criteria", "Write the code", "Run the tests"]}
            )
        elif "THOUGHT:" in request.prompt:
            # Reflection revise step — return something stable so the
            # duplicate-revision stop condition triggers quickly.
            text = "Revised plan: add input validation and a retry loop."
        elif "rigorous critic" in (request.system or ""):
            text = "The plan ignores failure cases."
        elif "Evaluate whether" in (request.system or ""):
            text = "RESOLVED"
        else:
            text = (
                "Hello from Capsule Brain. I can decompose goals, verify "
                "outputs, and reflect on failures."
            )
        return LLMResult(
            text=text,
            model=model_cfg["model_name"],
            provider=self.name,
            latency_ms=12.5,
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )


def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def config(tmp: Path) -> dict:
    return {
        "memory": {"db_path": str(tmp / "memory.sqlite")},
        "memory_consolidator": {"enable": False},
        "llm_gateway": {
            "enable": True,
            "default_model": "fake",
            "models": {
                "fake": {
                    "provider": "fake",
                    "model_name": "fake-model-v1",
                    "capabilities": ["text", "json"],
                    "input_cost_per_million": 0.5,
                    "output_cost_per_million": 1.5,
                }
            },
        },
        "conversation": {
            "enable": True,
            "db_path": str(tmp / "conversation.sqlite"),
        },
        "feedback": {"enable": True},
        "reflection": {
            "enable": True,
            "db_path": str(tmp / "reflection.sqlite"),
            "max_iterations": 3,
        },
        "verification": {
            "enable": True,
            "db_path": str(tmp / "verification.sqlite"),
        },
        "execution": {"enable": False},
        "goal_planner": {"db_path": str(tmp / "goals.sqlite")},
        "redis_bridge": {"enable": False},
    }


async def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="capsule_demo_"))
    print(f"Working directory: {tmp}")

    app = build_application(config(tmp), llm_providers={"fake": FakeProvider()})
    bus = app.services.require("event_bus")

    # Collect every event so we can print the trace at the end.
    events_log: list[tuple[str, dict]] = []

    def make_collector(et: str):
        async def handler(event: EventEnvelope) -> None:
            events_log.append((et, dict(event.payload)))
        return handler

    interesting = [
        "conversation.response.created",
        "gui.response",
        "feedback.recorded",
        "verification.failed",
        "reflection.completed",
        "goal.created",
        "memory.created",
    ]
    for et in interesting:
        bus.subscribe(et, make_collector(et))

    banner("STEP 1 — Start the application")
    await app.start()
    health = await app.services.health()
    for name in sorted(health):
        h = health[name]
        print(f"  {name:22s} state={h.state.value:10s} healthy={h.healthy}")

    banner("STEP 2 — Operator sends a chat message")
    await bus.publish(
        EventEnvelope(
            event_type="conversation.message.create",
            source="operator",
            payload={"text": "What can you do?", "conversation_id": "demo-conv-1"},
        )
    )
    # Let async handlers drain.
    await asyncio.sleep(0.05)
    print(f"  FakeProvider calls so far: {app.services.require('llm_gateway').providers['fake'].calls}")

    banner("STEP 3 — Operator gives negative feedback on that response")
    # Find the response id from the events we captured.
    response_id = None
    for et, payload in events_log:
        if et == "conversation.response.created":
            response_id = payload["response_id"]
            break
    print(f"  Target response_id: {response_id}")
    await bus.publish(
        EventEnvelope(
            event_type="feedback.submit",
            source="operator",
            payload={
                "response_id": response_id,
                "classification": "negative",
                "text": "Too vague.",
            },
        )
    )
    await asyncio.sleep(0.05)

    banner("STEP 4 — Verification failure (empty subject)")
    await bus.publish(
        EventEnvelope(
            event_type="verification.request",
            source="operator",
            payload={"subject": "", "source": "demo", "metadata": {}},
        )
    )
    await asyncio.sleep(0.05)

    banner("STEP 5 — Goal request")
    await bus.publish(
        EventEnvelope(
            event_type="goal.request",
            source="operator",
            payload={"text": "Ship a verified Python module"},
        )
    )
    await asyncio.sleep(0.05)

    banner("STEP 6 — Event trace (in order)")
    for et, payload in events_log:
        summary = {}
        for k in ("response_id", "text", "classification", "status",
                  "session_id", "stop_reason", "id", "tasks"):
            if k in payload:
                v = payload[k]
                if isinstance(v, str) and len(v) > 60:
                    v = v[:57] + "..."
                summary[k] = v
        print(f"  {et:32s} {summary}")

    banner("STEP 7 — Persisted state")
    memory = app.services.require("memory")
    memories = await memory.recent(limit=20)
    print(f"  Memory records: {len(memories)}")
    for m in memories[:8]:
        print(f"    [{m.type.value:9s}] protected={int(m.protected)} "
              f"{m.text[:55]}")

    goal_planner = app.services.require("goal_planner")
    goals = await goal_planner.list_goals()
    print(f"\n  Goals: {len(goals)}")
    for g in goals:
        print(f"    {g['text']}")
        for t in g["tasks"]:
            print(f"      - [{t['status']}] {t['text']}")

    gateway = app.services.require("llm_gateway")
    usage = gateway.usage.totals
    print(f"\n  LLM usage: requests={usage.requests} "
          f"input_tokens={usage.input_tokens} "
          f"output_tokens={usage.output_tokens} "
          f"est_cost=${usage.estimated_cost_usd:.6f}")

    banner("STEP 8 — Shutdown")
    await app.stop()
    print("  All services stopped cleanly.")
    print(f"\nDemo artifacts left in: {tmp}")


if __name__ == "__main__":
    asyncio.run(main())

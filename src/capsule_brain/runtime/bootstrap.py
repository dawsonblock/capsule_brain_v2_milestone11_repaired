from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable

from capsule_brain.core.goal_planner_v2 import GoalPlannerV2
from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.memory.consolidation import MemoryConsolidator
from capsule_brain.memory.service import MemoryService
from capsule_brain.observability.tracing import Tracer, set_default_tracer
from capsule_brain.runtime.application import CapsuleApplication

Decomposer = Callable[[str], Awaitable[list[str]]]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and validate basic consistency.

    This is the canonical config-loading entry point. It catches the most
    common configuration error — enabling LLM-dependent services without the
    LLM gateway — before runtime startup.
    """
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict[str, Any]) -> None:
    """Validate configuration consistency before building the application."""
    llm_enabled = cfg.get("llm_gateway", {}).get("enable", False)

    # LLM-dependent services that are explicitly enabled require the LLM gateway.
    for service in ("conversation", "feedback", "reflection"):
        service_cfg = cfg.get(service, {})
        if service_cfg.get("enable", False) and not llm_enabled:
            raise ValueError(
                f"{service}.enable is true but llm_gateway.enable is false. "
                f"Either enable llm_gateway or set {service}.enable to false."
            )

    # Host execution requires explicit unsafe opt-in
    exec_cfg = cfg.get("execution", {})
    if (
        exec_cfg.get("enable", False)
        and exec_cfg.get("runner", "container") == "host"
        and not exec_cfg.get("unsafe_allow_host_execution", False)
    ):
        raise ValueError(
            "execution.runner is 'host' but unsafe_allow_host_execution is not "
            "set. Host execution is dangerous — set unsafe_allow_host_execution: "
            "true to acknowledge the risk, or use runner: container."
        )


async def default_decomposer(goal_text: str) -> list[str]:
    return [
        f"Define success criteria for: {goal_text}",
        f"Identify dependencies and constraints for: {goal_text}",
        f"Execute the first measurable step for: {goal_text}",
    ]


def build_application(
    cfg: dict[str, Any] | None = None,
    *,
    decomposer: Decomposer | None = None,
    llm_providers: dict | None = None,
) -> CapsuleApplication:
    cfg = dict(cfg or {})
    app = CapsuleApplication(cfg)

    # Initialize distributed tracing. When opentelemetry is installed and a
    # tracer provider is configured, spans are exported to OTel-compatible
    # backends (Jaeger, Phoenix, etc.). Otherwise the tracer is a no-op with
    # zero overhead. Tracing is always safe to enable — it degrades gracefully.
    tracing_cfg = cfg.get("tracing", {}) or {}
    if tracing_cfg.get("enable", False):
        tracer = Tracer.otel(
            service_name=str(tracing_cfg.get("service_name", "capsule-brain"))
        )
    else:
        tracer = Tracer()
    set_default_tracer(tracer)

    bus = LocalEventBus(cfg.get("event_bus", {}))
    app.services.register(bus)

    memory = MemoryService(bus, cfg.get("memory", {}))
    app.services.register(memory, requires=["event_bus"])

    memory_cfg = cfg.get("memory_consolidator", {})
    if memory_cfg.get("enable", True):
        consolidator = MemoryConsolidator(memory, memory_cfg)
        app.services.register(consolidator, requires=["memory"])

    resolved_decomposer = decomposer
    llm_cfg = cfg.get("llm_gateway", {})
    gateway = None

    if llm_cfg.get("enable", False):
        from capsule_brain.llm.gateway import LLMGateway
        from capsule_brain.llm.goal_decomposer import StructuredGoalDecomposer
        from capsule_brain.llm.tools import ToolRegistry

        gateway = LLMGateway(llm_cfg, providers=llm_providers)
        app.services.register(gateway)
        resolved_decomposer = StructuredGoalDecomposer(
            gateway,
            model=llm_cfg.get("goal_model"),
        )
        # Tool registry for native function-calling. Registered unconditionally
        # when the LLM gateway is enabled so services can register tools at
        # startup. The registry starts empty — services add tools via
        # registry.register(ToolSpec(...), handler).
        tool_registry = ToolRegistry()
        app.services.register(tool_registry)

    # ExperienceStore is a standalone SQLite store with no LLM dependency.
    # Register it unconditionally so feedback/conversation can rely on it
    # regardless of whether the LLM gateway is enabled.
    from capsule_brain.learning.experience_store import ExperienceStore

    learning_cfg = cfg.get("learning", {})
    experience_store = ExperienceStore(
        db_path=learning_cfg.get("db_path", "data/experience_v2.sqlite"),
        cfg=learning_cfg,
    )
    app.services.register(experience_store)

    conversation_cfg = cfg.get("conversation", {})
    conversation = None
    # Conversation defaults to enabled when LLM is available, disabled when
    # it is not. Explicitly enabling conversation without LLM is an error.
    if conversation_cfg.get("enable", gateway is not None):
        if gateway is None:
            raise RuntimeError(
                "ConversationService requires llm_gateway.enable=true"
            )
        from capsule_brain.conversation.service import ConversationService

        conversation = ConversationService(
            event_bus=bus,
            memory=memory,
            llm=gateway,
            cfg=conversation_cfg,
            experience_store=experience_store,
        )
        app.services.register(
            conversation,
            requires=["event_bus", "memory", "llm_gateway", "experience_store"],
        )

    feedback_cfg = cfg.get("feedback", {})
    # Feedback defaults to enabled when conversation is available.
    if feedback_cfg.get("enable", conversation is not None):
        if conversation is None:
            raise RuntimeError(
                "FeedbackService requires conversation.enable=true"
            )
        from capsule_brain.learning.feedback_service import FeedbackService

        feedback = FeedbackService(
            event_bus=bus,
            memory=memory,
            conversations=conversation.repository,
            experience_store=experience_store,
            cfg=feedback_cfg,
        )
        app.services.register(
            feedback,
            requires=["event_bus", "memory", "conversation", "experience_store"],
        )

    reflection_cfg = cfg.get("reflection", {})
    # Reflection defaults to enabled when LLM is available.
    if reflection_cfg.get("enable", gateway is not None):
        if gateway is None:
            raise RuntimeError(
                "ReflectionService requires llm_gateway.enable=true"
            )
        from capsule_brain.reflection.service import ReflectionService

        reflection = ReflectionService(
            event_bus=bus,
            llm=gateway,
            memory=memory,
            cfg=reflection_cfg,
            experience_store=experience_store,
            conversation_repository=(
                conversation.repository if conversation else None
            ),
        )
        app.services.register(
            reflection,
            requires=["event_bus", "memory", "llm_gateway"],
        )

    execution_cfg = cfg.get("execution", {})
    execution = None
    if execution_cfg.get("enable", False):
        from capsule_brain.execution.service import ExecutionService

        runner = None
        if execution_cfg.get("runner", "container") == "container":
            from capsule_brain.execution.container_runner import (
                ContainerExecutionRunner,
            )
            from capsule_brain.execution.models import ExecutionPolicy

            policy = ExecutionPolicy(
                allow=bool(execution_cfg.get("allow", False)),
                timeout_s=float(execution_cfg.get("timeout_s", 10.0)),
                max_output_chars=int(
                    execution_cfg.get("max_output_chars", 20000)
                ),
                allowed_commands=tuple(
                    execution_cfg.get(
                        "allowed_commands",
                        ["python", "pytest"],
                    )
                ),
                cwd_root=str(
                    execution_cfg.get("cwd_root", "sandbox")
                ),
            )
            runner = ContainerExecutionRunner(
                policy,
                engine=str(
                    execution_cfg.get("container_engine", "docker")
                ),
                image=str(
                    execution_cfg.get("container_image", "python:3.11-slim")
                ),
                memory=str(
                    execution_cfg.get("container_memory", "512m")
                ),
                memory_swap=str(
                    execution_cfg.get("container_memory_swap", "512m")
                ),
                cpus=str(
                    execution_cfg.get("container_cpus", "1.0")
                ),
                pids_limit=int(
                    execution_cfg.get("container_pids_limit", 128)
                ),
                nofile_limit=int(
                    execution_cfg.get("container_nofile_limit", 1024)
                ),
                pin_image_digest=bool(
                    execution_cfg.get(
                        "container_pin_image_digest", True
                    )
                ),
            )

        # Bounded async worker pool. When max_workers > 0, execution jobs are
        # queued and dispatched to at most max_workers concurrent runners,
        # preventing Docker daemon CPU/RAM thrashing under heavy multi-task
        # load. Set worker_pool.max_workers to 0 to disable the pool and use
        # direct runner.run() calls (the original behavior).
        worker_pool = None
        pool_cfg = execution_cfg.get("worker_pool", {}) or {}
        if int(pool_cfg.get("max_workers", 2)) > 0:
            from capsule_brain.execution.models import ExecutionPolicy
            from capsule_brain.execution.runner import ExecutionRunner
            from capsule_brain.execution.worker_pool import (
                ExecutionWorkerPool,
            )

            # The runner was already created above for the container case.
            # For the host case (runner is None), create a default runner
            # wrapping the same policy the ExecutionService will use.
            pool_runner = runner or ExecutionRunner(
                ExecutionPolicy(
                    allow=bool(execution_cfg.get("allow", False)),
                    timeout_s=float(execution_cfg.get("timeout_s", 10.0)),
                    max_output_chars=int(
                        execution_cfg.get("max_output_chars", 20000)
                    ),
                    allowed_commands=tuple(
                        execution_cfg.get(
                            "allowed_commands",
                            ["python", "pytest"],
                        )
                    ),
                    cwd_root=str(execution_cfg.get("cwd_root", "sandbox")),
                )
            )
            worker_pool = ExecutionWorkerPool(pool_runner, pool_cfg)

        execution = ExecutionService(
            event_bus=bus,
            cfg=execution_cfg,
            runner=runner,
            worker_pool=worker_pool,
        )
        app.services.register(execution, requires=["event_bus"])

    verification_cfg = cfg.get("verification", {})
    if verification_cfg.get("enable", True):
        from capsule_brain.verification.service import VerificationService

        verification = VerificationService(
            event_bus=bus,
            cfg=verification_cfg,
            execution_service=execution,
        )
        requires = ["event_bus"]
        if execution is not None:
            requires.append("execution")
        app.services.register(verification, requires=requires)

    goal_planner = GoalPlannerV2(
        event_bus=bus,
        decomposer=resolved_decomposer or default_decomposer,
        cfg=cfg.get("goal_planner", {}),
    )
    requirements = ["event_bus"]
    if llm_cfg.get("enable", False):
        requirements.append("llm_gateway")
    app.services.register(goal_planner, requires=requirements)

    # WorkflowService (DAG engine). Sits on top of the event bus and other
    # services. The default plan->generate->test->reflect workflow is
    # registered when enabled; it gracefully degrades when LLM/execution
    # services are unavailable.
    workflow_cfg = cfg.get("workflow", {}) or {}
    if workflow_cfg.get("enable", False):
        from capsule_brain.workflow.builtins import (
            build_plan_generate_test_reflect_workflow,
        )
        from capsule_brain.workflow.runner import WorkflowRunnerService

        workflow_runner = WorkflowRunnerService(
            event_bus=bus,
            cfg=workflow_cfg,
        )
        app.services.register(workflow_runner, requires=["event_bus"])

        # Register the default workflow. The workflow builder reads services
        # from the registry at execution time (not build time) so it sees
        # whatever services are currently registered.
        default_wf = build_plan_generate_test_reflect_workflow(
            services=app.services,
            max_iterations=int(
                workflow_cfg.get("max_reflect_iterations", 3)
            ),
            require_approval=bool(
                workflow_cfg.get("require_approval", False)
            ),
        )
        workflow_runner.register_workflow(default_wf)

    redis_cfg = cfg.get("redis_bridge", {})
    if redis_cfg.get("enable", False):
        from capsule_brain.events.redis_bridge import RedisBridge
        app.services.register(RedisBridge(bus, redis_cfg), requires=["event_bus"])

    return app

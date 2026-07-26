from __future__ import annotations

import logging
from typing import Any

from capsule_brain.llm.gateway import LLMGateway
from capsule_brain.llm.models import LLMRequest

from .models import WorkflowState
from .runner import Workflow, WorkflowNode

log = logging.getLogger(__name__)

# Cap reflect->generate loops so a stuck workflow cannot loop forever.
# The runner's max_steps is the hard cap; this is the soft cap specific to
# the plan->generate->test->reflect cycle.
DEFAULT_MAX_REFLECT_ITERATIONS = 3


def _safe_get(services: Any, name: str) -> Any:
    """Fetch a service from a registry/dict, returning None if absent."""
    if services is None:
        return None
    if hasattr(services, "get"):
        return services.get(name)
    if isinstance(services, dict):
        return services.get(name)
    return None


def build_plan_generate_test_reflect_workflow(
    *,
    services: Any,
    max_iterations: int = DEFAULT_MAX_REFLECT_ITERATIONS,
    require_approval: bool = False,
) -> Workflow:
    """Build the default Plan -> Generate -> Test -> Reflect workflow.

    Nodes:
    - ``plan``: ask the LLM to decompose the goal into a concrete plan.
    - ``generate``: ask the LLM to produce code from the plan.
    - ``test``: run the code via ExecutionService (if enabled) and capture
      stdout/stderr/exit_code into state.
    - ``approve``: optional human-in-the-loop checkpoint. Pauses before
      completing so an operator can review the generated code. Enabled when
      ``require_approval=True``.
    - ``reflect``: if the test failed and iterations remain, loop back to
      ``generate`` with the failure context. Otherwise, complete.

    The workflow degrades gracefully when services are unavailable:
    - No LLMGateway: plan/generate/reflect nodes record a stub in state.
    - No ExecutionService: test node skips execution and marks the code as
      "untested" (treated as passed so the workflow can complete).

    Args:
        services: a ServiceRegistry or dict providing ``llm_gateway`` and
            ``execution`` services. Either may be absent.
        max_iterations: max reflect->generate loops before giving up.
        require_approval: if True, insert an approval checkpoint before
            completion.
    """
    llm = _safe_get(services, "llm_gateway")
    execution = _safe_get(services, "execution")

    async def plan_node(state: WorkflowState) -> WorkflowState:
        if llm is None or not llm.is_running:
            state.plan = f"[offline] Plan steps for: {state.goal}"
            return state
        try:
            result = await llm.generate(
                LLMRequest(
                    prompt=(
                        f"Decompose this goal into a concise, actionable plan "
                        f"for code generation. Output the plan only.\n\n"
                        f"GOAL: {state.goal}"
                    ),
                    temperature=0.2,
                    system=(
                        "You are a planning module for an autonomous coding "
                        "agent. Produce a short, concrete plan."
                    ),
                ),
                route="coding",
            )
            state.plan = result.text.strip()
        except Exception as exc:
            log.warning("plan node LLM call failed: %s", exc)
            state.plan = f"[plan failed: {exc}] {state.goal}"
        return state

    async def generate_node(state: WorkflowState) -> WorkflowState:
        if llm is None or not llm.is_running:
            state.code = f"# offline mode: no code generated for {state.goal}"
            return state
        try:
            prompt = (
                f"PLAN:\n{state.plan}\n\n"
                f"GOAL:\n{state.goal}\n\n"
            )
            if state.errors:
                prompt += (
                    f"PREVIOUS ERRORS (fix these):\n"
                    f"{chr(10).join(state.errors[-3:])}\n\n"
                )
            prompt += "Output ONLY the Python code, no markdown fences."
            result = await llm.generate(
                LLMRequest(
                    prompt=prompt,
                    temperature=0.0,
                    system=(
                        "You are a code generation module. Output only "
                        "valid, runnable Python code."
                    ),
                ),
                route="coding",
            )
            state.code = result.text.strip()
        except Exception as exc:
            log.warning("generate node LLM call failed: %s", exc)
            state.code = f"# generate failed: {exc}"
        return state

    async def test_node(state: WorkflowState) -> WorkflowState:
        if execution is None or not execution.is_running:
            # No execution service: mark as untested and treat as a soft
            # pass so the workflow can complete without a reflect loop.
            state.stdout = ""
            state.stderr = "[execution unavailable: code not tested]"
            state.extra["exit_code"] = 0
            state.extra["test_passed"] = True
            return state
        try:
            from capsule_brain.execution.models import ExecutionRequest

            # Write the code to a temp file in the sandbox cwd_root and run
            # it with the configured interpreter. The execution service's
            # policy controls what commands are allowed.
            import os
            import tempfile

            cwd_root = execution.policy.cwd_root
            os.makedirs(cwd_root, exist_ok=True)
            tmp = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
                dir=cwd_root,
                prefix="wf_",
            )
            tmp.write(state.code)
            tmp.close()

            allowed = execution.policy.allowed_commands
            # Pick the first allowed python-like interpreter.
            interpreter = next(
                (c for c in allowed if "python" in c.lower()),
                "python",
            )
            request = ExecutionRequest(
                command=[interpreter, os.path.basename(tmp.name)],
                cwd=cwd_root,
                source="workflow",
                metadata={"run_id": state.run_id},
            )
            result = await execution.execute(request)
            state.stdout = result.stdout
            state.stderr = result.stderr
            # Stash the exit code in extra for the conditional edge.
            state.extra["exit_code"] = result.exit_code
            state.extra["test_passed"] = bool(result.passed)
        except Exception as exc:
            log.warning("test node execution failed: %s", exc)
            state.stderr = f"[test execution failed: {exc}]"
            state.extra["test_passed"] = False
        return state

    def test_passed(state: WorkflowState) -> bool:
        return bool(state.extra.get("test_passed", False))

    def after_test(state: WorkflowState) -> str | None:
        """Conditional edge after the test node.

        - If the test passed (or execution was unavailable), go to approve
          (if configured) or complete.
        - If the test failed and we have iterations left, go to reflect.
        - If the test failed and we're out of iterations, go to complete
          with the failure recorded.
        """
        if test_passed(state):
            return "approve" if require_approval else "complete"
        if state.iterations < max_iterations:
            return "reflect"
        return "complete"

    async def reflect_node(state: WorkflowState) -> WorkflowState:
        state.iterations += 1
        # Record the failure context for the next generate iteration.
        failure = (
            f"Iteration {state.iterations}: test failed. "
            f"stderr: {state.stderr[:500]}"
        )
        state.errors.append(failure)
        if llm is not None and llm.is_running:
            try:
                result = await llm.generate(
                    LLMRequest(
                        prompt=(
                            f"GOAL:\n{state.goal}\n\n"
                            f"PLAN:\n{state.plan}\n\n"
                            f"CODE:\n{state.code}\n\n"
                            f"STDERR:\n{state.stderr[:1000]}\n\n"
                            f"Produce a corrected version of the code. "
                            f"Output ONLY the code."
                        ),
                        temperature=0.0,
                        system=(
                            "You are a reflection module. Analyze the test "
                            "failure and output corrected code."
                        ),
                    ),
                    route="reflection",
                )
                state.code = result.text.strip()
            except Exception as exc:
                log.warning("reflect node LLM call failed: %s", exc)
                state.errors.append(f"reflect failed: {exc}")
        return state

    def after_reflect(state: WorkflowState) -> str:
        # Always loop back to test after reflect. The after_test edge will
        # decide whether to reflect again or complete.
        return "test"

    async def approve_node(state: WorkflowState) -> WorkflowState:
        # The checkpoint guard on this node handles the pause. The action
        # itself is a no-op; the runner parks before transitioning to
        # ``complete``.
        return state

    def approve_checkpoint(state: WorkflowState) -> bool:
        # Always pause at the approve node so the operator can review.
        return True

    async def complete_node(state: WorkflowState) -> WorkflowState:
        # Terminal node. No work to do; the runner marks the run completed.
        return state

    nodes = [
        WorkflowNode(
            "plan",
            plan_node,
            next_node="generate",
            description="Decompose the goal into a plan",
        ),
        WorkflowNode(
            "generate",
            generate_node,
            next_node="test",
            description="Generate code from the plan",
        ),
        WorkflowNode(
            "test",
            test_node,
            next_node=after_test,
            description="Run the generated code and capture results",
        ),
        WorkflowNode(
            "reflect",
            reflect_node,
            next_node=after_reflect,
            description="Reflect on test failure and produce a fix",
        ),
        WorkflowNode(
            "approve",
            approve_node,
            next_node="complete",
            checkpoint=approve_checkpoint if require_approval else None,
            description="Human-in-the-loop approval checkpoint",
        ),
        WorkflowNode(
            "complete",
            complete_node,
            next_node=None,
            description="Terminal node",
        ),
    ]

    return Workflow(
        name="plan_generate_test_reflect",
        nodes=nodes,
        entry="plan",
        description=(
            "Plan -> Generate -> Test -> (Pass? -> Approve? -> Complete | "
            "Fail? -> Reflect -> loop to Test)"
        ),
    )

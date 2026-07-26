from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from capsule_brain.llm.gateway import LLMGateway
from capsule_brain.llm.models import LLMRequest

from .models import ReflectionIteration, ReflectionSession


@dataclass(slots=True)
class StrategyContext:
    """Inputs passed to every reflection strategy.

    Strategies read ``seed`` and ``metadata`` to decide how to prompt the
    model, and write their iterations back into ``session``.
    """

    seed: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    session: ReflectionSession | None = None


class ReflectionStrategy(ABC):
    """A pluggable reflection strategy.

    The default ``ReflectionService`` loop runs critique -> revise -> evaluate
    regardless of failure type. Strategies let us specialize that loop:
    - Code syntax failures get a fast-path single repair call.
    - Pytest failures get test-driven repair prompts with assertion diffs.
    - Operator feedback gets the full multi-perspective reasoning loop.

    Each strategy returns the list of iterations it produced and the final
    text. The service is responsible for persisting the session and writing
    the reflection memory.
    """

    name: str = "default"

    @abstractmethod
    async def run(
        self,
        llm: LLMGateway,
        ctx: StrategyContext,
        *,
        max_iterations: int,
        route: str | None,
        model: str | None,
    ) -> tuple[list[ReflectionIteration], str, str]:
        """Run the strategy.

        Returns ``(iterations, final_text, stop_reason)``.
        """
        raise NotImplementedError


class DefaultReflectionStrategy(ReflectionStrategy):
    """The original critique -> revise -> evaluate loop.

    Preserved verbatim so existing behavior is unchanged when no specialized
    strategy matches. Used for operator feedback, goal reasoning, and any
    trigger without a specialized strategy.
    """

    name = "default"

    async def run(
        self,
        llm: LLMGateway,
        ctx: StrategyContext,
        *,
        max_iterations: int,
        route: str | None,
        model: str | None,
    ) -> tuple[list[ReflectionIteration], str, str]:
        current = ctx.seed
        prior_revisions: set[str] = {_normalize(ctx.seed)}
        iterations: list[ReflectionIteration] = []

        for idx in range(max_iterations):
            critique = await self._critique(llm, current, route=route, model=model)
            revision = await self._revise(llm, current, critique, route=route, model=model)
            evaluation = await self._evaluate(
                llm, current, revision, critique, route=route, model=model
            )
            resolved = _is_resolved(evaluation)
            iterations.append(
                ReflectionIteration(
                    index=idx,
                    critique=critique,
                    revision=revision,
                    evaluation=evaluation,
                    resolved=resolved,
                )
            )
            normalized = _normalize(revision)
            if resolved:
                return iterations, revision, "resolved"
            if normalized in prior_revisions:
                return iterations, revision, "duplicate_revision"
            prior_revisions.add(normalized)
            current = revision
        return iterations, current, "max_iterations"

    async def _critique(self, llm, thought, *, route, model):
        result = await llm.generate(
            LLMRequest(
                model=model,
                temperature=0.2,
                system=(
                    "You are a rigorous critic. Identify the single most important "
                    "flaw, missing assumption, or next step in the thought. Be concise."
                ),
                prompt=thought,
            ),
            route=route,
        )
        return result.text.strip()

    async def _revise(self, llm, thought, critique, *, route, model):
        result = await llm.generate(
            LLMRequest(
                model=model,
                temperature=0.3,
                system=(
                    "Revise the thought using the critique. Produce a materially "
                    "improved version, not commentary about revising it."
                ),
                prompt=f"THOUGHT:\n{thought}\n\nCRITIQUE:\n{critique}",
            ),
            route=route,
        )
        return result.text.strip()

    async def _evaluate(self, llm, previous, revision, critique, *, route, model):
        result = await llm.generate(
            LLMRequest(
                model=model,
                temperature=0.0,
                system=(
                    "Evaluate whether the revision resolves the critique. "
                    "Begin with exactly RESOLVED or CONTINUE, followed by one short reason."
                ),
                prompt=(
                    f"PREVIOUS:\n{previous}\n\nCRITIQUE:\n{critique}"
                    f"\n\nREVISION:\n{revision}"
                ),
            ),
            route=route,
        )
        return result.text.strip()


class CodeSyntaxStrategy(ReflectionStrategy):
    """Fast-path repair for Python syntax errors.

    Syntax failures have a single, well-defined fix: correct the syntax. A
    full critique -> revise -> evaluate loop wastes 3 LLM calls per iteration
    on what is essentially a one-shot repair. This strategy issues a single
    repair call with a focused prompt, then validates the result by
    attempting to compile it. If the fix compiles, we stop immediately.

    This reduces token usage and latency by ~60-80% for simple syntax fixes.
    """

    name = "code_syntax"

    async def run(
        self,
        llm: LLMGateway,
        ctx: StrategyContext,
        *,
        max_iterations: int,
        route: str | None,
        model: str | None,
    ) -> tuple[list[ReflectionIteration], str, str]:
        current = ctx.seed
        iterations: list[ReflectionIteration] = []

        for idx in range(max(max_iterations, 1)):
            result = await llm.generate(
                LLMRequest(
                    model=model,
                    temperature=0.0,
                    system=(
                        "You are a Python syntax repair tool. Output ONLY the corrected "
                        "Python code with no explanation, no markdown fences, and no "
                        "commentary. Preserve the original intent. Fix the syntax error(s) "
                        "shown in the input."
                    ),
                    prompt=current,
                ),
                route=route,
            )
            revision = result.text.strip()
            # Validate by attempting to compile. If it parses, we are done.
            resolved = _compiles_clean(revision)
            iterations.append(
                ReflectionIteration(
                    index=idx,
                    critique="syntax_error_fast_path",
                    revision=revision,
                    evaluation="RESOLVED" if resolved else "CONTINUE",
                    resolved=resolved,
                )
            )
            if resolved:
                return iterations, revision, "resolved"
            # Feed the still-broken revision back for another attempt.
            current = revision
        return iterations, current, "max_iterations"


class PytestFailureStrategy(ReflectionStrategy):
    """Test-driven repair for pytest assertion failures.

    Pytest failures are not syntax errors — the code runs but produces wrong
    output. The most valuable signal is the assertion diff (expected vs
    actual). This strategy extracts the diff from the seed/metadata and
    injects it into a focused repair prompt, then re-evaluates by checking
    whether the revision plausibly addresses the failing assertion.

    Uses a 2-call loop (repair + self-check) instead of the default 3-call
    critique/revise/evaluate loop, saving ~33% of tokens per iteration.
    """

    name = "pytest_failure"

    async def run(
        self,
        llm: LLMGateway,
        ctx: StrategyContext,
        *,
        max_iterations: int,
        route: str | None,
        model: str | None,
    ) -> tuple[list[ReflectionIteration], str, str]:
        assertion_diff = _extract_assertion_diff(ctx.seed, ctx.metadata)
        current = ctx.seed
        prior_revisions: set[str] = {_normalize(ctx.seed)}
        iterations: list[ReflectionIteration] = []

        for idx in range(max(max_iterations, 1)):
            repair_prompt = self._build_repair_prompt(current, assertion_diff)
            repair = await llm.generate(
                LLMRequest(
                    model=model,
                    temperature=0.0,
                    system=(
                        "You are a test-driven code repair tool. The code below failed a "
                        "pytest assertion. Output ONLY the corrected code with no "
                        "explanation or markdown fences. Make the failing assertion pass "
                        "while preserving intent."
                    ),
                    prompt=repair_prompt,
                ),
                route=route,
            )
            revision = repair.text.strip()

            check = await llm.generate(
                LLMRequest(
                    model=model,
                    temperature=0.0,
                    system=(
                        "Does the revised code plausibly make the failing assertion pass? "
                        "Begin with exactly RESOLVED or CONTINUE, followed by one short reason."
                    ),
                    prompt=(
                        f"FAILING ASSERTION:\n{assertion_diff}\n\n"
                        f"REVISED CODE:\n{revision}"
                    ),
                ),
                route=route,
            )
            evaluation = check.text.strip()
            resolved = _is_resolved(evaluation)
            iterations.append(
                ReflectionIteration(
                    index=idx,
                    critique="pytest_assertion_repair",
                    revision=revision,
                    evaluation=evaluation,
                    resolved=resolved,
                )
            )
            normalized = _normalize(revision)
            if resolved:
                return iterations, revision, "resolved"
            if normalized in prior_revisions:
                return iterations, revision, "duplicate_revision"
            prior_revisions.add(normalized)
            current = revision
        return iterations, current, "max_iterations"

    @staticmethod
    def _build_repair_prompt(seed: str, assertion_diff: str) -> str:
        parts = []
        if assertion_diff:
            parts.append(f"FAILING ASSERTION:\n{assertion_diff}\n")
        parts.append(f"CODE / OUTPUT:\n{seed}")
        parts.append("TASK: Produce the corrected code.")
        return "\n\n".join(parts)


class OperatorFeedbackStrategy(DefaultReflectionStrategy):
    """Multi-perspective qualitative reasoning for operator feedback.

    Operator feedback is qualitative — there is no deterministic check like
    "does it compile" or "does the assertion pass". The full critique ->
    revise -> evaluate loop is the right tool. This strategy is a thin
    specialization of the default strategy that tags itself distinctly for
    metrics and tracing.
    """

    name = "operator_feedback"


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------

def select_strategy(
    source: str,
    metadata: dict[str, Any],
    *,
    strategies: dict[str, ReflectionStrategy] | None = None,
) -> ReflectionStrategy:
    """Pick a reflection strategy based on the trigger source and metadata.

    Selection rules (first match wins):
    1. If metadata explicitly requests a strategy via ``strategy`` key, use it.
    2. If source is ``verification`` and metadata indicates a syntax error,
       use CodeSyntaxStrategy.
    3. If source is ``verification`` and metadata indicates a pytest failure,
       use PytestFailureStrategy.
    4. If source is ``feedback``, use OperatorFeedbackStrategy.
    5. Otherwise, use DefaultReflectionStrategy.

    Unknown strategy names fall back to the default so a misconfigured
    metadata field never crashes reflection.
    """
    registry = strategies or _default_strategies()

    explicit = str(metadata.get("strategy", "")).strip().lower()
    if explicit and explicit in registry:
        return registry[explicit]

    source_l = source.lower()
    if source_l == "verification":
        check_details = metadata.get("check_details") or {}
        verifier = str(check_details.get("verifier") or metadata.get("verifier") or "").lower()
        summary = str(metadata.get("summary") or check_details.get("summary") or "").lower()
        # PythonSyntaxVerifier failures get the fast path.
        if "python_syntax" in verifier or "syntaxerror" in summary or "syntax error" in summary:
            return registry["code_syntax"]
        # pytest / execution-backed failures get the test-driven repair path.
        if "pytest" in verifier or "pytest" in summary or "assertion" in summary:
            return registry["pytest_failure"]
    if source_l == "feedback":
        return registry["operator_feedback"]

    return registry["default"]


def _default_strategies() -> dict[str, ReflectionStrategy]:
    return {
        "default": DefaultReflectionStrategy(),
        "code_syntax": CodeSyntaxStrategy(),
        "pytest_failure": PytestFailureStrategy(),
        "operator_feedback": OperatorFeedbackStrategy(),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_resolved(evaluation: str) -> bool:
    return evaluation.strip().upper().startswith("RESOLVED")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _compiles_clean(code: str) -> bool:
    """Return True if ``code`` compiles as Python with no SyntaxError."""
    try:
        compile(code, "<reflection>", "exec")
        return True
    except SyntaxError:
        return False
    except Exception:
        # Value errors (e.g. null bytes) are not syntax fixes we can verify.
        return False


def _extract_assertion_diff(seed: str, metadata: dict[str, Any]) -> str:
    """Extract the most useful assertion signal from the seed or metadata.

    Looks for pytest's ``assert`` lines and ``E`` diff markers in the seed,
    and falls back to a ``failure_details`` field in metadata if present.
    """
    details = metadata.get("failure_details") or metadata.get("check_details") or {}
    if isinstance(details, dict):
        diff = details.get("assertion_diff") or details.get("diff")
        if diff:
            return str(diff)
    # Scan the seed for pytest assertion lines and the E-prefixed diff lines.
    lines = seed.splitlines()
    relevant: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("assert ") or stripped.startswith("E "):
            relevant.append(stripped)
        elif "AssertionError" in stripped:
            relevant.append(stripped)
    return "\n".join(relevant[:20]) if relevant else ""

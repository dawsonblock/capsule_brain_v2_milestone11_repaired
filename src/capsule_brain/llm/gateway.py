from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

from capsule_brain.observability.tracing import get_default_tracer
from capsule_brain.runtime.service import CapsuleService, HealthStatus, ServiceState

from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from .errors import (
    LLMCapabilityError,
    LLMConfigurationError,
    LLMProviderError,
    LLMTimeoutError,
)
from .models import LLMRequest, LLMResult
from .provider_health import ProviderHealthRegistry
from .providers.base import LLMProvider
from .providers.openai_compatible import OpenAICompatibleProvider
from .routing import ModelRouter
from .telemetry import RequestTrace
from .tools import ToolCall, ToolRegistry, ToolResult
from .usage import UsageTracker


class LLMGateway(CapsuleService):
    name = "llm_gateway"

    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        providers: dict[str, LLMProvider] | None = None,
    ) -> None:
        super().__init__(cfg)
        self.models = dict(self.cfg.get("models", {}))
        self.default_model = self.cfg.get("default_model")
        self.timeout_s = float(self.cfg.get("timeout_s", 45.0))
        self.max_attempts = max(1, int(self.cfg.get("max_attempts", 2)))
        self.backoff_s = max(0.0, float(self.cfg.get("backoff_s", 0.25)))
        self.max_concurrency = max(1, int(self.cfg.get("max_concurrency", 4)))

        self.providers = providers or {"openai": OpenAICompatibleProvider()}
        self.router = ModelRouter(self.cfg.get("routing", {}))
        self.usage = UsageTracker()
        self.provider_health = ProviderHealthRegistry()

        cb_cfg = self.cfg.get("circuit_breaker", {})
        self._breaker_cfg = CircuitBreakerConfig(
            failure_threshold=max(1, int(cb_cfg.get("failure_threshold", 3))),
            recovery_timeout_s=max(
                0.1, float(cb_cfg.get("recovery_timeout_s", 30.0))
            ),
        )
        self._breakers: dict[str, CircuitBreaker] = defaultdict(
            lambda: CircuitBreaker(self._breaker_cfg)
        )

        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self._active_requests = 0
        self._completed_requests = 0
        self._last_trace: RequestTrace | None = None

    async def start(self) -> None:
        self.state = ServiceState.STARTING

        if not self.models:
            raise LLMConfigurationError("LLMGateway has no configured models")

        if not self.default_model:
            self.default_model = next(iter(self.models))

        if self.default_model not in self.models:
            raise LLMConfigurationError(
                f"Unknown default model: {self.default_model}"
            )

        for alias, model_cfg in self.models.items():
            provider_name = model_cfg.get("provider")
            if provider_name not in self.providers:
                raise LLMConfigurationError(
                    f"Model {alias!r} references unregistered provider {provider_name!r}"
                )

        self.state = ServiceState.RUNNING

    async def stop(self) -> None:
        self.state = ServiceState.STOPPING
        await asyncio.gather(
            *(provider.close() for provider in self.providers.values()),
            return_exceptions=True,
        )
        self.state = ServiceState.STOPPED

    def _resolve_model(
        self,
        alias: str,
        request: LLMRequest,
    ) -> tuple[dict[str, Any], LLMProvider]:
        if alias not in self.models:
            raise LLMConfigurationError(f"Unknown model alias: {alias}")

        cfg = self.models[alias]
        provider_name = cfg.get("provider")
        provider = self.providers.get(provider_name)

        if provider is None:
            raise LLMConfigurationError(
                f"Provider not registered: {provider_name}"
            )

        capabilities = set(cfg.get("capabilities", ["text"]))
        required = "json" if request.response_format == "json" else "text"
        if required not in capabilities:
            raise LLMCapabilityError(
                f"Model {alias} does not advertise capability: {required}"
            )
        # Tool-calling is an opt-in capability. Models that do not advertise
        # "tools" must not be sent tool schemas — the request would either be
        # rejected by the provider or silently ignored.
        if request.tools and "tools" not in capabilities:
            raise LLMCapabilityError(
                f"Model {alias} does not advertise capability: tools"
            )

        return cfg, provider

    async def _generate_with_model(
        self,
        alias: str,
        request: LLMRequest,
        trace: RequestTrace,
    ) -> LLMResult:
        model_cfg, provider = self._resolve_model(alias, request)
        breaker = self._breakers[alias]

        if breaker.is_open:
            raise LLMProviderError(f"Circuit open for model: {alias}")

        trace.model_alias = alias
        trace.provider = provider.name

        last_error: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            trace.attempts += 1

            try:
                async with self._semaphore:
                    self._active_requests += 1
                    try:
                        result = await asyncio.wait_for(
                            provider.generate(request, model_cfg),
                            timeout=self.timeout_s,
                        )
                    finally:
                        self._active_requests -= 1

                breaker.record_success()
                self.provider_health.success(provider.name)
                return replace(result, attempts=attempt)

            except asyncio.CancelledError:
                # Cancellation is control flow. Never convert it to a provider failure.
                raise

            except asyncio.TimeoutError:
                last_error = LLMTimeoutError(
                    f"LLM request exceeded {self.timeout_s:.1f}s"
                )

            except (LLMConfigurationError, LLMCapabilityError):
                raise

            except Exception as exc:
                last_error = exc

            breaker.record_failure()
            self.provider_health.failure(provider.name, last_error)

            if attempt < self.max_attempts:
                await asyncio.sleep(
                    self.backoff_s * (2 ** (attempt - 1))
                )

        if isinstance(last_error, LLMTimeoutError):
            raise last_error

        raise LLMProviderError(
            f"Model {alias!r} failed after {self.max_attempts} attempts"
        ) from last_error

    async def generate(
        self,
        request: LLMRequest,
        *,
        route: str | None = None,
    ) -> LLMResult:
        if self.state != ServiceState.RUNNING:
            raise RuntimeError("LLMGateway is not running")

        assert self.default_model is not None

        plan = self.router.resolve(route, request.model, self.default_model)
        trace = RequestTrace(route=plan.name, metadata=dict(request.metadata))
        # _last_trace is a best-effort diagnostic for health(); it is NOT the
        # authoritative provenance channel. The trace travels with the result
        # so concurrent requests cannot overwrite each other's provenance.
        self._last_trace = trace

        # OpenTelemetry span keyed by the request's correlation id (if any)
        # so LLM calls group with the originating operator event in the trace
        # waterfall.
        correlation_id = request.metadata.get("correlation_id")
        tracer = get_default_tracer()
        with tracer.span(
            "llm.generate",
            correlation_id=correlation_id,
            attributes={
                "route": plan.name,
                "requested_model": request.model or "",
                "response_format": request.response_format,
                "tool_count": len(request.tools),
            },
        ) as otel_span:
            last_error: Exception | None = None

            for index, alias in enumerate(plan.models):
                if index > 0:
                    trace.fallback_count += 1

                try:
                    result = await self._generate_with_model(alias, request, trace)
                    trace.finish()
                    self._completed_requests += 1

                    model_cfg = self.models[alias]
                    self.usage.record_success(
                        usage=result.usage,
                        model_cfg=model_cfg,
                        fallback_count=trace.fallback_count,
                    )
                    otel_span.set_attribute("model_alias", alias)
                    otel_span.set_attribute("attempts", trace.attempts)
                    otel_span.set_attribute("fallback_count", trace.fallback_count)
                    # Attach the authoritative trace to the result.
                    return replace(result, trace=trace)

                except asyncio.CancelledError:
                    trace.finish()
                    raise

                except (LLMConfigurationError, LLMCapabilityError):
                    trace.finish()
                    self.usage.record_failure()
                    raise

                except Exception as exc:
                    last_error = exc
                    continue

            trace.finish()
            self.usage.record_failure()

            if isinstance(last_error, LLMTimeoutError):
                raise last_error

            raise LLMProviderError(
                f"All models in route {plan.name!r} failed"
            ) from last_error

    async def generate_json(
        self,
        request: LLMRequest,
        *,
        route: str | None = None,
    ) -> tuple[dict[str, Any], LLMResult]:
        if request.response_format != "json":
            request = replace(request, response_format="json")

        result = await self.generate(request, route=route)

        try:
            parsed = json.loads(result.text)
        except json.JSONDecodeError as exc:
            raise LLMProviderError("Provider returned invalid JSON") from exc

        if not isinstance(parsed, dict):
            raise LLMProviderError(
                "Structured LLM response must be a JSON object"
            )

        return parsed, result

    async def generate_with_tools(
        self,
        request: LLMRequest,
        tool_registry: ToolRegistry,
        *,
        route: str | None = None,
        max_iterations: int = 4,
    ) -> tuple[LLMResult, list[ToolResult]]:
        """Run a tool-calling loop until the model stops requesting tools.

        Each iteration:
        1. Send the request (with tool schemas + any accumulated tool results)
           to the model.
        2. If the model returns tool_calls, execute them via ``tool_registry``
           and feed the results back as ``tool_results`` on the next request.
        3. Repeat until the model returns a plain text answer (no tool_calls)
           or ``max_iterations`` is reached.

        Returns the final LLMResult and the full list of ToolResults produced
        across all iterations. The final result's ``text`` is the model's
        natural-language answer; ``tool_calls`` is empty on the final result
        (any pending tool calls when max_iterations is exceeded are returned
        in the second-to-last result's tool_calls, but not executed).
        """
        if self.state != ServiceState.RUNNING:
            raise RuntimeError("LLMGateway is not running")
        if not tool_registry.specs():
            raise LLMConfigurationError(
                "generate_with_tools requires at least one registered tool"
            )
        max_iterations = max(1, int(max_iterations))

        # Replace the request's tools with the registry's full schema so the
        # model sees every available tool. We keep the caller's original
        # prompt/system/model intact.
        current_request = replace(
            request,
            tools=list(tool_registry.specs()),
            tool_results=list(request.tool_results),
        )

        all_tool_results: list[ToolResult] = []
        last_result: LLMResult | None = None

        for iteration in range(max_iterations):
            result = await self.generate(current_request, route=route)
            last_result = result

            if not result.tool_calls:
                # Model produced a final answer.
                return result, all_tool_results

            # Execute each requested tool call and collect results.
            for call in result.tool_calls:
                tool_result = await tool_registry.execute(call)
                all_tool_results.append(tool_result)
                current_request = replace(
                    current_request,
                    tool_results=current_request.tool_results
                    + [
                        {
                            "tool_call_id": tool_result.tool_call_id,
                            "name": tool_result.name,
                            "content": tool_result.content,
                        }
                    ],
                )

        # max_iterations reached with the model still requesting tools.
        # Return the last result as-is; callers can inspect last_result.tool_calls
        # to see what was not executed.
        assert last_result is not None
        return last_result, all_tool_results

    async def stream(
        self,
        request: LLMRequest,
        *,
        route: str | None = None,
    ) -> AsyncIterator[str]:
        if self.state != ServiceState.RUNNING:
            raise RuntimeError("LLMGateway is not running")

        assert self.default_model is not None
        plan = self.router.resolve(route, request.model, self.default_model)

        # Streaming fallback is only safe before the first token is emitted.
        last_error: Exception | None = None

        for alias in plan.models:
            model_cfg, provider = self._resolve_model(alias, request)
            breaker = self._breakers[alias]

            if breaker.is_open:
                last_error = LLMProviderError(
                    f"Circuit open for model: {alias}"
                )
                continue

            emitted = False

            try:
                async with self._semaphore:
                    self._active_requests += 1
                    try:
                        async with asyncio.timeout(self.timeout_s):
                            async for chunk in provider.stream(
                                request, model_cfg
                            ):
                                emitted = True
                                yield chunk
                    finally:
                        self._active_requests -= 1

                breaker.record_success()
                self.provider_health.success(provider.name)
                return

            except asyncio.CancelledError:
                raise

            except GeneratorExit:
                # Consumer stopped consuming early (e.g. UI cancelled
                # generation). The provider was producing successfully, so
                # record success — otherwise health metrics would falsely
                # show a degraded provider.
                if emitted:
                    breaker.record_success()
                    self.provider_health.success(provider.name)
                raise

            except Exception as exc:
                breaker.record_failure()
                self.provider_health.failure(provider.name, exc)
                last_error = exc

                if emitted:
                    raise LLMProviderError(
                        f"Streaming failed after output began for model {alias}"
                    ) from exc

        raise LLMProviderError("All streaming models failed") from last_error

    async def health(self) -> HealthStatus:
        usage = self.usage.totals
        breakers = {
            alias: {
                "open": breaker.is_open,
                "failures": breaker.failures,
            }
            for alias, breaker in self._breakers.items()
        }

        degraded = (
            any(item["degraded"] for item in self.provider_health.snapshot().values())
            or any(item["open"] for item in breakers.values())
        )

        state = (
            ServiceState.DEGRADED
            if self.state == ServiceState.RUNNING and degraded
            else self.state
        )

        return HealthStatus(
            state=state,
            details={
                "models": sorted(self.models),
                "default_model": self.default_model,
                "active_requests": self._active_requests,
                "completed_requests": self._completed_requests,
                "max_concurrency": self.max_concurrency,
                "providers": self.provider_health.snapshot(),
                "breakers": breakers,
                "usage": {
                    "requests": usage.requests,
                    "failures": usage.failures,
                    "fallbacks": usage.fallbacks,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "estimated_cost_usd": round(
                        usage.estimated_cost_usd, 8
                    ),
                },
                "last_trace": (
                    {
                        "request_id": self._last_trace.request_id,
                        "route": self._last_trace.route,
                        "model_alias": self._last_trace.model_alias,
                        "provider": self._last_trace.provider,
                        "attempts": self._last_trace.attempts,
                        "fallback_count": self._last_trace.fallback_count,
                        "latency_ms": round(
                            self._last_trace.latency_ms, 3
                        ),
                    }
                    if self._last_trace
                    else None
                ),
            },
        )

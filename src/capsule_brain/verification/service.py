from __future__ import annotations

from typing import Any

from capsule_brain.events.models import EventEnvelope
from capsule_brain.runtime.service import (
    CapsuleService,
    HealthStatus,
    ServiceState,
)

from .models import (
    VerificationCheck,
    VerificationResult,
    VerificationStatus,
    now,
)
from .repository import VerificationRepository
from .verifiers import (
    ConsistencyVerifier,
    JSONSyntaxVerifier,
    NonEmptyVerifier,
    PythonSyntaxVerifier,
    RequiredFieldsVerifier,
    Verifier,
)


class VerificationService(CapsuleService):
    name = "verification"

    def __init__(
        self,
        event_bus: Any,
        cfg: dict[str, Any] | None = None,
        repository: VerificationRepository | None = None,
        verifiers: list[Verifier] | None = None,
        execution_service: Any | None = None,
    ) -> None:
        super().__init__(cfg)
        self.event_bus = event_bus
        self.repository = (
            repository
            or VerificationRepository(
                self.cfg.get(
                    "db_path",
                    "data/verification_v2.sqlite",
                )
            )
        )
        self.execution_service = execution_service

        if verifiers is not None:
            self.verifiers = verifiers
        else:
            self.verifiers = [
                NonEmptyVerifier(),
                PythonSyntaxVerifier(),
                JSONSyntaxVerifier(),
                RequiredFieldsVerifier(),
                ConsistencyVerifier(),
            ]
            # When an ExecutionService is available, add execution-backed
            # verifiers so verification can actually compile/run code rather
            # than relying on static checks alone.
            if execution_service is not None:
                from .execution_verifiers import create_execution_verifiers

                exec_verifier_names = set(
                    self.cfg.get("execution_verifiers", [])
                    or ["python_compile", "pytest", "ruff", "mypy"]
                )
                self.verifiers.extend(
                    create_execution_verifiers(
                        execution_service,
                        include=exec_verifier_names,
                    )
                )

        self._unsubscribers: list = []
        self.runs = 0
        self.failures = 0

    async def start(self) -> None:
        self.state = ServiceState.STARTING
        await self.repository.start()
        self._unsubscribers.append(
            self.event_bus.subscribe(
                "verification.request",
                self._on_request,
            )
        )
        self.state = ServiceState.RUNNING

    async def stop(self) -> None:
        self.state = ServiceState.STOPPING
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers.clear()
        await self.repository.stop()
        self.state = ServiceState.STOPPED

    async def _on_request(
        self,
        event: EventEnvelope,
    ) -> None:
        try:
            await self.verify(
                subject=str(
                    event.payload.get("subject", "")
                ),
                source=str(
                    event.payload.get(
                        "source",
                        event.source,
                    )
                ),
                metadata=dict(
                    event.payload.get("metadata") or {}
                ),
            )
        except Exception as exc:
            self.failures += 1
            await self.event_bus.publish(
                EventEnvelope(
                    event_type="verification.error",
                    source=self.name,
                    payload={"error": str(exc)},
                )
            )

    async def verify(
        self,
        *,
        subject: str,
        source: str = "operator",
        metadata: dict[str, Any] | None = None,
    ) -> VerificationResult:
        self.runs += 1
        result = VerificationResult(
            subject=subject,
            source=source,
            metadata=dict(metadata or {}),
        )
        await self.repository.save(result)

        for verifier in self.verifiers:
            try:
                check = await verifier.verify(
                    subject,
                    result.metadata,
                )
            except Exception as exc:
                check = VerificationCheck(
                    verifier=verifier.name,
                    status=VerificationStatus.ERROR,
                    summary=f"Verifier error: {exc}",
                )
            result.checks.append(check)

        active = [
            check.status
            for check in result.checks
            if check.status != VerificationStatus.SKIP
        ]

        if any(
            status in {
                VerificationStatus.FAIL,
                VerificationStatus.ERROR,
            }
            for status in active
        ):
            result.status = VerificationStatus.FAIL
        elif active:
            result.status = VerificationStatus.PASS
        else:
            result.status = VerificationStatus.SKIP

        result.completed_at = now()
        await self.repository.save(result)

        failures = [
            check.summary
            for check in result.checks
            if check.status
            in {
                VerificationStatus.FAIL,
                VerificationStatus.ERROR,
            }
        ]

        event_type = (
            "verification.failed"
            if result.status == VerificationStatus.FAIL
            else "verification.completed"
        )
        await self.event_bus.publish(
            EventEnvelope(
                event_type=event_type,
                source=self.name,
                payload={
                    "verification_id": result.id,
                    "status": result.status.value,
                    "summary": (
                        " ".join(failures)
                        if failures
                        else (
                            f"Verification "
                            f"{result.status.value}."
                        )
                    ),
                    "subject": subject,
                    "source": source,
                    "metadata": result.metadata,
                },
            )
        )
        return result

    async def health(self) -> HealthStatus:
        return HealthStatus(
            state=self.state,
            details={
                "runs": self.runs,
                "failures": self.failures,
                "result_count": (
                    await self.repository.count()
                    if self.state == ServiceState.RUNNING
                    else 0
                ),
                "verifiers": [
                    verifier.name
                    for verifier in self.verifiers
                ],
            },
        )
